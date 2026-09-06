"""The map built while the site runs, rather than in a separate pass.

`vigil map build` opens every camera a second time and records for a minute.
The cameras are already open and already projecting, so every analysed frame
is a free sample of the ground — and the test that matters is that the map
actually improves over a run and that a camera which moves stops contributing
ground computed through a pose that is no longer true.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from vigil.domain.geo import CameraPose, LatLon
from vigil.kernel import native
from vigil.service.livemap import LiveMap

pytestmark = pytest.mark.skipif(not native.available(),
                                reason=f"a map needs the engine core: {native.fault()}")

POSE = CameraPose(LatLon(33.8938, 35.5018), 4.0, 0.0, -25.0, 0.0, 62.0, 36.0, 40.0)


def _frame(seed: int, size=(240, 320)) -> np.ndarray:
    """A frame with structure, so a median over several is not a flat grey."""
    rng = np.random.default_rng(seed)
    image = np.zeros((*size, 3), dtype=np.uint8)
    image[:, :, 0] = np.linspace(20, 200, size[1], dtype=np.uint8)[None, :]
    image[:, :, 1] = np.linspace(30, 180, size[0], dtype=np.uint8)[:, None]
    image[:, :, 2] = 120
    image = np.clip(image.astype(np.int16) + rng.integers(-8, 9, image.shape), 0, 255)
    return image.astype(np.uint8)


def test_the_map_fills_in_as_the_site_runs():
    live = LiveMap(POSE.position, rebuild_every_seconds=0.0)
    assert live.ground is None and "no map yet" in live.describe()

    covered = []
    for step in range(20):
        live.observe("gate", POSE, _frame(step), at_seconds=step * 3.0)
        ground = live.tick()
        if ground is not None:
            covered.append(int(ground.summary()["seen"]))
    live.close()

    assert covered, "twenty frames three seconds apart produced no map at all"
    assert covered[-1] > 0
    assert covered[-1] >= covered[0], (
        f"the map should not shrink as more is seen: {covered[0]} then {covered[-1]}")
    assert live.samples >= 15, f"only {live.samples} frames were folded in"


def test_frames_arriving_faster_than_the_sample_interval_are_skipped():
    """A hundred frames of the same half-second is one sample's worth of
    information, and folding all of them in makes the median confident about
    a moment rather than about the ground."""
    live = LiveMap(POSE.position)
    assert live.observe("gate", POSE, _frame(1), at_seconds=100.0)
    assert not live.observe("gate", POSE, _frame(2), at_seconds=100.05)
    assert live.observe("gate", POSE, _frame(3), at_seconds=200.0)
    assert live.samples == 2
    live.close()


def test_a_camera_that_moves_has_its_ground_rebuilt_rather_than_left_standing():
    """Samples projected through the old pose describe ground somewhere else.
    Discarding them is right; leaving the composite showing them is not."""
    live = LiveMap(POSE.position, rebuild_every_seconds=0.0)
    for step in range(12):
        live.observe("gate", POSE, _frame(step), at_seconds=step * 3.0)
    live.tick()
    assert live.ground is not None, "twelve frames should have produced a map"
    before = int(live.ground.summary()["seen"])
    assert before > 0

    # Re-aimed 40 degrees. Nothing it saw before is where it thought.
    moved = replace(POSE, heading=40.0)
    live.observe("gate", moved, _frame(99), at_seconds=100.0)
    ground = live.tick()
    assert ground is not None
    after = int(ground.summary()["seen"])
    assert after < before, (
        f"the old pose's ground is still being drawn: {before} cells before, {after} after")
    live.close()


def test_the_map_survives_a_restart_by_being_written(tmp_path: Path):
    from vigil.service.mapping import load_map

    live = LiveMap(POSE.position, tmp_path, rebuild_every_seconds=0.0)
    for step in range(12):
        live.observe("gate", POSE, _frame(step), at_seconds=step * 3.0)
    live.tick()
    assert live.save() is not None
    live.close()

    reloaded = load_map(tmp_path)
    assert reloaded is not None, "a map that will not load back is a map nobody has"
    assert reloaded.summary()["seen"] == live.ground.summary()["seen"]


def test_a_frame_the_map_cannot_use_never_reaches_the_camera_worker_as_an_error():
    """The map is the least important thing done with a frame and is not
    allowed to stop the rest."""
    live = LiveMap(POSE.position)
    assert live.observe("gate", POSE, np.zeros((0, 0, 3), dtype=np.uint8), at_seconds=1.0) is False
    assert live.observe("gate", POSE, None, at_seconds=2.0) is False
    live.close()


def test_the_runtime_builds_one_only_when_a_camera_is_placed(tmp_path, keychain):
    """No placement means no origin to hang a lattice on, and a map of a site
    whose cameras have no positions is a picture of nothing."""
    from vigil.adapters.detectors import MotionDetector
    from vigil.service.auth import Principal, Role
    from vigil.service.runtime import Runtime
    from vigil.service.site import SiteService
    from vigil.storage.store import Store

    admin = Principal("root", Role.ADMIN, "user")
    with Store(":memory:") as store:
        site = SiteService(store, keychain)
        runtime = Runtime(site, detector_factory=MotionDetector, map_dir=tmp_path)
        unplaced = site.add_camera("gate", "file:///x", by=admin)
        assert runtime._ensure_map([unplaced]) is None
        assert runtime.ground() is None and runtime.map_state() == "no map"

        placed = site.place_camera("gate", POSE, by=admin)
        assert runtime._ensure_map([placed]) is not None
        assert "no map yet" in runtime.map_state()
        runtime._map.close()
