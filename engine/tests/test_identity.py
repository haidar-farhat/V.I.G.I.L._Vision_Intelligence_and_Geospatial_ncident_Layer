"""Tests for the per-site identity switch and the wiring behind it.

`sentinel.faces` and `sentinel.registry` were tested and called by nothing;
`Pipeline` accepted a plate reader that nothing passed. So every assertion here
is about a *wire*: the switch on the site row, the node reading it, the plate
reader reaching the pipeline, the face engine being shown a person's box and
nothing else, a template reaching the register only when an operator names a
track. Each test is written so that removing the wire it covers fails it.

**No model exists on this machine and none may be downloaded.** Faces run on a
stand-in backend that returns vectors the test wrote — the pattern
`test_faces.py` established — and plates on a reader that records whether it
was called. The properties under test are all above the model: that it was
shown nothing while the switch was off, only a person's crop while it was on,
and that what it produced went nowhere unless somebody said so.

Most of these tests are about *not* doing something. That is the shape of a
biometric feature's failures.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from sentinel import logs
from sentinel.core import BoundingBox, CameraPose, Detection, LatLon, Track
from sentinel.detect import DetectorInfo
from sentinel.faces import (
    FRAMES_FOR_A_MATCH,
    MATCH_SIMILARITY,
    POSSIBLE_SIMILARITY,
    TEMPLATE_DIMENSIONS,
    FaceBox,
    Verdict,
)
from sentinel.node import (
    ACTOR,
    FACE_MODEL_FILES,
    FACE_STRIDE_FRAMES,
    FRAMES_KEPT_FOR_ENROLMENT,
    PLATE_MODEL_FILES,
    Node,
    NodeError,
    Update,
)
from sentinel.pipeline import FrameResult, PipelineStats
from sentinel.registry import Confidence, IdentifierKind, RegistryError, SubjectKind
from sentinel.site import DEFAULT_SITE_ID, Identity, Site
from sentinel.store import MIGRATIONS, Store

SITE = LatLon(33.8938, 35.5018)

#: Where the person is, in every synthetic frame below. On a 400x300 frame this
#: is a 100x75 crop, which is what the stand-in must be shown and all it may be
#: shown.
PERSON = BoundingBox(0.25, 0.50, 0.25, 0.25)
CROP_SHAPE = (75, 100, 3)
FRAME_SHAPE = (300, 400, 3)


@pytest.fixture(autouse=True)
def clean_logging():
    logs.reset()
    yield
    logs.reset()


@pytest.fixture
def empty_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A models directory with nothing in it, so "no models" is a fact and not
    whatever happens to be in the checkout's own folder."""
    directory = tmp_path / "models"
    directory.mkdir()
    monkeypatch.setenv("SENTINEL_MODELS_DIR", str(directory))
    return directory


# ------------------------------------------------------------ the stand-ins


def unit(*components: float) -> tuple[float, ...]:
    """A 128-float vector with these leading components and zeros after."""
    vector = [0.0] * TEMPLATE_DIMENSIONS
    for index, value in enumerate(components):
        vector[index] = value
    return tuple(vector)


def at_similarity(target: float) -> tuple[float, ...]:
    """A vector whose cosine with ``unit(1.0)`` is exactly ``target``."""
    angle = math.acos(target)
    return unit(math.cos(angle), math.sin(angle))


class StandInModels:
    """What YuNet and SFace would return, decided by the test rather than a file.

    Records a copy of every image it is handed — the copy matters, because the
    frame buffer a crop views may be reused — which is how "only inside the
    person's box" is checked: not that the crop looked right, but that the
    detector was never shown anything else. Each call to ``detect`` returns one
    face carrying the next ``(quality, vector)`` the test scripted, cycling.
    """

    def __init__(self, *faces: tuple[float, tuple[float, ...]]):
        self._faces = list(faces) or [(0.99, unit(1.0))]
        self._calls = 0
        self._pending: tuple[FaceBox, tuple[float, ...]] | None = None
        self.images: list[np.ndarray] = []
        self.embedded: list[FaceBox] = []

    @property
    def name(self) -> str:
        return "stand-in embedder"

    def detect(self, image):
        self.images.append(np.array(image, copy=True))
        score, vector = self._faces[self._calls % len(self._faces)]
        self._calls += 1
        box = FaceBox(
            x=4.0, y=4.0, w=64.0, h=64.0, score=score,
            raw=tuple([4.0, 4.0, 64.0, 64.0] + [0.0] * 11),
        )
        self._pending = (box, vector)
        return (box,)

    def embed(self, image, face):
        assert self._pending is not None and face is self._pending[0], (
            "the engine embedded a face the detector never found"
        )
        self.embedded.append(face)
        return self._pending[1]


class _PersonDetector:
    """A detector that labels one fixed box ``person`` on every frame.

    The reference scene's walkers are found by motion, which labels nothing,
    and a track nobody labelled a person must never be examined for a face. So
    the real-pipeline tests below use this instead: one person track for the
    length of the clip, at a box whose crop shape the test knows.
    """

    info = DetectorInfo(
        kind="fake", name="one-person-box", class_names={0: "person"}, classifies=True
    )

    def detect(self, image):
        return [Detection(bbox=PERSON, confidence=0.9, class_id=0)]


class _VehicleDetector:
    info = DetectorInfo(
        kind="fake", name="one-car-box", class_names={2: "car"}, classifies=True
    )

    def detect(self, image):
        return [Detection(bbox=PERSON, confidence=0.9, class_id=2)]


class _RecordingPlateReader:
    """Stands in for `PlateReader`: reads nothing, remembers being asked."""

    country = "GENERIC"

    def __init__(self):
        self.calls = 0

    def read(self, frame, vehicle_box, *, frame_index):
        self.calls += 1
        return []


class _FaceRunner:
    """Stands in for a camera whose pipeline publishes frames with images.

    Hands out the updates it was given, one per poll, in order. Everything the
    node reads from a runner is here, including the detector's labels — which
    is what decides whether a track is a person at all.
    """

    detector_info = DetectorInfo(
        kind="fake", name="labels-people",
        class_names={0: "person", 2: "car"}, classifies=True,
    )

    def __init__(self, updates):
        self._updates = list(updates)
        self.is_running = True
        self.fault = None
        self.stats = None
        self.skipped = 0
        self.analysis_fps = 30.0
        self.dropped_frames = 0
        self.reconnects = 0

    def take_latest(self):
        return self._updates.pop(0) if self._updates else None

    def take_segments(self):
        return []

    def take_events(self):
        return []

    def ask_to_stop(self):
        self.is_running = False

    def stop(self, timeout=None):
        self.is_running = False
        return True

    @property
    def seconds_since_frame(self):
        return 0.0

    @property
    def seconds_since_started(self):
        return 1.0


def bright_person_frame() -> np.ndarray:
    """A dark frame with one bright rectangle exactly where the person is."""
    frame = np.zeros(FRAME_SHAPE, dtype=np.uint8)
    frame[150:225, 100:200] = 255
    return frame


def _track(track_id: int, class_id: int, first: int, last: int) -> Track:
    return Track(
        id=track_id, class_id=class_id, bbox=PERSON, confidence=0.9, hits=3,
        first_seen_millis=first, last_seen_millis=last, position=None,
        speed_mps=None, heading_degrees=None,
    )


def _frame(
    index: int, *, tracks=(), ended=(), image="bright", first: int = 1_000,
) -> Update:
    """One published frame: ``tracks`` is ``[(track_id, class_id), ...]``."""
    last = first + index * 40
    result = FrameResult(
        index=index, timestamp_millis=last, source_id="gate",
        detections=(), ended=tuple(ended), events=(), plates=(),
        tracks=tuple(_track(tid, cid, first, last) for tid, cid in tracks),
        image=bright_person_frame() if image == "bright" else image,
    )
    return Update(result=result, analysis_fps=30.0, skipped=0, stats=PipelineStats())


def _person_frames(track_id: int, indices, **kw) -> list[Update]:
    return [_frame(i, tracks=[(track_id, 0)], **kw) for i in indices]


def _attach(node: Node, camera_id: str, runner) -> None:
    """Give a camera a stand-in runner, the way `Node.start` would.

    `start` snapshots the site's switch onto the record when it builds the
    runner; a runner attached by hand needs the same, or the record says faces
    were off when it started and the node — correctly — examines nothing.
    """
    record = node.camera(camera_id)
    record.identity = node.identity
    record.runner = runner


def _faces_on(tmp_path: Path, backend, **kw) -> Node:
    node = Node(tmp_path / "n.db", node_id="site-a", face_backend=backend, **kw)
    node.set_identity(Identity(faces=True), reason="contractor list for the test")
    return node


def _rows(node: Node, action: str):
    return [row for row in node.store.audit_trail(limit=500) if row["action"] == action]


def _enrol_from_track(node: Node, camera_id: str = "gate", track_id: int = 1, **kw) -> str:
    return node.enrol_person(
        kw.pop("name", "Ali Hassan"), camera_id, track_id,
        basis=kw.pop("basis", "staff list, consent on file"), **kw,
    )


@contextlib.contextmanager
def _listening_to_the_node(level: int = logging.WARNING):
    """Everything `node.py` logs at or above ``level``; `caplog` cannot, because
    `logs.configure` sets ``propagate=False`` on the `sentinel` tree."""
    said: list[logging.LogRecord] = []

    class Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            said.append(record)

    logger = logging.getLogger("sentinel.node")
    handler = Collect()
    handler.setLevel(level)
    was = logger.level
    logger.addHandler(handler)
    logger.setLevel(min(was, level) if was else level)
    try:
        yield said
    finally:
        logger.removeHandler(handler)
        logger.setLevel(was)


# ---------------------------------------------------------------- migration 9


def make_site(**overrides) -> Site:
    fields = dict(id=DEFAULT_SITE_ID, name="Beirut yard", origin=SITE)
    fields.update(overrides)
    return Site(**fields)


def test_the_identity_switch_and_the_declared_flag_each_carry_a_way_back():
    """Migration 9 put the switch on the site row; migration 10 added the
    `declared` flag that keeps the node's own placeholder row from freezing an
    origin nobody chose. Both must be undoable on an air-gapped machine."""
    by_version = {migration.version: migration for migration in MIGRATIONS}
    assert by_version[9].name == "site_identity"
    assert by_version[10].name == "site_declared"
    assert by_version[11].name == "camera_recording"
    assert by_version[12].name == "users"
    assert MIGRATIONS[-1].version == 12, "a newer migration arrived; check it below too"
    for migration in MIGRATIONS:
        assert migration.down.strip(), (
            f"migration {migration.version} ({migration.name}) has no way back — "
            "an upgrade with no way back is a gamble on an air-gapped machine"
        )


def test_a_site_saved_with_plates_on_reads_back_so():
    with Store(":memory:") as store:
        store.save_site(make_site(identity=Identity(plates=True)))
        restored = store.site()
        assert restored is not None
        assert restored.identity == Identity(plates=True, faces=False, face_crops=False)

        store.save_site(make_site(identity=Identity(faces=True, face_crops=True)))
        restored = store.site()
        assert restored.identity == Identity(plates=False, faces=True, face_crops=True)


def test_re_saving_a_site_updates_its_switch_rather_than_keeping_the_first():
    # The upsert has to carry the three columns, or a switch flipped after the
    # site was first saved silently stays where it was.
    with Store(":memory:") as store:
        store.save_site(make_site())
        store.save_site(make_site(identity=Identity(plates=True)))

        assert store.site().identity.plates is True
        assert len(store.sites()) == 1


def test_the_identity_columns_refuse_a_third_value():
    with Store(":memory:") as store:
        store.save_site(make_site())
        with pytest.raises(sqlite3.IntegrityError):
            store._connection.execute("UPDATE sites SET identity_faces = 2")


def test_the_site_row_survives_its_switch_and_flag_leaving_and_reads_as_off_and_declared():
    """Down twice, then up again, on a database with a site in it.

    The downs must take the flags and leave the site: a build that predates
    them cannot honour them, and losing the site's origin along with them would
    re-anchor every zone on the plan view. Coming back up, a row written before
    the columns existed — which is what the rolled-back row now is — must read
    as everything off, and as *declared*. Off is the only honest reading of
    "nobody asked" for a switch that starts biometric processing; declared is
    the safe reading for the flag, because a placeholder read as declared
    freezes an origin while a declared row read as a placeholder moves one.
    """
    with Store(":memory:") as store:
        store.save_site(make_site(identity=Identity(plates=True, faces=True)))
        before = store.applied_versions()

        # Down through everything above the switch, in order, until the
        # switch itself goes. Each migration's own down is exercised.
        undone_names = []
        while True:
            undone = store.rollback()
            assert undone is not None, "ran out of migrations before reaching site_identity"
            undone_names.append(undone.name)
            if undone.name == "site_identity":
                break
        assert undone_names == ["users", "camera_recording", "site_declared", "site_identity"]
        assert "declared" not in store.column_names("sites"), "the flag survived its own down"
        columns = store.column_names("sites")
        assert not any(c.startswith("identity_") for c in columns), (
            "the flags survived their own down"
        )
        survived = store._connection.execute("SELECT name FROM sites").fetchone()
        assert survived is not None and survived[0] == "Beirut yard", (
            "rolling back the switch took the site with it"
        )

        store.migrate()

        assert store.applied_versions() == before
        restored = store.site()
        assert restored is not None
        assert restored.name == "Beirut yard"
        assert restored.origin.lat == pytest.approx(SITE.lat)
        assert restored.identity == Identity(), (
            "a row written before the switch existed came back with something on"
        )
        assert restored.declared is True, (
            "a row written before the flag existed must read as declared: the "
            "other reading moves an origin somebody chose"
        )


def test_a_site_row_written_by_hand_with_the_old_columns_reads_as_off():
    # The same property from the other side: a row inserted with no mention of
    # the identity columns at all — what every pre-9 deployment holds.
    with Store(":memory:") as store:
        store._connection.execute(
            "INSERT INTO sites (id, name, origin_lat, origin_lon, created_at, updated_at) "
            "VALUES ('default', 'Old yard', 33.9, 35.5, 1, 1)"
        )
        restored = store.site()
        assert restored is not None
        assert restored.identity == Identity()
        assert restored.identity.describe() == "off"


# ------------------------------------------------------------- the switch


def test_the_default_site_is_never_persisted_and_anchors_on_the_first_placed_camera(
    tmp_path: Path, reference_video: Path
):
    pose = CameraPose(position=SITE, mount_height=6.0, heading=180.0, pitch=-22.0)
    with Node(tmp_path / "n.db") as node:
        assert node.site().origin == LatLon(0.0, 0.0), "no camera, so nowhere"
        node.add_camera(reference_video, camera_id="gate", pose=pose)

        site = node.site()

        assert site.id == DEFAULT_SITE_ID
        assert site.origin == SITE
        assert site.identity == Identity()
        assert node.store.site() is None, "looking at the site wrote one"


def test_set_identity_persists_the_switch_and_audits_both_states_with_the_reason(
    tmp_path: Path,
):
    reason = "contractor list agreed with the site manager, 2026-09-01"
    with Node(tmp_path / "n.db", node_id="gatehouse") as node:
        returned = node.set_identity(Identity(plates=True, faces=True), reason=reason)

        assert returned.identity == Identity(plates=True, faces=True)
        assert node.identity == Identity(plates=True, faces=True)
        assert node.site().identity == Identity(plates=True, faces=True)
        assert node.store.site() is not None, "the switch was not written to the row"

        rows = _rows(node, "site.identity")

    assert len(rows) == 1
    row = rows[0]
    print(row["detail"], row["before_json"], row["after_json"])
    assert row["subject"] == DEFAULT_SITE_ID
    assert row["actor"] == ACTOR
    assert row["node_id"] == "gatehouse"
    assert reason in row["detail"], "the reason did not reach the audit row"
    assert json.loads(row["before_json"]) == {"plates": False, "faces": False, "face_crops": False}
    assert json.loads(row["after_json"]) == {"plates": True, "faces": True, "face_crops": False}
    assert row["chain_hash"], "the row that turned faces on is not chained"

    # A restarted node comes back with the switch it was left with. Off again
    # after a restart sounds safe and is not: a register the operator switched
    # on would silently match nobody on Monday.
    with Node(tmp_path / "n.db", node_id="gatehouse") as node:
        assert node.identity == Identity(plates=True, faces=True)


def test_a_switch_flipped_for_no_reason_is_refused_and_nothing_changes(tmp_path: Path):
    with Node(tmp_path / "n.db") as node:
        for blank in ("", "   "):
            with pytest.raises(NodeError, match="reason"):
                node.set_identity(Identity(faces=True), reason=blank)

        assert node.identity == Identity()
        assert node.store.site() is None
        assert _rows(node, "site.identity") == []


def test_identity_status_reads_off_when_nothing_is_on(tmp_path: Path):
    with Node(tmp_path / "n.db") as node:
        assert node.identity_status == "off"
        assert "identity        off" in node.summary()


def test_identity_status_names_the_stand_in_when_faces_run_on_one(tmp_path: Path):
    with _faces_on(tmp_path, StandInModels()) as node:
        status = node.identity_status
        print(status)

    assert status.startswith("faces: on")
    assert "stand-in" in status
    assert "no models" not in status


def test_identity_status_names_the_missing_face_models_and_where_they_belong(
    tmp_path: Path, empty_models: Path, reference_video: Path
):
    """Faces on and no models is a status line, not a failure to start.

    The camera has to run — the operator may be waiting for the files to
    arrive on a connected machine — and the line has to say which files and
    which directory, or "faces: on" would describe a configuration rather than
    the system.
    """
    with Node(tmp_path / "n.db", detector_factory=_PersonDetector) as node:
        node.set_identity(Identity(faces=True), reason="staff list")
        status = node.identity_status
        print(status)

        node.add_camera(reference_video, camera_id="gate")
        node.run_forever()
        frames = node.camera("gate").runner.stats.frames

    assert "faces: on but no models" in status
    assert str(empty_models) in status
    for name in FACE_MODEL_FILES:
        assert name in status
    assert "yunet.onnx" in status and "sface.onnx" in status
    assert frames > 0, "the camera did not run with faces on and no models"


def test_identity_status_names_the_missing_plate_models_and_hands_no_reader_out(
    tmp_path: Path, empty_models: Path, reference_video: Path
):
    with Node(tmp_path / "n.db", detector_factory=_VehicleDetector) as node:
        node.set_identity(Identity(plates=True), reason="gate access list")
        status = node.identity_status
        print(status)

        node.add_camera(reference_video, camera_id="gate")
        node.start()
        reader = node.camera("gate").runner._plate_reader
        node.stop()

    assert "plates: on but no models" in status
    assert str(empty_models) in status
    for name in PLATE_MODEL_FILES:
        assert name in status
    assert reader is None, "a reader was handed out with no models to build it from"


def test_face_crops_is_recorded_and_audited_and_does_nothing_else(tmp_path: Path):
    # The flag that needs the most justification is recorded the moment it is
    # set, and refused any effect until the code that would keep a crop under
    # its own retention exists. The status must say so, or "faces" on a strip
    # would leave an operator believing photographs are being kept.
    with _faces_on(tmp_path, StandInModels()) as node:
        node.set_identity(Identity(faces=True, face_crops=True), reason="test")
        status = node.identity_status
        print(status)
        after = json.loads(_rows(node, "site.identity")[0]["after_json"])  # newest first

        node.add_camera(tmp_path / "unused.mp4", camera_id="gate")
        _attach(node, "gate", _FaceRunner(_person_frames(1, [0, 5, 10])))
        for _ in range(3):
            node.poll()
        held = node.templates_for("gate", 1)
        subjects = node.store.register.subjects()

    assert after["face_crops"] is True
    assert "face crops" in status and "not" in status.split("face crops")[1]
    assert held, "the fixture examined nobody, so it proves nothing about crops"
    assert subjects == (), "something was stored without an operator enrolling it"


def test_setting_identity_while_cameras_run_says_what_applies_when(
    tmp_path: Path, reference_video: Path
):
    reader = _RecordingPlateReader()
    with Node(
        tmp_path / "n.db", realtime=True, plate_reader=reader,
        detector_factory=_VehicleDetector,
    ) as node:
        node.add_camera(reference_video, camera_id="gate")
        node.start()
        with _listening_to_the_node() as said:
            node.set_identity(Identity(plates=True), reason="access list")
        running_reader = node.camera("gate").runner._plate_reader
        node.stop()

    warnings = [r.getMessage() for r in said if "started from now on" in r.getMessage()]
    assert len(warnings) == 1, "nothing told the operator the running camera is unchanged"
    assert "gate" in warnings[0]
    assert running_reader is None, "a running pipeline was given a reader it cannot take"
    assert reader.calls == 0


# ------------------------------------------------------------- faces off


def test_with_faces_off_the_backend_is_shown_nothing_over_the_reference_video(
    tmp_path: Path, reference_video: Path
):
    """Off means nothing ran. Not "the name column is hidden": no pixel reached
    the detector, however many person tracks went past it."""
    backend = StandInModels()
    with Node(
        tmp_path / "n.db", keep_images=True, face_backend=backend,
        detector_factory=_PersonDetector,
    ) as node:
        assert node.identity.faces is False
        node.add_camera(reference_video, camera_id="gate")
        node.run_forever()

        stats = node.camera("gate").runner.stats
        assert stats.frames > 0 and stats.distinct_objects >= 1, (
            "no person track was produced, so the switch was never tested"
        )
        assert node.identity_of("gate", 1) is None
        assert node.templates_for("gate", 1) == ()

    assert backend.images == [], f"faces are off and the detector saw {len(backend.images)} image(s)"
    assert backend.embedded == []


# -------------------------------------------------------------- faces on


def test_with_faces_on_the_backend_sees_only_the_person_crop_and_never_the_frame(
    tmp_path: Path,
):
    backend = StandInModels()
    with _faces_on(tmp_path, backend) as node:
        node.add_camera(tmp_path / "unused.mp4", camera_id="gate")
        # A person and a car on the same frame: one crop, of the person.
        _attach(node, "gate", _FaceRunner([_frame(0, tracks=[(1, 0), (2, 2)])]))

        node.poll()

    assert len(backend.images) == 1, "the car was examined for a face, or the person was not"
    shown = backend.images[0]
    print("frame:", FRAME_SHAPE, "shown to the detector:", shown.shape)
    assert shown.shape == CROP_SHAPE
    assert shown.shape != FRAME_SHAPE
    assert int(shown.min()) == 255, "the crop came from somewhere other than the person"


def test_faces_on_over_the_reference_video_examines_only_the_person_box(
    tmp_path: Path, reference_video: Path
):
    """The real pipeline, the real clip, a fixed person box: every image the
    backend saw is the crop of that box, and there are far fewer of them than
    frames because the work is rate-limited by frame index per track."""
    backend = StandInModels()
    # keep_images is deliberately off: a daemon keeps no images, and face work
    # has to get its frame anyway or a switched-on face path would run over
    # results with nothing to look at.
    with Node(
        tmp_path / "n.db", keep_images=False, face_backend=backend,
        detector_factory=_PersonDetector,
    ) as node:
        node.set_identity(Identity(faces=True), reason="staff list")
        node.add_camera(reference_video, camera_id="gate")
        node.run_forever()
        frames = node.camera("gate").runner.stats.frames
        status = node.identity_status

    # 640x480 frames, and PERSON is a quarter of each side.
    shapes = {image.shape for image in backend.images}
    print(f"{frames} frames, {len(backend.images)} examined, shapes {shapes}")
    assert backend.images, "faces are on and the detector was shown nothing"
    assert shapes == {(120, 160, 3)}
    assert (480, 640, 3) not in shapes
    assert len(backend.images) <= math.ceil(frames / FACE_STRIDE_FRAMES), (
        "more frames were examined than the stride allows"
    )
    assert "without an image" not in status, (
        "frames reached the face work with no image although faces were on"
    )


def test_face_work_runs_at_most_once_per_stride_per_track(tmp_path: Path):
    backend = StandInModels()
    with _faces_on(tmp_path, backend) as node:
        node.add_camera(tmp_path / "unused.mp4", camera_id="gate")
        _attach(node, "gate", _FaceRunner(_person_frames(1, range(2 * FACE_STRIDE_FRAMES))))

        for _ in range(2 * FACE_STRIDE_FRAMES):
            node.poll()

    print(f"{2 * FACE_STRIDE_FRAMES} consecutive frames, {len(backend.images)} examined")
    assert len(backend.images) == 2, "the per-track stride is not being honoured"


def test_templates_are_held_up_to_the_cap_and_dropped_when_the_track_ends(tmp_path: Path):
    backend = StandInModels()
    frames = _person_frames(1, [i * FACE_STRIDE_FRAMES for i in range(FRAMES_KEPT_FOR_ENROLMENT + 3)])
    with _faces_on(tmp_path, backend) as node:
        node.add_camera(tmp_path / "unused.mp4", camera_id="gate")
        _attach(node, "gate", _FaceRunner(frames + [_frame(999, ended=[1])]))

        counts = []
        for _ in frames:
            node.poll()
            counts.append(len(node.templates_for("gate", 1)))
        held = node.templates_for("gate", 1)
        node.poll()  # the frame on which the track ended
        after = node.templates_for("gate", 1)
        identity_after = node.identity_of("gate", 1)
        subjects = node.store.register.subjects()

    print("held per poll:", counts)
    assert counts[:3] == [1, 2, 3]
    assert len(held) == FRAMES_KEPT_FOR_ENROLMENT
    assert max(counts) == FRAMES_KEPT_FOR_ENROLMENT
    assert after == (), "the track ended and its templates outlived it"
    assert identity_after is None
    assert subjects == (), "a template was persisted without an enrolment"


def test_enrolling_a_person_stores_the_best_template_by_id_and_never_the_name(
    tmp_path: Path,
):
    best_vector = unit(0.0, 0.0, 3.0, 4.0)
    backend = StandInModels((0.70, unit(1.0)), (0.99, best_vector), (0.80, unit(0.0, 1.0)))
    with _faces_on(tmp_path, backend) as node:
        node.add_camera(tmp_path / "unused.mp4", camera_id="gate")
        _attach(node, "gate", _FaceRunner(_person_frames(1, [0, 5, 10])))
        for _ in range(3):
            node.poll()
        assert len(node.templates_for("gate", 1)) == 3

        subject_id = _enrol_from_track(node, name="Ali Hassan", notes="day shift")

        register = node.store.register
        subject = register.subject(subject_id)
        identifiers = register.identifiers(subject_id)
        rows = _rows(node, "person.enrolled")

    assert subject is not None and subject.kind is SubjectKind.PERSON
    assert subject.display_name == "Ali Hassan" and subject.notes == "day shift"
    assert len(identifiers) == 1
    stored = identifiers[0]
    assert stored.kind is IdentifierKind.FACE_TEMPLATE
    assert stored.model == "stand-in embedder"
    assert stored.quality == pytest.approx(0.99)
    assert (stored.source_camera, stored.source_track) == ("gate", 1)
    assert stored.enrolled_by == ACTOR and stored.basis == "staff list, consent on file"
    decoded = np.frombuffer(stored.template, dtype=np.float32)
    expected = np.asarray(best_vector, dtype=np.float64)
    expected /= np.linalg.norm(expected)
    assert decoded.shape == (TEMPLATE_DIMENSIONS,)
    assert np.allclose(decoded, expected, atol=1e-6), "the stored vector is not the best-quality face"

    assert len(rows) == 1
    row = rows[0]
    print(row["subject"], row["detail"])
    assert row["subject"] == subject_id
    assert "gate" in row["detail"] and "track 1" in row["detail"]
    for column in ("subject", "detail"):
        assert "Ali" not in (row[column] or ""), f"the name reached the audit log in {column}"
        assert "Hassan" not in (row[column] or "")
        assert "0.6" not in (row[column] or "") and "0.8," not in (row[column] or ""), (
            "a vector component reached the audit log"
        )


def test_a_second_track_with_a_matching_face_becomes_a_match_and_a_sighting(
    tmp_path: Path,
):
    """Enrol from one track, then a second track of the same face.

    Below `FRAMES_FOR_A_MATCH` faces the verdict is capped at POSSIBLE and the
    sighting says so; from the third face the track is a MATCH, the sighting is
    replaced with the certain claim and cites the enrolment, and each claim was
    audited once by subject id.
    """
    backend = StandInModels((0.95, unit(1.0)))
    with _faces_on(tmp_path, backend) as node:
        node.add_camera(tmp_path / "unused.mp4", camera_id="gate")
        gate = node.camera("gate")
        _attach(node, "gate", _FaceRunner(_person_frames(1, [0]) + [_frame(1, ended=[1])]))
        node.poll()
        subject_id = _enrol_from_track(node)
        node.poll()
        enrolment = node.store.register.identifiers(subject_id)[0]

        second = _person_frames(2, [100, 105, 110], first=5_000)
        gate.runner = _FaceRunner(second)
        verdicts = []
        for _ in second:
            node.poll()
            verdicts.append(node.identity_of("gate", 2).verdict)

        identity = node.identity_of("gate", 2)
        history = node.store.register.history(subject_id)
        audits = _rows(node, "person.sighted")

    print("verdicts:", [v.value for v in verdicts], "final:", identity.describe())
    assert verdicts[: FRAMES_FOR_A_MATCH - 1] == [Verdict.POSSIBLE] * (FRAMES_FOR_A_MATCH - 1), (
        "too few faces were allowed to be a match"
    )
    assert identity.verdict is Verdict.MATCH
    assert identity.person_id == subject_id and identity.name == "Ali Hassan"
    assert identity.frames == FRAMES_FOR_A_MATCH
    assert identity.score == pytest.approx(1.0)

    assert len(history) == 1, "one track became several sightings"
    sighting = history[0]
    assert (sighting.camera_id, sighting.track_id) == ("gate", 2)
    assert sighting.confidence is Confidence.MATCH
    assert sighting.score == pytest.approx(1.0)
    assert sighting.identifier_id == enrolment.id, "the sighting does not say why"
    assert (sighting.first_seen_millis, sighting.last_seen_millis) == (5_000, 5_000 + 110 * 40)

    # Track 2's two claims, once each and in the order they were made — the
    # trail is newest first, so it is put back in writing order here. Track 1
    # was enrolled from and then matched itself on the frame after, which is a
    # sighting too.
    on_track_two = sorted(
        (row for row in audits if "track 2" in row["detail"]), key=lambda r: r["id"]
    )
    assert [("possible match" in r["detail"], "match" in r["detail"]) for r in on_track_two] == [
        (True, True), (False, True)
    ], "each claim about the encounter was not audited exactly once"
    for row in audits:
        assert row["subject"] == subject_id
        assert "Ali" not in row["detail"] and "Hassan" not in row["detail"]


def test_a_lower_similarity_yields_possible_and_a_possible_sighting_never_a_match(
    tmp_path: Path,
):
    midpoint = (MATCH_SIMILARITY + POSSIBLE_SIMILARITY) / 2
    backend = StandInModels((0.95, unit(1.0)))
    with _faces_on(tmp_path, backend) as node:
        node.add_camera(tmp_path / "unused.mp4", camera_id="gate")
        gate = node.camera("gate")
        _attach(node, "gate", _FaceRunner(_person_frames(1, [0])))
        node.poll()
        subject_id = _enrol_from_track(node)

        backend._faces = [(0.95, at_similarity(midpoint))]
        frames = _person_frames(2, [100 + 5 * i for i in range(FRAMES_FOR_A_MATCH + 2)], first=5_000)
        gate.runner = _FaceRunner(frames)
        seen = []
        for _ in frames:
            node.poll()
            seen.append(node.identity_of("gate", 2))

        history = [s for s in node.store.register.history(subject_id) if s.track_id == 2]
        audits = [r for r in _rows(node, "person.sighted") if "track 2" in r["detail"]]

    print("scores:", [round(i.score, 3) for i in seen], "verdicts:", [i.verdict.value for i in seen])
    assert all(i.verdict is Verdict.POSSIBLE for i in seen), "a middling score became a match"
    assert all(POSSIBLE_SIMILARITY <= i.score < MATCH_SIMILARITY for i in seen)
    assert seen[-1].frames > FRAMES_FOR_A_MATCH, "the cap, not the score, held it at POSSIBLE"
    assert len(history) == 1
    assert history[0].confidence is Confidence.POSSIBLE
    assert history[0].score == pytest.approx(midpoint, abs=1e-6)
    assert len(audits) == 1 and "possible match" in audits[0]["detail"]


def test_a_stranger_is_neither_sighted_nor_enrolled(tmp_path: Path):
    backend = StandInModels((0.95, unit(1.0)))
    with _faces_on(tmp_path, backend) as node:
        node.add_camera(tmp_path / "unused.mp4", camera_id="gate")
        gate = node.camera("gate")
        _attach(node, "gate", _FaceRunner(_person_frames(1, [0])))
        node.poll()
        subject_id = _enrol_from_track(node)

        backend._faces = [(0.95, unit(0.0, 1.0))]  # orthogonal: nobody enrolled
        frames = _person_frames(2, [100, 105, 110, 115], first=5_000)
        gate.runner = _FaceRunner(frames)
        for _ in frames:
            node.poll()

        identity = node.identity_of("gate", 2)
        history = node.store.register.history(subject_id)
        subjects = node.store.register.subjects()

    print(identity.describe())
    assert identity.verdict is Verdict.NONE
    assert identity.person_id is None and identity.name is None
    assert identity.score is not None and identity.score < POSSIBLE_SIMILARITY
    assert [s.track_id for s in history] == [], "a stranger was recorded as somebody"
    assert [s.id for s in subjects] == [subject_id], "observation enrolled somebody"


def test_a_sighting_cites_the_enrolment_the_track_resembles_most(tmp_path: Path):
    # Two templates for one person, and a track that resembles the second.
    # "Why does it think this is her" has to point at the right row.
    backend = StandInModels((0.95, unit(1.0)))
    with _faces_on(tmp_path, backend) as node:
        node.add_camera(tmp_path / "unused.mp4", camera_id="gate")
        gate = node.camera("gate")
        _attach(node, "gate", _FaceRunner(_person_frames(1, [0])))
        node.poll()
        subject_id = _enrol_from_track(node)

        backend._faces = [(0.95, unit(0.0, 0.0, 1.0))]
        gate.runner = _FaceRunner(_person_frames(2, [100], first=3_000))
        node.poll()
        same = node.enrol_person("Ali Hassan", "gate", 2, basis="second photo", subject_id=subject_id)
        assert same == subject_id
        enrolments = node.store.register.identifiers(subject_id)
        assert len(enrolments) == 2

        gate.runner = _FaceRunner(_person_frames(3, [200, 205, 210], first=9_000))
        for _ in range(3):
            node.poll()
        sighting = next(s for s in node.store.register.history(subject_id) if s.track_id == 3)

    assert sighting.confidence is Confidence.MATCH
    assert sighting.identifier_id == enrolments[1].id, "the wrong enrolment was cited"


def test_enrol_person_refuses_plainly_when_off_when_nothing_is_held_and_when_unnamed(
    tmp_path: Path,
):
    with Node(tmp_path / "n.db", face_backend=StandInModels()) as node:
        node.add_camera(tmp_path / "unused.mp4", camera_id="gate")
        with pytest.raises(RegistryError, match="faces are off") as off:
            node.enrol_person("Ali Hassan", "gate", 1, basis="staff list")
        print("off:", off.value)

        node.set_identity(Identity(faces=True), reason="staff list")
        with pytest.raises(RegistryError, match="no face held") as nothing:
            node.enrol_person("Ali Hassan", "gate", 1, basis="staff list")
        print("nothing held:", nothing.value)

        _attach(node, "gate", _FaceRunner(_person_frames(1, [0])))
        node.poll()
        assert node.templates_for("gate", 1)
        with pytest.raises(RegistryError, match="display name") as blank:
            node.enrol_person("   ", "gate", 1, basis="staff list")
        print("blank:", blank.value)
        with pytest.raises(RegistryError, match="basis"):
            node.enrol_person("Ali Hassan", "gate", 1, basis="")

        assert node.store.register.subjects() == ()
        assert _rows(node, "person.enrolled") == []


def test_forgetting_a_subject_deletes_it_and_audits_counts_only(tmp_path: Path):
    backend = StandInModels((0.95, unit(1.0)))
    with _faces_on(tmp_path, backend) as node:
        node.add_camera(tmp_path / "unused.mp4", camera_id="gate")
        _attach(node, "gate", _FaceRunner(_person_frames(1, [0, 5, 10, 15])))
        node.poll()
        subject_id = _enrol_from_track(node)
        for _ in range(3):
            node.poll()
        assert node.identity_of("gate", 1).person_id == subject_id
        assert node.store.register.history(subject_id)

        forgotten = node.forget_subject(subject_id)

        register = node.store.register
        assert register.subject(subject_id) is None
        assert register.identifiers(subject_id) == ()
        assert register.history(subject_id) == ()
        assert node.identity_of("gate", 1) is None, "a live track kept naming a forgotten person"
        rows = _rows(node, "subject.forgotten")

    assert forgotten.found is True and forgotten.identifiers_deleted == 1
    assert forgotten.sightings_unlinked == 1
    assert len(rows) == 1
    row = rows[0]
    print(row["detail"])
    assert row["subject"] == subject_id
    assert "identifiers_deleted" in row["detail"] and "sightings_unlinked" in row["detail"]
    assert "Ali" not in row["detail"] and "Hassan" not in row["detail"]


def test_pinning_a_subject_is_audited_both_ways(tmp_path: Path):
    backend = StandInModels()
    with _faces_on(tmp_path, backend) as node:
        node.add_camera(tmp_path / "unused.mp4", camera_id="gate")
        _attach(node, "gate", _FaceRunner(_person_frames(1, [0])))
        node.poll()
        subject_id = _enrol_from_track(node)

        node.pin_subject(subject_id, True)
        assert node.store.register.subject(subject_id).pinned is True
        node.pin_subject(subject_id, False)
        assert node.store.register.subject(subject_id).pinned is False

        pinned = _rows(node, "subject.pinned")
        unpinned = _rows(node, "subject.unpinned")
        with pytest.raises(RegistryError):
            node.pin_subject("nobody", True)

    assert [r["subject"] for r in pinned] == [subject_id]
    assert [r["subject"] for r in unpinned] == [subject_id]
    assert '"pinned":true' in pinned[0]["detail"].replace(" ", "")
    assert '"pinned":false' in unpinned[0]["detail"].replace(" ", "")
    assert "Ali" not in pinned[0]["detail"]


def test_enrol_vehicle_stores_the_plate_normalised_and_audits_by_id_only(tmp_path: Path):
    with Node(tmp_path / "n.db") as node:
        subject_id = node.enrol_vehicle(
            "Contractor van", "b-7421", basis="site access list",
            camera_id="gate", track_id=3,
        )

        register = node.store.register
        subject = register.subject(subject_id)
        identifiers = register.identifiers(subject_id)
        rows = _rows(node, "vehicle.enrolled")
        found = register.find_plate("B 7421")

    assert subject.kind is SubjectKind.VEHICLE and subject.display_name == "Contractor van"
    assert len(identifiers) == 1
    stored = identifiers[0]
    assert stored.kind is IdentifierKind.PLATE
    assert stored.plate == "B7421", "the plate was not normalised by the register's format"
    assert stored.raw_text == "b-7421"
    assert (stored.source_camera, stored.source_track) == ("gate", 3)
    assert found is not None and found.id == subject_id
    assert len(rows) == 1 and rows[0]["subject"] == subject_id
    for column in ("subject", "detail"):
        assert "7421" not in (rows[0][column] or ""), f"the plate reached the audit log in {column}"
        assert "Contractor" not in (rows[0][column] or "")


# ---------------------------------------------------------------- plates


def test_with_plates_on_the_injected_reader_reaches_the_pipeline_and_is_called(
    tmp_path: Path, reference_video: Path
):
    reader = _RecordingPlateReader()
    with Node(tmp_path / "n.db", plate_reader=reader, detector_factory=_VehicleDetector) as node:
        node.set_identity(Identity(plates=True), reason="gate access list")
        assert node.identity_status.startswith("plates: on")
        node.add_camera(reference_video, camera_id="gate")
        node.run_forever()
        frames = node.camera("gate").runner.stats.frames

    print(f"{frames} frames, reader called {reader.calls} time(s)")
    assert frames > 0
    assert reader.calls > 0, "plates are on and the reader never reached the pipeline"


def test_with_plates_off_the_reader_is_never_called(tmp_path: Path, reference_video: Path):
    reader = _RecordingPlateReader()
    with Node(tmp_path / "n.db", plate_reader=reader, detector_factory=_VehicleDetector) as node:
        assert node.identity.plates is False
        node.add_camera(reference_video, camera_id="gate")
        node.run_forever()
        frames = node.camera("gate").runner.stats.frames
        handed = node.camera("gate").runner._plate_reader

    assert frames > 0
    assert handed is None, "the reader reached the pipeline with plates off"
    assert reader.calls == 0, f"plates are off and the reader was called {reader.calls} time(s)"


def test_the_plate_reader_is_built_from_the_models_directory_only_when_plates_are_on(
    tmp_path: Path, empty_models: Path, reference_video: Path, monkeypatch: pytest.MonkeyPatch
):
    """Without an injected reader the node builds one from the operator's files.

    The files here are not models — nothing here may load a real one — so the
    constructor is stood in for by one that records what it was asked to build
    and refuses to be asked while plates are off.
    """
    from sentinel import node as node_module

    for name in PLATE_MODEL_FILES:
        (empty_models / name).write_bytes(b"not a model, and never fetched")
    built = []

    def fake_reader(models):
        built.append(models)
        return _RecordingPlateReader()

    monkeypatch.setattr(node_module, "PlateReader", fake_reader)

    with Node(tmp_path / "n.db", detector_factory=_VehicleDetector) as node:
        node.add_camera(reference_video, camera_id="gate")
        node.start()
        node.stop()
        assert built == [], "a reader was constructed with plates off"

        node.set_identity(Identity(plates=True), reason="gate access list")
        status = node.identity_status
        node.start()
        handed = node.camera("gate").runner._plate_reader
        node.stop()

    print(status)
    assert len(built) == 1, "plates are on and no reader was built for the camera"
    assert built[0].detector_path == empty_models / "plate-detector.onnx"
    assert built[0].recogniser_path == empty_models / "plate-recogniser.onnx"
    assert built[0].charset_path == empty_models / "plate-charset.txt"
    # Behind the switch, so that plates *off* stops it at its next read rather
    # than at the camera's next restart; the reader itself is the one built.
    assert isinstance(handed, node_module._SwitchedPlateReader), type(handed)
    assert isinstance(handed.reader, _RecordingPlateReader)
    assert "plates: on" in status and "no models" not in status


# ------------------------------------------------------ frames and switches


def test_face_work_skips_frames_that_carry_no_image_and_says_so(tmp_path: Path):
    backend = StandInModels()
    with _faces_on(tmp_path, backend) as node:
        node.add_camera(tmp_path / "unused.mp4", camera_id="gate")
        _attach(node, "gate", _FaceRunner(_person_frames(1, [0, 5], image=None)))

        node.poll()
        node.poll()
        status = node.identity_status

    print(status)
    assert backend.images == []
    assert "2 frame(s) arrived without an image" in status


def test_turning_faces_off_stops_face_work_at_the_next_poll_and_drops_what_was_held(
    tmp_path: Path,
):
    backend = StandInModels()
    with _faces_on(tmp_path, backend) as node:
        node.add_camera(tmp_path / "unused.mp4", camera_id="gate")
        _attach(node, "gate", _FaceRunner(_person_frames(1, [0, 5, 10, 15])))
        node.poll()
        node.poll()
        assert len(node.templates_for("gate", 1)) == 2
        examined = len(backend.images)

        node.set_identity(Identity(), reason="contractor list withdrawn")

        assert node.templates_for("gate", 1) == (), "templates outlived the switch"
        assert node.identity_of("gate", 1) is None
        node.poll()
        node.poll()
        rows = _rows(node, "site.identity")

    assert len(backend.images) == examined, "faces were examined after the site said no"
    assert len(rows) == 2 and json.loads(rows[0]["after_json"])["faces"] is False  # newest first


def test_a_camera_restarted_in_the_same_process_forgets_the_old_runs_tracks(tmp_path: Path):
    # Track ids start again with each runner. Track 1 of the new run is a
    # different person from track 1 of the old, and must not inherit its faces.
    backend = StandInModels()
    with _faces_on(tmp_path, backend) as node:
        node.add_camera(tmp_path / "unused.mp4", camera_id="gate")
        gate = node.camera("gate")
        _attach(node, "gate", _FaceRunner(_person_frames(1, [0, 5, 10])))
        for _ in range(3):
            node.poll()
        assert len(node.templates_for("gate", 1)) == 3

        gate.runner = _FaceRunner(_person_frames(1, [0]))
        assert node.templates_for("gate", 1) == (), "the new run inherited the old run's faces"
        node.poll()
        held = node.templates_for("gate", 1)
        # The lookup above is keyed by run, so it would read empty even if the
        # old run's templates were still sitting in memory. They must not be: a
        # camera restarted every few minutes for a month would otherwise hold
        # the faces of every track of every run that ever ended, which is
        # biometric data kept for tracks that no longer exist.
        runs_held = {key[1] for key in node._face_tracks if key[0] == "gate"}

    assert len(held) == 1
    assert runs_held == {gate.run}, f"face state survived from earlier runs: {runs_held}"


def test_the_whole_engine_still_imports_no_qt_with_identity_wired():
    # The identity path pulls faces, plates and the registry into the node. None
    # of them may bring Qt with them: a worker in a cupboard has no display.
    code = (
        "import sys, pkgutil, importlib, sentinel; "
        "[importlib.import_module('sentinel.' + m.name) "
        " for m in pkgutil.iter_modules(sentinel.__path__) "
        " if m.name != '__main__']; "
        "import sentinel.node, sentinel.faces, sentinel.plates, sentinel.registry; "
        "bad = sorted(m for m in sys.modules "
        "if m.startswith('PySide') or m.startswith('shiboken')); "
        "print(bad); sys.exit(1 if bad else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )

    assert result.returncode == 0, f"the engine dragged in Qt: {result.stdout.strip()}"


# --------------------------------------------- the switch reaches a running reader

_A_POSE = CameraPose(
    position=SITE, mount_height=6.0, heading=180.0, pitch=-22.0,
    horizontal_fov=62.0, vertical_fov=36.0, range_meters=90.0,
)



def test_the_switched_reader_reads_only_while_the_site_says_plates_are_on():
    """`_SwitchedPlateReader` is what a pipeline is handed, so that plates
    *off* reaches a running camera at its next read. The pipeline's own
    bookkeeping carries on over an empty answer."""
    import threading

    from sentinel import node as node_module

    inner = _RecordingPlateReader()
    switch = threading.Event()
    switch.set()
    wrapped = node_module._SwitchedPlateReader(inner, switch)
    assert wrapped.country == inner.country
    assert wrapped.reader is inner

    wrapped.read(None, None, frame_index=1)
    assert inner.calls == 1

    switch.clear()
    assert wrapped.read(None, None, frame_index=2) == ()
    assert inner.calls == 1, "the site said no and the reader was still asked"

    switch.set()
    wrapped.read(None, None, frame_index=3)
    assert inner.calls == 2


def test_turning_plates_off_clears_the_switch_every_running_reader_watches(tmp_path: Path):
    with Node(tmp_path / "n.db", plate_reader=_RecordingPlateReader()) as node:
        node.set_identity(Identity(plates=True), reason="gate access list")
        assert node._plates_on.is_set()
        node.set_identity(Identity(plates=False), reason="no longer needed")
        assert not node._plates_on.is_set()


# ------------------------------------------------ the node's placeholder site row


def test_the_switch_written_before_any_camera_is_placed_does_not_freeze_the_origin(
    tmp_path: Path, reference_video: Path
):
    """A node that turned plates on before its first camera was placed once
    kept the site at (0, 0) for good, with every camera placed afterwards
    anchored on the Gulf of Guinea. The row `set_identity` writes carries the
    switch and is marked undeclared; the origin keeps following the cameras."""
    with Node(tmp_path / "n.db", plate_reader=_RecordingPlateReader()) as node:
        node.set_identity(Identity(plates=True), reason="gate access list")
        assert node.store.site().declared is False, "the node's own row claimed to be declared"
        assert node.site().origin == LatLon(0.0, 0.0), "nothing is placed, so there is nowhere"

        node.add_camera(reference_video, camera_id="gate")
        node.place_camera("gate", _A_POSE)

        derived = node.site()
        assert derived.identity.plates is True, "the switch was lost with the origin"
        assert derived.origin == _A_POSE.position, "the origin froze on the placeholder"
        assert derived.declared is False


def test_a_declared_site_is_returned_as_it_is_whatever_the_cameras_say(
    tmp_path: Path, reference_video: Path
):
    elsewhere = LatLon(SITE.lat + 0.01, SITE.lon + 0.01)
    with Node(tmp_path / "n.db") as node:
        node.store.save_site(make_site(origin=elsewhere))
        node.add_camera(reference_video, camera_id="gate")
        node.place_camera("gate", _A_POSE)
        site = node.site()
        assert site.declared is True
        assert site.origin == elsewhere, "a declared origin was second-guessed from the cameras"
