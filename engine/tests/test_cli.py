"""Tests for the headless command line.

The console has never been the only way to run this, but until `cli.py` it was.
These cover the whole path a container or a scheduled job takes: parse a camera
placement off a command line, analyse a real encoded file, raise events,
correlate them into an incident, persist it, and export it as evidence — all
with no Qt anywhere.

The argument parsing tests carry more weight than they look like they do. Every
one of them is a way an operator can put a camera somewhere it is not, and a
misplaced camera produces coordinates that are indistinguishable from measured
ones on a map somebody will be sent to.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from sentinel import cli, logs
from sentinel.store import Store


@pytest.fixture(autouse=True)
def clean_logging():
    logs.reset()
    yield
    logs.reset()


@pytest.fixture
def database(tmp_path: Path) -> Path:
    return tmp_path / "sentinel.db"


# ------------------------------------------------------------ camera placement


def test_a_placement_is_parsed_into_a_pose():
    pose = cli._pose("33.8938,35.5018,6,180,-22")

    assert pose.position.lat == pytest.approx(33.8938)
    assert pose.position.lon == pytest.approx(35.5018)
    assert pose.mount_height == 6.0
    assert pose.heading == 180.0
    assert pose.pitch == -22.0


def test_the_optics_can_be_given_and_default_otherwise():
    default = cli._pose("33.8938,35.5018,6,180,-22")
    explicit = cli._pose("33.8938,35.5018,6,180,-22,62,36,90")

    assert (default.horizontal_fov, default.vertical_fov) == (70.0, 40.0)
    assert (explicit.horizontal_fov, explicit.vertical_fov, explicit.range_meters) == (
        62.0, 36.0, 90.0,
    )


@pytest.mark.parametrize(
    "text, because",
    [
        ("33.8938,35.5018,6,180", "too few fields"),
        ("33.8938,35.5018,6,180,-22,62", "six fields is neither shape"),
        ("north,35.5018,6,180,-22", "not a number"),
        ("91,35.5018,6,180,-22", "latitude out of range"),
        ("33.8938,181,6,180,-22", "longitude out of range"),
        ("33.8938,35.5018,0,180,-22", "a mast of zero height"),
        ("33.8938,35.5018,-3,180,-22", "a mast below the ground"),
        ("33.8938,35.5018,6,180,0", "level: sees no ground to project on"),
        ("33.8938,35.5018,6,180,15", "tilted up: sees no ground at all"),
    ],
)
def test_an_impossible_placement_is_refused(text: str, because: str):
    # Refused rather than clamped. A camera the system silently "fixes" reports
    # positions that look exactly like measured ones.
    with pytest.raises(argparse.ArgumentTypeError):
        cli._pose(text)


def test_a_zone_is_parsed_into_a_named_polygon():
    zone = cli._zone("Loading Yard:33.8940,35.5016;33.8940,35.5020;33.8936,35.5018")

    assert zone.name == "Loading Yard"
    assert zone.id == "loading-yard"
    assert len(zone.ring) == 3


@pytest.mark.parametrize(
    "text",
    [
        "no-colon-here",
        "Yard:",
        "Yard:33.8940,35.5016;33.8940,35.5020",
        "Yard:33.8940,35.5016;north,35.5020;33.8936,35.5018",
    ],
)
def test_an_impossible_zone_is_refused(text: str):
    with pytest.raises(argparse.ArgumentTypeError):
        cli._zone(text)


def test_the_rules_match_what_is_configured():
    # Without a zone there is nothing to be inside, so the zone rules would be
    # dead weight and would let a run report "0 events" for a reason that has
    # nothing to do with the footage.
    without = {type(rule).__name__ for rule in cli._rules([])}
    with_zone = {
        type(rule).__name__
        for rule in cli._rules([cli._zone("Yard:33.894,35.5016;33.894,35.502;33.8936,35.5018")])
    }

    assert without == {"RapidMovementRule"}
    assert "ZoneEntryRule" in with_zone
    assert without < with_zone


# ---------------------------------------------------------------- a whole run


def zone_in_front_of(pose, radius: float = 12.0) -> str:
    """A zone the camera can actually adjudicate, in `--zone` syntax.

    Placed just past the near edge of the real footprint, which is not the same
    as the stated range — the same calculation `sentinel coverage` prints, and
    the reason that command exists.
    """
    from sentinel.core import destination_point, field_of_view, haversine_distance

    footprint = field_of_view(pose, arc_segments=16)
    near = min(haversine_distance(pose.position, point) for point in footprint)
    centre = destination_point(pose.position, pose.heading, near + radius)
    ring = [destination_point(centre, bearing, radius) for bearing in (0.0, 90.0, 180.0, 270.0)]
    return "Yard:" + ";".join(f"{point.lat:.6f},{point.lon:.6f}" for point in ring)


def test_a_whole_run_produces_an_incident_and_an_evidence_package(
    reference_video: Path, reference_pose, database: Path, tmp_path: Path
):
    placement = (
        f"{reference_pose.position.lat},{reference_pose.position.lon},"
        f"{reference_pose.mount_height},{reference_pose.heading},{reference_pose.pitch},"
        f"{reference_pose.horizontal_fov},{reference_pose.vertical_fov},"
        f"{reference_pose.range_meters}"
    )
    export = tmp_path / "evidence"

    code = cli.main([
        "--database", str(database), "--quiet",
        "run", str(reference_video),
        "--id", "gate",
        "--place", placement,
        "--zone", zone_in_front_of(reference_pose),
        "--export", str(export),
    ])

    assert code == 0

    with Store(database) as store:
        incidents = store.incidents()
        assert len(incidents) == 1, "one intrusion is one incident"
        assert store.event_count() > 1, "several events fed it"

    packages = list(export.iterdir())
    assert len(packages) == 1
    manifest = json.loads((packages[0] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["files"], "the package must list what is in it"


def test_a_run_without_a_placement_still_works_and_locates_nothing(
    reference_video: Path, database: Path
):
    # No default position exists, deliberately. Objects are tracked and reported
    # as not placed rather than pinned to a nominal origin somebody would be
    # sent to.
    code = cli.main([
        "--database", str(database), "--quiet", "run", str(reference_video),
    ])

    assert code == 0
    with Store(database) as store:
        assert store.incident_count() == 0


def test_a_missing_file_fails_before_anything_is_opened(database: Path, tmp_path: Path):
    code = cli.main([
        "--database", str(database), "--quiet", "run", str(tmp_path / "absent.mp4"),
    ])

    assert code == 2


def test_ids_must_be_given_once_per_source_or_not_at_all(
    reference_video: Path, database: Path
):
    code = cli.main([
        "--database", str(database), "--quiet",
        "run", str(reference_video), str(reference_video), "--id", "only-one",
    ])

    assert code == 2


# ------------------------------------------------------------------ inspection


def test_an_incident_survives_the_process_that_raised_it(
    reference_video: Path, reference_pose, database: Path, tmp_path: Path, capsys
):
    # Until `Store.incident` existed, an incident could only be exported while
    # the process that raised it was still running — which makes an evidence
    # package something you must remember to produce at the time rather than
    # something you can produce when somebody asks.
    placement = (
        f"{reference_pose.position.lat},{reference_pose.position.lon},"
        f"{reference_pose.mount_height},{reference_pose.heading},{reference_pose.pitch},"
        f"{reference_pose.horizontal_fov},{reference_pose.vertical_fov},"
        f"{reference_pose.range_meters}"
    )
    cli.main([
        "--database", str(database), "--quiet", "run", str(reference_video),
        "--place", placement, "--zone", zone_in_front_of(reference_pose),
    ])

    with Store(database) as store:
        incident_id = store.incidents()[0]["id"]
        rebuilt = store.incident(incident_id)

    assert rebuilt is not None
    assert rebuilt.id == incident_id
    assert rebuilt.events, "the events must come back with it"
    assert rebuilt.risk.factors, "and so must the reasoning"

    # And a wholly separate invocation can export it.
    destination = tmp_path / "later"
    code = cli.main([
        "--database", str(database), "--quiet",
        "export", incident_id, "--to", str(destination),
    ])

    assert code == 0
    assert (destination / incident_id / "incident.json").is_file()


def test_exporting_an_unknown_incident_says_so(database: Path, tmp_path: Path):
    Store(database).close()

    code = cli.main([
        "--database", str(database), "--quiet",
        "export", "inc_nothing", "--to", str(tmp_path),
    ])

    assert code == 2


def test_coverage_reports_the_ground_a_camera_actually_sees(capsys):
    # The stated range is not the coverage. This command exists because the
    # first thing anybody does is put a zone where the camera cannot see.
    code = cli.main(["coverage", "--place", "33.8938,35.5018,6,180,-22,62,36,90"])

    printed = capsys.readouterr().out

    assert code == 0
    assert "ground covered" in printed
    assert "--zone" in printed, "it must hand back something usable"
    # 6 m mast at 22 degrees with a 36 degree vertical field: 7 m to 86 m,
    # whatever the datasheet says.
    assert "90 m" in printed and "7." in printed


def test_coverage_refuses_a_camera_that_sees_no_ground(capsys):
    # `_pose` refuses a non-negative pitch, so this is the other way to get
    # there: tilted down, but not enough for the bottom of the frame to reach
    # the ground within any range.
    code = cli.main(["coverage", "--place", "33.8938,35.5018,6,180,-0.01,62,0.001,90"])

    assert code == 1
    assert "sees no ground" in capsys.readouterr().err


def test_where_answers_where_the_files_are(capsys):
    code = cli.main(["where"])

    printed = capsys.readouterr().out

    assert code == 0
    for expected in ("data directory", "database", "logs", "evidence"):
        assert expected in printed


# --------------------------------------------------------------------- errors


def test_an_interrupt_is_not_reported_as_a_crash(monkeypatch, capsys):
    # Interrupting a long run is a normal thing to do. Everything already
    # written stays written, because each event and incident is its own
    # transaction.
    monkeypatch.setattr(cli, "_where", lambda args: (_ for _ in ()).throw(KeyboardInterrupt))

    assert cli.main(["where"]) == 130
    assert "interrupted" in capsys.readouterr().err


def test_an_operator_gets_a_sentence_and_a_developer_gets_the_traceback(
    monkeypatch, capsys
):
    def explode(args):
        raise RuntimeError("the disk is full")

    monkeypatch.setattr(cli, "_where", explode)

    assert cli.main(["where"]) == 1
    printed = capsys.readouterr().err
    assert "the disk is full" in printed
    assert "--verbose" in printed, "it must say how to get more"


def test_two_cameras_do_not_share_a_background_model(
    reference_video: Path, database: Path, monkeypatch
):
    # MOG2 carries a per-pixel model of *its* scene. Sharing one detector
    # between two cameras corrupts both models and every detection that comes
    # out of them — and it corrupts them quietly, because the output is still
    # detection-shaped.
    # Patched where `detector_for` resolves it, which is the module that owns
    # the class — not on `cli`, which stopped naming it when the choice of
    # detector moved behind the factory. A test hooked to the old spelling went
    # on passing while counting nothing at all.
    from sentinel import detect as module

    built: list[object] = []
    original = module.MotionDetector

    def counting(*args, **kwargs):
        detector = original(*args, **kwargs)
        built.append(detector)
        return detector

    monkeypatch.setattr(module, "MotionDetector", counting)

    cli.main([
        "--database", str(database), "--quiet",
        "run", str(reference_video), str(reference_video),
    ])

    assert len(built) == 2
    assert built[0] is not built[1]


# ------------------------------------------------------------- local cameras


def test_devices_lists_what_the_operating_system_reports(monkeypatch, capsys):
    from sentinel import devices

    monkeypatch.setattr(
        devices, "list_cameras",
        lambda: [
            devices.LocalCamera(0, "Integrated Camera", "USB-ONE", "Media Foundation"),
            devices.LocalCamera(1, "Logitech C920", "USB-TWO", "Media Foundation"),
        ],
    )

    assert cli.main(["--quiet", "devices"]) == 0

    printed = capsys.readouterr().out
    assert "Integrated Camera" in printed
    assert "device:0" in printed and "device:1" in printed
    assert "USB-TWO" in printed
    # An unconfirmed index must say so, and say how to resolve it.
    assert "assumed" in printed
    assert "--probe" in printed


def test_devices_opens_nothing_without_probe(monkeypatch):
    import cv2

    from sentinel import devices

    def forbidden(*args, **kwargs):
        raise AssertionError("listing devices opened a camera")

    monkeypatch.setattr(cv2, "VideoCapture", forbidden)
    monkeypatch.setattr(devices, "list_cameras", lambda: [])

    assert cli.main(["--quiet", "devices"]) == 0


def test_devices_on_a_machine_with_none_suggests_probing(monkeypatch, capsys):
    from sentinel import devices

    monkeypatch.setattr(devices, "list_cameras", lambda: [])

    assert cli.main(["--quiet", "devices"]) == 0
    assert "--probe" in capsys.readouterr().out


def test_a_device_source_is_not_mistaken_for_a_missing_file(monkeypatch, database: Path):
    # `device:0` has no "://" and is not a path that exists, so the file check
    # rejected it outright before this was handled.
    opened: list[str] = []

    class FakePipeline:
        def __init__(self, source, detector, **kwargs):
            opened.append(source.source_id)
            self.stats = __import__(
                "sentinel.pipeline", fromlist=["PipelineStats"]
            ).PipelineStats()

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def run(self):
            return iter(())

    monkeypatch.setattr(cli, "Pipeline", FakePipeline)

    assert cli.main(["--database", str(database), "--quiet", "run", "device:0"]) == 0
    assert opened == ["cam-01"]


@pytest.mark.parametrize("source", ["device:front", "device:", "device:-1"])
def test_a_malformed_device_source_is_refused_before_anything_opens(
    source: str, database: Path, capsys
):
    # A mistyped index is a command-line mistake, so it exits 2 with a sentence.
    # It must never quietly become index 0: that would point a camera at
    # somewhere nobody chose, and every position it reported would be wrong.
    code = cli.main(["--database", str(database), "--quiet", "run", source])

    assert code == 2
    assert "device:N" in capsys.readouterr().err


# --------------------------------------------------------- bounding a live run


def test_a_live_run_can_be_bounded_by_frames(reference_video: Path, database: Path):
    # A camera has no end. Without a bound a headless run never returns — and
    # Ctrl-C is not available to a scheduled job or a container, which on
    # Windows cannot even be sent an interrupt from outside.
    code = cli.main([
        "--database", str(database), "--quiet",
        "run", str(reference_video), "--frames", "12",
    ])

    assert code == 0


def test_frames_one_processes_one_frame_not_none(reference_video: Path, database: Path):
    # The bound is checked *after* the frame, so `--frames 1` does one frame.
    seen: list[int] = []
    from sentinel.pipeline import Pipeline

    original = Pipeline.run

    def counting(self):
        for result in original(self):
            seen.append(result.index)
            yield result

    import sentinel.pipeline as pipeline_module

    pipeline_module.Pipeline.run = counting
    try:
        assert cli.main([
            "--database", str(database), "--quiet",
            "run", str(reference_video), "--frames", "1",
        ]) == 0
    finally:
        pipeline_module.Pipeline.run = original

    assert len(seen) == 1


@pytest.mark.parametrize(
    "arguments",
    [
        ["--for", "0"],
        ["--for", "-5"],
        ["--frames", "0"],
        ["--frames", "-1"],
    ],
)
def test_an_impossible_bound_is_refused(arguments, reference_video: Path, database: Path):
    code = cli.main([
        "--database", str(database), "--quiet", "run", str(reference_video), *arguments,
    ])

    assert code == 2


def test_an_unbounded_live_source_says_so_before_it_starts(
    reference_video: Path, database: Path, monkeypatch, capsys
):
    # Said before it starts, not discovered afterwards by an operator whose
    # scheduled job never finished.
    monkeypatch.setattr(cli.VideoSource, "is_live", property(lambda self: True))

    class Stopped(RuntimeError):
        pass

    class FakePipeline:
        def __init__(self, *args, **kwargs):
            from sentinel.pipeline import PipelineStats

            self.stats = PipelineStats()

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def run(self):
            return iter(())

    monkeypatch.setattr(cli, "Pipeline", FakePipeline)

    cli.main([
        "--database", str(database), "--quiet", "run", str(reference_video),
    ])

    printed = capsys.readouterr().err
    assert "unbounded" in printed
    assert "--for" in printed and "--frames" in printed
