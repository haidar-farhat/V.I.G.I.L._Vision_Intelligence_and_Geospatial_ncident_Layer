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
import re
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

    # Opened and run — a missing-file refusal is 2 and opens nothing. The run
    # then got no frame from a live source, which is a starved run: 1.
    assert cli.main(["--database", str(database), "--quiet", "run", "device:0"]) == 1
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


# --------------------------------------------------- which zones a run watches


YARD = "Yard:33.8940,35.5016;33.8940,35.5020;33.8936,35.5020;33.8936,35.5016"
DOCK = "Dock:33.8950,35.5016;33.8950,35.5020;33.8946,35.5020;33.8946,35.5016"


class CapturingPipeline:
    """Stands in for `Pipeline` and keeps what the CLI built it with.

    The question these tests ask is "which zones, with which filters, reached
    the pipeline" — a question about the wiring, which a real decode of the
    reference scene would answer slowly and by inference from event counts.
    """

    built: list[dict] = []

    def __init__(self, source, detector, **kwargs):
        from sentinel.pipeline import PipelineStats

        CapturingPipeline.built.append(dict(kwargs, source_id=source.source_id))
        self.stats = PipelineStats()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def run(self):
        return iter(())


@pytest.fixture
def capturing_pipeline(monkeypatch):
    CapturingPipeline.built = []
    monkeypatch.setattr(cli, "Pipeline", CapturingPipeline)
    return CapturingPipeline.built


def test_a_class_filter_is_parsed_into_a_zone_name_and_its_labels():
    name, classes = cli._zone_classes("Loading Yard = person, car ")

    assert name == "Loading Yard"
    assert classes == frozenset({"person", "car"})


@pytest.mark.parametrize("text", ["Yard", "=person", "Yard=", "Yard=,"])
def test_a_class_filter_without_a_zone_or_without_labels_is_refused(text: str):
    # `Yard=` is a filter somebody forgot to finish far more often than a
    # decision to watch everything, and leaving the option off already means
    # that.
    with pytest.raises(argparse.ArgumentTypeError):
        cli._zone_classes(text)


def test_zone_classes_reach_the_zone_the_run_watches_and_are_stored_with_it(
    reference_video: Path, database: Path, capturing_pipeline
):
    code = cli.main([
        "--database", str(database), "--quiet",
        "run", str(reference_video),
        "--zone", YARD, "--zone", DOCK,
        "--zone-classes", "Yard=person", "--zone-classes", "yard=car",
    ])

    assert code == 0
    zones = {zone.name: zone for zone in capturing_pipeline[0]["zones"]}
    # Two filters naming the same zone — by name and by its slug — combine,
    # rather than the second silently dropping the class the first typed.
    assert zones["Yard"].classes == frozenset({"person", "car"})
    assert zones["Dock"].classes == frozenset(), "an unfiltered zone watches anything"

    # The filter is written back with the zone, so this is the way a filter
    # gets set without opening the console.
    with Store(database) as store:
        stored = {zone.name: zone for zone in store.zones()}
    assert stored["Yard"].classes == frozenset({"person", "car"})


def test_zone_classes_naming_a_zone_no_zone_declared_is_refused(
    reference_video: Path, database: Path, capturing_pipeline, capsys
):
    # Refused even though the database holds a zone of that name: a run that
    # rewrote a stored zone's filter would change what the console watches
    # from then on, with no audit row naming who did it.
    with Store(database) as store:
        store.save_zone(cli._zone(DOCK))

    code = cli.main([
        "--database", str(database), "--quiet",
        "run", str(reference_video),
        "--zone", YARD, "--zone-classes", "Dock=person",
    ])

    assert code == 2
    printed = capsys.readouterr().err
    assert "Dock" in printed and "--zone" in printed
    assert capturing_pipeline == [], "nothing was opened"
    with Store(database) as store:
        assert store.zones()[0].classes == frozenset(), "the stored zone was not touched"


def test_zone_classes_reach_the_node_too(database: Path, monkeypatch, tmp_path: Path):
    built: list[dict] = []

    class CapturingNode:
        def __init__(self, database, **kwargs):
            built.append(kwargs)
            self.store = Store(database)
            self.cameras = ()
            self.zones = tuple(kwargs["zones"])
            self.rules = ()

        def add_camera(self, source, *, camera_id=None, pose=None):
            pass

        def run_forever(self, *, until=None):
            pass

        def summary(self):
            return ""

        def close(self):
            self.store.close()

    monkeypatch.setattr(cli, "Node", CapturingNode)
    video = tmp_path / "gate.mp4"
    video.write_bytes(b"")

    code = cli.main([
        "--database", str(database), "--quiet",
        "node", str(video), "--for", "1",
        "--zone", YARD, "--zone-classes", "Yard=person",
    ])

    assert code == 0
    assert [zone.classes for zone in built[0]["zones"]] == [frozenset({"person"})]

    refused = cli.main([
        "--database", str(database), "--quiet",
        "node", str(video), "--for", "1", "--zone-classes", "Yard=person",
    ])
    assert refused == 2
    assert len(built) == 1, "a refused command builds no node"


def test_a_run_with_no_zone_restores_the_stored_zones_and_says_what_they_watch(
    reference_video: Path, database: Path, capturing_pipeline, capsys
):
    # The gap the real camera found: a database whose one zone watched
    # `person`, a run that built its zones from --zone alone, and a report of
    # "No events" from a run that was watching nothing.
    from dataclasses import replace

    with Store(database) as store:
        store.save_zone(replace(cli._zone(YARD), classes=frozenset({"person"})))
        store.save_zone(cli._zone(DOCK))

    code = cli.main([
        "--database", str(database), "--quiet", "run", str(reference_video),
    ])

    assert code == 0
    watched = {zone.name: zone.classes for zone in capturing_pipeline[0]["zones"]}
    assert watched == {"Yard": frozenset({"person"}), "Dock": frozenset()}

    printed = capsys.readouterr()
    assert "2 restored from the database" in printed.out
    assert "Yard" in printed.out and "watches person" in printed.out
    assert "Dock" in printed.out and "every class" in printed.out
    # And the reason this run can raise nothing in Yard, said before it starts.
    assert "Yard" in printed.err and "--model" in printed.err


def test_a_run_given_zones_uses_only_those_and_says_which_stored_ones_it_left(
    reference_video: Path, database: Path, capturing_pipeline, capsys
):
    with Store(database) as store:
        store.save_zone(cli._zone(DOCK))

    code = cli.main([
        "--database", str(database), "--quiet",
        "run", str(reference_video), "--zone", YARD,
    ])

    assert code == 0
    assert [zone.name for zone in capturing_pipeline[0]["zones"]] == ["Yard"]

    printed = capsys.readouterr().out
    assert "1 from --zone" in printed
    assert "not used this run: 1 stored zone(s)" in printed and "Dock" in printed
    with Store(database) as store:
        assert sorted(zone.name for zone in store.zones()) == ["Dock", "Yard"], (
            "explicit wins for the run; the stored zone is not deleted"
        )


def test_a_run_on_a_database_with_no_zones_says_so(
    reference_video: Path, database: Path, capturing_pipeline, capsys
):
    assert cli.main([
        "--database", str(database), "--quiet", "run", str(reference_video),
    ]) == 0

    assert capturing_pipeline[0]["zones"] == []
    assert "zones       none" in capsys.readouterr().out


# ------------------------------------------------ retention reaches the register


DAY = 86_400_000


def enrol_two_vans(database: Path, *, days_ago: float) -> tuple[str, str]:
    """A van enrolled `days_ago`, and a pinned one enrolled the same day.

    Returns the two identifier ids. Plates rather than faces because a plate
    needs no encoder; the sweep prices the kinds differently but selects them
    the same way.
    """
    import time

    from sentinel.registry import Plate

    then = int(time.time() * 1000) - int(days_ago * DAY)
    with Store(database) as store:
        gone = store.register.enrol(
            subject_id="veh-old", display_name="Old van",
            identifier=Plate("B 7421"), actor="operator:nadia",
            basis="site access list", now_millis=then,
        )
        kept = store.register.enrol(
            subject_id="veh-pinned", display_name="Pinned van",
            identifier=Plate("C 1122"), actor="operator:nadia",
            basis="site access list", now_millis=then,
        )
        store.register.set_pinned("veh-pinned", True, actor="operator:nadia")
    return gone.identifier.id, kept.identifier.id


def test_retention_reports_the_register_sweep_without_apply_and_touches_nothing(
    database: Path, capsys
):
    enrol_two_vans(database, days_ago=400)

    code = cli.main([
        "--database", str(database), "--quiet",
        "retention", "--min-free-gib", "0",
    ])

    assert code == 0
    printed = capsys.readouterr().out
    assert "2 identifier(s) examined" in printed
    assert "would delete    1 identifier(s)" in printed
    assert "kept        1 identifier(s) past retention" in printed
    assert "Add --apply" in printed

    with Store(database) as store:
        assert len(store.register.identifiers("veh-old")) == 1, "a report deleted"
        assert all(
            row["action"] != "register.sweep" for row in store.audit_trail()
        ), "a report audited a sweep that did not happen"


def test_retention_sweeps_an_expired_identifier_keeps_a_pinned_one_and_audits_it(
    database: Path, capsys
):
    gone, kept = enrol_two_vans(database, days_ago=400)

    code = cli.main([
        "--database", str(database), "--quiet",
        "retention", "--apply", "--min-free-gib", "0",
    ])

    assert code == 0
    printed = capsys.readouterr().out
    assert "deleted    1 identifier(s)" in printed
    assert "kept        1 identifier(s) past retention" in printed

    with Store(database) as store:
        assert store.register.identifiers("veh-old") == ()
        assert [row.id for row in store.register.identifiers("veh-pinned")] == [kept]
        assert store.register.subject("veh-old") is not None, (
            "the name survives; forgetting it is a person's decision"
        )
        sweeps = [row for row in store.audit_trail() if row["action"] == "register.sweep"]

    assert len(sweeps) == 1
    detail = json.loads(sweeps[0]["detail"])
    assert detail["deleted_ids"] == [gone]
    assert detail["kept_pinned_ids"] == [kept]
    assert "7421" not in sweeps[0]["detail"], "ids and counts, never a plate"


def test_the_report_promises_exactly_what_apply_then_does(database: Path, capsys):
    # The dry run reads the register with its own selection, because the sweep
    # has none. The two must agree, or the report understates the sweep.
    enrol_two_vans(database, days_ago=400)
    arguments = ["--database", str(database), "--quiet", "retention", "--min-free-gib", "0"]

    cli.main([*arguments, "--plate-days", "500"])
    inside = capsys.readouterr().out
    cli.main([*arguments, "--plate-days", "500", "--apply"])
    applied_inside = capsys.readouterr().out

    assert "would delete    0 identifier(s)" in inside
    assert "deleted    0 identifier(s)" in applied_inside

    cli.main(arguments)
    report = capsys.readouterr().out
    cli.main([*arguments, "--apply"])
    applied = capsys.readouterr().out

    assert "would delete    1 identifier(s)" in report
    assert "deleted    1 identifier(s)" in applied


def test_a_negative_identifier_retention_is_refused(database: Path):
    assert cli.main([
        "--database", str(database), "--quiet", "retention", "--face-days", "-1",
    ]) == 2


class _Source:
    def __init__(self, live: bool):
        self.is_live = live
        self.display_url = "device:0"


def test_a_live_run_that_got_no_frames_is_called_starved():
    """Two processes on one camera: the second reconnected, analysed one frame in
    ten seconds, printed its summary and exited 0 — a run that worked, to a
    scheduled job. Measured through the packaged binary."""
    assert cli._starvation(_Source(live=True), 0, 12.0) is not None
    assert "another program" in cli._starvation(_Source(live=True), 1, 10.0)
    # A healthy camera, a short bounded run, and a file are not starved.
    assert cli._starvation(_Source(live=True), 150, 10.0) is None
    assert cli._starvation(_Source(live=True), 1, 1.0) is None
    assert cli._starvation(_Source(live=False), 0, 60.0) is None


def test_a_starved_live_run_exits_non_zero_and_says_so(
    reference_video: Path, database: Path, monkeypatch, capsys
):
    from sentinel import decode
    from sentinel.pipeline import Pipeline

    def nothing(self):
        return iter(())

    monkeypatch.setattr(Pipeline, "run", nothing)
    monkeypatch.setattr(decode.VideoSource, "is_live", property(lambda self: True))

    code = cli.main([
        "--database", str(database), "--quiet",
        "run", str(reference_video), "--frames", "3",
    ])

    assert code == 1
    assert "STARVED" in capsys.readouterr().err


# ---------------------------------------------------------------- the basemap


def placement_of(pose) -> str:
    """A pose in `--place` syntax, optics included, so it comes back exactly."""
    return (
        f"{pose.position.lat},{pose.position.lon},{pose.mount_height},"
        f"{pose.heading},{pose.pitch},{pose.horizontal_fov},{pose.vertical_fov},"
        f"{pose.range_meters}"
    )


def test_basemap_build_on_the_reference_video_writes_both_files_and_exits_0(
    reference_video: Path, reference_pose, tmp_path: Path, capsys
):
    from sentinel.basemap import load_basemap

    out = tmp_path / "basemap"
    code = cli.main([
        "--quiet", "basemap", "build", str(reference_video),
        "--id", "gate", "--place", placement_of(reference_pose),
        "--for", "3", "--cell", "0.5", "--out", str(out),
    ])
    printed = capsys.readouterr()

    assert code == 0, printed.err
    assert (out / "basemap.png").is_file() and (out / "basemap.json").is_file()
    asset = load_basemap(out)
    assert asset is not None
    # The stored poses are the ones given, exactly — not re-rounded on the way.
    assert asset.poses == {"gate": reference_pose}
    assert asset.cameras == ("gate",)
    assert asset.covered_cells > 1_000
    # Thinned to at most four frames a second of media time: three seconds of
    # a 15 fps file is a dozen frames, not forty-five.
    assert 9 <= asset.frames_used <= 13
    # Never a frame, and never the source: a camera URL carries a credential.
    text = (out / "basemap.json").read_text(encoding="utf-8")
    assert reference_video.name not in text
    assert "basemap" in printed.out
    assert str(out / "basemap.png") in printed.out and str(out / "basemap.json") in printed.out
    assert "gate" in printed.out

    # And `show` reads it back, verifying the fingerprint on the way.
    assert cli.main(["--quiet", "basemap", "show", "--dir", str(out)]) == 0
    shown = capsys.readouterr().out
    assert asset.fingerprint in shown
    assert "gate" in shown and "cell" in shown


def test_basemap_show_with_no_asset_exits_1(tmp_path: Path, capsys):
    code = cli.main(["--quiet", "basemap", "show", "--dir", str(tmp_path / "nothing")])

    assert code == 1
    printed = capsys.readouterr().out
    assert "no basemap" in printed
    assert "basemap build" in printed, "it must say how to get one"


def test_basemap_show_refuses_a_tampered_asset(reference_pose, tmp_path: Path, capsys):
    import numpy as np

    from sentinel.basemap import BasemapBuilder, save_basemap

    builder = BasemapBuilder(cell_size_m=1.0)
    for index in range(3):
        builder.feed("gate", reference_pose, np.full((480, 640, 3), 90, np.uint8), 1_000.0 + index)
    _, json_path = save_basemap(builder.build(now=1_100.0), tmp_path)
    document = json.loads(json_path.read_text(encoding="utf-8"))
    document["poses"]["gate"]["heading"] = 90.0
    json_path.write_text(json.dumps(document), encoding="utf-8")

    code = cli.main(["--quiet", "basemap", "show", "--dir", str(tmp_path)])

    assert code == 1
    assert "fingerprint" in capsys.readouterr().err


def test_basemap_build_refuses_a_camera_without_a_bound_before_opening_it(
    monkeypatch, capsys
):
    import cv2

    def forbidden(*args, **kwargs):
        raise AssertionError("a camera was opened")

    monkeypatch.setattr(cv2, "VideoCapture", forbidden)

    code = cli.main([
        "--quiet", "basemap", "build", "device:0", "--place", "33.8938,35.5018,6,180,-22",
    ])

    assert code == 2
    assert "--for" in capsys.readouterr().err


@pytest.mark.parametrize(
    "arguments",
    [
        ["--place", "33.8938,35.5018,6,180,-22", "--place", "33.8938,35.5018,6,180,-22"],
        ["--place", "33.8938,35.5018,6,180,-22", "--id", "a", "--id", "b"],
        ["--place", "33.8938,35.5018,6,180,-22", "--for", "0"],
        ["--place", "33.8938,35.5018,6,180,-22", "--cell", "0"],
    ],
)
def test_an_impossible_basemap_build_is_refused_before_anything_opens(
    arguments, reference_video: Path, tmp_path: Path
):
    code = cli.main([
        "--quiet", "basemap", "build", str(reference_video), *arguments,
        "--out", str(tmp_path / "bm"),
    ])

    assert code == 2
    assert not (tmp_path / "bm").exists()


def test_basemap_build_refuses_a_missing_file(tmp_path: Path):
    code = cli.main([
        "--quiet", "basemap", "build", str(tmp_path / "absent.mp4"),
        "--place", "33.8938,35.5018,6,180,-22",
    ])

    assert code == 2


def test_a_starved_basemap_build_exits_1_and_writes_nothing(
    reference_video: Path, tmp_path: Path, monkeypatch, capsys
):
    import time

    from sentinel import decode

    monkeypatch.setattr(decode.VideoSource, "is_live", property(lambda self: True))

    class Silent:
        """A camera another program holds: it opens and never delivers."""

        def __init__(self, source):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self, timeout=10.0):
            time.sleep(min(timeout, 0.02))
            return None

    monkeypatch.setattr(decode, "LiveStream", Silent)

    code = cli.main([
        "--quiet", "basemap", "build", str(reference_video),
        "--place", "33.8938,35.5018,6,180,-22", "--for", "0.2",
        "--out", str(tmp_path / "bm"),
    ])

    assert code == 1
    assert "STARVED" in capsys.readouterr().err
    assert not (tmp_path / "bm").exists()


def test_a_basemap_build_from_a_camera_that_sees_no_ground_exits_1(
    reference_video: Path, tmp_path: Path, capsys
):
    # `_pose` refuses a non-negative pitch, so this is the other way to point a
    # camera at nothing: tilted down, but not enough for the bottom of the
    # frame to reach the ground within any range.
    code = cli.main([
        "--quiet", "basemap", "build", str(reference_video),
        "--place", "33.8938,35.5018,6,180,-0.01,62,0.001,90", "--for", "1",
        "--out", str(tmp_path / "bm"),
    ])
    printed = capsys.readouterr()

    assert code == 1
    assert "no ground" in printed.err
    assert not (tmp_path / "bm").exists()
    # Frames were read and taken, and none reached a median — and the count
    # says so. It used to report the frames offered as "fed to the median".
    read = re.search(r"(\d+) frame\(s\) read in", printed.out)
    assert read is not None and int(read.group(1)) > 0, printed.out
    assert "0 fed to the median" in printed.out


def test_basemap_build_refuses_a_repeated_id_before_anything_opens(
    reference_video: Path, tmp_path: Path, monkeypatch, capsys
):
    """Two sources under one id would be merged or refused mid-build; refused first."""
    import cv2

    def forbidden(*args, **kwargs):
        raise AssertionError("a source was opened")

    monkeypatch.setattr(cv2, "VideoCapture", forbidden)

    code = cli.main([
        "--quiet", "basemap", "build", str(reference_video), str(reference_video),
        "--id", "gate", "--id", "gate", "--place", "33.8938,35.5018,6,180,-22",
        "--out", str(tmp_path / "bm"),
    ])

    assert code == 2
    assert "gate" in capsys.readouterr().err
    assert not (tmp_path / "bm").exists()


def test_feed_basemap_refuses_a_live_source_without_a_bound_before_opening_it(
    reference_video: Path, reference_pose, monkeypatch
):
    """A contract, not an `assert`: it held under `python -O` as `started + None`."""
    from sentinel import decode
    from sentinel.decode import VideoSource

    class Forbidden:
        def __init__(self, source):
            raise AssertionError("the stream was opened")

    monkeypatch.setattr(decode, "LiveStream", Forbidden)
    source = VideoSource(str(reference_video), source_id="gate", live=True)

    with pytest.raises(ValueError, match="--for"):
        cli._feed_basemap(object(), source, reference_pose, duration=None, per_second=4.0)
