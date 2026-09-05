"""Headless analysis: ``python -m sentinel``.

The console is a *viewer*. This is the same pipeline with no window, which is
what a container, a scheduled job, or a developer chasing a bug over one file
actually wants.

**What this is not.** It is not a daemon. It processes the sources it is given,
in order, to completion, and exits — a file is evidence and every frame of it is
processed, so a replay reproduces the original result exactly. Running cameras
continuously and unattended is a different program that does not exist yet; see
[ROADMAP.md](../../ROADMAP.md) item 1.2. Pointing this at an RTSP URL works and
will run until the stream ends or you interrupt it, but nothing supervises it.

Everything it writes goes to the same database the console reads, so the usual
sequence is: analyse here, look at the incidents there.

    python -m sentinel run gate.mp4 --place 33.8938,35.5018,6,180,-22
    python -m sentinel incidents
    python -m sentinel export INC-... --to ./evidence
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

from . import devices, logs, paths, telemetry
from .recording import RetentionPolicy, apply_retention
from .registry import Identifier, Register
from .registry import RetentionPolicy as IdentifierRetentionPolicy
from .core import CameraPose, LatLon
from .decode import VideoSource
from .detect import MotionDetector
from .evidence import (
    DEFAULT_LEAD_SECONDS,
    DEFAULT_TRAIL_SECONDS,
    ExportError,
    coverage_for,
    export_incident,
)
from .events import Severity, default_rules
from .node import Node
from .incidents import Correlator
from .pipeline import Pipeline
from .store import Store, default_database_path
from .zones import Zone, ZoneKind

_log = logs.get(__name__)

#: How the actor is recorded in the audit log. There is no authentication yet,
#: so there is nobody to name; recording the truth is better than inventing an
#: operator, because an audit trail with a false entry is worse than none.
ACTOR = "cli (unauthenticated)"


# ------------------------------------------------------------------- arguments


def _pose(text: str) -> CameraPose:
    """``lat,lon,height,heading,pitch[,hfov,vfov,range]``.

    Positional and terse because it is typed, and because every field is
    mandatory up to pitch: a camera without one of them cannot be placed, and a
    default would put an object on a map somewhere nobody measured.
    """
    parts = [part.strip() for part in text.split(",")]
    if len(parts) not in (5, 8):
        raise argparse.ArgumentTypeError(
            "--place takes lat,lon,height,heading,pitch or "
            "lat,lon,height,heading,pitch,hfov,vfov,range"
        )

    try:
        values = [float(part) for part in parts]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"--place: {error}") from error

    lat, lon, height, heading, pitch = values[:5]
    if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 180.0:
        raise argparse.ArgumentTypeError("--place: latitude or longitude out of range")
    if height <= 0.0:
        raise argparse.ArgumentTypeError("--place: mount height must be above zero")
    if pitch >= 0.0:
        raise argparse.ArgumentTypeError(
            "--place: pitch is negative looking down, which is the normal "
            "mounting. A camera level or tilted up sees no ground to project on."
        )

    optics = values[5:] or [70.0, 40.0, 90.0]
    return CameraPose(
        position=LatLon(lat, lon),
        mount_height=height,
        heading=heading,
        pitch=pitch,
        horizontal_fov=optics[0],
        vertical_fov=optics[1],
        range_meters=optics[2],
    )


def _zone(text: str) -> Zone:
    """``name:lat,lon;lat,lon;lat,lon[;...]`` — a named polygon, three vertices up."""
    name, sep, ring = text.partition(":")
    if not sep or not ring:
        raise argparse.ArgumentTypeError("--zone takes name:lat,lon;lat,lon;lat,lon")

    points = []
    for vertex in ring.split(";"):
        try:
            lat, lon = (float(part) for part in vertex.split(","))
        except ValueError as error:
            raise argparse.ArgumentTypeError(f"--zone {name}: {error}") from error
        points.append(LatLon(lat, lon))

    if len(points) < 3:
        raise argparse.ArgumentTypeError(
            f"--zone {name}: a polygon needs at least three vertices"
        )

    return Zone(
        id=_zone_id(name),
        name=name.strip(),
        kind=ZoneKind.RESTRICTED,
        ring=tuple(points),
        enter_after_millis=600,
    )


def _zone_id(name: str) -> str:
    """The id a `--zone` name becomes, in one place.

    `--zone-classes` finds its zone by the same rule, so ``Loading Yard`` and
    ``loading-yard`` name the same polygon on both options. Two spellings of
    the slug would let a filter typed for a zone silently attach to nothing.
    """
    return name.strip().lower().replace(" ", "-")


def _zone_classes(text: str) -> tuple[str, frozenset[str]]:
    """``NAME=label,label`` — which detector labels one `--zone` acts on.

    The labels are the model's own strings, matched exactly, because a model
    file's names are the only vocabulary a site has. An empty list is refused
    rather than read as "everything": leaving the option off already means
    that, and ``Yard=`` is far more often a filter somebody forgot to finish
    than a decision to watch every class.
    """
    name, sep, labels = text.partition("=")
    if not sep or not name.strip():
        raise argparse.ArgumentTypeError(
            "--zone-classes takes NAME=label,label — the NAME of a --zone in the "
            "same command, then the detector's own labels"
        )
    classes = frozenset(label.strip() for label in labels.split(",") if label.strip())
    if not classes:
        raise argparse.ArgumentTypeError(
            f"--zone-classes {name.strip()}: no labels. Leave the option off to "
            "watch every class the detector reports."
        )
    return name.strip(), classes


def _apply_zone_classes(
    zones: list[Zone], filters: list[tuple[str, frozenset[str]]]
) -> tuple[list[Zone], str | None]:
    """Attach each `--zone-classes` filter to the `--zone` it names.

    Returns the zones with their filters, or the sentence to refuse with. The
    filter attaches only to a zone declared by ``--zone`` *in this command*,
    never to one the database already holds: a run that quietly rewrote a
    stored zone's filter would change what the console watches from then on,
    with no console open and no audit row naming the operator who did it.

    Two filters naming the same zone are combined, because ``Yard=person``
    followed by ``Yard=car`` reads as both; a later one replacing an earlier
    one would drop a class the operator typed.
    """
    by_id = {zone.id: zone for zone in zones}
    for name, classes in filters:
        zone = by_id.get(_zone_id(name))
        if zone is None:
            declared = ", ".join(zone.name for zone in zones) or "none"
            return zones, (
                f"--zone-classes names {name!r}, but no --zone in this command "
                f"declares it (declared: {declared}). The filter applies only to "
                "a zone given by --zone in the same command; a zone stored in the "
                "database keeps the filter set in the console."
            )
        by_id[zone.id] = replace(zone, classes=zone.classes | classes)
    return [by_id[zone.id] for zone in zones], None


def _watches(zone: Zone) -> str:
    """What one zone acts on, in the words the summary line uses."""
    if zone.classes:
        return "watches " + ", ".join(sorted(zone.classes))
    return "watches every class the detector reports"


def _describe_zones(
    zones: list[Zone], *, explicit: bool, unused_stored: list[Zone], model_given: bool
) -> None:
    """Say which zones this invocation runs with, and what each one watches.

    Printed before the first frame, because the end-of-run summary can only
    count events, and "No events" from a run whose zones all filtered for a
    label the detector never produces reads as an empty scene. It read that
    way once on a real camera: a database whose one zone watched ``person``,
    a ``run`` that ignored stored zones entirely, and a report that said
    nothing crossed a rule when nothing was being watched at all.
    """
    if not zones:
        print(
            "zones       none — the database holds none and no --zone was given, "
            "so no zone rule runs this time"
        )
        return

    origin = "from --zone" if explicit else "restored from the database"
    print(f"zones       {len(zones)} {origin}")
    for zone in zones:
        print(f"  {zone.name:<20} {_watches(zone)}")

    if explicit and unused_stored:
        # Explicit wins, and the operator sees what that cost.
        names = ", ".join(zone.name for zone in unused_stored)
        print(
            f"  not used this run: {len(unused_stored)} stored zone(s) --zone did "
            f"not name ({names})"
        )

    if not model_given and any(zone.classes for zone in zones):
        # The rule that matters most: a filtered zone never fires from a motion
        # detector, which cannot say what it saw. A run that watched nothing
        # for that reason would otherwise report "No events" as an empty scene.
        filtered = ", ".join(zone.name for zone in zones if zone.classes)
        print(
            f"  WARNING: {filtered} filter by class, but without --model the "
            "motion detector names nothing, so these zones cannot fire this run.",
            file=sys.stderr,
        )


def _ring(text: str) -> list[LatLon]:
    """``lat,lon;lat,lon;lat,lon[;...]`` — a boundary, three vertices up.

    The same vertex syntax as `--zone` without the name, so an operator who has
    typed one has typed the other.
    """
    points = []
    for vertex in text.split(";"):
        try:
            lat, lon = (float(part) for part in vertex.split(","))
        except ValueError as error:
            raise argparse.ArgumentTypeError(f"--site: {error}") from error
        points.append(LatLon(lat, lon))

    if len(points) < 3:
        raise argparse.ArgumentTypeError(
            "--site: a boundary needs at least three vertices"
        )
    return points


def _detector(args: argparse.Namespace):
    """The detector this invocation asked for.

    `detector_for` reads the model to decide what it is, so `--model` on a
    segmentation graph gives masks and on a detection graph gives boxes,
    without the operator having to say which they handed over.
    """
    from .detect import detector_for

    model = getattr(args, "model", None)
    if model is None:
        return detector_for(None, detect_scale=args.detect_scale)
    return detector_for(model)


#: One definition, in `events.py`, shared by the CLI, the node and the console.
#: There were two copies and a third was about to appear; rule sets that drift
#: produce two deployments that disagree about what an incident is.
_rules = default_rules


# --------------------------------------------------------------------- run


#: A live run that averaged fewer analysed frames per second than this, after
#: at least this many seconds, was not watching. One frame in ten seconds is
#: what a camera another program holds looks like — the driver hands over a
#: frame on reconnect and nothing after — and until this existed such a run
#: printed its summary and exited 0, which to a scheduled job is a run that
#: worked. Measured: two `sentinel run device:0` processes at once, the
#: second reconnected once, analysed one frame, and exited 0.
STARVED_BELOW_FPS = 1.0
STARVED_AFTER_SECONDS = 5.0


def _starvation(source, frames: int, elapsed: float) -> str | None:
    """Why a live run's frame count means it was not watching, or ``None``."""
    if not source.is_live:
        return None
    if frames == 0:
        return (
            f"no frame arrived from {source.display_url} in {elapsed:.0f}s. "
            "Is another program using the camera?"
        )
    if elapsed >= STARVED_AFTER_SECONDS and frames / elapsed < STARVED_BELOW_FPS:
        return (
            f"{frames} frame(s) in {elapsed:.0f}s ({frames / elapsed:.1f}/s) from "
            f"{source.display_url}. A camera delivering this little is usually "
            "held by another program."
        )
    return None


def _run(args: argparse.Namespace) -> int:
    explicit, refusal = _apply_zone_classes(
        list(args.zone or []), list(args.zone_classes or [])
    )
    if refusal is not None:
        print(f"error: {refusal}", file=sys.stderr)
        return 2

    # Every argument is checked before any source is constructed. An earlier
    # version validated `--id` *after* the loop that indexed it, so the wrong
    # number of ids raised IndexError and the operator got a traceback instead
    # of the sentence saying what they had typed wrong.
    if args.id and len(args.id) != len(args.source):
        print(
            f"error: {len(args.id)} --id for {len(args.source)} source(s). "
            "Give one per source, or none and take cam-01, cam-02, ...",
            file=sys.stderr,
        )
        return 2
    if args.place and len(args.place) not in (1, len(args.source)):
        print(
            f"error: {len(args.place)} --place for {len(args.source)} source(s). "
            "Give one, applied to every source, or one per source.",
            file=sys.stderr,
        )
        return 2
    if not 0.1 <= args.detect_scale <= 1.0:
        print("error: --detect-scale must be between 0.1 and 1.0", file=sys.stderr)
        return 2
    if args.duration is not None and args.duration <= 0:
        print("error: --for must be greater than zero", file=sys.stderr)
        return 2
    if args.frames is not None and args.frames < 1:
        print("error: --frames must be at least 1", file=sys.stderr)
        return 2
    if args.segment_seconds <= 0:
        print("error: --segment-seconds must be greater than zero", file=sys.stderr)
        return 2

    record_to = None
    if args.record is not None:
        record_to = Path(args.record) if args.record else paths.recordings_directory()
        record_to.mkdir(parents=True, exist_ok=True)

    sources: list[VideoSource] = []
    for index, source in enumerate(args.source, start=1):
        # A local file is checked here rather than inside the decoder, so a typo
        # fails before any camera is opened rather than half way through a run.
        # A device is neither a file nor a URL: `VideoSource` validates the
        # index at construction, and whether it opens is a question only the
        # operating system can answer.
        if devices.is_device_source(source):
            # A malformed index is a mistyped command line, not a failure: exit
            # 2 with the sentence, the same as every other bad argument, rather
            # than letting the exception become a traceback at an operator.
            try:
                devices.device_index(source)
            except devices.DeviceError as error:
                print(f"error: {error}", file=sys.stderr)
                return 2
        elif "://" not in source and not Path(source).exists():
            print(f"error: no such file: {source}", file=sys.stderr)
            return 2
        sources.append(
            VideoSource(source, source_id=args.id[index - 1] if args.id else f"cam-{index:02d}")
        )

    store = Store(args.database or default_database_path())
    all_events = []
    recording_failed = False
    starved = False

    try:
        # Explicit wins; otherwise the database's own zones, the way `Node`
        # and the console already behave. Until this matched, a run with no
        # --zone on a database whose zones carried filters watched nothing and
        # reported "No events" as though the scene were empty.
        stored = store.zones()
        if explicit:
            zones = explicit
            named = {zone.id for zone in explicit}
            unused = [zone for zone in stored if zone.id not in named]
        else:
            zones, unused = stored, []
        rules = _rules(zones)
        print()
        _describe_zones(
            zones, explicit=bool(explicit), unused_stored=unused,
            model_given=args.model is not None,
        )

        for index, source in enumerate(sources):
            pose = None
            if args.place:
                pose = args.place[index] if len(args.place) > 1 else args.place[0]

            store.save_camera(
                source.source_id, source.source_id, source.display_url, pose=pose
            )

            if source.is_live and args.duration is None and args.frames is None:
                # Said before it starts, not discovered afterwards. A live source
                # has no end, and on Windows an external SIGINT does not reach a
                # Python process at all — measured — so a scheduled job or a
                # container that started one without a bound has no way to stop
                # it short of killing the process.
                print(
                    f"\n{source.source_id}: live source, unbounded. It runs until "
                    "the stream ends or you press Ctrl-C at this terminal. For a "
                    "scheduled job or a container, use --for SECONDS or --frames N.",
                    file=sys.stderr,
                )

            with Pipeline(
                source,
                # One detector per camera, never shared. MOG2 carries a
                # per-pixel model of *its* scene; feeding it two cameras
                # corrupts both models and every detection that comes out of
                # them. The console gets this right by construction — one
                # worker per camera — and this had to be made to match. A model
                # holds no per-scene state, but one session per camera keeps the
                # rule uniform and lets them run genuinely in parallel.
                _detector(args),
                record_to=record_to,
                segment_seconds=args.segment_seconds,
                # Indexed as each segment closes rather than at the end, so a
                # run that is interrupted still leaves findable footage.
                on_segment=store.save_segment if record_to else None,
                pose=pose,
                zones=zones,
                rules=rules,
                node_id=args.node,
            ) as pipeline:
                events = []
                deadline = (
                    time.monotonic() + args.duration
                    if args.duration is not None
                    else None
                )

                started = time.monotonic()
                for count, result in enumerate(pipeline.run(), start=1):
                    events.extend(result.events)

                    # Checked after the frame, so `--frames 1` processes one
                    # frame rather than none.
                    if args.frames is not None and count >= args.frames:
                        _log.info("%s: stopping after %d frames", source.source_id, count)
                        break
                    if deadline is not None and time.monotonic() >= deadline:
                        _log.info(
                            "%s: stopping after %.0fs", source.source_id, args.duration
                        )
                        break

                store.save_events(events)
                all_events.extend(events)

                print(f"\n{source.source_id}  ({source.display_url})")
                print(pipeline.stats.summary())

                # A live source that delivered nothing, or next to nothing, is
                # a run that did not watch — said on stderr beside the summary
                # and carried into the exit code, because a scheduled job
                # reads only the exit code.
                starvation = _starvation(
                    source, pipeline.stats.frames, time.monotonic() - started
                )
                if starvation is not None:
                    print(f"  STARVED             {starvation}", file=sys.stderr)
                    starved = True

                # Only when recording was actually asked for. Reading the
                # attribute otherwise says nothing and couples this reporting
                # to a detail of the pipeline that a caller passing its own
                # pipeline-shaped object need not provide.
                recorder = pipeline.recorder if record_to is not None else None
                if recorder is not None:
                    recording_stats = recorder.stats
                    print(
                        f"recorded              {recording_stats.segments_written}"
                        f" segment(s), "
                        f"{recording_stats.bytes_written / 1024**2:.1f} MiB"
                    )
                    if recording_stats.frames_dropped:
                        print(
                            f"  dropped             {recording_stats.frames_dropped}"
                            f" frame(s)"
                            f" ({recording_stats.dropped_fraction:.0%}) — the writer"
                            " could not keep up"
                        )
                    if recording_stats.fault is not None:
                        # On stdout as well as in the log, because an operator
                        # reading the run's report must not have to also read
                        # the log to find out the recording stopped.
                        print(
                            f"  RECORDING FAILED    {recording_stats.fault}",
                            file=sys.stderr,
                        )
                        recording_failed = True

        # Only what --zone declared is written back. The restored zones came
        # from this table and rewriting them would say an edit happened.
        for zone in explicit:
            store.save_zone(zone)

        if not all_events:
            print("\nNo events. Nothing crossed a rule.")
            # A failed recording is a failed run even when the analysis found
            # nothing: a scheduled job that exits 0 is a job nobody looks at.
            return 1 if recording_failed or starved else 0

        # Across every source, deliberately: a camera correlating only its own
        # events raises one incident per camera for one intrusion, which is the
        # duplication this whole stage exists to remove.
        correlator = Correlator(zone_kinds={zone.id: zone.kind for zone in zones})
        incidents = correlator.correlate(all_events)

        for incident in incidents:
            store.save_incident(incident)
            store.audit(ACTOR, "incident.opened", incident.id, incident.summary)

        _report(all_events, incidents)

        if args.export:
            _export_all(incidents, Path(args.export), store)

        return 1 if recording_failed or starved else 0
    finally:
        store.close()
        for source in sources:
            source.close()


def _report(events, incidents) -> None:
    reduction = 100.0 * (1.0 - len(incidents) / len(events)) if events else 0.0

    print()
    print("=" * 66)
    print(f"  {len(events)} event(s)  ->  {len(incidents)} incident(s)"
          f"   ({reduction:.0f}% less for a person to read)")
    print("=" * 66)

    for incident in incidents:
        print()
        print(f"  {incident.id}")
        print(f"  {incident.severity.value:<8} risk {incident.risk.score}/100"
              f"   {len(incident.events)} event(s)"
              f"   {len(incident.cameras)} camera(s)")
        print(f"  {incident.summary}")
        # The factors, not just the score. A bare number invites an operator to
        # calibrate against it without understanding it, and then to ignore it
        # the first time it is wrong.
        for factor in incident.risk.factors:
            print(f"    · {factor.points:+5.1f}  {factor.name}: {factor.because}")


def _export_one(incident, destination: Path, store: Store, lead: float, trail: float):
    """Export one incident *with the footage that shows it*.

    This function exists because for a while the two export paths did neither
    half of it. Recording worked, coverage worked, preservation worked, and
    nothing in the shipped code ever called any of them — so every package came
    out with no video, and `preserved` was never set on any segment, which left
    retention free to delete the exact footage an incident depended on. The
    mechanism that prevents that was real, tested, and unreachable.

    Preservation happens *before* the copy and is kept even if the copy then
    fails. Over-preserving costs disk; under-preserving destroys evidence.
    """
    coverage = coverage_for(store, incident, lead_seconds=lead, trail_seconds=trail)

    clips = [segment.path for cover in coverage for segment in cover.segments]
    if clips:
        preserved = store.preserve_segments(clips)
        store.audit(
            ACTOR, "recording.preserved", incident.id,
            f"{preserved} segment(s) held as evidence and exempted from retention",
        )

    export = export_incident(
        incident, destination, exported_by=ACTOR, footage=coverage
    )
    store.audit(ACTOR, "incident.exported", incident.id, str(export.directory))
    return export, coverage


def _describe_coverage(coverage) -> None:
    """Say what the package has, and — the part that matters — what it lacks."""
    for cover in coverage:
        if cover.is_complete and cover.segments:
            print(f"      {cover.camera_id}: {len(cover.segments)} clip(s), complete")
            continue
        if not cover.segments:
            # A camera the incident names with nothing recorded is a finding,
            # not an absence, and it is invisible unless said out loud.
            print(f"      {cover.camera_id}: NO FOOTAGE", file=sys.stderr)
            continue
        missing = sum(end - start for start, end in cover.gaps) / 1000
        print(
            f"      {cover.camera_id}: {len(cover.segments)} clip(s), "
            f"{cover.covered_fraction:.0%} covered — {missing:.0f}s missing",
            file=sys.stderr,
        )


def _export_all(incidents, destination: Path, store: Store,
                lead: float = DEFAULT_LEAD_SECONDS,
                trail: float = DEFAULT_TRAIL_SECONDS) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    print()
    for incident in incidents:
        try:
            export, coverage = _export_one(incident, destination, store, lead, trail)
        except ExportError as error:
            print(f"  export failed for {incident.id}: {error}", file=sys.stderr)
            continue
        print(f"  {incident.id}  ->  {export.directory}")
        print(f"      manifest sha256  {export.manifest_sha256}")
        _describe_coverage(coverage)
    print("\n  Record those digests separately. They are what makes a package "
          "checkable later.")


# ------------------------------------------------------------------ inspection


def _incidents(args: argparse.Namespace) -> int:
    store = Store(args.database or default_database_path())
    try:
        rows = store.incidents(limit=args.limit)
        if not rows:
            print("No incidents recorded.")
            return 0
        for row in rows:
            print(f'{row["id"]}  {row["severity"]:<8} '
                  f'risk {row["risk_score"]:>5.1f}/100  '
                  f'{row["distinct_object_count"]} object(s)  {row["summary"]}')
        print()
        print(f"{len(rows)} incident(s). Export one with: "
              "sentinel export <id> --to <dir>")
        return 0
    finally:
        store.close()


def _export(args: argparse.Namespace) -> int:
    store = Store(args.database or default_database_path())
    try:
        incident = store.incident(args.id)
        if incident is None:
            print(f"error: no incident {args.id}", file=sys.stderr)
            return 2

        destination = Path(args.to)
        destination.mkdir(parents=True, exist_ok=True)
        export, coverage = _export_one(
            incident, destination, store, args.lead, args.trail
        )

        print(f"{len(export.files)} files written to {export.directory}")
        print(f"manifest sha256  {export.manifest_sha256}")
        _describe_coverage(coverage)
        return 0
    except ExportError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    finally:
        store.close()


def _site_coverage(args: argparse.Namespace) -> int:
    """What every placed camera covers of the site, and what it misses.

    The question an installer actually has, and the one a plan view cannot
    answer by eye: six cameras drawn as six overlapping wedges look like
    thorough coverage, and the four-metre corridor between two of them looks
    like nothing at all until somebody walks down it.
    """
    from .coverage import CoverageError, analyse

    store = Store(args.database or default_database_path())
    try:
        placed = {}
        unplaced = []
        for row in store.cameras():
            pose = store.camera_pose(row["id"])
            if pose is None:
                unplaced.append(row["id"])
            else:
                placed[row["id"]] = pose

        if not placed:
            print(
                "No camera on this node is placed, so there is no coverage to "
                "compute. Place them with `sentinel run --place` or in the "
                "console.",
                file=sys.stderr,
            )
            return 1

        try:
            result = analyse(args.site, placed)
        except CoverageError as error:
            print(f"error: {error}", file=sys.stderr)
            return 2

        print(result.describe())
        if unplaced:
            print()
            print(f"not counted, because unplaced   {', '.join(unplaced)}")

        if result.gaps:
            print()
            print("UNMONITORED AREAS, largest first:")
            for index, gap in enumerate(result.gaps[:10], start=1):
                centre = gap.ring[0]
                print(
                    f"  {index}. {gap.area_m2:,.0f} m²  near "
                    f"{centre.lat:.6f},{centre.lon:.6f}"
                )
            # Non-zero, because an uncovered site is a finding and a scheduled
            # check that exits 0 is a check nobody reads.
            return 1
        return 0
    finally:
        store.close()


def _coverage(args: argparse.Namespace) -> int:
    """What a placed camera can actually see, and where to put a zone.

    Exists because the first thing anybody does with `--zone` is put one
    somewhere the camera cannot see, get zero events, and conclude the detector
    is broken. A camera's *stated* range is not its coverage: a 6 m mast tilted
    22 degrees with a 36 degree vertical field covers 7 m to 86 m however large
    the number on the datasheet is, and it sees no ground at all at the mast.
    """
    from .core import destination_point, field_of_view, haversine_distance

    if getattr(args, "site", None):
        return _site_coverage(args)
    if args.place is None:
        print(
            "error: give --place for one camera's coverage, or --site for the "
            "whole node's",
            file=sys.stderr,
        )
        return 2

    pose = args.place
    footprint = field_of_view(pose, arc_segments=24)
    if not footprint:
        print(
            "This camera sees no ground at all. The bottom of the frame is above "
            "the horizon, so nothing can be projected and every object would be "
            "reported as not placed.",
            file=sys.stderr,
        )
        return 1

    distances = [haversine_distance(pose.position, point) for point in footprint]
    near, far = min(distances), max(distances)

    print(f"camera at        {pose.position.lat:.6f}, {pose.position.lon:.6f}")
    print(f"mast             {pose.mount_height:.1f} m, pitch {pose.pitch:.1f}°, "
          f"heading {pose.heading:.0f}°")
    print(f"stated range     {pose.range_meters:.0f} m")
    print(f"ground covered   {near:.1f} m to {far:.1f} m ahead")
    print()
    print("The near edge is where positions are most accurate: uncertainty grows")
    print("super-linearly with distance, so a zone there is one the system can")
    print("genuinely adjudicate rather than one it will mostly report UNCERTAIN.")
    print()

    radius = args.zone_radius
    centre = destination_point(pose.position, pose.heading, near + radius)
    ring = [destination_point(centre, bearing, radius) for bearing in (0.0, 90.0, 180.0, 270.0)]
    vertices = ";".join(f"{point.lat:.6f},{point.lon:.6f}" for point in ring)
    print(f"A {radius:.0f} m square just past that near edge:")
    print()
    print(f'  --zone "{args.zone_name}:{vertices}"')
    return 0


def _bytes(count: int) -> str:
    """Bytes at a scale a person can read.

    Fixed GiB made a real 2.8 MiB of preserved evidence print as "0.00 GiB",
    directly above the line saying three segments were preserved — two true
    statements that read as a contradiction.
    """
    for unit, size in (("GiB", 1024**3), ("MiB", 1024**2), ("KiB", 1024)):
        if count >= size:
            return f"{count / size:.2f} {unit}"
    return f"{count} bytes"


def _node(args: argparse.Namespace) -> int:
    """Run cameras unattended until stopped.

    The difference from `run` is not the analysis — it is the same pipeline —
    but the shape. `run` processes what it is given to completion and exits, so
    a replay is reproducible. `node` keeps cameras going, correlates across all
    of them on a cadence, and is what a machine with no display in a cupboard
    actually does.
    """
    zones, refusal = _apply_zone_classes(
        list(args.zone or []), list(args.zone_classes or [])
    )
    if refusal is not None:
        print(f"error: {refusal}", file=sys.stderr)
        return 2

    if args.place and len(args.place) not in (1, len(args.source)):
        print(
            f"error: {len(args.place)} --place for {len(args.source)} source(s). "
            "Give one, applied to every source, or one per source.",
            file=sys.stderr,
        )
        return 2
    if args.id and len(args.id) != len(args.source):
        print("error: --id must be given once per source, or not at all",
              file=sys.stderr)
        return 2
    if args.for_seconds is not None and args.for_seconds <= 0:
        print("error: --for must be greater than zero", file=sys.stderr)
        return 2

    record_to = None
    if args.record is not None:
        record_to = Path(args.record) if args.record else paths.recordings_directory()
        record_to.mkdir(parents=True, exist_ok=True)

    node = Node(
        args.database or default_database_path(),
        node_id=args.node,
        zones=zones,
        record_to=record_to,
        # `--record` has always meant every camera this node runs, and the
        # per-camera flag the console sets does not narrow it.
        record_every_camera=record_to is not None,
        segment_seconds=args.segment_seconds,
        detector_factory=lambda: _detector(args),
    )

    try:
        for index, source in enumerate(args.source):
            pose = None
            if args.place:
                pose = args.place[index] if len(args.place) > 1 else args.place[0]
            node.add_camera(
                source,
                camera_id=args.id[index] if args.id else None,
                pose=pose,
            )

        deadline = time.monotonic() + args.for_seconds if args.for_seconds else None
        print(f"node {args.node}: {len(node.cameras)} camera(s), "
              f"{len(node.zones)} zone(s), {len(node.rules)} rule(s)")
        # `Node` restored the stored zones itself when --zone was absent; what
        # it is actually watching is `node.zones`, whichever way it got them.
        # The explicit ones are already saved, so the stored zones it did not
        # name are the ones left over in the table.
        named = {zone.id for zone in zones}
        _describe_zones(
            list(node.zones), explicit=bool(zones),
            unused_stored=[z for z in node.store.zones() if z.id not in named],
            model_given=args.model is not None,
        )
        if deadline is None:
            # On Windows an external SIGINT does not reach a Python process at
            # all — measured — so a scheduled job or a container needs --for.
            print("Running until every camera ends or you press Ctrl-C. "
                  "For a scheduled job use --for SECONDS.", file=sys.stderr)

        node.run_forever(
            until=(lambda _: time.monotonic() >= deadline) if deadline else None
        )

        print()
        print(node.summary())
        return 0
    finally:
        node.close()


def _register_expiry(
    register: Register, now_millis: int, policy: IdentifierRetentionPolicy
) -> tuple[list[Identifier], list[Identifier]]:
    """What a register sweep would delete and what a pin would save — read only.

    The sweep has no dry run of its own, and the first thing anybody should do
    with a retention policy is find out what it would have eaten; a report that
    could only say "add --apply to find out" would be no report. The selection
    is the sweep's — retention by kind from the same policy, age from the same
    enrolment time, the subject's pin the only exemption — and
    `test_cli` holds the two to the same answer, so this cannot drift into
    promising a smaller sweep than the one --apply then performs.
    """
    expired: list[Identifier] = []
    pinned: list[Identifier] = []
    for subject in register.subjects():
        for identifier in register.identifiers(subject.id):
            retention = policy.retention_millis(identifier.kind)
            if retention is None or identifier.age_millis(now_millis) < retention:
                continue
            (pinned if subject.pinned else expired).append(identifier)
    return expired, pinned


def _retention(args: argparse.Namespace) -> int:
    """Report or apply the retention policy: recorded video, then the register.

    Reports by default. The first thing anybody should do with a retention
    policy is find out what it would have eaten, and a command whose default
    deletes video is a command that deletes video by accident.

    The register is swept by the same command, after the video, because the
    two are one decision: a deployment where the footage expires on schedule
    and the biometric templates it was taken from do not is the wrong way
    round, and it stayed that way here for as long as the sweep was a tested
    method nothing called.
    """
    policy = RetentionPolicy(
        max_age_days=args.keep_days if args.keep_days > 0 else None,
        max_bytes=int(args.max_gib * 1024**3) if args.max_gib else None,
        min_free_bytes=int(args.min_free_gib * 1024**3) if args.min_free_gib else None,
    )
    if args.face_days < 0 or args.plate_days < 0:
        print("error: --face-days and --plate-days are durations; neither can be "
              "negative", file=sys.stderr)
        return 2
    identifiers = IdentifierRetentionPolicy(
        face_template_days=args.face_days, plate_days=args.plate_days
    )

    store = Store(args.database or default_database_path())
    try:
        total = store.recorded_bytes()
        preserved = store.recorded_bytes(preserved=True)
        count = store.recording_count()

        print(f"policy      {policy.describe()}")
        print(f"recorded    {count} segment(s), {_bytes(total)}")
        print(f"preserved   {_bytes(preserved)} — evidence, never deleted")
        print()

        result = apply_retention(store, policy, actor=ACTOR, dry_run=not args.apply)

        verb = "deleted" if args.apply else "would delete"
        print(f"{verb}    {len(result.deleted)} segment(s), {_bytes(result.freed_bytes)}")
        if result.kept_preserved:
            print(f"kept        {result.kept_preserved} preserved segment(s)")
        if result.already_missing:
            print(f"missing     {result.already_missing} indexed file(s) were already gone")
        if result.failed:
            print(f"failed      {len(result.failed)} file(s) could not be deleted")

        # The register, whatever the video sweep managed. A full disk is a
        # reason to stop deleting video, not a reason to keep a face template
        # past the day it was promised to go.
        now_millis = int(time.time() * 1000)
        register = store.register
        if args.apply:
            sweep = register.sweep_expired(now_millis, identifiers)
            # Audited even when it deleted nothing: the row is the evidence
            # that the sweep ran, which is what a data-protection audit asks
            # for first. Ids and counts only, never a name or a plate.
            store.audit(ACTOR, sweep.action, None, sweep.detail())
            expired, pinned = list(sweep.deleted), list(sweep.kept_pinned)
            examined = sweep.examined
        else:
            expired, pinned = _register_expiry(register, now_millis, identifiers)
            examined = sum(
                len(register.identifiers(subject.id)) for subject in register.subjects()
            )

        print()
        print(f"register    {identifiers.describe()} — {examined} identifier(s) examined")
        print(f"{verb}    {len(expired)} identifier(s) past retention")
        if pinned:
            # Reported, never passed over: an identifier past its retention
            # that is still here is exactly what an audit asks about, and the
            # answer is "an operator pinned that subject".
            print(f"kept        {len(pinned)} identifier(s) past retention, because "
                  "an operator pinned the subject")

        if result.shortfall:
            print()
            print(f"  {result.shortfall}")
            return 1
        if not args.apply and (result.deleted or expired):
            print()
            print("  Nothing was deleted. Add --apply to do it.")
        return 0
    finally:
        store.close()


def _devices(args: argparse.Namespace) -> int:
    """Cameras attached to this machine, as the operating system reports them.

    Listing opens nothing. `--probe` opens each one briefly to confirm which
    index is which and what resolution it gives — which is a deliberate act, and
    on macOS is what triggers the operating system's permission prompt, so it is
    a flag rather than the default.
    """
    found = devices.discover(probe_indices=args.probe)

    if not found:
        print("No cameras. The operating system reports none attached.")
        if not args.probe:
            print("\nSome cameras are not listed by the device registry but do "
                  "open. Try: sentinel devices --probe")
        return 0

    print(f"{len(found)} camera(s), through {devices.preferred_backend()}:")
    print()
    for camera in found:
        print(f"  {camera.label}")
        print(f"      use    {camera.source}")
        if camera.identifier:
            print(f"      id     {camera.identifier}")
        if camera.index_confirmed:
            print(f"      opens  {camera.backend}")
    print()

    if any(not camera.index_confirmed for camera in found):
        # Said plainly, because acting on a wrong index attributes an intrusion
        # to the wrong side of a building.
        print("An index marked assumed has not been opened, so it is this")
        print("machine's enumeration order and not a fact. Confirm it with")
        print("`sentinel devices --probe`, or in the console, which shows you a")
        print("frame — two identical cameras cannot be told apart any other way.")
        print()

    print("Then: sentinel run device:0 --place lat,lon,height,heading,pitch")
    return 0


def _where(args: argparse.Namespace) -> int:
    """Answer "where does this thing keep my files", which is asked constantly."""
    from .version import describe

    # First, because "which build is this" is the other question every bug
    # report needs answered and until now nothing printed it.
    print(f"build            {describe()}")
    print(f"data directory   {paths.data_directory()}")
    print(f"models directory {paths.models_directory()}")
    print(f"database         {args.database or default_database_path()}")
    print(f"logs             {paths.log_directory()}")
    print(f"evidence         {paths.evidence_directory()}")
    print(f"recordings       {paths.recordings_directory()}")
    print(f"packaged build   {paths.is_frozen()}")
    print()
    print(f"Override the lot with {paths.DATA_DIR_VARIABLE}.")
    return 0


# --------------------------------------------------------------------- basemap


#: Frames per second, per camera, that feed the basemap median. The median
#: needs its samples separated in time — a walker has to *leave* a cell between
#: two of them to be outvoted — and sampling one frame onto the default
#: quarter-metre grid costs 79 ms, measured, so feeding every frame of a 15 fps
#: camera would both fill the ring with one half-second and fall behind it.
BASEMAP_FRAMES_PER_SECOND = 4.0


def _source_problem(source: str) -> str | None:
    """Why a source string cannot be opened, said before anything is opened.

    The two checks `run` makes inline: a malformed device index is a mistyped
    command, and a file that is not there is a typo. Both are worth a sentence
    and exit 2 rather than a traceback half way through opening cameras.
    """
    if devices.is_device_source(source):
        try:
            devices.device_index(source)
        except devices.DeviceError as error:
            return str(error)
        return None
    if "://" not in source and not Path(source).exists():
        return f"no such file: {source}"
    return None


def _feed_basemap(
    builder, source: VideoSource, pose: CameraPose, *, duration: float | None,
    per_second: float,
) -> tuple[int, float]:
    """Feed one source to the builder at most ``per_second`` frames a second.

    Thinned on the source's own clock — the wall clock for a camera, media
    time for a file — so a recording of the yard is sub-sampled exactly as the
    camera that made it would have been. ``duration`` bounds a camera by the
    wall clock and a file by its media time; ``None`` lets a file run to its
    end. A camera is read through `LiveStream`, the same reader `run` uses, so
    a dropped frame is a gap and not an ending.

    Returns frames *read* and seconds elapsed, which is what starvation is
    judged on: a camera delivering one frame in ten seconds is held by another
    program whether or not that frame was fed.
    """
    from .decode import LiveStream

    interval = 1000.0 / per_second
    started = time.monotonic()
    read = 0
    last_kept: int | None = None

    def wanted(timestamp_millis: int) -> bool:
        nonlocal last_kept
        if last_kept is not None and timestamp_millis - last_kept < interval:
            return False
        last_kept = timestamp_millis
        return True

    if source.is_live:
        if duration is None:
            # A contract, not an assertion: an `assert` here vanished under
            # `python -O` and the next line computed ``started + None``. The
            # command line refuses a camera without --for before opening it;
            # this is for any other caller.
            raise ValueError(
                f"{source.display_url} is a live source, which has no end; "
                "_feed_basemap needs a duration for it (the command line "
                "refuses a camera without --for before opening it)"
            )
        deadline = started + duration
        with LiveStream(source) as stream:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                # Bounded by what is left rather than the reader's own ten
                # seconds: a --for 5 that then waited ten more for a silent
                # camera would not be a bound.
                frame = stream.read(timeout=min(1.0, remaining))
                if frame is None:
                    continue
                read += 1
                if wanted(frame.timestamp_millis):
                    # A live frame is already stamped with the wall clock.
                    builder.feed(
                        source.source_id, pose, frame.image, frame.timestamp_millis / 1000.0
                    )
    else:
        source.open()
        for frame in source:
            if duration is not None and frame.timestamp_millis > duration * 1000.0:
                break
            read += 1
            if wanted(frame.timestamp_millis):
                # A file carries no wall clock of its own. What is honest is
                # when this system looked at it, which is what the asset's
                # per-cell age then measures from.
                builder.feed(source.source_id, pose, frame.image, time.time())
    return read, time.monotonic() - started


def _basemap_build(args: argparse.Namespace) -> int:
    """Build the site's basemap from its own cameras and keep it.

    The same sources and placements `run` takes, fed one after another for
    ``--for`` seconds each — the shape `run` has, one decode thread at a time,
    so two cameras and ``--for 60`` is a two-minute build. Nothing but the
    ground raster and its provenance is written: no frame, and no source URL,
    because a camera URL carries a credential and a basemap is meant to be
    copied about.
    """
    from .basemap import BasemapBuilder, BasemapError, basemap_directory, save_basemap
    from .decode import is_live_source

    if len(args.place) not in (1, len(args.source)):
        print(
            f"error: {len(args.place)} --place for {len(args.source)} source(s). "
            "Give one, applied to every source, or one per source.",
            file=sys.stderr,
        )
        return 2
    if args.id and len(args.id) != len(args.source):
        print(
            f"error: {len(args.id)} --id for {len(args.source)} source(s). "
            "Give one per source, or none and take cam-01, cam-02, ...",
            file=sys.stderr,
        )
        return 2
    if args.id and len(set(args.id)) != len(args.id):
        # Refused here, before a source is opened. Two sources under one id
        # were either refused by the builder half way through the build, with
        # the first source already fed for --for seconds, or — with the same
        # placement — silently merged into one camera's median.
        repeated = sorted({camera_id for camera_id in args.id if args.id.count(camera_id) > 1})
        print(
            f"error: --id {', '.join(repeated)} given more than once. Every source "
            "needs its own id: each cell of the basemap names the camera it came from.",
            file=sys.stderr,
        )
        return 2
    if args.duration is not None and args.duration <= 0:
        print("error: --for must be greater than zero", file=sys.stderr)
        return 2
    if args.cell <= 0:
        print("error: --cell must be greater than zero", file=sys.stderr)
        return 2
    for source in args.source:
        problem = _source_problem(source)
        if problem is not None:
            print(f"error: {problem}", file=sys.stderr)
            return 2
        if args.duration is None and is_live_source(source):
            # Refused rather than warned about, unlike `run`: an unbounded run
            # at least keeps analysing, but a basemap is built once at the
            # end, and a build that never ends builds nothing.
            print(
                f"error: {source} is a camera, which has no end. Give --for SECONDS "
                "so the build can finish; a minute is a reasonable start.",
                file=sys.stderr,
            )
            return 2

    sources = [
        VideoSource(source, source_id=args.id[index] if args.id else f"cam-{index + 1:02d}")
        for index, source in enumerate(args.source)
    ]
    builder = BasemapBuilder(cell_size_m=args.cell)
    starved = False

    try:
        for index, source in enumerate(sources):
            pose = args.place[index] if len(args.place) > 1 else args.place[0]
            print(f"\n{source.source_id}  ({source.display_url})")
            read, elapsed = _feed_basemap(
                builder, source, pose, duration=args.duration,
                per_second=BASEMAP_FRAMES_PER_SECOND,
            )
            # Sampled, not offered: a camera that sees no ground has frames
            # taken and none of them reach a median, and this line said they had.
            fed = builder.frames_sampled().get(source.source_id, 0)
            print(f"  {read} frame(s) read in {elapsed:.0f}s, {fed} fed to the median")

            starvation = _starvation(source, read, elapsed)
            if starvation is not None:
                print(f"  STARVED             {starvation}", file=sys.stderr)
                starved = True

        for camera_id in builder.blind_cameras():
            print(
                f"  {camera_id}: sees no ground — the bottom of its frame is above "
                "the horizon — and contributed nothing",
                file=sys.stderr,
            )

        try:
            asset = builder.build()
        except BasemapError as error:
            print(f"\nerror: {error}", file=sys.stderr)
            return 1

        directory = Path(args.out) if args.out else basemap_directory()
        png_path, json_path = save_basemap(asset, directory)

        print()
        print(asset.describe())
        for camera_id in asset.cameras:
            print(f"  {camera_id:<12} {asset.cells_from(camera_id):,} cell(s)")
        print()
        print(f"written   {png_path}")
        print(f"          {json_path}")
        print(
            "Only the ground plane is correct. Anything with height is smeared "
            "along the ray from the camera that saw it, so this is a basemap, "
            "not a photograph."
        )
        return 1 if starved else 0
    finally:
        for source in sources:
            source.close()


def _basemap_show(args: argparse.Namespace) -> int:
    """Describe the basemap on disk — after verifying it is the one it claims to be."""
    from .basemap import BasemapError, basemap_directory, load_basemap

    directory = Path(args.dir) if args.dir else basemap_directory()
    try:
        asset = load_basemap(directory)
    except BasemapError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if asset is None:
        print(
            f"no basemap in {directory}. Build one with: sentinel basemap build "
            "SOURCE --place lat,lon,height,heading,pitch --for 60"
        )
        return 1

    print(asset.describe())
    for camera_id in asset.cameras:
        print(f"  {camera_id:<12} {asset.cells_from(camera_id):,} cell(s)")
    print(f"fingerprint  {asset.fingerprint}")
    print(f"files        {directory}")
    return 0


# ---------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sentinel",
        description=(
            "Sentinel Vision — headless analysis. Decodes, detects, tracks, "
            "projects, applies rules, correlates events into incidents and "
            "records the result. No network, ever."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  sentinel run gate.mp4 --place 33.8938,35.5018,6,180,-22\n"
            "  sentinel run north.mp4 south.mp4 --id north --id south \\\n"
            "      --place 33.8938,35.5018,6,180,-22 \\\n"
            "      --place 33.8942,35.5018,6,0,-22 \\\n"
            "      --zone 'Yard:33.8940,35.5016;33.8940,35.5020;"
            "33.8936,35.5020;33.8936,35.5016'\n"
            "  sentinel run gate.mp4 --model gate.onnx --zone 'Yard:...' "
            "--zone-classes Yard=person,car\n"
            "  sentinel node gate.mp4 north.mp4 --record --for 3600\n"
            "  sentinel devices --probe\n"
            "  sentinel run device:0 --place 33.8938,35.5018,3,90,-15\n"
            "  sentinel basemap build device:0 --place 33.8938,35.5018,3,90,-15 --for 60\n"
            "  sentinel incidents\n"
            "  sentinel export INC-abc123 --to ./evidence\n"
            "  sentinel where\n"
        ),
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="developer logging: DEBUG, with thread, file and line",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true",
        help="warnings and errors only",
    )
    parser.add_argument(
        "--database", type=Path, default=None,
        help="database to use (default: the per-user data directory)",
    )

    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser(
        "run", help="analyse one or more sources and record what happened"
    )
    run.add_argument(
        "source", nargs="+",
        help="video files, rtsp:// URLs, or device:N for a camera attached to "
             "this machine (see `sentinel devices`)",
    )
    run.add_argument(
        "--id", action="append", default=None,
        help="camera id, once per source (default: cam-01, cam-02, ...)",
    )
    run.add_argument(
        "--place", action="append", type=_pose, default=None,
        help=(
            "lat,lon,height,heading,pitch[,hfov,vfov,range] — once, or once per "
            "source. Without it objects are tracked but not located, and no "
            "position is invented."
        ),
    )
    run.add_argument(
        "--zone", action="append", type=_zone, default=None,
        help=(
            "name:lat,lon;lat,lon;lat,lon — a restricted polygon. Without any, "
            "the zones the database already holds are used, filters included."
        ),
    )
    run.add_argument(
        "--zone-classes", action="append", type=_zone_classes, default=None,
        metavar="NAME=label,label",
        help=(
            "which detector labels a --zone in this command acts on, e.g. "
            "Yard=person,car. Repeatable. Without it a zone fires for anything. "
            "A filtered zone needs --model: the motion detector names nothing."
        ),
    )
    run.add_argument(
        "--for", dest="duration", type=float, default=None, metavar="SECONDS",
        help=(
            "stop a live source after this long. A camera has no end, so "
            "without this a headless run never returns — and Ctrl-C is not "
            "available to a scheduled job or a container. Ignored by files, "
            "which stop on their own."
        ),
    )
    run.add_argument(
        "--frames", type=int, default=None, metavar="N",
        help="stop after N frames. Bounds a live run by work rather than by time.",
    )
    run.add_argument(
        "--record", metavar="DIR", nargs="?", const="", default=None,
        help=(
            "record video to DIR, or to the data directory when given no value. "
            "Roughly 17.5 GB per camera per day at 640x480/15fps — see "
            "`sentinel retention`."
        ),
    )
    run.add_argument(
        "--segment-seconds", type=float, default=60.0, metavar="SECONDS",
        help=(
            "length of each recorded clip (default 60). This is the upper bound "
            "on what a power cut costs, because a container killed mid-write may "
            "not play at all."
        ),
    )
    run.add_argument(
        "--model", type=Path, default=None, metavar="FILE",
        help=(
            "an ONNX model to detect with, instead of motion. A model with mask "
            "outputs segments — one instance per object, with a ground-contact "
            "point taken from its own lowest pixel. Operator-supplied: nothing "
            "is ever downloaded. See devtools/export_model.py."
        ),
    )
    run.add_argument(
        "--detect-scale", type=float, default=0.75,
        help="detection resolution scale (default 0.75: 1.7x faster and "
             "slightly better recall — see docs/OVERVIEW.md)",
    )
    run.add_argument("--node", default="local", help="node id recorded on every event")
    run.add_argument(
        "--export", metavar="DIR", default=None,
        help="write an evidence package per incident into DIR",
    )
    run.set_defaults(handler=_run)

    listing = commands.add_parser("incidents", help="list recorded incidents")
    listing.add_argument("--limit", type=int, default=50)
    listing.set_defaults(handler=_incidents)

    export = commands.add_parser("export", help="export one incident as evidence")
    export.add_argument("id", help="incident id, from `sentinel incidents`")
    export.add_argument("--to", required=True, metavar="DIR")
    export.add_argument(
        "--lead", type=float, default=DEFAULT_LEAD_SECONDS, metavar="SECONDS",
        help=(
            "how much recorded footage to include from before the incident "
            f"opened (default {DEFAULT_LEAD_SECONDS:g}). An event fires after "
            "somebody is already inside a zone, so what explains it starts "
            "earlier."
        ),
    )
    export.add_argument(
        "--trail", type=float, default=DEFAULT_TRAIL_SECONDS, metavar="SECONDS",
        help=(
            "and from after it closed (default "
            f"{DEFAULT_TRAIL_SECONDS:g}) — what somebody does on the way out is "
            "evidence too."
        ),
    )
    export.set_defaults(handler=_export)

    coverage = commands.add_parser(
        "coverage",
        help="what a camera can see — or, with --site, what the whole node misses",
    )
    coverage.add_argument(
        "--place", type=_pose, default=None,
        help="one camera's pose, for what it alone covers and where a zone fits",
    )
    coverage.add_argument(
        "--site", type=_ring, default=None, metavar="lat,lon;lat,lon;...",
        help=(
            "the site boundary. Given this, reports what every *placed* camera "
            "on this node covers of it and — the useful half — what it does not. "
            "Exits non-zero when anything is uncovered, so a scheduled check "
            "says something."
        ),
    )
    coverage.add_argument("--zone-radius", type=float, default=12.0, metavar="METRES")
    coverage.add_argument("--zone-name", default="Restricted Area A")
    coverage.set_defaults(handler=_coverage)

    node = commands.add_parser(
        "node", help="run cameras unattended, with no display — what a worker runs"
    )
    node.add_argument("source", nargs="+", help="video files, rtsp:// URLs, or device:N")
    node.add_argument("--id", action="append", default=None,
                      help="camera id, once per source")
    node.add_argument("--place", action="append", type=_pose, default=None,
                      help="lat,lon,height,heading,pitch[,hfov,vfov,range]")
    node.add_argument("--zone", action="append", type=_zone, default=None,
                      help="name:lat,lon;lat,lon;lat,lon (default: the stored zones)")
    node.add_argument("--zone-classes", action="append", type=_zone_classes,
                      default=None, metavar="NAME=label,label",
                      help="which detector labels a --zone in this command acts on")
    node.add_argument("--record", metavar="DIR", nargs="?", const="", default=None,
                      help="record video to DIR, or to the data directory")
    node.add_argument("--segment-seconds", type=float, default=60.0)
    node.add_argument("--detect-scale", type=float, default=0.75)
    node.add_argument("--model", type=Path, default=None, metavar="FILE",
                      help="an ONNX detection or segmentation model to use instead of motion")
    node.add_argument("--node", default="local", help="this node's id")
    node.add_argument(
        "--for", dest="for_seconds", type=float, default=None, metavar="SECONDS",
        help="stop after this long. A camera has no end, and Ctrl-C is not "
             "available to a scheduled job or a container",
    )
    node.set_defaults(handler=_node)

    retention = commands.add_parser(
        "retention",
        help="delete recorded video and enrolled identifiers the policy no longer covers",
    )
    retention.add_argument(
        "--face-days", type=float, default=30.0, metavar="DAYS",
        help=(
            "delete face templates enrolled longer ago than this (default 30). "
            "A pinned subject is the only exemption. There is no value meaning "
            "forever: keeping a biometric indefinitely is a decision that needs "
            "a name and an audit row, not a flag."
        ),
    )
    retention.add_argument(
        "--plate-days", type=float, default=365.0, metavar="DAYS",
        help="delete plates enrolled longer ago than this (default 365)",
    )
    retention.add_argument(
        "--keep-days", type=float, default=14.0, metavar="DAYS",
        help="delete recordings older than this (default 14; 0 means no age limit)",
    )
    retention.add_argument(
        "--max-gib", type=float, default=None, metavar="GIB",
        help="delete oldest until the total is under this",
    )
    retention.add_argument(
        "--min-free-gib", type=float, default=5.0, metavar="GIB",
        help=(
            "delete oldest until this much of the volume is free (default 5). A "
            "disk at 100%% stops the database too, not only the recording."
        ),
    )
    retention.add_argument(
        "--apply", action="store_true",
        help="actually delete. Without it, this reports what would go and touches nothing",
    )
    retention.set_defaults(handler=_retention)

    listing = commands.add_parser(
        "devices", help="cameras attached to this machine, through the OS's own API"
    )
    listing.add_argument(
        "--probe", action="store_true",
        help="open each camera briefly to confirm its index and resolution",
    )
    listing.set_defaults(handler=_devices)

    where = commands.add_parser("where", help="print every path this build uses")
    where.set_defaults(handler=_where)

    basemap = commands.add_parser(
        "basemap",
        help="build the site's own basemap from its cameras, or show the one it has",
    )
    basemap_commands = basemap.add_subparsers(dest="basemap_command", required=True)
    build = basemap_commands.add_parser(
        "build",
        help="feed each source for a while and keep the median ground as the basemap",
        description=(
            "Samples each camera's frames onto a metric ground grid through the "
            "same projection its positions come from, keeps a running median per "
            "cell so whatever moved through is removed, composites the cameras "
            "cell by cell, and writes basemap.png and basemap.json. Only the "
            "ground plane is correct; anything with height is smeared along its "
            "ray. No frame is written, and no source URL."
        ),
    )
    build.add_argument(
        "source", nargs="+",
        help="video files, rtsp:// URLs, or device:N — the same sources `run` takes",
    )
    build.add_argument(
        "--id", action="append", default=None,
        help="camera id, once per source (default: cam-01, cam-02, ...)",
    )
    build.add_argument(
        "--place", action="append", type=_pose, required=True,
        help=(
            "lat,lon,height,heading,pitch[,hfov,vfov,range] — once, or once per "
            "source. Required: the basemap is the inverse of this projection."
        ),
    )
    build.add_argument(
        "--for", dest="duration", type=float, default=None, metavar="SECONDS",
        help=(
            "feed each camera for this long; required for a camera, which has "
            "no end. For a file, this much of the recording (default: all of it)."
        ),
    )
    build.add_argument(
        "--cell", type=float, default=0.25, metavar="METRES",
        help="ground cell size (default 0.25)",
    )
    build.add_argument(
        "--out", default=None, metavar="DIR",
        help="where to write basemap.png and basemap.json (default: basemap/ "
             "under the data directory)",
    )
    build.set_defaults(handler=_basemap_build)

    show = basemap_commands.add_parser(
        "show", help="describe the basemap on disk, after verifying its fingerprint"
    )
    show.add_argument("--dir", default=None, metavar="DIR",
                      help="where it was written (default: basemap/ under the data directory)")
    show.set_defaults(handler=_basemap_show)

    return parser


def main(argv: list[str] | None = None) -> int:
    # Before the parser, because the parser's imports are not the point — the
    # point is that this runs before anything heavy loads. ONNX Runtime reads
    # ORT_DISABLE_TELEMETRY when its native library initialises, which is
    # earlier than any Python call can reach it.
    telemetry.silence()

    args = build_parser().parse_args(argv)

    logs.configure(
        level="DEBUG" if args.verbose else ("WARNING" if args.quiet else "INFO"),
        developer=args.verbose,
    )

    try:
        return args.handler(args)
    except KeyboardInterrupt:
        # Interrupting a run is a normal thing to do to a long one, not a crash.
        # Everything already written stays written; nothing is half-committed,
        # because each event and incident is its own transaction.
        _log.info("interrupted; everything already recorded is kept")
        print("\ninterrupted", file=sys.stderr)
        return 130
    except Exception as error:  # noqa: BLE001
        # An operator cannot act on a traceback, and one printed at them hides
        # the sentence that says what to do. A developer gets both.
        _log.error("%s: %s", type(error).__name__, error, exc_info=True)
        print(f"\nerror: {error}", file=sys.stderr)
        if not args.verbose:
            print("Run again with --verbose for the traceback.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
