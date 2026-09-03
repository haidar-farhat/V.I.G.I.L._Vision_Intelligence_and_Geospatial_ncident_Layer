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
from pathlib import Path

from . import devices, logs, paths, telemetry
from .recording import RetentionPolicy, apply_retention
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
        id=name.strip().lower().replace(" ", "-"),
        name=name.strip(),
        kind=ZoneKind.RESTRICTED,
        ring=tuple(points),
        enter_after_millis=600,
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


def _run(args: argparse.Namespace) -> int:
    zones = list(args.zone or [])
    rules = _rules(zones)

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

    try:
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

        for zone in zones:
            store.save_zone(zone)

        if not all_events:
            print("\nNo events. Nothing crossed a rule.")
            # A failed recording is a failed run even when the analysis found
            # nothing: a scheduled job that exits 0 is a job nobody looks at.
            return 1 if recording_failed else 0

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

        return 1 if recording_failed else 0
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
    zones = list(args.zone or [])

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


def _retention(args: argparse.Namespace) -> int:
    """Report or apply the recording retention policy.

    Reports by default. The first thing anybody should do with a retention
    policy is find out what it would have eaten, and a command whose default
    deletes video is a command that deletes video by accident.
    """
    policy = RetentionPolicy(
        max_age_days=args.keep_days if args.keep_days > 0 else None,
        max_bytes=int(args.max_gib * 1024**3) if args.max_gib else None,
        min_free_bytes=int(args.min_free_gib * 1024**3) if args.min_free_gib else None,
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
        if result.shortfall:
            print()
            print(f"  {result.shortfall}")
            return 1
        if not args.apply and result.deleted:
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
    print(f"data directory   {paths.data_directory()}")
    print(f"database         {args.database or default_database_path()}")
    print(f"logs             {paths.log_directory()}")
    print(f"evidence         {paths.evidence_directory()}")
    print(f"recordings       {paths.recordings_directory()}")
    print(f"packaged build   {paths.is_frozen()}")
    print()
    print(f"Override the lot with {paths.DATA_DIR_VARIABLE}.")
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
            "  sentinel node gate.mp4 north.mp4 --record --for 3600\n"
            "  sentinel devices --probe\n"
            "  sentinel run device:0 --place 33.8938,35.5018,3,90,-15\n"
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
        help="name:lat,lon;lat,lon;lat,lon — a restricted polygon",
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
                      help="name:lat,lon;lat,lon;lat,lon")
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
        "retention", help="delete recorded video the policy no longer covers"
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
