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
from pathlib import Path

from . import logs, paths
from .core import CameraPose, LatLon
from .decode import VideoSource
from .detect import MotionDetector
from .evidence import ExportError, export_incident
from .events import (
    AfterHoursRule,
    LoiteringRule,
    RapidMovementRule,
    Severity,
    ZoneEntryRule,
)
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


def _rules(zones: list[Zone]) -> list:
    """The rule set, matched to what is actually configured.

    Without a zone there is nothing to be inside, so the zone rules would be
    dead weight and would let the run report "0 events" for a reason that has
    nothing to do with the footage.
    """
    if not zones:
        return [RapidMovementRule(speed_mps=6.0)]
    return [
        ZoneEntryRule(),
        AfterHoursRule(),
        LoiteringRule(dwell_millis=8000),
        RapidMovementRule(speed_mps=6.0),
    ]


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

    sources: list[VideoSource] = []
    for index, source in enumerate(args.source, start=1):
        # A local file is checked here rather than inside the decoder, so a typo
        # fails before any camera is opened rather than half way through a run.
        if "://" not in source and not Path(source).exists():
            print(f"error: no such file: {source}", file=sys.stderr)
            return 2
        sources.append(
            VideoSource(source, source_id=args.id[index - 1] if args.id else f"cam-{index:02d}")
        )

    store = Store(args.database or default_database_path())
    all_events = []

    try:
        for index, source in enumerate(sources):
            pose = None
            if args.place:
                pose = args.place[index] if len(args.place) > 1 else args.place[0]

            store.save_camera(
                source.source_id, source.source_id, source.display_url, pose=pose
            )

            with Pipeline(
                source,
                # One detector per camera, never shared. MOG2 carries a
                # per-pixel model of *its* scene; feeding it two cameras
                # corrupts both models and every detection that comes out of
                # them. The console gets this right by construction — one
                # worker per camera — and this had to be made to match.
                MotionDetector(detect_scale=args.detect_scale),
                pose=pose,
                zones=zones,
                rules=rules,
                node_id=args.node,
            ) as pipeline:
                events = []
                for result in pipeline.run():
                    events.extend(result.events)

                store.save_events(events)
                all_events.extend(events)

                print(f"\n{source.source_id}  ({source.display_url})")
                print(pipeline.stats.summary())

        for zone in zones:
            store.save_zone(zone)

        if not all_events:
            print("\nNo events. Nothing crossed a rule.")
            return 0

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

        return 0
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


def _export_all(incidents, destination: Path, store: Store) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    print()
    for incident in incidents:
        try:
            export = export_incident(incident, destination, exported_by=ACTOR)
        except ExportError as error:
            print(f"  export failed for {incident.id}: {error}", file=sys.stderr)
            continue
        store.audit(ACTOR, "incident.exported", incident.id, str(export.directory))
        print(f"  {incident.id}  ->  {export.directory}")
        print(f"      manifest sha256  {export.manifest_sha256}")
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
        export = export_incident(incident, destination, exported_by=ACTOR)
        store.audit(ACTOR, "incident.exported", incident.id, str(export.directory))

        print(f"{len(export.files)} files written to {export.directory}")
        print(f"manifest sha256  {export.manifest_sha256}")
        return 0
    except ExportError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
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


def _where(args: argparse.Namespace) -> int:
    """Answer "where does this thing keep my files", which is asked constantly."""
    print(f"data directory   {paths.data_directory()}")
    print(f"database         {args.database or default_database_path()}")
    print(f"logs             {paths.log_directory()}")
    print(f"evidence         {paths.evidence_directory()}")
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
    run.add_argument("source", nargs="+", help="video files, or rtsp:// URLs")
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
    export.set_defaults(handler=_export)

    coverage = commands.add_parser(
        "coverage",
        help="what a placed camera can actually see, and a zone that fits inside it",
    )
    coverage.add_argument("--place", type=_pose, required=True)
    coverage.add_argument("--zone-radius", type=float, default=12.0, metavar="METRES")
    coverage.add_argument("--zone-name", default="Restricted Area A")
    coverage.set_defaults(handler=_coverage)

    where = commands.add_parser("where", help="print every path this build uses")
    where.set_defaults(handler=_where)

    return parser


def main(argv: list[str] | None = None) -> int:
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
