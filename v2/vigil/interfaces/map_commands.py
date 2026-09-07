"""`vigil map`: build the site's ground map from the cameras that watch it.

The product path for `vigil.service.mapping`. v1's lesson, in its own words:
`orthophoto.py` could already turn a frame into a top-down patch, run a median
over many of them and composite several cameras — and until `basemap.py`
existed, nothing called any of it. Correct, tested code that no product path
reaches is the repository's recurring defect, and a mapping service with no
command is another instance of it.

The capture loop lives in the service, not here. An interface does not open a
camera: the layering rule says so, and the reason it says so is that a capture
loop inside a CLI is one nothing else can reuse and no test can run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..domain.geo import LatLon
from ..kernel import native
from ..logs import get as _get_logger
from ..service.coverage import CoverageError, analyse, boundary_from_cameras
from ..service.mapping import (
    DEFAULT_CELL_SIZE_M, DEFAULT_DEPTH, MIN_SAMPLES, MappingError, build_from_cameras, load_map,
    save_map, visualise,
)

_log = _get_logger(__name__)


def _map_directory(settings) -> Path:
    return settings.data_dir / "map"


def _site_origin(cameras) -> LatLon:
    """One tangent-plane origin for the whole site.

    The mean of the placed cameras rather than the first of them: the tangent
    plane's error grows with distance from its origin, so putting it in the
    middle halves the worst case. It has to be *stable* across builds, which
    is why it comes from the cameras rather than from whichever one happened
    to start first.
    """
    placed = [c for c in cameras if c.pose is not None]
    if not placed:
        raise MappingError("no camera has been placed, so there is no ground to map")
    return LatLon(
        sum(c.pose.position.lat for c in placed) / len(placed),
        sum(c.pose.position.lon for c in placed) / len(placed),
    )


def _build(ctx) -> int:
    site = ctx.site
    cameras = [c for c in site.cameras(ctx.principal) if c.pose is not None]
    if ctx.args.camera:
        wanted = set(ctx.args.camera)
        cameras = [c for c in cameras if c.id in wanted]
        missing = wanted - {c.id for c in cameras}
        if missing:
            print(f"not placed, or not a camera: {', '.join(sorted(missing))}")
            return 2
    if not cameras:
        print("no placed camera to map from. A camera has to know where it is and where it looks "
              "before anything it sees can go on a map. "
              "Place one with `vigil cameras place ID lat,lon,height,heading,pitch`.")
        return 2

    sources = {c.id: (site.source_with_credentials(c), c.pose) for c in cameras}
    print(f"mapping {len(sources)} camera(s) for {ctx.args.seconds:.0f} s at {ctx.args.cell} m cells")
    try:
        report = build_from_cameras(
            sources, _site_origin(cameras), seconds=float(ctx.args.seconds),
            cell_size_m=ctx.args.cell, depth=ctx.args.depth,
            minimum_samples=ctx.args.min_samples,
        )
    except MappingError as error:
        print(str(error))
        return 2

    for camera_id, fault in sorted(report.faults.items()):
        print(f"{camera_id}: {fault}")
    for camera_id, count in sorted(report.skipped.items()):
        print(f"{camera_id}: skipped {count} unusable frame(s)")
    for camera_id, count in sorted(report.samples.items()):
        print(f"{camera_id}: {count} sample(s)")

    directory = _map_directory(ctx.settings)
    save_map(report.ground, directory)
    print(report.ground.describe())
    print(f"written to {directory}")
    _write_png(ctx, report.ground)
    return 0


def _show(ctx) -> int:
    """What the stored map claims, and how much of it is worth believing."""
    ground = load_map(_map_directory(ctx.settings))
    if ground is None:
        print("no map has been built, or the stored one does not hash to its own fingerprint. "
              "Build one with `vigil map build`.")
        return 1
    print(f"cameras   {', '.join(ground.cameras) or '—'}")
    print(f"grid      {ground.grid.rows} x {ground.grid.cols} cells at {ground.grid.cell_size_m} m")
    print(f"ground    {ground.describe()}")
    print()
    # A histogram, because "mean confidence 0.6" hides the difference between a
    # map that is uniformly mediocre and one that is excellent near the masts
    # and worthless past them — and only the second is usable.
    seen = ground.valid > 0
    if seen.any():
        edges = [0.0, 0.2, 0.35, 0.6, 0.8, 1.01]
        counts, _ = np.histogram(ground.confidence[seen], bins=edges)
        total = max(1, int(seen.sum()))
        for label, low, high, count in zip(
            ["unusable", "poor", "fair", "good", "excellent"], edges, edges[1:], counts
        ):
            share = count / total
            print(f"  {label:<10} {low:.2f}-{min(high, 1.0):.2f}  {count:>8}  {share:>6.1%} "
                  f"{'#' * int(share * 40)}")
    if ctx.args.json:
        print(json.dumps(ground.summary(), indent=2))
    _write_png(ctx, ground)
    return 0


def _write_png(ctx, ground) -> None:
    if not getattr(ctx.args, "png", None):
        return
    import cv2

    cv2.imwrite(str(Path(ctx.args.png)), visualise(ground))
    print(f"picture written to {ctx.args.png}")


def _coverage(ctx) -> int:
    """What these cameras reach, and what an installer has not covered yet."""
    cameras = {c.id: c.pose for c in ctx.site.cameras(ctx.principal) if c.pose is not None}
    if not cameras:
        print("no placed camera. Coverage is a question about where cameras point, so they have "
              "to be placed first.")
        return 2
    try:
        if ctx.args.boundary:
            boundary = [LatLon(*(float(v) for v in pair.split(",")))
                        for pair in ctx.args.boundary.split(";")]
        else:
            boundary = boundary_from_cameras(cameras)
            print("no boundary given, so one was drawn around the cameras. That flatters them: "
                  "it is defined by where they point. Pass --boundary for the real fence.")
        report = analyse(boundary, cameras)
    except CoverageError as error:
        print(str(error))
        return 2
    except ValueError:
        print("a boundary is `lat,lon;lat,lon;lat,lon[;...]`")
        return 2
    print(report.describe())
    return 0


def _dataset(ctx) -> int:
    """Export the corpus running this product has already produced."""
    from ..adapters.detectors import detector_for
    from ..service.dataset import DatasetError, collect, write
    from ..service.search import moment

    destination = Path(ctx.args.to or (ctx.settings.data_dir / "dataset"))
    detector = None
    model = ctx.settings.default_model()
    if model is not None and not ctx.args.no_predictions:
        detector = detector_for(model)
    since = None
    if ctx.args.since:
        try:
            since = moment(ctx.args.since)
        except ValueError as error:
            print(f"error: {error}")
            return 2
    try:
        export = collect(ctx.store, detector, since_millis=since, limit=ctx.args.limit,
                         frames_per_incident=ctx.args.per_incident)
    except DatasetError as error:
        print(str(error))
        return 2
    info = detector.info if detector is not None else None
    write(export, destination,
          model_sha256=getattr(info, "model_sha256", None),
          class_names=getattr(info, "class_names", None))
    print(export.describe())
    print(f"written to {destination}")
    print()
    print("Next: open it in CVAT or Label Studio (both run offline) and *correct* the boxes.")
    print("The split is by day and is in data.yaml. Do not re-split it at random —")
    print("consecutive video frames are near-duplicates and a random split leaks.")
    return 0


def _map(ctx) -> int:
    if not native.available():
        print(f"building a map needs the engine core, which is not loaded: {native.fault()}")
        print("Build it with `python tasks.py core`.")
        return 2
    return _build(ctx) if ctx.args.map_command == "build" else _show(ctx)


def add_arguments(commands) -> None:
    parser = commands.add_parser(
        "map", help="the site's ground map, built from the cameras that watch it")
    sub = parser.add_subparsers(dest="map_command", required=True)
    build = sub.add_parser("build", help="watch the cameras and write what the ground looks like")
    build.add_argument("--seconds", type=float, default=60.0,
                       help="how long to watch. The median needs time between samples, not frames")
    build.add_argument("--camera", action="append",
                       help="only this camera; repeatable. Default is every placed camera")
    build.add_argument("--cell", type=float, default=DEFAULT_CELL_SIZE_M, help="ground cell, metres")
    build.add_argument("--depth", type=int, default=DEFAULT_DEPTH, help="samples kept per cell")
    build.add_argument("--min-samples", type=int, default=MIN_SAMPLES, dest="min_samples",
                       help="fewest samples before a cell is drawn at all")
    build.add_argument("--png", help="also write the map as a picture")
    show = sub.add_parser("show", help="what the stored map claims")
    show.add_argument("--png", help="write the map as a picture")
    show.add_argument("--json", action="store_true")
    parser.set_defaults(handler=_map)

    coverage = commands.add_parser(
        "coverage", help="which ground these cameras reach, and which they do not")
    coverage.add_argument("--boundary", help="the site's fence: lat,lon;lat,lon;lat,lon[;...]")
    coverage.set_defaults(handler=_coverage)

    dataset = commands.add_parser(
        "dataset", help="export the frames, pre-labels and human judgements already on disk")
    ds = dataset.add_subparsers(dest="dataset_command", required=True)
    export = ds.add_parser("export", help="images, YOLO pre-labels and a manifest, split by day")
    export.add_argument("--to", help="where to write it (default: the data directory)")
    export.add_argument("--since", help="2h, 3d, a date, or a full ISO moment")
    export.add_argument("--limit", type=int, default=200, help="incidents to walk")
    export.add_argument("--per-incident", type=int, default=3, dest="per_incident",
                        help="frames around each incident")
    export.add_argument("--no-predictions", action="store_true", dest="no_predictions",
                        help="images only, no pre-labels from the detector")
    dataset.set_defaults(handler=_dataset)
