"""A world, and cameras that look at it.

The reference scene in ``scene.py`` is drawn directly in image space. That is
enough to test one camera, and it cannot test two: two views of a scene invented
separately are not two views of one scene, and a correlation test built on them
would be checking that two pictures agree by construction.

So here the objects have positions **on the ground**, in metres, and each camera
renders what it would see of them through the same projection the system uses to
interpret real footage. If the geometry is wrong, the two cameras disagree about
where the same person is and the correlation fails — which is exactly the failure
this is meant to be able to detect.

That makes it the strongest available check of the specification's central claim
short of real hardware: *three cameras seeing one person is one incident*. The
people are generated, but the disagreement between cameras is not — it comes from
real projection through real poses.

The rendering is deliberately plain. This exists to test geometry and
correlation, not to fool a detector.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from sentinel.core import CameraPose, LatLon, destination_point, image_coordinates

WIDTH = 640
HEIGHT = 480
FPS = 15
DURATION_SECONDS = 14
FRAME_COUNT = FPS * DURATION_SECONDS

#: Where the site is. Arbitrary, but fixed, so map assertions have real numbers.
ORIGIN = LatLon(33.8938, 35.5018)

#: How tall the people are. Used both to draw them and to work out how large
#: they should appear, so apparent size and ground position stay consistent.
PERSON_HEIGHT_M = 1.75


@dataclass(frozen=True, slots=True)
class WorldWalker:
    """One person, walking a straight line across the ground.

    Positions are in metres east and north of :data:`ORIGIN`, which is the frame
    the geometry works in. Nothing here is expressed in pixels: a walker does not
    know what a camera is.
    """

    name: str
    start_east: float
    start_north: float
    end_east: float
    end_north: float
    #: Fraction of the sequence spent walking; the rest is spent standing.
    walk_fraction: float = 1.0
    enters_at: float = 0.0
    colour: tuple[int, int, int] = (64, 58, 92)

    def position_at(self, progress: float) -> LatLon | None:
        if progress < self.enters_at:
            return None

        span = max(1e-6, 1.0 - self.enters_at)
        local = (progress - self.enters_at) / span
        travelled = min(local / self.walk_fraction, 1.0) if self.walk_fraction > 0 else 1.0

        east = self.start_east + (self.end_east - self.start_east) * travelled
        north = self.start_north + (self.end_north - self.start_north) * travelled
        return _offset(ORIGIN, east, north)

    def speed_mps(self) -> float:
        distance = math.hypot(
            self.end_east - self.start_east, self.end_north - self.start_north
        )
        walking_seconds = DURATION_SECONDS * (1.0 - self.enters_at) * self.walk_fraction
        return distance / walking_seconds if walking_seconds > 0 else 0.0


def _offset(origin: LatLon, east: float, north: float) -> LatLon:
    """Metres east and north of a point."""
    moved = destination_point(origin, 0.0, north) if north else origin
    return destination_point(moved, 90.0, east) if east else moved


#: One person, walking east across the ground in front of both cameras. Chosen so
#: the correlation question is unambiguous: whatever the cameras report, there is
#: exactly one object, and any answer other than one is wrong.
LONE_WALKER = (
    WorldWalker("walker", start_east=-14.0, start_north=18.0,
                end_east=14.0, end_north=18.0, colour=(64, 58, 92)),
)

#: Two people crossing the same ground at once, for the case where one incident
#: should contain two objects rather than collapsing them.
TWO_WALKERS = (
    WorldWalker("east-bound", -14.0, 20.0, 14.0, 20.0, colour=(64, 58, 92)),
    WorldWalker("west-bound", 14.0, 15.0, -14.0, 15.0, colour=(52, 74, 60)),
)


def camera(
    *,
    east: float,
    north: float,
    heading: float,
    mount_height: float = 6.0,
    pitch: float = -18.0,
) -> CameraPose:
    """A camera placed relative to the site origin."""
    return CameraPose(
        position=_offset(ORIGIN, east, north),
        mount_height=mount_height,
        heading=heading,
        pitch=pitch,
        horizontal_fov=62.0,
        vertical_fov=36.0,
        range_meters=90.0,
    )


#: Two cameras watching the same stretch of ground from different sides. Their
#: fields of view overlap in the middle, which is where the hand-off happens.
CAMERA_WEST = camera(east=-16.0, north=0.0, heading=25.0)
CAMERA_EAST = camera(east=16.0, north=0.0, heading=-25.0)


def _background(rng: np.random.Generator) -> np.ndarray:
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[:] = (58, 62, 68)
    noise = rng.normal(0.0, 7.0, frame.shape[:2])
    return np.clip(frame + noise[..., None], 0, 255).astype(np.uint8)


def render(
    pose: CameraPose,
    walkers: tuple[WorldWalker, ...],
    index: int,
    background: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """One frame as this camera would see this world."""
    progress = index / max(1, FRAME_COUNT - 1)
    frame = background.copy()

    for walker in walkers:
        point = walker.position_at(progress)
        if point is None:
            continue

        feet = image_coordinates(pose, point, 0.0)
        head = image_coordinates(pose, point, PERSON_HEIGHT_M)
        if feet is None or head is None or not feet.in_frame:
            continue

        # Apparent height comes from projecting the same world point at two
        # heights, so size and ground position cannot drift apart. A detector
        # reading the bottom edge of this box therefore recovers the true
        # ground position, which is the property under test.
        bottom = feet.v * HEIGHT
        top = head.v * HEIGHT
        height_px = max(6.0, bottom - top)
        width_px = max(3.0, height_px * 0.34)
        centre_x = feet.u * WIDTH

        left = int(centre_x - width_px / 2)
        top_px = int(top)
        right = int(centre_x + width_px / 2)
        bottom_px = int(bottom)

        torso_top = top_px + int(height_px * 0.20)
        hip = top_px + int(height_px * 0.58)

        cv2.rectangle(frame, (left, torso_top), (right, hip), walker.colour, -1)
        cv2.rectangle(
            frame, (left, hip), (right, bottom_px),
            tuple(max(0, int(c * 0.62)) for c in walker.colour), -1,
        )
        cv2.rectangle(
            frame, (left, torso_top), (right, torso_top + max(1, int(height_px * 0.12))),
            tuple(min(255, int(c * 1.7) + 24) for c in walker.colour), -1,
        )
        cv2.circle(
            frame, (int(centre_x), top_px + int(height_px * 0.10)),
            max(2, int(width_px * 0.44)), (128, 148, 176), -1,
        )

    noise = rng.normal(0.0, 3.0, frame.shape)
    return np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def write_view(
    path: Path,
    pose: CameraPose,
    walkers: tuple[WorldWalker, ...] = LONE_WALKER,
    *,
    seed: int = 11,
) -> Path:
    """Encode one camera's view of the world to a real video file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    background = _background(rng)

    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT)
    )
    if not writer.isOpened():
        raise RuntimeError(f"No encoder available for {path.suffix}")

    try:
        for index in range(FRAME_COUNT):
            writer.write(render(pose, walkers, index, background, rng))
    finally:
        writer.release()

    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"The encoder produced no output at {path}")
    return path


def truth_at(walkers: tuple[WorldWalker, ...], index: int) -> dict[str, LatLon]:
    """Where each walker actually is, on the ground, in frame ``index``.

    This is the ground truth spatial accuracy is measured against — a real world
    position, not a box in a picture.
    """
    progress = index / max(1, FRAME_COUNT - 1)
    positions = {}
    for walker in walkers:
        point = walker.position_at(progress)
        if point is not None:
            positions[walker.name] = point
    return positions
