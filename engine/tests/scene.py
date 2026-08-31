"""A synthetic camera scene, encoded to a real video file.

Be clear about what this is and is not. The **file** is real: a genuine
container written by a real encoder and read back by a real decoder, so the
decode path under test is the same one a camera exercises. The **scene** is
synthetic — generated geometry, not photographed reality.

That distinction matters because a synthetic scene is easy on a detector.
Nothing here proves the system works on real footage; it proves the pipeline
carries frames, detections, tracks and positions end to end without lying about
them. Anything stronger has to wait for a camera.

The scene is built to be honest work for the stages below it:

- Perspective: objects further up the frame are smaller and move more slowly in
  pixels for the same ground speed, which is what makes ground projection worth
  doing at all.
- Sensor noise and a slow illumination drift, so a background model has to cope
  with something rather than a static image.
- One object that walks behind an occluder and comes out the other side, which
  is the case that decides whether a tracker keeps an identity or invents one.
- Two objects that cross, which is where naive association swaps them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

WIDTH = 640
HEIGHT = 480
FPS = 15
DURATION_SECONDS = 12
FRAME_COUNT = FPS * DURATION_SECONDS

#: Where the ground plane meets the sky in this synthetic view. Objects above it
#: are unprojectable, which is the property the geometry layer must respect.
HORIZON_Y = 120


@dataclass(frozen=True, slots=True)
class Walker:
    """One moving object, described in normalised image space.

    Motion is expressed as a path through the frame rather than a world
    trajectory: this is a picture generator, and pretending otherwise would
    quietly bake the projection model into its own test data.
    """

    name: str
    start: tuple[float, float]
    end: tuple[float, float]
    #: Fraction of the sequence spent walking; the rest is spent standing still.
    walk_fraction: float = 1.0
    #: Delay before this object appears, as a fraction of the sequence.
    enters_at: float = 0.0
    colour: tuple[int, int, int] = (60, 60, 70)


#: Three objects, chosen to exercise the three failure modes that matter.
WALKERS = (
    # Walks the full depth of the scene, far to near: its box grows by a factor
    # of four, which breaks trackers that assume scale stability.
    Walker("approaching", start=(0.30, 0.32), end=(0.42, 0.86), colour=(64, 58, 92)),
    # Crosses the first one's path.
    Walker("crossing", start=(0.80, 0.45), end=(0.18, 0.62), colour=(52, 74, 60)),
    # Enters late, walks briefly, then stands still for the rest of the
    # sequence: the loitering case, where "speed unknown" and "speed zero" must
    # not be confused.
    Walker(
        "loiterer",
        start=(0.62, 0.70),
        end=(0.55, 0.78),
        walk_fraction=0.35,
        enters_at=0.25,
        colour=(78, 62, 54),
    ),
)

#: A static occluder the approaching walker passes behind.
OCCLUDER = (0.34, 0.52, 0.10, 0.13)  # x, y, w, h normalised


def _ground_texture(rng: np.random.Generator) -> np.ndarray:
    """A textured ground plane under a plain sky.

    Texture matters: a uniform background makes frame differencing look far
    better than it is.
    """
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    # Sky: flat and bright.
    frame[:HORIZON_Y] = (150, 140, 128)

    # Ground: darker with distance, so the scene has depth cues.
    for y in range(HORIZON_Y, HEIGHT):
        depth = (y - HORIZON_Y) / (HEIGHT - HORIZON_Y)
        value = int(48 + 46 * depth)
        frame[y] = (value, value + 4, value + 8)

    # Gravel-like speckle, coarser in the foreground where a real camera
    # resolves more detail.
    ground = frame[HORIZON_Y:]
    noise = rng.normal(0.0, 9.0, ground.shape[:2])
    scale = np.linspace(0.35, 1.6, ground.shape[0])[:, None]
    ground[:] = np.clip(ground + (noise * scale)[..., None], 0, 255).astype(np.uint8)

    # A path and two markings, so there is real structure to hold still.
    cv2.line(frame, (int(WIDTH * 0.12), HEIGHT), (int(WIDTH * 0.44), HORIZON_Y), (96, 98, 104), 2)
    cv2.line(frame, (int(WIDTH * 0.92), HEIGHT), (int(WIDTH * 0.56), HORIZON_Y), (96, 98, 104), 2)
    cv2.rectangle(frame, (int(WIDTH * 0.05), HEIGHT - 40), (int(WIDTH * 0.16), HEIGHT - 8),
                  (88, 90, 96), -1)

    return frame


def _scale_at(y_normalised: float) -> float:
    """How large an object of fixed real height appears at this image row.

    Linear in the distance below the horizon: crude, but it is the same
    monotonic relationship real perspective has, which is all the stages under
    test depend on.
    """
    y = y_normalised * HEIGHT
    depth = max(y - HORIZON_Y, 6.0) / (HEIGHT - HORIZON_Y)
    return 0.18 + 0.82 * depth


def walker_box(walker: Walker, progress: float) -> tuple[int, int, int, int] | None:
    """Pixel box for one walker at a point in the sequence, or ``None`` if absent.

    Also the ground truth: tests compare what the pipeline reports against these
    boxes, so this function is the specification of what is actually there.
    """
    if progress < walker.enters_at:
        return None

    span = max(1e-6, 1.0 - walker.enters_at)
    local = (progress - walker.enters_at) / span
    travelled = min(local / walker.walk_fraction, 1.0) if walker.walk_fraction > 0 else 1.0

    x = walker.start[0] + (walker.end[0] - walker.start[0]) * travelled
    y = walker.start[1] + (walker.end[1] - walker.start[1]) * travelled

    scale = _scale_at(y)
    height = int(160 * scale)
    width = max(4, int(height * 0.36))

    # A slight gait bob, so the box is not perfectly steady frame to frame.
    bob = int(math.sin(progress * math.pi * 24) * max(1.0, height * 0.012))

    left = int(x * WIDTH - width / 2)
    top = int(y * HEIGHT - height) + bob
    return left, top, width, height


def _draw_walker(
    frame: np.ndarray, walker: Walker, box: tuple[int, int, int, int], phase: float
) -> None:
    """Draw one walker with internal structure.

    Head, torso and legs are drawn as distinct tones rather than one flat
    rectangle. This is not decoration: a uniformly shaded shape is an
    unrealistically hard case for a background model, which absorbs the
    unchanging interior of a slow-moving solid block and leaves only its leading
    edge. Real objects have internal texture, and testing against a shape that
    does not would mean tuning the detector for a problem nobody has.
    """
    left, top, width, height = box
    centre = left + width // 2

    torso_top = top + int(height * 0.20)
    hip = top + int(height * 0.58)

    torso = walker.colour
    legs = tuple(max(0, int(c * 0.62)) for c in walker.colour)
    skin = (128, 148, 176)

    cv2.rectangle(frame, (left, torso_top), (left + width, hip), torso, -1)

    # Legs, swinging apart and together, so the silhouette changes shape frame
    # to frame the way a walking person's does.
    swing = int(math.sin(phase * math.pi * 2) * max(1.0, width * 0.30))
    leg_w = max(2, int(width * 0.34))
    cv2.rectangle(frame, (centre - leg_w - swing, hip),
                  (centre - swing, top + height), legs, -1)
    cv2.rectangle(frame, (centre + swing, hip),
                  (centre + leg_w + swing, top + height), legs, -1)

    # A lighter band across the shoulders, which is what most clothing does and
    # what gives a background model something to key on.
    cv2.rectangle(frame, (left, torso_top),
                  (left + width, torso_top + max(1, int(height * 0.10))),
                  tuple(min(255, int(c * 1.7) + 22) for c in walker.colour), -1)

    head_radius = max(2, int(width * 0.42))
    cv2.circle(frame, (centre, top + int(height * 0.12)), head_radius, skin, -1)

    # A contact shadow, which is what makes the ground-contact point of a box
    # ambiguous in real footage too.
    shadow = frame.copy()
    cv2.ellipse(
        shadow,
        (left + width // 2, top + height),
        (max(3, width // 2), max(2, height // 22)),
        0, 0, 360, (20, 20, 24), -1,
    )
    cv2.addWeighted(shadow, 0.55, frame, 0.45, 0, frame)


def render_frame(index: int, background: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One frame of the scene."""
    progress = index / max(1, FRAME_COUNT - 1)
    frame = background.copy()

    # Slow illumination drift, as a real scene has. A background model that
    # cannot absorb this produces phantom detections across the whole image.
    drift = 1.0 + 0.05 * math.sin(progress * math.pi * 2)
    frame = np.clip(frame.astype(np.float32) * drift, 0, 255).astype(np.uint8)

    for walker in WALKERS:
        box = walker_box(walker, progress)
        if box is not None:
            _draw_walker(frame, walker, box, index / 6.0)

    # The occluder is drawn last so it covers whatever passes behind it.
    ox, oy, ow, oh = OCCLUDER
    cv2.rectangle(
        frame,
        (int(ox * WIDTH), int(oy * HEIGHT)),
        (int((ox + ow) * WIDTH), int((oy + oh) * HEIGHT)),
        (72, 66, 58),
        -1,
    )

    # Sensor noise, applied after everything else, as a sensor would.
    noise = rng.normal(0.0, 3.5, frame.shape)
    return np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def write_scene(path: Path, *, seed: int = 7) -> Path:
    """Encode the scene to a real video file and return its path.

    Deterministic for a given seed, so a test that fails can be re-run against
    identical input.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    background = _ground_texture(rng)

    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT)
    )
    if not writer.isOpened():
        raise RuntimeError(
            f"No encoder available for {path.suffix}. OpenCV was built with "
            "FFmpeg, so this usually means the container is unsupported."
        )

    try:
        for index in range(FRAME_COUNT):
            writer.write(render_frame(index, background, rng))
    finally:
        writer.release()

    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"The encoder produced no output at {path}")

    return path


def ground_truth(index: int) -> dict[str, tuple[int, int, int, int]]:
    """Which objects are where in frame ``index``, in pixels."""
    progress = index / max(1, FRAME_COUNT - 1)
    boxes = {}
    for walker in WALKERS:
        box = walker_box(walker, progress)
        if box is not None:
            boxes[walker.name] = box
    return boxes
