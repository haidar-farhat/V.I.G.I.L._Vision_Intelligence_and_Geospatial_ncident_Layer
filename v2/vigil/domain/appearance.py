"""What a tracked thing looks like, as a number two frames apart can compare.

# Why this exists

v1 measured the problem and wrote it down: twenty seconds of one person and
some furniture on a laptop camera counted **3, 10, 4 and 11 "distinct
objects"** across four runs. The tracker associated on position alone, so
whenever the detector lost somebody for longer than the gap budget they came
back as a new id, and every count, every group risk factor and every loitering
clock downstream treated the new id as a new person.

v1's answer was `reid.py`: link the fragments *after the fact*, the way the
correlator links tracks across cameras. That reconciles the count and it
cannot do anything else — the event was already raised against fragment #7,
the loitering clock already restarted, and the console already drew a fresh
box. v1 said so in its own docstring.

v2 dropped `reid.py` entirely and kept the tracker, so v2 had the
fragmentation and none of the mitigation.

This is the descriptor, moved to where it can actually prevent the split: the
tracker holds one per track and consults it *during* association, so a person
who reappears is matched to their own track rather than reconciled with it an
hour later.

# Why a colour histogram and not a learned embedding

Nothing here may be downloaded and a re-identification network is a download.
More to the point, a masked HSV histogram is measurably enough for the job it
is being given: it does not have to identify a stranger, it has to distinguish
the four people currently in one camera's frame over the next fifteen seconds.
`tools/measure_fragmentation.py` is the measurement, and it runs on this.

**What it cannot do, stated plainly.** It cannot tell two people in the same
dark coat apart — colour is all it has. So similarity never links on its own:
time and place are conditions rather than tie-breakers, and a red coat leaving
one side of the frame while a red coat arrives at the other is two people.

# Where the pixels are read

Only the *type* and the comparison live here. Turning a crop into a descriptor
is pixel work and lives in `vigil.perception.appearance`, because the domain
does not import OpenCV and a tracker should not be doing colour conversion.

# The gallery

v1 kept one exponentially-averaged descriptor per track. That follows a person
turning, and it also *forgets* what their front looked like by the time it has
learned their back — so somebody who turns around and turns back matches worse
than they did. This keeps a short gallery and compares against the closest
member of it, which is what a person's several appearances actually are.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Hue and saturation together, value separately. A full 3-D cube spreads a
#: distant person's few hundred pixels across a thousand bins until the cosine
#: between two frames of the *same* object is noise; two dense marginals tell
#: one coat from another.
HUE_BINS = 16
SATURATION_BINS = 8
#: Value is kept because most clothing is near grey, where hue is undefined
#: and saturation nil — value is the only channel separating a black coat from
#: a white shirt. It carries half the histogram's mass rather than all of it
#: because illumination drift moves it and moves nothing else.
VALUE_BINS = 8

#: Fewer pixels than this is not an appearance, it is a colour sample of the
#: noise floor, and comparing two of them links anything to anything.
MINIMUM_PIXELS = 32

#: How many past appearances a track keeps. Six at 15 fps with the sampling
#: below spans several seconds and a person's turn; more costs memory for
#: views that are no longer what they look like.
GALLERY_DEPTH = 6

#: Ceiling for appearance used *inside a frame*, where geometry already
#: decides and appearance only shades the cost.
#:
#: Loose on purpose, and the looseness was measured. At 0.45 this gate vetoed
#: correct pairs whose crop was momentarily contaminated by whoever was
#: standing in front of them, and the tracker was then forced onto its second
#: choice: four people milling in a small space cost 105 identity switches
#: with appearance against 95 without. At 0.70 the same scenario reads 93 —
#: better than geometry alone — because appearance can no longer overrule the
#: Mahalanobis gate and the overlap, only tip a choice between candidates
#: both of them already accept.
MAX_APPEARANCE_DISTANCE = 0.70

#: Ceiling for appearance used to *re-identify* a track after it was lost.
#:
#: Tight, for the opposite reason: across a gap of seconds the box has moved
#: and geometry is barely evidence at all, so appearance is what is deciding.
#: A gate this side of the separation between two people (measured at 0.56 on
#: the synthetic scenes) is what stops a red coat leaving one side of the
#: frame and a blue coat arriving at the other from being called one person.
#:
#: **Both numbers are calibrated on synthetic scenes.** Real clothing under
#: real light will not separate as cleanly, and the honest expectation is that
#: this one needs raising and the one above needs watching. See
#: PRODUCTION_READINESS.md.
MAX_REIDENTIFY_DISTANCE = 0.35


@dataclass(frozen=True, slots=True)
class Appearance:
    """A unit-length colour descriptor and the pixel count behind it."""

    vector: np.ndarray
    pixels: int

    @property
    def usable(self) -> bool:
        return self.pixels >= MINIMUM_PIXELS and bool(np.any(self.vector))

    def distance(self, other: "Appearance") -> float:
        """Cosine distance in `[0, 2]`, 0 for identical descriptors."""
        if not self.usable or not other.usable:
            return float("inf")
        return float(1.0 - np.dot(self.vector, other.vector))


def _unit(histogram: np.ndarray) -> np.ndarray:
    total = float(np.linalg.norm(histogram))
    return histogram / total if total > 0 else histogram


@dataclass(slots=True)
class Gallery:
    """The recent appearances of one track, newest last."""

    entries: list[Appearance] = field(default_factory=list)
    depth: int = GALLERY_DEPTH

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def usable(self) -> bool:
        return bool(self.entries)

    def observe(self, appearance: Appearance) -> None:
        """Keep a usable descriptor, drop the oldest when full.

        An unusable one — too few pixels, a box mostly off the frame — is
        discarded rather than stored. A descriptor of forty pixels of noise
        does not become useful by being averaged with a good one; it makes the
        good one worse.
        """
        if not appearance.usable:
            return
        self.entries.append(appearance)
        if len(self.entries) > self.depth:
            del self.entries[0]

    def distance(self, appearance: Appearance) -> float:
        """Distance to the *closest* remembered appearance.

        The minimum rather than the mean: a track's gallery holds a person's
        front and their back, and the mean of those two is a person who does
        not exist and matches neither.
        """
        if not appearance.usable or not self.entries:
            return float("inf")
        return min(entry.distance(appearance) for entry in self.entries)

    def merge(self, other: "Gallery") -> None:
        """Absorb another track's appearances, for a re-identified fragment."""
        for entry in other.entries:
            self.observe(entry)


#: Between-object distances kept to calibrate a scene. Two hundred spans
#: several minutes of a busy view and a whole night of a quiet one, and it is
#: a ring so the figure follows the light rather than averaging noon into
#: midnight.
SEPARATION_SAMPLES = 200

#: Fewest samples before the measured threshold is trusted over the shipped
#: one. Below this the quantile is dominated by whichever two objects happened
#: to be in view.
MIN_SEPARATION_SAMPLES = 30

#: Quantile of the between-object distribution used as the ceiling. A
#: candidate must be closer than 95% of the pairs this scene has proved are
#: *different* objects — so the stated worst case is a 5% chance that any one
#: comparison admits a stranger, before the margin below is applied.
SEPARATION_QUANTILE = 5.0

#: How much better the best candidate must be than the runner-up, as a
#: fraction of the measured spread. Without it, two objects that look equally
#: like a lost track let the solver pick one, and picking is guessing.
SEPARATION_MARGIN = 0.5


@dataclass(slots=True)
class SceneSeparation:
    """How far apart *different* objects look in this particular scene.

    # Why this exists

    `MAX_REIDENTIFY_DISTANCE` was calibrated on synthetic colour blocks, which
    separated two coats at a cosine distance of 0.56. `tools/calibrate.py` ran
    the same measurement on twenty seconds of real video and found the
    *different-object* median at **0.105** — so the shipped gate of 0.35 would
    have admitted nearly everything and merged two objects rather than telling
    them apart. A fragment is visible on screen; a merge is not.

    No fixed number survives that, because the right one is a property of the
    scene: a yard holding a red van and a white car separates cleanly, and a
    corridor of people in dark coats does not. So it is measured instead of
    assumed, and the ground truth for it needs no labels at all — **two tracks
    visible in the same frame are certainly different objects**, because one
    object cannot be in two places.

    The tracker feeds every concurrent pair in here and asks what a stranger
    normally scores. A candidate has to beat that to be called the same
    object.

    # What it does when it cannot tell

    Refuses. If this scene's own measurements say two different objects
    routinely look identical, no threshold exists that both re-identifies one
    object and keeps two apart, and the honest behaviour is to decline and let
    the track fragment — which an operator can see — rather than merge two
    people into one, which nobody can.
    """

    samples: list[float] = field(default_factory=list)

    def observe(self, distance: float) -> None:
        """Record how far apart two *concurrently visible* objects looked."""
        if not (0.0 <= distance < 2.0):
            return
        self.samples.append(float(distance))
        if len(self.samples) > SEPARATION_SAMPLES:
            del self.samples[0]

    @property
    def measured(self) -> bool:
        return len(self.samples) >= MIN_SEPARATION_SAMPLES

    def ceiling(self) -> float:
        """The furthest a candidate may be and still count as the same object.

        The shipped constant until this scene has said otherwise, and the
        smaller of the two afterwards — a scene may prove that the default is
        too generous, and none may raise it.
        """
        if not self.measured:
            return MAX_REIDENTIFY_DISTANCE
        import numpy as np

        return float(min(MAX_REIDENTIFY_DISTANCE,
                         np.percentile(np.asarray(self.samples), SEPARATION_QUANTILE)))

    def margin(self) -> float:
        """How much better the winner must be than the runner-up.

        Scaled by what this scene's spread actually is, so a view where
        everything looks alike demands a larger lead than one where nothing
        does.
        """
        return self.ceiling() * SEPARATION_MARGIN

    def describe(self) -> str:
        if not self.measured:
            return (f"not yet measured ({len(self.samples)}/{MIN_SEPARATION_SAMPLES} samples); "
                    f"using the shipped ceiling of {MAX_REIDENTIFY_DISTANCE}")
        import numpy as np

        a = np.asarray(self.samples)
        return (f"different objects in this scene: median {np.median(a):.3f}, "
                f"p{SEPARATION_QUANTILE:.0f} {np.percentile(a, SEPARATION_QUANTILE):.3f}; "
                f"re-identifying below {self.ceiling():.3f} with a {self.margin():.3f} margin")
