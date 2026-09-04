"""The site: one fixed origin, one boundary, one clock.

Everything geographic here has until now been anchored to whatever happened to
be placed first. The plan view takes its origin from the first camera, so
removing that camera re-anchors the frame and every zone, track and footprint
jumps on screen — nothing moved, the ruler did. Coverage takes its origin from
the first vertex of whatever boundary it was handed, so listing the same site
from a different corner builds a different tangent plane and returns areas that
differ in the last digits for no reason anybody could explain.

A site row is what stops an origin from being a side effect of insertion order.
It is also the thing basemaps, floor levels, plans and exports will each have to
agree with, and they can only agree with something that is written down.

Three fields, each because something above needs it:

**An origin and a frame kind.** A ``GEOGRAPHIC`` site's origin is a real
coordinate: latitudes shown against it mean what they say. A ``LOCAL`` site is a
floor plan or a hand sketch whose origin is a fixed but arbitrary point, and
every coordinate an interface would print for it is invented precision. The kind
is stored so that interface can refuse to print them, rather than each screen
guessing.

**An IANA time-zone name.** A schedule typed as 18:00–06:00 means 18:00 at the
site. Storing an offset instead would be wrong for half the year, in the dark,
on the night the clocks change — see :meth:`Site.clock`.

**A boundary ring, or nothing.** Optional on purpose: a site whose outline has
not been drawn is a normal state, and an empty ring stored as though it were a
real one would report the whole site as one uncovered gap. Stored as JSON in the
same shape zones store theirs, so one reader serves both.

:class:`SiteFrame` is the conversion between latitude/longitude and metres east
and north of that origin. It is intended to replace both
``sentinel.coverage._Frame`` and ``MapView._to_local``, which are today the same
arithmetic written twice in two files that cannot see each other. Neither is
edited here: until the plan view actually reads a stored origin, swapping the
class underneath it would move nothing, and a test asserts the two agree so they
cannot drift in the meantime. It is built from ``haversine_distance``,
``bearing_degrees`` and ``destination_point`` — never from metres-per-degree
constants of its own. Those constants exist, in Rust, and are tested there; a
second copy in Python would be a second answer, and the two would come to differ
for reasons nobody would find.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import tzinfo
from enum import Enum
from typing import Sequence

from .core import LatLon, bearing_degrees, destination_point, haversine_distance

#: The id of the site every deployment has before anybody names a second one.
#: One site per node today; the id exists so that stops being an assumption
#: baked into every query.
DEFAULT_SITE_ID = "default"

#: What a site's clock is when nobody has declared one. UTC rather than the
#: machine's own zone: a schedule evaluated in a zone the operator never chose
#: is wrong in a way nothing on screen explains, and UTC is at least wrong in a
#: way somebody can spot.
DEFAULT_TIMEZONE = "UTC"


class SiteError(ValueError):
    """The site record cannot be used as given."""


class FrameKind(str, Enum):
    """Whether this site's coordinates mean anything outside the drawing.

    Recorded rather than inferred from whether an origin looks plausible. Every
    latitude is plausible, including the one somebody typed to get a floor plan
    onto the screen, and a screen that guesses will eventually print a
    fabricated coordinate beside a real one with nothing to tell them apart.
    """

    #: The origin is a surveyed or mapped coordinate. Latitudes mean what they
    #: say, and may be exported, printed, and handed to somebody driving there.
    GEOGRAPHIC = "GEOGRAPHIC"
    #: The origin is a fixed but arbitrary point on a plan. Distances and areas
    #: are real; coordinates are not, and must not be shown as though they were.
    LOCAL = "LOCAL"


@dataclass(frozen=True, slots=True)
class Site:
    """The place being watched, as one record.

    ``origin`` anchors the metric frame and is deliberately not derived from the
    cameras: a derived origin moves when the thing it was derived from is
    deleted, which is the plan view bug this record exists to end.

    ``boundary`` is an open ring of at least three points, or empty. Empty means
    *not drawn yet*, and must not be read as an outline enclosing nothing — the
    difference is between "coverage cannot be computed" and "none of this site is
    covered", and only one of those is worth alarming about.
    """

    id: str
    name: str
    origin: LatLon
    frame: FrameKind = FrameKind.GEOGRAPHIC
    #: An IANA name such as ``Asia/Beirut``. Never a fixed offset — see
    #: :meth:`clock`.
    timezone: str = DEFAULT_TIMEZONE
    boundary: tuple[LatLon, ...] = ()

    def __post_init__(self) -> None:
        """Refuse a ring of one or two points at the door.

        A two-point boundary reaches shapely as a line, whose area is zero, so
        every coverage figure computed against it is a division by zero or a
        confident 0% — a site reported as entirely unwatched because somebody
        clicked twice and stopped.
        """
        if 0 < len(self.boundary) < 3:
            raise SiteError(
                f"a site boundary needs at least three points; "
                f"{len(self.boundary)} were given. Leave it empty for a site "
                "whose outline has not been drawn yet."
            )

    @property
    def is_georeferenced(self) -> bool:
        """Whether a coordinate from this site may be shown to somebody.

        The question every screen that prints a latitude has to ask. A position
        in a LOCAL frame is a position on a drawing, and presenting it as a place
        on the Earth is the same class of lie as presenting a camera's own
        position as the location of what it saw.
        """
        return self.frame is FrameKind.GEOGRAPHIC

    @property
    def has_boundary(self) -> bool:
        """Whether coverage has anything to subtract footprints from."""
        return len(self.boundary) >= 3

    def clock(self) -> tzinfo:
        """The zone this site's schedules are written in.

        An IANA name rather than a stored offset, because an offset is wrong for
        half the year: a site recorded as UTC+3 in August is UTC+2 in January,
        and an "after hours" window that moves by an hour on the night the clocks
        change disarms the site at exactly the hour nobody is watching it.

        Raises :class:`SiteError` rather than falling back to UTC when the name
        cannot be resolved — which on Windows means the optional ``tzdata``
        package is missing. A silent fallback is how an evaluator came to arm at
        21:00 local instead of 18:00 with nothing on screen saying so; a caller
        that wants to carry on anyway has to choose that itself, somewhere the
        operator can see the choice.
        """
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

        try:
            return ZoneInfo(self.timezone)
        except (ZoneInfoNotFoundError, ValueError, KeyError) as error:
            raise SiteError(
                f"the site's time zone {self.timezone!r} cannot be resolved on "
                "this machine, so its schedules cannot be read in the site's "
                "clock. On Windows the tz database is not shipped with Python: "
                "install the 'tzdata' package."
            ) from error

    def metric_frame(self) -> "SiteFrame":
        """The frame every module measuring this site is meant to share.

        Handed out by the site rather than constructed per caller, because two
        frames on two different origins are two answers to the same question —
        and the place they disagree, the far corner of a large site, is the place
        nobody checks.
        """
        return SiteFrame(self.origin)


class SiteFrame:
    """Metres east and north of a site origin, and back.

    Planar geometry needs metres. A union or an area computed in degrees is
    wrong by the cosine of the latitude, which at this project's own test site
    is an 18% error that looks entirely plausible. A site is hundreds of metres
    across, not hundreds of kilometres, so a local tangent plane about a fixed
    origin is exact enough — it round-trips a point a kilometre out to well
    inside a centimetre — and it needs no projection library.

    Intended to replace ``sentinel.coverage._Frame`` and ``MapView._to_local``,
    which do this same arithmetic in two other files today. Those are left alone
    here: replacing them is a separate change with its own tests, and until the
    plan view reads a stored origin, swapping the class underneath it would move
    nothing. A test asserts this frame and ``coverage._Frame`` agree on the same
    point, so the two cannot drift apart in the meantime.

    The conversion goes through ``haversine_distance``, ``bearing_degrees`` and
    ``destination_point`` — the Rust core's geodesy, tested there — rather than
    metres-per-degree constants of its own, so there is exactly one answer in the
    system to "how far apart are these two points".
    """

    __slots__ = ("origin",)

    def __init__(self, origin: LatLon):
        self.origin = origin

    @classmethod
    def of(cls, site: Site) -> "SiteFrame":
        """The frame of a stored site, on its recorded origin.

        The whole point of the site record: an origin that outlives the first
        camera placed on it.
        """
        return cls(site.origin)

    def to_xy(self, point: LatLon) -> tuple[float, float]:
        """Metres east and north of the origin.

        Identical to ``coverage._Frame.to_xy`` deliberately, down to the
        zero-distance branch: the bearing from a point to itself is arbitrary,
        and multiplying an arbitrary bearing by a zero distance is harmless only
        until somebody changes the multiplication.
        """
        distance = haversine_distance(self.origin, point)
        if distance == 0.0:
            return (0.0, 0.0)
        bearing = math.radians(bearing_degrees(self.origin, point))
        return (distance * math.sin(bearing), distance * math.cos(bearing))

    def to_latlon(self, x: float, y: float) -> LatLon:
        """The coordinate that many metres east and north of the origin.

        The exact inverse of :meth:`to_xy`, and it has to be: an outline drawn on
        screen is converted one way and stored the other, so a conversion losing
        a millimetre a trip would walk a site's boundary off its own fence over a
        season of edits.
        """
        distance = math.hypot(x, y)
        if distance == 0.0:
            return self.origin
        return destination_point(
            self.origin, math.degrees(math.atan2(x, y)) % 360.0, distance
        )

    def ring_to_xy(self, ring: Sequence[LatLon]) -> list[tuple[float, float]]:
        """A whole ring in one call, for handing to shapely.

        Every caller of this conversion is really converting a polygon, and the
        comprehension that does it was about to be written for a fourth time.
        """
        return [self.to_xy(point) for point in ring]

    def ring_to_latlon(
        self, points: Sequence[tuple[float, float]]
    ) -> tuple[LatLon, ...]:
        """A shapely ring back to coordinates, in the order it went in."""
        return tuple(self.to_latlon(x, y) for x, y in points)

    def __repr__(self) -> str:
        return f"SiteFrame(origin={self.origin!r})"
