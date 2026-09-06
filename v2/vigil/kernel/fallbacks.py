"""The same algorithms as the core, in NumPy, for when the core is absent.

Every function here mirrors one in `core/src/`. That is a duplication, and it
is safe for exactly one reason: `tests/test_native.py` holds each pair to
**identical** answers — not close ones — over randomised inputs, ties
included. Without that the fallback would be a second, subtly different
product that runs on machines where the build did not.

They live apart from `native` because `native` is about the boundary — loading
a library, checking an ABI, marshalling pointers — and these are about
arithmetic. Nothing here touches ctypes.

Where a mirror would be dishonest, there is none. The ground rasteriser has no
NumPy path: v1 measured one at 79 ms per frame per camera, and quietly running
eighty times slower is not a fallback, so `MapBuilder` refuses instead.
"""

from __future__ import annotations

import math

import numpy as np

#: Mirrors `core/src/triangulate.rs`. The core's copy is the definition; these
#: exist so the fallback refuses the same pairs the core would, with the core
#: absent.
MAX_GAP_M = 3.0
MIN_PARALLAX_DEG = 5.0

#: Refusal codes, mirroring `core/src/triangulate.rs`'s `Refusal`. Defined in
#: `native` as the ABI's numbers and repeated here so a fallback returns the
#: same code the core would; `domain.triangulation` maps them to names.
PARALLEL, TOO_LITTLE_PARALLAX, BEHIND, TOO_FAR_APART = -2, -3, -4, -5

def _assign_numpy(cost: np.ndarray) -> np.ndarray:
    """Jonker-Volgenant in NumPy: the same algorithm, one axis vectorised.

    Kept short and deliberately not clever. It exists so a checkout without a
    Rust toolchain still associates *optimally* — a greedy stand-in here would
    mean the fallback silently reintroduced the identity swaps the whole
    rewrite was for.
    """
    rows, cols = cost.shape
    if rows > cols:
        return _transpose_assignment(_assign_numpy(np.ascontiguousarray(cost.T)), rows, cols)
    finite = np.where(np.isfinite(cost), cost, 1.0e9)
    u = np.zeros(rows + 1)
    v = np.zeros(cols + 1)
    p = np.zeros(cols + 1, dtype=np.int64)
    way = np.zeros(cols + 1, dtype=np.int64)
    for i in range(1, rows + 1):
        p[0] = i
        j0 = 0
        minv = np.full(cols + 1, np.inf)
        used = np.zeros(cols + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            free = ~used[1:]
            if not free.any():
                break
            current = finite[i0 - 1] - u[i0] - v[1:]
            better = free & (current < minv[1:])
            minv[1:][better] = current[better]
            way[1:][better] = j0
            candidates = np.where(free, minv[1:], np.inf)
            j1 = int(np.argmin(candidates)) + 1
            delta = candidates[j1 - 1]
            if not np.isfinite(delta):
                break
            u[p[used]] += delta
            v[used] -= delta
            minv[~used] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    out = np.full(rows, -1, dtype=np.int64)
    for j in range(1, cols + 1):
        if p[j]:
            out[p[j] - 1] = j - 1
    return out

def _transpose_assignment(by_column: np.ndarray, rows: int, cols: int) -> np.ndarray:
    out = np.full(rows, -1, dtype=np.int64)
    for col, row in enumerate(by_column):
        if row >= 0:
            out[row] = col
    return out

def _triangulate_numpy(a_o, a_d, b_o, b_d, angular_sigma_deg, min_parallax_deg,
                       out) -> tuple[int, np.ndarray]:
    """The same arithmetic without the core. `tests/test_native.py` holds the
    two to agreement, which is what makes having both safe."""
    na, nb = np.linalg.norm(a_d), np.linalg.norm(b_d)
    if not np.isfinite(na) or not np.isfinite(nb) or na < 1e-12 or nb < 1e-12:
        raise NativeError("a ray with a direction of no length")
    a_d, b_d = a_d / na, b_d / nb
    d = float(a_d @ b_d)
    denominator = 1.0 - d * d
    if denominator < 1e-12:
        return PARALLEL, out
    parallax = math.degrees(math.acos(min(1.0, max(-1.0, d))))
    if parallax > 90.0:
        parallax = 180.0 - parallax
    if parallax < min_parallax_deg:
        return TOO_LITTLE_PARALLAX, out
    w = a_o - b_o
    e, f = float(a_d @ w), float(b_d @ w)
    s = (d * f - e) / denominator
    t = (f - d * e) / denominator
    if s <= 0.0 or t <= 0.0:
        return BEHIND, out
    pa, pb = a_o + a_d * s, b_o + b_d * t
    gap = float(np.linalg.norm(pa - pb))
    if gap > MAX_GAP_M:
        return TOO_FAR_APART, out
    point = 0.5 * (pa + pb)
    sigma = 0.5 * (s + t) * math.radians(angular_sigma_deg) / max(1e-9, math.sin(math.radians(parallax)))
    out[:] = (point[0], point[1], point[2], parallax, gap, s, t, sigma)
    return 0, out



def _fit_plane_numpy(cloud, threshold_m, iterations, seed, out) -> np.ndarray | None:
    # The same xorshift as the core, so the same points draw the same triples
    # and the two implementations agree exactly rather than approximately.
    state = (int(seed) | 1) & 0xFFFFFFFFFFFFFFFF

    def draw(n: int) -> int:
        nonlocal state
        state ^= (state << 13) & 0xFFFFFFFFFFFFFFFF
        state ^= state >> 7
        state ^= (state << 17) & 0xFFFFFFFFFFFFFFFF
        return state % n

    best, best_count = None, -1
    for _ in range(max(1, iterations)):
        i, j, k = draw(len(cloud)), draw(len(cloud)), draw(len(cloud))
        if i == j or j == k or i == k:
            continue
        plane = _plane_through(cloud[i], cloud[j], cloud[k])
        if plane is None:
            continue
        count = int(np.count_nonzero(np.abs(cloud @ plane[0] - plane[1]) <= threshold_m))
        if count > best_count:
            best, best_count = plane, count
    if best is None:
        best = _least_squares_plane(cloud)
        if best is None:
            return None
    inliers = cloud[np.abs(cloud @ best[0] - best[1]) <= threshold_m]
    if len(inliers) < 3:
        return None
    plane = _least_squares_plane(inliers)
    if plane is None:
        return None
    normal, offset = plane
    heights = inliers @ normal - offset
    tilt = (0.0, 0.0) if abs(normal[2]) < 1e-9 else (-normal[0] / normal[2], -normal[1] / normal[2])
    out[:] = (normal[0], normal[1], normal[2], offset, len(inliers),
              float(np.sqrt(np.mean(heights ** 2))), tilt[0], tilt[1])
    return out

def _plane_through(a, b, c):
    return _upward(np.cross(b - a, c - a), a)

def _least_squares_plane(points):
    """Least squares over `z = ax + by + c`; level when the points are in a line."""
    centre = points.mean(axis=0)
    d = points - centre
    sxx, sxy, syy = float(d[:, 0] @ d[:, 0]), float(d[:, 0] @ d[:, 1]), float(d[:, 1] @ d[:, 1])
    sxz, syz = float(d[:, 0] @ d[:, 2]), float(d[:, 1] @ d[:, 2])
    determinant = sxx * syy - sxy * sxy
    if abs(determinant) < 1e-12:
        return _upward(np.array([0.0, 0.0, 1.0]), centre)
    a = (sxz * syy - syz * sxy) / determinant
    b = (syz * sxx - sxz * sxy) / determinant
    return _upward(np.array([-a, -b, 1.0]), centre)

def _upward(normal, through):
    length = float(np.linalg.norm(normal))
    if not np.isfinite(length) or length < 1e-12:
        return None
    unit = normal / length
    if unit[2] < 0:
        unit = -unit
    return unit, float(unit @ through)

def _suppress_numpy(xyxy, scores, classes, iou_threshold, soft, sigma, floor):
    """The same algorithm without the core, held to it by `tests/test_native.py`.

    The ordering is **descending score, then ascending index**, matching the
    Rust exactly. `argsort` is not stable by default, so equal scores — which
    a quantised model emits constantly — could otherwise come out in either
    order and the two implementations would disagree for a reason neither
    could be blamed for.
    """
    keep: list[int] = []
    for label in np.unique(classes):
        members = np.flatnonzero(classes == label)
        local = (_soft_nms_numpy if soft else _nms_numpy)(
            xyxy[members], scores[members], iou_threshold, sigma, floor)
        keep.extend(int(members[i]) for i in local)
    return np.array(sorted(keep, key=lambda i: (-scores[i], i)), dtype=np.int64)

def _order(scores: np.ndarray) -> np.ndarray:
    return np.lexsort((np.arange(len(scores)), -scores))

def _iou_against(xyxy, areas, i, rest):
    xx1 = np.maximum(xyxy[i, 0], xyxy[rest, 0])
    yy1 = np.maximum(xyxy[i, 1], xyxy[rest, 1])
    xx2 = np.minimum(xyxy[i, 2], xyxy[rest, 2])
    yy2 = np.minimum(xyxy[i, 3], xyxy[rest, 3])
    inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
    return inter / (areas[i] + areas[rest] - inter + 1e-9)

def _nms_numpy(xyxy, scores, iou_threshold, _sigma, _floor):
    areas = (xyxy[:, 2] - xyxy[:, 0]).clip(0) * (xyxy[:, 3] - xyxy[:, 1]).clip(0)
    order = list(_order(scores))
    keep: list[int] = []
    while order:
        i = int(order.pop(0))
        keep.append(i)
        if not order:
            break
        rest = np.asarray(order, dtype=np.int64)
        order = [int(j) for j in rest[_iou_against(xyxy, areas, i, rest) <= iou_threshold]]
    return keep

def _soft_nms_numpy(xyxy, scores, iou_threshold, sigma, floor):
    areas = (xyxy[:, 2] - xyxy[:, 0]).clip(0) * (xyxy[:, 3] - xyxy[:, 1]).clip(0)
    working = scores.astype(np.float64).copy()
    taken = np.zeros(len(scores), dtype=bool)
    keep: list[int] = []
    while True:
        live = np.flatnonzero(~taken & (working >= floor))
        if not len(live):
            break
        # Highest working score, ties to the lower index — the same rule the
        # Rust applies, written out rather than left to a sort's default.
        best = int(live[np.lexsort((live, -working[live]))[0]])
        taken[best] = True
        keep.append(best)
        rest = np.flatnonzero(~taken & (working >= floor))
        if not len(rest):
            break
        overlap = _iou_against(xyxy, areas, best, rest)
        decayed = overlap > iou_threshold
        working[rest[decayed]] *= np.exp(-(overlap[decayed] ** 2) / sigma)
    return keep
