import type { Vec2 } from '@sentinel/shared-types';
import { pointToSegment } from './vec.ts';

/**
 * Planar polygon predicates, in local metres.
 *
 * Rings are implicitly closed: the last vertex connects back to the first, and
 * the first vertex is never repeated at the end.
 */

/**
 * Crossing-number point-in-polygon.
 *
 * The half-open vertical test `(yi > y) !== (yj > y)` counts each vertex exactly
 * once, so a point level with a vertex is classified consistently instead of
 * flickering - which matters when a track walks along a zone edge and would
 * otherwise generate a burst of spurious enter/exit events.
 */
export const pointInPolygon = (point: Vec2, ring: readonly Vec2[]): boolean => {
  const n = ring.length;
  if (n < 3) return false;

  let inside = false;
  for (let i = 0, j = n - 1; i < n; j = i, i += 1) {
    const a = ring[i];
    const b = ring[j];
    if (a === undefined || b === undefined) continue;

    const intersects =
      a.y > point.y !== b.y > point.y &&
      point.x < ((b.x - a.x) * (point.y - a.y)) / (b.y - a.y) + a.x;

    if (intersects) inside = !inside;
  }
  return inside;
};

/** Signed area. Positive is counter-clockwise. */
export const signedArea = (ring: readonly Vec2[]): number => {
  const n = ring.length;
  if (n < 3) return 0;

  let sum = 0;
  for (let i = 0, j = n - 1; i < n; j = i, i += 1) {
    const a = ring[i];
    const b = ring[j];
    if (a === undefined || b === undefined) continue;
    sum += (b.x + a.x) * (b.y - a.y);
  }
  return sum / 2;
};

export const area = (ring: readonly Vec2[]): number => Math.abs(signedArea(ring));

/** Area-weighted centroid. Falls back to the vertex mean for degenerate rings. */
export const centroid = (ring: readonly Vec2[]): Vec2 => {
  const n = ring.length;
  if (n === 0) return { x: 0, y: 0 };

  const a2 = signedArea(ring);
  if (a2 === 0) {
    let sx = 0;
    let sy = 0;
    let counted = 0;
    for (const p of ring) {
      if (p === undefined) continue;
      sx += p.x;
      sy += p.y;
      counted += 1;
    }
    return counted === 0 ? { x: 0, y: 0 } : { x: sx / counted, y: sy / counted };
  }

  let cx = 0;
  let cy = 0;
  for (let i = 0, j = n - 1; i < n; j = i, i += 1) {
    const p = ring[i];
    const q = ring[j];
    if (p === undefined || q === undefined) continue;
    const f = q.x * p.y - p.x * q.y;
    cx += (q.x + p.x) * f;
    cy += (q.y + p.y) * f;
  }
  return { x: cx / (6 * a2), y: cy / (6 * a2) };
};

/** Perpendicular distance from a point to the ring's boundary (never negative). */
export const distanceToRing = (point: Vec2, ring: readonly Vec2[]): number => {
  const n = ring.length;
  if (n === 0) return Number.POSITIVE_INFINITY;
  if (n === 1) {
    const only = ring[0];
    return only === undefined ? Number.POSITIVE_INFINITY : Math.hypot(point.x - only.x, point.y - only.y);
  }

  let best = Number.POSITIVE_INFINITY;
  for (let i = 0, j = n - 1; i < n; j = i, i += 1) {
    const a = ring[i];
    const b = ring[j];
    if (a === undefined || b === undefined) continue;
    const d = pointToSegment(point, a, b).distance;
    if (d < best) best = d;
  }
  return best;
};

/** Shortest distance from a point to an open polyline. */
export const distanceToPolyline = (point: Vec2, path: readonly Vec2[]): number => {
  if (path.length === 0) return Number.POSITIVE_INFINITY;
  if (path.length === 1) {
    const only = path[0];
    return only === undefined ? Number.POSITIVE_INFINITY : Math.hypot(point.x - only.x, point.y - only.y);
  }

  let best = Number.POSITIVE_INFINITY;
  for (let i = 0; i < path.length - 1; i += 1) {
    const a = path[i];
    const b = path[i + 1];
    if (a === undefined || b === undefined) continue;
    const d = pointToSegment(point, a, b).distance;
    if (d < best) best = d;
  }
  return best;
};

const orientation = (a: Vec2, b: Vec2, c: Vec2): number => {
  const v = (b.y - a.y) * (c.x - b.x) - (b.x - a.x) * (c.y - b.y);
  if (v > 0) return 1;
  if (v < 0) return -1;
  return 0;
};

/**
 * Proper segment intersection test.
 *
 * Deliberately excludes collinear-touching and endpoint-grazing cases: a track
 * whose interpolated path merely touches a tripwire should not fire it. Only an
 * unambiguous crossing counts.
 */
export const segmentsIntersect = (p1: Vec2, p2: Vec2, q1: Vec2, q2: Vec2): boolean => {
  const o1 = orientation(p1, p2, q1);
  const o2 = orientation(p1, p2, q2);
  const o3 = orientation(q1, q2, p1);
  const o4 = orientation(q1, q2, p2);
  return o1 !== o2 && o3 !== o4 && o1 !== 0 && o2 !== 0 && o3 !== 0 && o4 !== 0;
};

/**
 * Which side of the directed line a->b the point falls on.
 * +1 left, -1 right, 0 exactly on the line.
 */
export const sideOfLine = (a: Vec2, b: Vec2, point: Vec2): -1 | 0 | 1 => {
  const v = (b.x - a.x) * (point.y - a.y) - (b.y - a.y) * (point.x - a.x);
  if (v > 0) return 1;
  if (v < 0) return -1;
  return 0;
};

/** Axis-aligned rectangle as a ring, given opposite corners. */
export const rectangleRing = (a: Vec2, b: Vec2): readonly Vec2[] => {
  const minX = Math.min(a.x, b.x);
  const maxX = Math.max(a.x, b.x);
  const minY = Math.min(a.y, b.y);
  const maxY = Math.max(a.y, b.y);
  return [
    { x: minX, y: minY },
    { x: maxX, y: minY },
    { x: maxX, y: maxY },
    { x: minX, y: maxY },
  ];
};

/** Approximate a circle as a ring, for rendering and for uniform ring handling. */
export const circleRing = (center: Vec2, radius: number, segments = 32): readonly Vec2[] => {
  const points: Vec2[] = [];
  const count = Math.max(3, Math.trunc(segments));
  for (let i = 0; i < count; i += 1) {
    const angle = (i / count) * Math.PI * 2;
    points.push({ x: center.x + radius * Math.cos(angle), y: center.y + radius * Math.sin(angle) });
  }
  return points;
};
