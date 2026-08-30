import type { Vec2 } from '@sentinel/shared-types';

/** Planar vector helpers. All pure, all allocation-light, all deterministic. */

export const vec = (x: number, y: number): Vec2 => ({ x, y });

export const add = (a: Vec2, b: Vec2): Vec2 => ({ x: a.x + b.x, y: a.y + b.y });
export const sub = (a: Vec2, b: Vec2): Vec2 => ({ x: a.x - b.x, y: a.y - b.y });
export const scale = (a: Vec2, k: number): Vec2 => ({ x: a.x * k, y: a.y * k });

export const dot = (a: Vec2, b: Vec2): number => a.x * b.x + a.y * b.y;
/** 2D cross product (z-component). Sign gives the turn direction. */
export const cross = (a: Vec2, b: Vec2): number => a.x * b.y - a.y * b.x;

export const lengthSq = (a: Vec2): number => a.x * a.x + a.y * a.y;
export const length = (a: Vec2): number => Math.hypot(a.x, a.y);

export const distance = (a: Vec2, b: Vec2): number => Math.hypot(b.x - a.x, b.y - a.y);
export const distanceSq = (a: Vec2, b: Vec2): number => {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  return dx * dx + dy * dy;
};

export const normalize = (a: Vec2): Vec2 => {
  const len = length(a);
  return len === 0 ? { x: 0, y: 0 } : { x: a.x / len, y: a.y / len };
};

export const lerp = (a: Vec2, b: Vec2, t: number): Vec2 => ({
  x: a.x + (b.x - a.x) * t,
  y: a.y + (b.y - a.y) * t,
});

export const DEG_TO_RAD = Math.PI / 180;
export const RAD_TO_DEG = 180 / Math.PI;

export const toRadians = (deg: number): number => deg * DEG_TO_RAD;
export const toDegrees = (rad: number): number => rad * RAD_TO_DEG;

/** Wrap any angle into [0, 360). */
export const normalizeDegrees = (deg: number): number => {
  const wrapped = deg % 360;
  return wrapped < 0 ? wrapped + 360 : wrapped;
};

/** Smallest signed difference a - b, in (-180, 180]. */
export const angleDifference = (a: number, b: number): number => {
  const diff = normalizeDegrees(a - b);
  return diff > 180 ? diff - 360 : diff;
};

export const clamp = (value: number, min: number, max: number): number =>
  value < min ? min : value > max ? max : value;

/**
 * Shortest distance from a point to a segment, plus where along the segment the
 * closest approach falls (`t` in 0..1). Used by corridor zones and by trajectory
 * simplification.
 */
export const pointToSegment = (
  p: Vec2,
  a: Vec2,
  b: Vec2,
): { readonly distance: number; readonly t: number; readonly closest: Vec2 } => {
  const ab = sub(b, a);
  const lenSq = lengthSq(ab);
  if (lenSq === 0) {
    return { distance: distance(p, a), t: 0, closest: a };
  }
  const t = clamp(dot(sub(p, a), ab) / lenSq, 0, 1);
  const closest = add(a, scale(ab, t));
  return { distance: distance(p, closest), t, closest };
};
