import type { GeoBounds, LatLon, Vec2 } from '@sentinel/shared-types';
import { DEG_TO_RAD, RAD_TO_DEG, normalizeDegrees, toRadians } from './vec.ts';

/**
 * Geodesy for site-scale work.
 *
 * A security site spans hundreds of metres, not thousands of kilometres, so all
 * zone and track mathematics happens in a **local tangent plane** anchored at a
 * site origin: metres east (x) and metres north (y). Planar geometry there is
 * exact enough to be indistinguishable from ellipsoidal results at this scale,
 * and it keeps the polygon and intersection code free of spherical trigonometry.
 *
 * Distances and bearings between arbitrary points still use haversine on a sphere,
 * because those are used for camera topology edges that may span a whole site.
 */

/** WGS84 mean radius, metres. */
export const EARTH_RADIUS_M = 6_371_008.8;

/**
 * Metres per degree of latitude at a given latitude.
 * Series expansion of the WGS84 meridian arc; sub-centimetre over a site.
 */
export const metersPerDegreeLatitude = (latitudeDeg: number): number => {
  const lat = toRadians(latitudeDeg);
  return (
    111_132.92 -
    559.82 * Math.cos(2 * lat) +
    1.175 * Math.cos(4 * lat) -
    0.0023 * Math.cos(6 * lat)
  );
};

/** Metres per degree of longitude at a given latitude. */
export const metersPerDegreeLongitude = (latitudeDeg: number): number => {
  const lat = toRadians(latitudeDeg);
  return 111_412.84 * Math.cos(lat) - 93.5 * Math.cos(3 * lat) + 0.118 * Math.cos(5 * lat);
};

/**
 * A local east-north-up frame anchored at `origin`.
 *
 * Created once per site (or per zone cluster) and reused: the conversion factors
 * are latitude-dependent, and recomputing them per point would be both slower and
 * subtly inconsistent between a polygon and the point being tested against it.
 */
export type LocalFrame = {
  readonly origin: LatLon;
  readonly metersPerLat: number;
  readonly metersPerLon: number;
};

export const createLocalFrame = (origin: LatLon): LocalFrame => ({
  origin,
  metersPerLat: metersPerDegreeLatitude(origin.lat),
  metersPerLon: metersPerDegreeLongitude(origin.lat),
});

/** Geographic -> local metres (x = east, y = north). */
export const toLocal = (frame: LocalFrame, point: LatLon): Vec2 => ({
  x: (point.lon - frame.origin.lon) * frame.metersPerLon,
  y: (point.lat - frame.origin.lat) * frame.metersPerLat,
});

/** Local metres -> geographic. Exact inverse of `toLocal`. */
export const toLatLon = (frame: LocalFrame, local: Vec2): LatLon => ({
  lat: frame.origin.lat + local.y / frame.metersPerLat,
  lon: frame.origin.lon + local.x / frame.metersPerLon,
});

export const toLocalRing = (frame: LocalFrame, ring: readonly LatLon[]): readonly Vec2[] =>
  ring.map((p) => toLocal(frame, p));

/** Great-circle distance in metres. */
export const haversineDistance = (a: LatLon, b: LatLon): number => {
  const lat1 = a.lat * DEG_TO_RAD;
  const lat2 = b.lat * DEG_TO_RAD;
  const dLat = lat2 - lat1;
  const dLon = (b.lon - a.lon) * DEG_TO_RAD;
  const sinLat = Math.sin(dLat / 2);
  const sinLon = Math.sin(dLon / 2);
  const h = sinLat * sinLat + Math.cos(lat1) * Math.cos(lat2) * sinLon * sinLon;
  return 2 * EARTH_RADIUS_M * Math.asin(Math.min(1, Math.sqrt(h)));
};

/** Initial compass bearing from `a` to `b`, degrees, 0 = north. */
export const bearingDegrees = (a: LatLon, b: LatLon): number => {
  const lat1 = a.lat * DEG_TO_RAD;
  const lat2 = b.lat * DEG_TO_RAD;
  const dLon = (b.lon - a.lon) * DEG_TO_RAD;
  const y = Math.sin(dLon) * Math.cos(lat2);
  const x = Math.cos(lat1) * Math.sin(lat2) - Math.sin(lat1) * Math.cos(lat2) * Math.cos(dLon);
  return normalizeDegrees(Math.atan2(y, x) * RAD_TO_DEG);
};

/** The point reached by travelling `distanceMeters` from `origin` on `bearingDeg`. */
export const destinationPoint = (
  origin: LatLon,
  bearingDeg: number,
  distanceMeters: number,
): LatLon => {
  const angular = distanceMeters / EARTH_RADIUS_M;
  const bearing = bearingDeg * DEG_TO_RAD;
  const lat1 = origin.lat * DEG_TO_RAD;
  const lon1 = origin.lon * DEG_TO_RAD;

  const sinLat2 =
    Math.sin(lat1) * Math.cos(angular) + Math.cos(lat1) * Math.sin(angular) * Math.cos(bearing);
  const lat2 = Math.asin(Math.min(1, Math.max(-1, sinLat2)));
  const lon2 =
    lon1 +
    Math.atan2(
      Math.sin(bearing) * Math.sin(angular) * Math.cos(lat1),
      Math.cos(angular) - Math.sin(lat1) * sinLat2,
    );

  return {
    lat: lat2 * RAD_TO_DEG,
    // Normalise longitude into [-180, 180).
    lon: (((lon2 * RAD_TO_DEG + 540) % 360) - 180),
  };
};

export const boundsOf = (points: readonly LatLon[]): GeoBounds | null => {
  const first = points[0];
  if (first === undefined) return null;

  let minLat = first.lat;
  let maxLat = first.lat;
  let minLon = first.lon;
  let maxLon = first.lon;

  for (let i = 1; i < points.length; i += 1) {
    const p = points[i];
    if (p === undefined) continue;
    if (p.lat < minLat) minLat = p.lat;
    if (p.lat > maxLat) maxLat = p.lat;
    if (p.lon < minLon) minLon = p.lon;
    if (p.lon > maxLon) maxLon = p.lon;
  }
  return { minLat, minLon, maxLat, maxLon };
};

export const boundsContain = (bounds: GeoBounds, point: LatLon): boolean =>
  point.lat >= bounds.minLat &&
  point.lat <= bounds.maxLat &&
  point.lon >= bounds.minLon &&
  point.lon <= bounds.maxLon;
