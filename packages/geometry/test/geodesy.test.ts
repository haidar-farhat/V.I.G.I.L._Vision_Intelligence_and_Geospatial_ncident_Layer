import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import type { LatLon } from '@sentinel/shared-types';
import {
  bearingDegrees,
  boundsContain,
  boundsOf,
  createLocalFrame,
  destinationPoint,
  haversineDistance,
  toLatLon,
  toLocal,
} from '../src/geodesy.ts';

/** A fixed site anchor used across the geometry suite. */
const SITE: LatLon = { lat: 33.8938, lon: 35.5018 };

const closeTo = (actual: number, expected: number, tolerance: number, label: string): void => {
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    `${label}: expected ${expected} +/- ${tolerance}, got ${actual}`,
  );
};

describe('local frame', () => {
  test('round-trips a geographic point through local metres', () => {
    const frame = createLocalFrame(SITE);
    const point: LatLon = { lat: SITE.lat + 0.0012, lon: SITE.lon - 0.0008 };

    const local = toLocal(frame, point);
    const back = toLatLon(frame, local);

    closeTo(back.lat, point.lat, 1e-12, 'latitude round-trip');
    closeTo(back.lon, point.lon, 1e-12, 'longitude round-trip');
  });

  test('origin maps to the local origin', () => {
    const frame = createLocalFrame(SITE);
    const local = toLocal(frame, SITE);
    closeTo(local.x, 0, 1e-9, 'x');
    closeTo(local.y, 0, 1e-9, 'y');
  });

  test('local axes are east-positive and north-positive', () => {
    const frame = createLocalFrame(SITE);
    const east = toLocal(frame, { lat: SITE.lat, lon: SITE.lon + 0.001 });
    const north = toLocal(frame, { lat: SITE.lat + 0.001, lon: SITE.lon });

    assert.ok(east.x > 0, 'increasing longitude must increase x');
    closeTo(east.y, 0, 1e-9, 'pure east movement has no y');
    assert.ok(north.y > 0, 'increasing latitude must increase y');
    closeTo(north.x, 0, 1e-9, 'pure north movement has no x');
  });

  test('local distances agree with haversine over site scale', () => {
    const frame = createLocalFrame(SITE);
    const target = destinationPoint(SITE, 47, 250);

    const local = toLocal(frame, target);
    const planar = Math.hypot(local.x, local.y);
    const geodesic = haversineDistance(SITE, target);

    // Under a centimetre of disagreement at 250 m: the planar approximation is
    // far more accurate than the metre-scale position uncertainty it carries.
    closeTo(planar, geodesic, 0.01, 'planar vs geodesic distance');
  });
});

describe('haversine and bearing', () => {
  test('distance to self is zero', () => {
    closeTo(haversineDistance(SITE, SITE), 0, 1e-9, 'self distance');
  });

  test('destination point lands at the requested distance and bearing', () => {
    for (const bearing of [0, 45, 90, 180, 271, 359]) {
      const target = destinationPoint(SITE, bearing, 500);
      closeTo(haversineDistance(SITE, target), 500, 0.001, `distance on bearing ${bearing}`);
      closeTo(bearingDegrees(SITE, target), bearing, 0.001, `bearing ${bearing}`);
    }
  });

  test('cardinal bearings point the expected way', () => {
    const north = destinationPoint(SITE, 0, 100);
    const east = destinationPoint(SITE, 90, 100);
    assert.ok(north.lat > SITE.lat, 'north increases latitude');
    closeTo(north.lon, SITE.lon, 1e-9, 'north does not change longitude');
    assert.ok(east.lon > SITE.lon, 'east increases longitude');
  });
});

describe('bounds', () => {
  test('computes bounds and containment', () => {
    const points: LatLon[] = [
      { lat: 1, lon: 2 },
      { lat: -3, lon: 5 },
      { lat: 4, lon: -1 },
    ];
    const bounds = boundsOf(points);
    assert.notEqual(bounds, null);
    assert.deepEqual(bounds, { minLat: -3, minLon: -1, maxLat: 4, maxLon: 5 });
    assert.ok(boundsContain(bounds!, { lat: 0, lon: 0 }));
    assert.ok(!boundsContain(bounds!, { lat: 10, lon: 0 }));
  });

  test('empty input yields no bounds rather than a degenerate box', () => {
    assert.equal(boundsOf([]), null);
  });
});
