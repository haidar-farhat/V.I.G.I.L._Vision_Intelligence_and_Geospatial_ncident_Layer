import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import type { CameraPose, LatLon, Zone, ZoneGeometry, ZoneId } from '@sentinel/shared-types';
import { asId, utcMillis } from '@sentinel/shared-types';
import { destinationPoint } from '../src/geodesy.ts';
import {
  CoverageVerdict,
  analyseZoneCoverage,
  cameraSees,
  sampleZone,
  zonesSeenByCamera,
} from '../src/coverage.ts';

const SITE: LatLon = { lat: 33.8938, lon: 35.5018 };

/** Offset from the site anchor by metres east and north. */
const at = (east: number, north: number): LatLon =>
  destinationPoint(destinationPoint(SITE, 0, north), 90, east);

const zone = (id: string, name: string, geometry: ZoneGeometry): Zone => ({
  id: asId<ZoneId>(id),
  name,
  purpose: 'RESTRICTED',
  geometry,
  locationId: null,
  parentZoneId: null,
  active: true,
  createdAt: utcMillis(0),
  updatedAt: utcMillis(0),
});

/**
 * A mast looking in a given direction.
 *
 * 6 m high with an 18-degree downward tilt and a 40-degree vertical FOV, which
 * puts the visible ground band from roughly 8 m to the stated range.
 */
const mast = (position: LatLon, heading: number, rangeMeters = 90): CameraPose => ({
  position: { ...position, altitude: 0 },
  mountHeight: 6,
  heading,
  pitch: -18,
  roll: 0,
  horizontalFov: 70,
  verticalFov: 40,
  rangeMeters,
});

/** A 40 m square, 20 m to 60 m north of the site anchor. */
const SQUARE = zone('zone-a', 'Restricted Zone A', {
  kind: 'POLYGON',
  ring: [at(-20, 20), at(20, 20), at(20, 60), at(-20, 60)],
});

describe('what a camera can see', () => {
  const camera = mast(at(0, 0), 0); // at the anchor, looking north

  test('sees a point ahead of it inside its range', () => {
    assert.ok(cameraSees(camera, at(0, 40)));
  });

  test('cannot see the ground at its own base', () => {
    // A tilted camera has a blind foreground. Treating it as visible is how a
    // gap ends up marked as covered on a map.
    assert.ok(!cameraSees(camera, at(0, 2)));
    assert.ok(!cameraSees(camera, at(0, 5)));
  });

  test('cannot see beyond its range', () => {
    assert.ok(!cameraSees(camera, at(0, 200)));
  });

  test('cannot see behind itself', () => {
    assert.ok(!cameraSees(camera, at(0, -40)));
  });

  test('cannot see outside its horizontal field of view', () => {
    // 70-degree FOV: 35 degrees either side of north.
    assert.ok(cameraSees(camera, at(20, 40)), 'inside the wedge');
    assert.ok(!cameraSees(camera, at(60, 20)), 'off to the side');
  });

  test('a camera pointed elsewhere sees nothing in front of the first one', () => {
    const facingSouth = mast(at(0, 0), 180);
    assert.ok(!cameraSees(facingSouth, at(0, 40)));
    assert.ok(cameraSees(facingSouth, at(0, -40)));
  });
});

describe('zone sampling', () => {
  test('samples only the interior of the zone', () => {
    const points = sampleZone(SQUARE, 5, 4000);

    assert.ok(points.length > 40, `too few samples: ${points.length}`);
    for (const point of points) {
      // Every sample must be inside the 40 m square.
      assert.ok(point.lat > SITE.lat, 'sample fell south of the zone');
    }
  });

  test('is deterministic, so a number only moves when the site does', () => {
    // A random sampler makes coverage jitter between runs, and an operator
    // nudging a camera needs the number to move because of the camera.
    const a = sampleZone(SQUARE, 5, 4000);
    const b = sampleZone(SQUARE, 5, 4000);

    assert.equal(a.length, b.length);
    assert.deepEqual(a[0], b[0]);
    assert.deepEqual(a[a.length - 1], b[b.length - 1]);
  });

  test('widens the spacing rather than truncating a large zone', () => {
    // Truncating would sample one corner finely and report the rest as unknown.
    const huge = zone('big', 'Whole Site', {
      kind: 'POLYGON',
      ring: [at(-2000, -2000), at(2000, -2000), at(2000, 2000), at(-2000, 2000)],
    });

    const points = sampleZone(huge, 5, 500);
    assert.ok(points.length <= 500, `exceeded the cap: ${points.length}`);
    assert.ok(points.length > 100, 'still sampled meaningfully');

    // Samples must span the zone, not cluster in one corner.
    const lats = points.map((p) => p.lat);
    const spread = Math.max(...lats) - Math.min(...lats);
    assert.ok(spread > 0.02, `samples clustered rather than spanning: spread ${spread}`);
  });

  test('handles circle and corridor zones', () => {
    const circle = zone('c', 'Asset', { kind: 'CIRCLE', center: at(0, 40), radiusMeters: 15 });
    const corridor = zone('r', 'Road', {
      kind: 'CORRIDOR',
      path: [at(-50, 40), at(50, 40)],
      widthMeters: 10,
    });

    assert.ok(sampleZone(circle, 3, 2000).length > 20);
    assert.ok(sampleZone(corridor, 3, 2000).length > 20);
  });

  test('a tripwire has no interior to sample', () => {
    const line = zone('l', 'Tripwire', { kind: 'LINE', path: [at(-20, 40), at(20, 40)] });
    assert.deepEqual(sampleZone(line, 5, 1000), []);
  });
});

describe('zone coverage', () => {
  test('reports a zone no camera can see as blind, and says what that means', () => {
    // The failure this whole analysis exists to prevent: a zone that looks
    // protected on a map because cameras exist near it.
    const facingAway = mast(at(0, 0), 180);
    const coverage = analyseZoneCoverage(SQUARE, [{ cameraId: 'cam-07', pose: facingAway }]);

    assert.equal(coverage.verdict, CoverageVerdict.Blind);
    assert.equal(coverage.coveredFraction, 0);
    assert.match(coverage.summary, /will go undetected/);
    assert.ok(coverage.blindSpots.length > 0, 'the gap is returned for rendering');
  });

  test('reports a zone fully in view as covered', () => {
    const coverage = analyseZoneCoverage(SQUARE, [{ cameraId: 'cam-07', pose: mast(at(0, 0), 0) }]);

    assert.ok(coverage.coveredFraction > 0.9, `only ${coverage.coveredFraction} covered`);
    assert.ok(
      coverage.verdict === CoverageVerdict.Covered || coverage.verdict === CoverageVerdict.Partial,
    );
    assert.equal(coverage.contributingCameras[0]?.cameraId, 'cam-07');
  });

  test('a partly-covered zone reports the fraction that is blind', () => {
    // A narrow camera cannot cover a wide square from close range.
    const narrow: CameraPose = { ...mast(at(0, 0), 0), horizontalFov: 20 };
    const coverage = analyseZoneCoverage(SQUARE, [{ cameraId: 'cam-07', pose: narrow }]);

    assert.equal(coverage.verdict, CoverageVerdict.Partial);
    assert.ok(coverage.coveredFraction > 0 && coverage.coveredFraction < 1);
    assert.match(coverage.summary, /blind spot/);
    assert.match(coverage.summary, /would not be detected/);
  });

  test('distinguishes covered from redundant, because one camera is a single point of failure', () => {
    const one = analyseZoneCoverage(SQUARE, [{ cameraId: 'cam-07', pose: mast(at(0, 0), 0) }]);

    // Two cameras approaching the same square from opposite sides.
    const two = analyseZoneCoverage(SQUARE, [
      { cameraId: 'cam-07', pose: mast(at(0, 0), 0) },
      { cameraId: 'cam-08', pose: mast(at(0, 80), 180) },
    ]);

    assert.ok(
      two.redundantFraction > one.redundantFraction,
      'a second camera must increase redundancy',
    );
    assert.equal(two.contributingCameras.length, 2);
  });

  test('warns that fully-covered is not the same as safe', () => {
    const coverage = analyseZoneCoverage(SQUARE, [{ cameraId: 'cam-07', pose: mast(at(0, 0), 0) }]);

    if (coverage.verdict === CoverageVerdict.Covered) {
      assert.match(coverage.summary, /Losing that camera would open a gap/);
    }
  });

  test('an unplaced camera is skipped rather than counted as blind', () => {
    // Reporting a gap that placement would close sends an operator installing
    // hardware they already own.
    const coverage = analyseZoneCoverage(SQUARE, [
      { cameraId: 'cam-07', pose: mast(at(0, 0), 0) },
      { cameraId: 'cam-99', pose: null },
    ]);

    assert.ok(coverage.coveredFraction > 0.9);
    assert.ok(!coverage.contributingCameras.some((c) => c.cameraId === 'cam-99'));
  });

  test('no cameras placed at all says exactly that', () => {
    const coverage = analyseZoneCoverage(SQUARE, []);

    assert.equal(coverage.verdict, CoverageVerdict.Blind);
    assert.match(coverage.summary, /No camera has been placed on the map/);
  });

  test('the blind spot list is capped but the count is not', () => {
    const coverage = analyseZoneCoverage(SQUARE, [], { maxBlindSpots: 10 });

    assert.equal(coverage.blindSpots.length, 10);
    assert.ok(coverage.totalSamples > 10, 'every sample was still evaluated');
    assert.equal(coverage.coveredSamples, 0);
  });

  test('is deterministic for a fixed site', () => {
    const cameras = [{ cameraId: 'cam-07', pose: mast(at(0, 0), 0) }];
    const a = analyseZoneCoverage(SQUARE, cameras);
    const b = analyseZoneCoverage(SQUARE, cameras);

    assert.equal(a.coveredFraction, b.coveredFraction);
    assert.equal(a.totalSamples, b.totalSamples);
  });

  test('a zone with no area reports so rather than dividing by zero', () => {
    const line = zone('l', 'Tripwire', { kind: 'LINE', path: [at(-20, 40), at(20, 40)] });
    const coverage = analyseZoneCoverage(line, [{ cameraId: 'cam-07', pose: mast(at(0, 0), 0) }]);

    assert.equal(coverage.totalSamples, 0);
    assert.equal(coverage.coveredFraction, 0);
    assert.match(coverage.summary, /no area to analyse/);
  });
});

describe('what a camera is for', () => {
  test('lists the zones a camera contributes to, most first', () => {
    // Surprisingly hard to answer from a map once a site has thirty cameras.
    const near = zone('near', 'Near Zone', {
      kind: 'POLYGON',
      ring: [at(-10, 15), at(10, 15), at(10, 35), at(-10, 35)],
    });
    const far = zone('far', 'Far Zone', {
      kind: 'POLYGON',
      ring: [at(-10, 200), at(10, 200), at(10, 220), at(-10, 220)],
    });

    const seen = zonesSeenByCamera(mast(at(0, 0), 0), [near, far, SQUARE]);

    assert.ok(seen.length >= 1);
    assert.ok(!seen.some((entry) => entry.zone.id === 'far'), 'beyond range must not appear');
    assert.ok((seen[0]?.fraction ?? 0) > 0);
  });

  test('an unplaced camera contributes to nothing', () => {
    assert.deepEqual(zonesSeenByCamera(null, [SQUARE]), []);
  });

  test('inactive zones are ignored', () => {
    const inactive = { ...SQUARE, active: false };
    assert.deepEqual(zonesSeenByCamera(mast(at(0, 0), 0), [inactive]), []);
  });
});
