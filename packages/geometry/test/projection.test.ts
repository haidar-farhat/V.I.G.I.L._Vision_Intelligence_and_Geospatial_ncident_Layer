import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import type { BoundingBox, CameraPose, LatLon } from '@sentinel/shared-types';
import { PositionSource } from '@sentinel/shared-types';
import {
  bearingInFov,
  farGroundDistance,
  fieldOfViewWedge,
  groundContactPoint,
  nearGroundDistance,
  projectDetection,
  projectToGround,
  rayAngles,
} from '../src/projection.ts';
import { bearingDegrees, haversineDistance } from '../src/geodesy.ts';
import { angleDifference } from '../src/vec.ts';

const SITE: LatLon = { lat: 33.8938, lon: 35.5018 };

/**
 * A 10 m mast looking due north, tilted 45 degrees down. At that tilt the centre
 * of the image lands exactly one mount-height away, which makes every expected
 * value in these tests checkable by hand.
 */
const POSE: CameraPose = {
  position: { ...SITE, altitude: 0 },
  mountHeight: 10,
  heading: 0,
  pitch: -45,
  roll: 0,
  horizontalFov: 60,
  verticalFov: 34,
  rangeMeters: 120,
};

const closeTo = (actual: number, expected: number, tolerance: number, label: string): void => {
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    `${label}: expected ${expected} +/- ${tolerance}, got ${actual}`,
  );
};

describe('ray angles', () => {
  test('image centre follows the camera heading and pitch exactly', () => {
    const { bearingDeg, elevationDeg } = rayAngles(POSE, 0.5, 0.5);
    closeTo(bearingDeg, 0, 1e-9, 'centre bearing');
    closeTo(elevationDeg, -45, 1e-9, 'centre elevation');
  });

  test('image edges span the stated field of view', () => {
    const left = rayAngles(POSE, 0, 0.5);
    const right = rayAngles(POSE, 1, 0.5);
    // Bearings wrap, so compare the left edge as a negative offset.
    closeTo(left.bearingDeg - 360, -30, 1e-6, 'left edge bearing');
    closeTo(right.bearingDeg, 30, 1e-6, 'right edge bearing');
  });

  test('bottom of frame looks further down than the top', () => {
    const top = rayAngles(POSE, 0.5, 0);
    const bottom = rayAngles(POSE, 0.5, 1);
    assert.ok(bottom.elevationDeg < top.elevationDeg, 'bottom must be steeper');
    closeTo(top.elevationDeg, -45 + 17, 1e-6, 'top edge elevation');
    closeTo(bottom.elevationDeg, -45 - 17, 1e-6, 'bottom edge elevation');
  });
});

describe('ground projection', () => {
  test('45 degree depression lands at exactly one mount height', () => {
    const projection = projectToGround(POSE, 0.5, 0.5);
    assert.notEqual(projection, null);
    closeTo(projection!.groundDistanceMeters, 10, 1e-9, 'ground distance');
    closeTo(projection!.bearingDeg, 0, 1e-9, 'bearing');
    closeTo(haversineDistance(POSE.position, projection!.position), 10, 0.001, 'geodesic distance');
    closeTo(bearingDegrees(POSE.position, projection!.position), 0, 0.001, 'geodesic bearing');
  });

  test('a ray at or above the horizon has no ground intersection', () => {
    const levelCamera: CameraPose = { ...POSE, pitch: 0 };
    assert.equal(projectToGround(levelCamera, 0.5, 0.5), null);

    const upward: CameraPose = { ...POSE, pitch: 10 };
    assert.equal(projectToGround(upward, 0.5, 0.5), null);
  });

  test('projections beyond the pose range are rejected, not clamped', () => {
    // A very shallow ray lands far beyond the 120 m stated range.
    const shallow: CameraPose = { ...POSE, pitch: -3 };
    assert.equal(projectToGround(shallow, 0.5, 0.5), null, 'must reject out-of-range');

    const unbounded = projectToGround(shallow, 0.5, 0.5, { enforceRange: false });
    assert.notEqual(unbounded, null, 'range enforcement is opt-out for FOV rendering');
    assert.ok(unbounded!.groundDistanceMeters > POSE.rangeMeters);
  });

  test('uncertainty grows sharply as the ray flattens toward the horizon', () => {
    const near = projectToGround({ ...POSE, pitch: -60 }, 0.5, 0.5);
    const mid = projectToGround({ ...POSE, pitch: -30 }, 0.5, 0.5);
    const far = projectToGround({ ...POSE, pitch: -10 }, 0.5, 0.5);

    assert.notEqual(near, null);
    assert.notEqual(mid, null);
    assert.notEqual(far, null);

    assert.ok(
      near!.uncertaintyMeters < mid!.uncertaintyMeters,
      'a steeper view must be more certain',
    );
    assert.ok(
      mid!.uncertaintyMeters < far!.uncertaintyMeters,
      'a flatter view must be less certain',
    );

    // The whole point of modelling this: error is super-linear in distance, so a
    // detection near the horizon must never be shown with the same confidence as
    // one at the camera's feet.
    const nearRatio = near!.uncertaintyMeters / near!.groundDistanceMeters;
    const farRatio = far!.uncertaintyMeters / far!.groundDistanceMeters;
    assert.ok(farRatio > nearRatio * 2, 'relative error must worsen with distance');
  });

  test('uncertainty scales with the stated angular uncertainty', () => {
    const tight = projectToGround(POSE, 0.5, 0.5, { angularUncertaintyDeg: 0.5 });
    const loose = projectToGround(POSE, 0.5, 0.5, { angularUncertaintyDeg: 2 });
    assert.ok(tight!.uncertaintyMeters < loose!.uncertaintyMeters);
    closeTo(loose!.uncertaintyMeters / tight!.uncertaintyMeters, 4, 1e-6, 'linear in sigma');
  });
});

describe('detection projection', () => {
  const box: BoundingBox = { x: 0.45, y: 0.4, w: 0.1, h: 0.2 };

  test('uses the bottom-centre of the box as the ground contact point', () => {
    const contact = groundContactPoint(box);
    closeTo(contact.x, 0.5, 1e-9, 'horizontal centre');
    closeTo(contact.y, 0.6, 1e-9, 'bottom edge');
  });

  test('produces a ground projection for a usable pose', () => {
    const estimate = projectDetection(POSE, box);
    assert.notEqual(estimate, null);
    assert.equal(estimate!.source, PositionSource.GroundProjection);
    assert.ok(estimate!.radiusMeters > 0, 'uncertainty is always stated');
  });

  test('falls back to the camera position rather than inventing one', () => {
    const levelCamera: CameraPose = { ...POSE, pitch: 5 };
    const estimate = projectDetection(levelCamera, box);

    assert.notEqual(estimate, null);
    assert.equal(estimate!.source, PositionSource.CameraFallback);
    assert.deepEqual(estimate!.point, levelCamera.position);
    // The uncertainty must cover the whole field of view, not imply precision.
    assert.equal(estimate!.radiusMeters, levelCamera.rangeMeters);
  });

  test('an unplaced camera yields no position at all', () => {
    assert.equal(projectDetection(null, box), null);
  });
});

describe('field of view footprint', () => {
  test('near edge is closer than the far edge', () => {
    const near = nearGroundDistance(POSE);
    const far = farGroundDistance(POSE);
    assert.notEqual(far, null);
    assert.ok(near > 0, 'a downward camera has a blind foreground');
    assert.ok(near < far!, 'the bottom of the frame is nearer than the top');
  });

  test('a camera whose top edge is above the horizon sees to infinity', () => {
    const shallow: CameraPose = { ...POSE, pitch: -5, verticalFov: 34 };
    assert.equal(farGroundDistance(shallow), null);
  });

  test('wedge is a closed annular sector inside the stated range', () => {
    const wedge = fieldOfViewWedge(POSE, 8);
    assert.ok(wedge.length > 8, 'both arcs are present');

    for (const point of wedge) {
      const distance = haversineDistance(POSE.position, point);
      assert.ok(
        distance <= POSE.rangeMeters + 0.5,
        `wedge point at ${distance.toFixed(1)} m exceeds the stated range`,
      );
    }

    // Every wedge point must lie within the horizontal field of view. The arc
    // endpoints sit exactly on the FOV edge, so allow a hair of tolerance for the
    // round-trip through geodesic coordinates.
    const edgeToleranceDeg = 1e-6;
    for (const point of wedge) {
      if (haversineDistance(POSE.position, point) < 0.01) continue;
      const offset = Math.abs(angleDifference(bearingDegrees(POSE.position, point), POSE.heading));
      assert.ok(
        offset <= POSE.horizontalFov / 2 + edgeToleranceDeg,
        `wedge point ${offset.toFixed(6)} deg off-axis exceeds the half-FOV`,
      );
    }
  });

  test('bearings are classified against the field of view', () => {
    assert.ok(bearingInFov(POSE, 0), 'straight ahead is in view');
    assert.ok(bearingInFov(POSE, 25), 'inside the right half');
    assert.ok(bearingInFov(POSE, 335), 'inside the left half, across the wrap');
    assert.ok(!bearingInFov(POSE, 45), 'outside the right half');
    assert.ok(!bearingInFov(POSE, 180), 'behind the camera');
  });

  test('the blind foreground is excluded from the footprint', () => {
    const wedge = fieldOfViewWedge(POSE, 8);
    const near = nearGroundDistance(POSE);
    const closest = Math.min(...wedge.map((p) => haversineDistance(POSE.position, p)));
    closeTo(closest, near, 0.5, 'closest footprint point equals the near ground distance');
  });
});
