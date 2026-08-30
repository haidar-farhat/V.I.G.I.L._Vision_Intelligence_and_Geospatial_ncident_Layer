import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import type {
  BoundingBox,
  CameraId,
  CameraPose,
  Detection,
  ModelId,
  UtcMillis,
} from '@sentinel/shared-types';
import { asId, utcMillis } from '@sentinel/shared-types';
import { Tracker, boxCenter, iou } from '../src/tracker.ts';

const CAMERA = asId<CameraId>('cam-07');
const MODEL = asId<ModelId>('model-test');

const detection = (
  box: BoundingBox,
  at: number,
  objectClass = 'person',
  confidence = 0.9,
): Detection => ({
  cameraId: CAMERA,
  frameAt: utcMillis(at),
  objectClass,
  confidence,
  box,
  modelId: MODEL,
});

/** A box of fixed size whose left edge walks across the frame. */
const walking = (x: number): BoundingBox => ({ x, y: 0.5, w: 0.1, h: 0.3 });

const POSE: CameraPose = {
  position: { lat: 33.8938, lon: 35.5018, altitude: 0 },
  mountHeight: 10,
  heading: 0,
  pitch: -45,
  roll: 0,
  horizontalFov: 60,
  verticalFov: 34,
  rangeMeters: 200,
};

/** Floating-point comparison; normalised box maths never lands on exact decimals. */
const closeTo = (actual: number, expected: number, tolerance = 1e-9): void => {
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    `expected ${expected} +/- ${tolerance}, got ${actual}`,
  );
};

describe('iou', () => {
  test('identical boxes overlap completely', () => {
    closeTo(iou(walking(0.2), walking(0.2)), 1);
  });

  test('disjoint boxes do not overlap', () => {
    assert.equal(iou(walking(0.0), walking(0.9)), 0);
  });

  test('partial overlap is between zero and one', () => {
    const overlap = iou(walking(0.2), walking(0.25));
    assert.ok(overlap > 0 && overlap < 1, `expected partial overlap, got ${overlap}`);
  });

  test('box centre is the geometric middle', () => {
    const centre = boxCenter({ x: 0.2, y: 0.4, w: 0.2, h: 0.2 });
    closeTo(centre.x, 0.3);
    closeTo(centre.y, 0.5);
  });
});

describe('track lifecycle', () => {
  test('a track is not reported until it is confirmed', () => {
    const tracker = new Tracker(CAMERA, { minHitsToConfirm: 2 });

    const first = tracker.update([detection(walking(0.2), 0)], utcMillis(0));
    assert.equal(first.tracks.length, 0, 'a single detection is not yet a track');

    const second = tracker.update([detection(walking(0.22), 200)], utcMillis(200));
    assert.equal(second.tracks.length, 1, 'confirmed on the second hit');
  });

  test('a moving object keeps one stable identity', () => {
    const tracker = new Tracker(CAMERA, { minHitsToConfirm: 2 });
    const ids = new Set<string>();

    for (let step = 0; step < 20; step += 1) {
      const at = utcMillis(step * 200);
      const update = tracker.update([detection(walking(0.05 + step * 0.03), step * 200)], at);
      for (const track of update.tracks) ids.add(track.id);
    }

    assert.equal(ids.size, 1, 'a single walk must not fragment into multiple tracks');
    assert.equal(tracker.activeTrackCount, 1);
  });

  test('two objects get two tracks', () => {
    const tracker = new Tracker(CAMERA, { minHitsToConfirm: 1 });

    for (let step = 0; step < 5; step += 1) {
      const at = step * 200;
      tracker.update(
        [
          detection({ x: 0.1 + step * 0.02, y: 0.5, w: 0.08, h: 0.25 }, at),
          detection({ x: 0.7 - step * 0.02, y: 0.5, w: 0.08, h: 0.25 }, at),
        ],
        utcMillis(at),
      );
    }

    assert.equal(tracker.activeTrackCount, 2);
  });

  test('different object classes never merge', () => {
    const tracker = new Tracker(CAMERA, { minHitsToConfirm: 1 });

    // Same position, different class: must be two distinct tracks.
    tracker.update([detection(walking(0.3), 0, 'person')], utcMillis(0));
    tracker.update([detection(walking(0.3), 200, 'car')], utcMillis(200));

    assert.equal(tracker.activeTrackCount, 2, 'a person track must not absorb a car detection');
  });
});

describe('gap tolerance', () => {
  test('a track survives an occlusion and keeps its identity', () => {
    const tracker = new Tracker(CAMERA, { minHitsToConfirm: 2, maxGapMillis: 2000 });

    // Establish a track moving steadily right.
    let id: string | undefined;
    for (let step = 0; step < 4; step += 1) {
      const at = step * 200;
      const update = tracker.update([detection(walking(0.1 + step * 0.05), at)], utcMillis(at));
      id = update.tracks[0]?.id ?? id;
    }
    assert.ok(id !== undefined, 'track established');

    // Occluded for three frames: no detections at all.
    for (let step = 4; step < 7; step += 1) {
      const at = step * 200;
      const update = tracker.update([], utcMillis(at));
      assert.equal(update.ended.length, 0, 'must not close the track during a short gap');
      assert.equal(update.observations[0]?.interpolated, true, 'coasted positions are flagged');
    }

    // Reacquired where the prediction says it should be.
    const at = 7 * 200;
    const update = tracker.update([detection(walking(0.1 + 7 * 0.05), at)], utcMillis(at));

    assert.equal(update.tracks.length, 1);
    assert.equal(update.tracks[0]?.id, id, 'the same object must keep the same track id');
  });

  test('a track is closed once the gap exceeds the limit', () => {
    const tracker = new Tracker(CAMERA, { minHitsToConfirm: 1, maxGapMillis: 1000 });
    tracker.update([detection(walking(0.2), 0)], utcMillis(0));

    const stillAlive = tracker.update([], utcMillis(900));
    assert.equal(stillAlive.ended.length, 0);

    const expired = tracker.update([], utcMillis(1500));
    assert.equal(expired.ended.length, 1, 'the track is closed after the gap limit');
    assert.equal(tracker.activeTrackCount, 0);
  });

  test('coasted observations are marked so nothing mistakes them for evidence', () => {
    const tracker = new Tracker(CAMERA, { minHitsToConfirm: 1, maxGapMillis: 2000 });

    const real = tracker.update([detection(walking(0.2), 0)], utcMillis(0));
    assert.equal(real.observations[0]?.interpolated, false);

    const coasted = tracker.update([], utcMillis(300));
    assert.equal(coasted.observations[0]?.interpolated, true);
  });
});

describe('determinism', () => {
  test('the same input always produces the same tracks', () => {
    const run = (): string => {
      const tracker = new Tracker(CAMERA, { minHitsToConfirm: 2 });
      const output: string[] = [];

      for (let step = 0; step < 12; step += 1) {
        const at = step * 200;
        const update = tracker.update(
          [
            detection({ x: 0.1 + step * 0.03, y: 0.5, w: 0.08, h: 0.25 }, at),
            detection({ x: 0.6 - step * 0.02, y: 0.4, w: 0.08, h: 0.25 }, at),
          ],
          utcMillis(at),
        );
        for (const track of update.tracks) {
          output.push(`${track.id}@${track.currentBox.x.toFixed(4)}`);
        }
      }
      return output.join('|');
    };

    assert.equal(run(), run(), 'tracking must be reproducible for the same scenario');
  });
});

describe('ground motion', () => {
  test('speed and heading stay null without a camera pose', () => {
    const tracker = new Tracker(CAMERA, { minHitsToConfirm: 1 });
    const update = tracker.update([detection(walking(0.2), 0)], utcMillis(0));

    assert.equal(update.tracks[0]?.speedMps, null);
    assert.equal(update.tracks[0]?.headingDegrees, null);
    assert.equal(update.tracks[0]?.currentPosition, null, 'no pose means no map position');
  });

  test('a posed camera produces a ground position and a heading', () => {
    const tracker = new Tracker(CAMERA, { minHitsToConfirm: 1, pose: POSE });

    // Walk down the frame, which moves the object toward the camera on the ground.
    let last: UtcMillis = utcMillis(0);
    for (let step = 0; step < 12; step += 1) {
      const at = utcMillis(step * 500);
      last = at;
      tracker.update(
        [detection({ x: 0.45, y: 0.2 + step * 0.05, w: 0.08, h: 0.15 }, at)],
        at,
      );
    }

    const tracks = tracker.tracks();
    const track = tracks[0];
    assert.ok(track !== undefined, 'track exists');
    assert.notEqual(track.currentPosition, null, 'a posed camera yields a map position');
    assert.notEqual(track.speedMps, null, 'ground speed is derived once it has moved');
    assert.notEqual(track.headingDegrees, null, 'heading is derived once it has moved');
    assert.ok((track.speedMps ?? 0) > 0, 'a walking object has non-zero speed');
    assert.ok(last > utcMillis(0));
  });

  test('a stationary object reports zero speed and no heading', () => {
    const tracker = new Tracker(CAMERA, { minHitsToConfirm: 1, pose: POSE });

    for (let step = 0; step < 10; step += 1) {
      const at = utcMillis(step * 500);
      tracker.update([detection({ x: 0.45, y: 0.6, w: 0.08, h: 0.15 }, at)], at);
    }

    const track = tracker.tracks()[0];
    assert.equal(track?.speedMps, 0, 'standing still is zero, not noise');
    assert.equal(
      track?.headingDegrees,
      null,
      'a heading derived from jitter is worse than no heading',
    );
  });
});

describe('reset', () => {
  test('reset closes every live track', () => {
    const tracker = new Tracker(CAMERA, { minHitsToConfirm: 1 });
    tracker.update(
      [detection(walking(0.2), 0), detection({ x: 0.7, y: 0.5, w: 0.1, h: 0.3 }, 0)],
      utcMillis(0),
    );

    const ended = tracker.reset();
    assert.equal(ended.length, 2);
    assert.equal(tracker.activeTrackCount, 0);
  });
});
