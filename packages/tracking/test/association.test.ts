import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import type { CameraId, CameraTopologyEdge, Track, TrackId } from '@sentinel/shared-types';
import { asId, utcMillis } from '@sentinel/shared-types';
import {
  appearanceSimilarity,
  bestAssociation,
  scoreAssociation,
  topologyIndex,
  travelTimePlausibility,
} from '../src/association.ts';

const CAM_07 = asId<CameraId>('cam-07');
const CAM_08 = asId<CameraId>('cam-08');
const CAM_99 = asId<CameraId>('cam-99');

/** Camera 07 connects to 08: 90 m apart, about 90 seconds on foot. */
const EDGE: CameraTopologyEdge = {
  fromCameraId: CAM_07,
  toCameraId: CAM_08,
  distanceMeters: 90,
  minTravelSeconds: 40,
  expectedTravelSeconds: 90,
  maxTravelSeconds: 240,
  confidence: 0.9,
  bidirectional: true,
};

const TOPOLOGY = topologyIndex([EDGE]);

const track = (
  id: string,
  cameraId: CameraId,
  firstSeen: number,
  lastSeen: number,
  overrides: Partial<Track> = {},
): Track => ({
  id: asId<TrackId>(id),
  cameraId,
  objectClass: 'person',
  firstSeen: utcMillis(firstSeen),
  lastSeen: utcMillis(lastSeen),
  confidence: 0.9,
  observationCount: 20,
  currentBox: { x: 0.4, y: 0.5, w: 0.1, h: 0.3 },
  currentPosition: null,
  trajectory: [],
  speedMps: 1.4,
  headingDegrees: 90,
  zoneIds: [],
  embedding: null,
  active: false,
  ...overrides,
});

describe('travel time plausibility', () => {
  test('peaks at the expected duration', () => {
    assert.equal(travelTimePlausibility(EDGE, 90), 1);
  });

  test('falls off either side of the expectation', () => {
    const early = travelTimePlausibility(EDGE, 60);
    const late = travelTimePlausibility(EDGE, 150);
    assert.ok(early > 0 && early < 1, `early: ${early}`);
    assert.ok(late > 0 && late < 1, `late: ${late}`);
  });

  test('is zero outside the physically possible window', () => {
    assert.equal(travelTimePlausibility(EDGE, 5), 0, 'faster than anyone can walk it');
    assert.equal(travelTimePlausibility(EDGE, 600), 0, 'far too late to be the same trip');
  });
});

/** Cosine similarity never lands on exact decimals. */
const closeTo = (actual: number | null, expected: number, tolerance = 1e-9): void => {
  assert.ok(actual !== null, 'expected a similarity, got null');
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    `expected ${expected} +/- ${tolerance}, got ${actual}`,
  );
};

describe('appearance similarity', () => {
  test('identical vectors are maximally similar', () => {
    closeTo(appearanceSimilarity([1, 0, 1], [1, 0, 1]), 1);
  });

  test('opposite vectors are minimally similar', () => {
    closeTo(appearanceSimilarity([1, 1], [-1, -1]), 0);
  });

  test('missing or mismatched embeddings yield no opinion rather than a guess', () => {
    assert.equal(appearanceSimilarity(null, [1, 2]), null);
    assert.equal(appearanceSimilarity([1, 2], null), null);
    assert.equal(appearanceSimilarity([1, 2], [1, 2, 3]), null, 'different lengths');
    assert.equal(appearanceSimilarity([], []), null, 'empty');
    assert.equal(appearanceSimilarity([0, 0], [1, 1]), null, 'a zero vector says nothing');
  });
});

describe('scoreAssociation', () => {
  test('scores a plausible hand-off highly and explains why', () => {
    const departed = track('a', CAM_07, 0, 10_000);
    const arrived = track('b', CAM_08, 100_000, 120_000);

    const association = scoreAssociation(departed, arrived, TOPOLOGY);
    assert.notEqual(association, null);
    assert.ok(association!.score > 0.9, `expected a strong score, got ${association!.score}`);

    const codes = association!.reasons.map((r) => r.code);
    assert.ok(codes.includes('CLASS_MATCH'));
    assert.ok(codes.includes('TOPOLOGY_EDGE_KNOWN'));
    assert.ok(codes.includes('TRAVEL_TIME_PLAUSIBLE'));
    assert.ok(association!.reasons.length >= 3, 'a score is never shown without its reasons');
  });

  test('rejects a different object class outright', () => {
    const departed = track('a', CAM_07, 0, 10_000);
    const arrived = track('b', CAM_08, 100_000, 120_000, { objectClass: 'car' });
    assert.equal(scoreAssociation(departed, arrived, TOPOLOGY), null);
  });

  test('rejects an arrival that precedes the departure', () => {
    const departed = track('a', CAM_07, 0, 100_000);
    const arrived = track('b', CAM_08, 50_000, 60_000);
    assert.equal(scoreAssociation(departed, arrived, TOPOLOGY), null, 'time cannot run backwards');
  });

  test('rejects two tracks on the same camera', () => {
    const departed = track('a', CAM_07, 0, 10_000);
    const arrived = track('b', CAM_07, 100_000, 120_000);
    assert.equal(scoreAssociation(departed, arrived, TOPOLOGY), null);
  });

  test('rejects a gap beyond the hard ceiling', () => {
    const departed = track('a', CAM_07, 0, 10_000);
    const arrived = track('b', CAM_08, 60 * 60 * 1000, 60 * 60 * 1000 + 5000);
    assert.equal(scoreAssociation(departed, arrived, TOPOLOGY), null);
  });

  test('a physically impossible travel time is rejected, not merely down-weighted', () => {
    const departed = track('a', CAM_07, 0, 10_000);
    // Arrives 15 seconds later, on a route whose minimum is 40 seconds.
    const tooFast = track('b', CAM_08, 25_000, 30_000);
    assert.equal(
      scoreAssociation(departed, tooFast, TOPOLOGY),
      null,
      'nobody crosses 90 m in 15 s; this must never reach a review queue',
    );

    // And arriving far outside the window is a different journey, not this one.
    const tooLate = track('c', CAM_08, 400_000, 410_000);
    assert.equal(scoreAssociation(departed, tooLate, TOPOLOGY), null);
  });

  test('an unknown route scores lower than a known one', () => {
    const departed = track('a', CAM_07, 0, 10_000, { embedding: [1, 0, 0] });
    const arrivedKnown = track('b', CAM_08, 100_000, 120_000, { embedding: [1, 0, 0] });
    const arrivedUnknown = track('c', CAM_99, 100_000, 120_000, { embedding: [1, 0, 0] });

    const known = scoreAssociation(departed, arrivedKnown, TOPOLOGY);
    const unknown = scoreAssociation(departed, arrivedUnknown, TOPOLOGY);

    assert.notEqual(known, null);
    assert.notEqual(unknown, null, 'strong corroboration still yields a weak candidate');
    assert.ok(
      unknown!.score < known!.score,
      `an unconfigured route must never outscore a known one (${unknown!.score} vs ${known!.score})`,
    );
    assert.ok(unknown!.reasons.some((r) => r.code === 'TOPOLOGY_EDGE_UNKNOWN'));
  });

  test('an unknown route alone is not enough to associate', () => {
    const departed = track('a', CAM_07, 0, 10_000);
    const arrived = track('c', CAM_99, 100_000, 120_000);
    assert.equal(
      scoreAssociation(departed, arrived, TOPOLOGY),
      null,
      'class and heading alone do not connect two unrelated cameras',
    );
  });

  test('opposing directions of travel weaken the association', () => {
    const departed = track('a', CAM_07, 0, 10_000, { headingDegrees: 90 });
    const sameWay = track('b', CAM_08, 100_000, 120_000, { headingDegrees: 90 });
    const opposite = track('c', CAM_08, 100_000, 120_000, { headingDegrees: 270 });

    const aligned = scoreAssociation(departed, sameWay, TOPOLOGY);
    const against = scoreAssociation(departed, opposite, TOPOLOGY);

    assert.notEqual(aligned, null);
    assert.notEqual(against, null);
    assert.ok(against!.score < aligned!.score, 'walking the other way is weaker evidence');
    assert.ok(against!.reasons.some((r) => r.code === 'DIRECTION_INCONSISTENT'));
  });

  test('the bidirectional edge works in reverse', () => {
    const departed = track('a', CAM_08, 0, 10_000);
    const arrived = track('b', CAM_07, 100_000, 120_000);

    const association = scoreAssociation(departed, arrived, TOPOLOGY);
    assert.notEqual(association, null);
    assert.ok(association!.reasons.some((r) => r.code === 'TOPOLOGY_EDGE_KNOWN'));
  });

  test('a missing embedding does not cap the achievable score', () => {
    const departed = track('a', CAM_07, 0, 10_000);
    const arrived = track('b', CAM_08, 100_000, 120_000);

    const withoutEmbedding = scoreAssociation(departed, arrived, TOPOLOGY);
    const withEmbedding = scoreAssociation(
      { ...departed, embedding: [1, 0, 0] },
      { ...arrived, embedding: [1, 0, 0] },
      TOPOLOGY,
    );

    assert.notEqual(withoutEmbedding, null);
    assert.notEqual(withEmbedding, null);
    // A deployment with no appearance model must still be able to score highly.
    assert.ok(withoutEmbedding!.score > 0.9);
    assert.ok(withEmbedding!.score > 0.9);
  });

  test('the score is always accompanied by contributions that sum toward it', () => {
    const departed = track('a', CAM_07, 0, 10_000);
    const arrived = track('b', CAM_08, 100_000, 120_000);

    const association = scoreAssociation(departed, arrived, TOPOLOGY);
    assert.notEqual(association, null);

    const total = association!.reasons.reduce((sum, r) => sum + r.contribution, 0);
    assert.ok(total > 0, 'every reason carries its weight, so the score can be audited');
  });
});

describe('bestAssociation', () => {
  test('picks the strongest candidate, not the first', () => {
    const arriving = track('arrive', CAM_08, 100_000, 120_000);
    const weak = track('weak', CAM_07, 0, 5_000, { headingDegrees: 270 });
    const strong = track('strong', CAM_07, 0, 10_000, { headingDegrees: 90 });

    const best = bestAssociation(arriving, [weak, strong], TOPOLOGY);
    assert.notEqual(best, null);
    assert.equal(best!.fromTrackId, strong.id);
  });

  test('returns nothing when no candidate is plausible', () => {
    const arriving = track('arrive', CAM_08, 100_000, 120_000);
    const wrongClass = track('car', CAM_07, 0, 10_000, { objectClass: 'car' });

    assert.equal(bestAssociation(arriving, [wrongClass], TOPOLOGY), null);
    assert.equal(bestAssociation(arriving, [], TOPOLOGY), null);
  });
});
