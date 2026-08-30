import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import type {
  CameraId,
  EventId,
  NodeId,
  RuleId,
  SecurityEvent,
  TrackAssociation,
  TrackId,
  Zone,
  ZoneId,
} from '@sentinel/shared-types';
import { asId, utcMillis } from '@sentinel/shared-types';
import { Correlator, deriveIncidentId } from '../src/correlation.ts';
import type { RiskContext } from '../src/risk.ts';

const NODE = asId<NodeId>('node-1');
const CAM_07 = asId<CameraId>('cam-07');
const CAM_08 = asId<CameraId>('cam-08');
const CAM_09 = asId<CameraId>('cam-09');
const ZONE_A = asId<ZoneId>('zone-restricted-a');
const ZONE_B = asId<ZoneId>('zone-loading');

const zone = (id: ZoneId, name: string, purpose: Zone['purpose']): Zone => ({
  id,
  name,
  purpose,
  geometry: { kind: 'POLYGON', ring: [] },
  locationId: null,
  parentZoneId: null,
  active: true,
  createdAt: utcMillis(0),
  updatedAt: utcMillis(0),
});

const ZONES = new Map<ZoneId, Zone>([
  [ZONE_A, zone(ZONE_A, 'Restricted Zone A', 'RESTRICTED')],
  [ZONE_B, zone(ZONE_B, 'Loading Area', 'LOADING')],
]);

const riskContext = (): RiskContext => ({
  zones: ZONES,
  afterHours: { startMinute: 22 * 60, endMinute: 6 * 60 },
  utcOffsetMinutes: 0,
  assessedAt: utcMillis(0),
});

let sequence = 0;
const event = (
  cameraId: CameraId,
  at: number,
  trackId: string,
  zoneId: ZoneId = ZONE_A,
  overrides: Partial<SecurityEvent> = {},
): SecurityEvent => {
  sequence += 1;
  return {
    id: asId<EventId>(`event-${sequence}`),
    type: 'PersonEnteredZone',
    severity: 'MEDIUM',
    status: 'NEW',
    occurredAt: utcMillis(at),
    recordedAt: utcMillis(at),
    cameraId,
    nodeId: NODE,
    zoneIds: [zoneId],
    trackIds: [asId<TrackId>(trackId)],
    objectClass: 'person',
    confidence: 0.9,
    position: null,
    ruleId: asId<RuleId>('rule-1'),
    modelId: null,
    evidenceIds: [],
    incidentId: null,
    summary: 'a person entered the zone',
    detail: { zone: ZONES.get(zoneId)?.name ?? '' },
    ...overrides,
  };
};

const association = (
  fromTrack: string,
  toTrack: string,
  fromCamera: CameraId,
  toCamera: CameraId,
  score: number,
): TrackAssociation => ({
  fromTrackId: asId<TrackId>(fromTrack),
  toTrackId: asId<TrackId>(toTrack),
  fromCameraId: fromCamera,
  toCameraId: toCamera,
  departedAt: utcMillis(0),
  arrivedAt: utcMillis(1000),
  score,
  reasons: [],
});

describe('the defining behaviour: no alert storms', () => {
  test('three cameras seeing one person produce ONE incident, not three alerts', () => {
    const correlator = new Correlator(riskContext());

    // The same person, handed off across three cameras, each hand-off scored.
    correlator.addAssociation(association('t-07', 't-08', CAM_07, CAM_08, 0.87));
    correlator.addAssociation(association('t-08', 't-09', CAM_08, CAM_09, 0.81));

    const a = correlator.ingest(event(CAM_07, 0, 't-07'));
    const b = correlator.ingest(event(CAM_08, 160_000, 't-08'));
    const c = correlator.ingest(event(CAM_09, 320_000, 't-09'));

    assert.equal(correlator.openIncidentCount, 1, 'this is the whole point of the product');
    assert.equal(a.incident.id, b.incident.id);
    assert.equal(b.incident.id, c.incident.id);

    assert.equal(a.opened, true, 'the first event opened it');
    assert.equal(b.opened, false);
    assert.equal(b.reason, 'ASSOCIATED_TRACK');

    const incident = c.incident;
    assert.equal(incident.eventIds.length, 3, 'all three events are supporting evidence');
    assert.deepEqual([...incident.cameraIds].sort(), [CAM_07, CAM_08, CAM_09].sort());
  });

  test('the same track re-triggering does not open new incidents', () => {
    const correlator = new Correlator(riskContext());

    for (let i = 0; i < 5; i += 1) {
      correlator.ingest(event(CAM_07, i * 10_000, 't-07'));
    }

    assert.equal(correlator.openIncidentCount, 1);
    assert.equal(correlator.incidents()[0]?.eventIds.length, 5);
  });

  test('a replayed event is not counted twice', () => {
    const correlator = new Correlator(riskContext());
    const e = event(CAM_07, 0, 't-07');

    correlator.ingest(e);
    correlator.ingest(e);
    correlator.ingest(e);

    // Worker reconnect replays are at-least-once; correlation must be idempotent.
    assert.equal(correlator.openIncidentCount, 1);
    assert.equal(correlator.incidents()[0]?.eventIds.length, 1);
  });
});

describe('separation of genuinely unrelated activity', () => {
  test('unrelated tracks in different zones open separate incidents', () => {
    const correlator = new Correlator(riskContext());

    correlator.ingest(event(CAM_07, 0, 't-a', ZONE_A));
    correlator.ingest(event(CAM_08, 1000, 't-b', ZONE_B));

    assert.equal(correlator.openIncidentCount, 2, 'genuinely separate situations stay separate');
  });

  test('a weak association does not merge two incidents', () => {
    const correlator = new Correlator(riskContext(), { minAssociationScore: 0.6 });
    correlator.addAssociation(association('t-a', 't-b', CAM_07, CAM_08, 0.35));

    correlator.ingest(event(CAM_07, 0, 't-a', ZONE_A));
    correlator.ingest(event(CAM_08, 160_000, 't-b', ZONE_B));

    assert.equal(
      correlator.openIncidentCount,
      2,
      'a weak hypothesis is worth showing, not worth restructuring the queue',
    );
  });

  test('activity after the quiet period opens a fresh incident', () => {
    const correlator = new Correlator(riskContext(), { quietPeriodMillis: 60_000 });

    correlator.ingest(event(CAM_07, 0, 't-a'));
    correlator.ingest(event(CAM_07, 500_000, 't-b'));

    assert.equal(correlator.openIncidentCount, 1, 'the first incident has been closed out');
    assert.equal(
      correlator.incidents()[0]?.openedAt,
      500_000,
      'the surviving incident is the recent one',
    );
  });

  test('an incident cannot grow without bound', () => {
    const correlator = new Correlator(riskContext(), {
      quietPeriodMillis: 60_000,
      maxSpanMillis: 120_000,
    });

    // A continuous trickle on one track, well inside the quiet period.
    for (let at = 0; at <= 300_000; at += 30_000) {
      correlator.ingest(event(CAM_07, at, 't-a'));
    }

    const incidents = correlator.incidents();
    assert.equal(incidents.length, 1);
    const span = (incidents[0]?.updatedAt ?? 0) - (incidents[0]?.openedAt ?? 0);
    assert.ok(
      span <= 120_000,
      `an incident must not swallow a whole night's activity (span ${span} ms)`,
    );
  });
});

describe('correlation reasons', () => {
  test('sharing a track is the strongest link', () => {
    const correlator = new Correlator(riskContext());
    correlator.ingest(event(CAM_07, 0, 't-a'));
    const second = correlator.ingest(event(CAM_07, 10_000, 't-a'));
    assert.equal(second.reason, 'SAME_TRACK');
  });

  test('same zone and time links circumstantially', () => {
    const correlator = new Correlator(riskContext());
    correlator.ingest(event(CAM_07, 0, 't-a', ZONE_A));
    const second = correlator.ingest(event(CAM_08, 5_000, 't-b', ZONE_A));
    assert.equal(second.reason, 'SAME_ZONE_AND_TIME');
  });

  test('a new situation is reported as such', () => {
    const correlator = new Correlator(riskContext());
    assert.equal(correlator.ingest(event(CAM_07, 0, 't-a')).reason, 'NEW_SITUATION');
  });
});

describe('incident assembly', () => {
  test('builds a chronological timeline', () => {
    const correlator = new Correlator(riskContext());
    correlator.addAssociation(association('t-07', 't-08', CAM_07, CAM_08, 0.9));

    const first = correlator.ingest(event(CAM_07, 0, 't-07'));
    correlator.ingest(event(CAM_08, 160_000, 't-08'));

    const timeline = correlator.timeline(first.incident.id);
    assert.ok(timeline.length >= 3, 'events plus the association entry');

    for (let i = 1; i < timeline.length; i += 1) {
      assert.ok((timeline[i]?.at ?? 0) >= (timeline[i - 1]?.at ?? 0), 'timeline must be ordered');
    }
    assert.ok(timeline.some((e) => e.kind === 'ASSOCIATION'));
  });

  test('titles an incident so it can be triaged from a list', () => {
    const correlator = new Correlator(riskContext());
    const result = correlator.ingest(event(CAM_07, 0, 't-a'));

    assert.match(result.incident.title, /person/i);
    assert.match(result.incident.title, /Restricted Zone A/);
    // Never who or why.
    assert.ok(!/intruder|criminal|suspect/i.test(result.incident.title));
  });

  test('notes multi-camera corroboration in the title', () => {
    const correlator = new Correlator(riskContext());
    correlator.addAssociation(association('t-07', 't-08', CAM_07, CAM_08, 0.9));

    correlator.ingest(event(CAM_07, 0, 't-07'));
    const second = correlator.ingest(event(CAM_08, 160_000, 't-08'));

    assert.match(second.incident.title, /2 cameras/);
  });

  test('exposes the events and associations backing an incident', () => {
    const correlator = new Correlator(riskContext());
    correlator.addAssociation(association('t-07', 't-08', CAM_07, CAM_08, 0.9));

    const first = correlator.ingest(event(CAM_07, 0, 't-07'));
    correlator.ingest(event(CAM_08, 160_000, 't-08'));

    assert.equal(correlator.eventsFor(first.incident.id).length, 2);
    assert.equal(correlator.associationsFor(first.incident.id).length, 1);
  });
});

describe('incident identity', () => {
  test('is deterministic, so replay reproduces the same incident', () => {
    const seed = event(CAM_07, 0, 't-a');
    assert.equal(deriveIncidentId(seed), deriveIncidentId(seed));
  });

  test('is readable enough for an operator to quote over a radio', () => {
    const id = deriveIncidentId(event(CAM_07, 0, 't-a'));
    assert.match(String(id), /^INC-[0-9A-F]{10}$/);
  });

  test('differs for genuinely different situations', () => {
    const a = deriveIncidentId(event(CAM_07, 0, 't-a'));
    const b = deriveIncidentId(event(CAM_08, 999_000, 't-b'));
    assert.notEqual(a, b);
  });
});
