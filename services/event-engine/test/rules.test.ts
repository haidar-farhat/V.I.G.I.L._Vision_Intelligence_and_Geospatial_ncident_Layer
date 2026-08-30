import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import type {
  CameraId,
  NodeId,
  Rule,
  RuleId,
  Track,
  TrackId,
  Zone,
  ZoneId,
  ZoneObservation,
} from '@sentinel/shared-types';
import { asId, utcMillis } from '@sentinel/shared-types';
import type { RuleContext } from '../src/rules.ts';
import { RuleEngine, deriveEventId, withinTimeWindow } from '../src/rules.ts';

const NODE = asId<NodeId>('node-1');
const CAM_07 = asId<CameraId>('cam-07');
const ZONE_A = asId<ZoneId>('zone-restricted-a');

const ZONE: Zone = {
  id: ZONE_A,
  name: 'Restricted Zone A',
  purpose: 'RESTRICTED',
  geometry: { kind: 'POLYGON', ring: [] },
  locationId: null,
  parentZoneId: null,
  active: true,
  createdAt: utcMillis(0),
  updatedAt: utcMillis(0),
};

const track = (overrides: Partial<Track> = {}): Track => ({
  id: asId<TrackId>('track-1'),
  cameraId: CAM_07,
  objectClass: 'person',
  firstSeen: utcMillis(0),
  lastSeen: utcMillis(1000),
  confidence: 0.9,
  observationCount: 10,
  currentBox: { x: 0.4, y: 0.5, w: 0.1, h: 0.3 },
  currentPosition: null,
  trajectory: [],
  speedMps: null,
  headingDegrees: null,
  zoneIds: [ZONE_A],
  embedding: null,
  active: true,
  ...overrides,
});

const rule = (overrides: Partial<Rule> = {}): Rule => ({
  id: asId<RuleId>('rule-restricted-entry'),
  name: 'Restricted entry',
  enabled: true,
  when: {
    objectClasses: ['person'],
    zoneIds: [ZONE_A],
    cameraIds: [],
    timeWindow: null,
    direction: 'ANY',
    minDurationMillis: 0,
    confidenceThreshold: 0.5,
    cooldownMillis: 0,
    minTrackCount: 1,
  },
  then: {
    eventType: 'PersonEnteredZone',
    severity: 'HIGH',
    notify: true,
    createIncident: true,
  },
  createdAt: utcMillis(0),
  updatedAt: utcMillis(0),
  ...overrides,
});

const context = (overrides: Partial<RuleContext> = {}): RuleContext => ({
  nodeId: NODE,
  cameraId: CAM_07,
  track: track(),
  observation: { zoneId: ZONE_A, transition: 'ENTERED', at: utcMillis(100_000) },
  zone: ZONE,
  concurrentTracksInZone: 1,
  utcOffsetMinutes: 0,
  ...overrides,
});

describe('time windows', () => {
  /** 2024-01-03 was a Wednesday. 00:00 UTC. */
  const WED_MIDNIGHT = utcMillis(Date.UTC(2024, 0, 3, 0, 0, 0));
  const at = (hours: number, minutes = 0): ReturnType<typeof utcMillis> =>
    utcMillis(WED_MIDNIGHT + hours * 3_600_000 + minutes * 60_000);

  test('matches a window inside a single day', () => {
    const window = { startMinute: 9 * 60, endMinute: 17 * 60, daysOfWeek: [] };
    assert.ok(withinTimeWindow(window, at(12), 0));
    assert.ok(!withinTimeWindow(window, at(8), 0));
    assert.ok(!withinTimeWindow(window, at(18), 0));
  });

  test('matches an after-hours window that crosses midnight', () => {
    // The common case, and the one a naive start<=t<end comparison gets wrong.
    const window = { startMinute: 22 * 60, endMinute: 6 * 60, daysOfWeek: [] };
    assert.ok(withinTimeWindow(window, at(23), 0), '23:00 is after hours');
    assert.ok(withinTimeWindow(window, at(2), 0), '02:00 is after hours');
    assert.ok(!withinTimeWindow(window, at(12), 0), 'midday is not');
    assert.ok(!withinTimeWindow(window, at(7), 0), '07:00 is not');
  });

  test('honours the site timezone rather than the host clock', () => {
    const window = { startMinute: 22 * 60, endMinute: 6 * 60, daysOfWeek: [] };
    // 20:00 UTC is 23:00 at a site three hours ahead: after hours there.
    assert.ok(!withinTimeWindow(window, at(20), 0), 'not after hours in UTC');
    assert.ok(withinTimeWindow(window, at(20), 180), 'but it is at the site');
  });

  test('restricts to specific days of the week', () => {
    const weekend = { startMinute: 0, endMinute: 1439, daysOfWeek: [0, 6] };
    assert.ok(!withinTimeWindow(weekend, at(12), 0), 'Wednesday is not the weekend');

    const midweek = { startMinute: 0, endMinute: 1439, daysOfWeek: [3] };
    assert.ok(withinTimeWindow(midweek, at(12), 0), 'Wednesday is day 3');
  });
});

describe('event identity', () => {
  test('is deterministic for the same occurrence', () => {
    const identity = {
      nodeId: NODE,
      cameraId: CAM_07,
      ruleId: asId<RuleId>('r1'),
      type: 'PersonEnteredZone' as const,
      trackId: asId<TrackId>('t1'),
      timeBucket: 100,
    };
    assert.equal(deriveEventId(identity), deriveEventId(identity));
  });

  test('differs when any identity field differs', () => {
    const base = {
      nodeId: NODE,
      cameraId: CAM_07,
      ruleId: asId<RuleId>('r1'),
      type: 'PersonEnteredZone' as const,
      trackId: asId<TrackId>('t1'),
      timeBucket: 100,
    };

    assert.notEqual(deriveEventId(base), deriveEventId({ ...base, timeBucket: 101 }));
    assert.notEqual(deriveEventId(base), deriveEventId({ ...base, cameraId: asId('cam-08') }));
    assert.notEqual(deriveEventId(base), deriveEventId({ ...base, trackId: asId('t2') }));
  });
});

describe('rule evaluation', () => {
  test('fires on a matching observation', () => {
    const engine = new RuleEngine();
    const result = engine.evaluate(rule(), context());

    assert.equal(result.fired, true);
    if (!result.fired) return;

    assert.equal(result.event.type, 'PersonEnteredZone');
    assert.equal(result.event.severity, 'HIGH');
    assert.equal(result.event.cameraId, CAM_07);
    assert.deepEqual(result.event.zoneIds, [ZONE_A]);
    assert.match(result.event.summary, /a person entered Restricted Zone A/);
  });

  test('describes what was observed, never who or why', () => {
    const engine = new RuleEngine();
    const result = engine.evaluate(rule(), context());
    assert.equal(result.fired, true);
    if (!result.fired) return;

    assert.ok(
      !/intruder|criminal|suspect|trespasser/i.test(result.event.summary),
      `summary must stay factual: "${result.event.summary}"`,
    );
  });

  test('does not fire for a disabled rule', () => {
    const engine = new RuleEngine();
    const result = engine.evaluate(rule({ enabled: false }), context());
    assert.equal(result.fired, false);
    if (result.fired) return;
    assert.equal(result.reason, 'DISABLED');
  });

  test('does not fire for the wrong object class', () => {
    const engine = new RuleEngine();
    const result = engine.evaluate(rule(), context({ track: track({ objectClass: 'car' }) }));
    assert.equal(result.fired, false);
    if (result.fired) return;
    assert.equal(result.reason, 'CLASS_MISMATCH');
  });

  test('a vehicle rule covers the vehicle family', () => {
    const engine = new RuleEngine();
    const vehicleRule = rule({
      when: { ...rule().when, objectClasses: ['vehicle'] },
      then: { ...rule().then, eventType: 'VehicleEnteredZone' },
    });

    for (const cls of ['car', 'truck', 'bus', 'motorcycle']) {
      const result = engine.evaluate(vehicleRule, context({ track: track({ objectClass: cls }) }));
      assert.equal(result.fired, true, `${cls} should satisfy a vehicle rule`);
    }

    const person = engine.evaluate(vehicleRule, context({ track: track({ objectClass: 'person' }) }));
    assert.equal(person.fired, false, 'a person is not a vehicle');
  });

  test('does not fire on the wrong transition', () => {
    const engine = new RuleEngine();
    // An "entered" rule must not fire on an exit.
    const result = engine.evaluate(
      rule(),
      context({ observation: { zoneId: ZONE_A, transition: 'EXITED', at: utcMillis(100_000) } }),
    );
    assert.equal(result.fired, false);
    if (result.fired) return;
    assert.equal(result.reason, 'TRANSITION_MISMATCH');
  });

  test('does not fire below the confidence threshold', () => {
    const engine = new RuleEngine();
    const result = engine.evaluate(
      rule({ when: { ...rule().when, confidenceThreshold: 0.95 } }),
      context(),
    );
    assert.equal(result.fired, false);
    if (result.fired) return;
    assert.equal(result.reason, 'BELOW_CONFIDENCE');
  });

  test('does not fire before the minimum dwell is reached', () => {
    const engine = new RuleEngine();
    const loitering = rule({
      when: { ...rule().when, minDurationMillis: 10_000 },
      then: { ...rule().then, eventType: 'ObjectLoitering' },
    });

    const tooEarly = engine.evaluate(
      loitering,
      context({
        observation: {
          zoneId: ZONE_A,
          transition: 'DWELLING',
          at: utcMillis(100_000),
          dwellMillis: 5_000,
        },
      }),
    );
    assert.equal(tooEarly.fired, false);
    if (tooEarly.fired) return;
    assert.equal(tooEarly.reason, 'DURATION_NOT_MET');

    const longEnough = engine.evaluate(
      loitering,
      context({
        observation: {
          zoneId: ZONE_A,
          transition: 'DWELLING',
          at: utcMillis(120_000),
          dwellMillis: 20_000,
        },
      }),
    );
    assert.equal(longEnough.fired, true);
    if (!longEnough.fired) return;
    assert.match(longEnough.event.summary, /remained in Restricted Zone A for 20 seconds/);
  });

  test('does not fire below the minimum track count', () => {
    const engine = new RuleEngine();
    const groupRule = rule({
      when: { ...rule().when, minTrackCount: 3 },
      then: { ...rule().then, eventType: 'CrowdDetected' },
    });

    const alone = engine.evaluate(groupRule, context({ concurrentTracksInZone: 1 }));
    assert.equal(alone.fired, false);
    if (alone.fired) return;
    assert.equal(alone.reason, 'TOO_FEW_TRACKS');

    assert.equal(engine.evaluate(groupRule, context({ concurrentTracksInZone: 3 })).fired, true);
  });

  test('respects the cooldown so one track cannot spam events', () => {
    const engine = new RuleEngine();
    const cooled = rule({ when: { ...rule().when, cooldownMillis: 60_000 } });

    const first = engine.evaluate(cooled, context({ observation: obs(100_000) }));
    assert.equal(first.fired, true);

    const suppressed = engine.evaluate(cooled, context({ observation: obs(110_000) }));
    assert.equal(suppressed.fired, false);
    if (suppressed.fired) return;
    assert.equal(suppressed.reason, 'IN_COOLDOWN');

    const afterCooldown = engine.evaluate(cooled, context({ observation: obs(200_000) }));
    assert.equal(afterCooldown.fired, true, 'the cooldown expires');
  });

  test('cooldown is per track, so one object cannot mask another', () => {
    const engine = new RuleEngine();
    const cooled = rule({ when: { ...rule().when, cooldownMillis: 60_000 } });

    engine.evaluate(cooled, context({ observation: obs(100_000) }));
    const other = engine.evaluate(
      cooled,
      context({ track: track({ id: asId<TrackId>('track-2') }), observation: obs(101_000) }),
    );

    assert.equal(other.fired, true, 'a second person must still raise an event');
  });

  test('enforces the time window', () => {
    const engine = new RuleEngine();
    const afterHours = rule({
      when: {
        ...rule().when,
        timeWindow: { startMinute: 22 * 60, endMinute: 6 * 60, daysOfWeek: [] },
      },
    });

    const midday = utcMillis(Date.UTC(2024, 0, 3, 12, 0, 0));
    const night = utcMillis(Date.UTC(2024, 0, 3, 2, 0, 0));

    const daytime = engine.evaluate(
      afterHours,
      context({ observation: { zoneId: ZONE_A, transition: 'ENTERED', at: midday } }),
    );
    assert.equal(daytime.fired, false);
    if (daytime.fired) return;
    assert.equal(daytime.reason, 'OUTSIDE_TIME_WINDOW');

    assert.equal(
      engine.evaluate(
        afterHours,
        context({ observation: { zoneId: ZONE_A, transition: 'ENTERED', at: night } }),
      ).fired,
      true,
    );
  });

  test('evaluateAll runs every registered rule', () => {
    const engine = new RuleEngine([
      rule(),
      rule({
        id: asId<RuleId>('rule-2'),
        name: 'Second rule',
        then: { ...rule().then, eventType: 'LineCrossed', severity: 'LOW' },
      }),
    ]);

    // Only the entry rule matches an ENTERED transition.
    const events = engine.evaluateAll(context());
    assert.equal(events.length, 1);
    assert.equal(events[0]?.type, 'PersonEnteredZone');
  });
});

const obs = (at: number): ZoneObservation => ({
  zoneId: ZONE_A,
  transition: 'ENTERED',
  at: utcMillis(at),
});
