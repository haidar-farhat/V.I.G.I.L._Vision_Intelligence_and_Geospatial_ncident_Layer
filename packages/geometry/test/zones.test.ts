import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import type { LatLon, Zone, ZoneGeometry } from '@sentinel/shared-types';
import { asId, utcMillis } from '@sentinel/shared-types';
import type { ZoneId } from '@sentinel/shared-types';
import { ZoneEngine, compileZone, zoneContains, zoneCrossing } from '../src/zones.ts';
import { destinationPoint } from '../src/geodesy.ts';

const SITE: LatLon = { lat: 33.8938, lon: 35.5018 };

/** Offset from the site anchor by metres east and north. */
const at = (east: number, north: number): LatLon => {
  const northed = destinationPoint(SITE, 0, north);
  return destinationPoint(northed, 90, east);
};

const makeZone = (id: string, geometry: ZoneGeometry): Zone => ({
  id: asId<ZoneId>(id),
  name: id,
  purpose: 'RESTRICTED',
  geometry,
  locationId: null,
  parentZoneId: null,
  active: true,
  createdAt: utcMillis(0),
  updatedAt: utcMillis(0),
});

/** A 40 m x 40 m square with its south-west corner at the site anchor. */
const SQUARE = makeZone('zone-square', {
  kind: 'POLYGON',
  ring: [at(0, 0), at(40, 0), at(40, 40), at(0, 40)],
});

describe('zone containment', () => {
  test('polygon contains interior points and excludes exterior ones', () => {
    const zone = compileZone(SQUARE);
    assert.ok(zoneContains(zone, at(20, 20)), 'centre is inside');
    assert.ok(zoneContains(zone, at(1, 1)), 'near corner is inside');
    assert.ok(!zoneContains(zone, at(-5, 20)), 'west of the zone is outside');
    assert.ok(!zoneContains(zone, at(60, 20)), 'east of the zone is outside');
    assert.ok(!zoneContains(zone, at(20, 100)), 'north of the zone is outside');
  });

  test('circle containment respects the radius', () => {
    const zone = compileZone(
      makeZone('zone-circle', { kind: 'CIRCLE', center: at(0, 0), radiusMeters: 25 }),
    );
    assert.ok(zoneContains(zone, at(0, 0)), 'centre');
    assert.ok(zoneContains(zone, at(20, 0)), 'inside the radius');
    assert.ok(!zoneContains(zone, at(30, 0)), 'outside the radius');
    // 3-4-5: exactly 25 m away, on the boundary of the polygonal approximation.
    assert.ok(!zoneContains(zone, at(40, 30)), 'well outside on the diagonal');
  });

  test('corridor containment is a buffered polyline', () => {
    const zone = compileZone(
      makeZone('zone-corridor', {
        kind: 'CORRIDOR',
        path: [at(0, 0), at(100, 0)],
        widthMeters: 10,
      }),
    );
    assert.ok(zoneContains(zone, at(50, 0)), 'on the centreline');
    assert.ok(zoneContains(zone, at(50, 4)), 'within half-width');
    assert.ok(!zoneContains(zone, at(50, 8)), 'beyond half-width');
    assert.ok(!zoneContains(zone, at(150, 0)), 'past the end of the path');
  });

  test('a line zone has no interior', () => {
    const zone = compileZone(
      makeZone('zone-line', { kind: 'LINE', path: [at(0, 0), at(0, 40)] }),
    );
    assert.ok(!zoneContains(zone, at(0, 20)), 'a tripwire contains nothing');
  });
});

describe('line crossing', () => {
  const tripwire = compileZone(
    makeZone('zone-wire', { kind: 'LINE', path: [at(0, -20), at(0, 20)] }),
  );

  test('detects a crossing and reports the side landed on', () => {
    const westToEast = zoneCrossing(tripwire, at(-10, 0), at(10, 0));
    const eastToWest = zoneCrossing(tripwire, at(10, 0), at(-10, 0));

    assert.ok(westToEast.crossed, 'west-to-east crosses');
    assert.ok(eastToWest.crossed, 'east-to-west crosses');
    assert.notEqual(
      westToEast.sign,
      eastToWest.sign,
      'opposite directions must report opposite signs',
    );
  });

  test('movement that does not reach the wire does not fire', () => {
    assert.ok(!zoneCrossing(tripwire, at(-10, 0), at(-5, 0)).crossed);
  });

  test('movement past the end of the wire does not fire', () => {
    assert.ok(!zoneCrossing(tripwire, at(-10, 50), at(10, 50)).crossed);
  });
});

describe('ZoneEngine transitions', () => {
  test('emits ENTERED once on entry and EXITED once on departure', () => {
    const engine = new ZoneEngine([SQUARE]);

    const outside = engine.update('track-1', at(-10, 20), utcMillis(0));
    assert.deepEqual(outside, [], 'no transition while outside');

    const entering = engine.update('track-1', at(20, 20), utcMillis(1000));
    assert.equal(entering.length, 1);
    assert.equal(entering[0]?.transition, 'ENTERED');

    const staying = engine.update('track-1', at(21, 20), utcMillis(1100));
    assert.deepEqual(staying, [], 'no repeat ENTERED while still inside');

    const leaving = engine.update('track-1', at(-10, 20), utcMillis(2000));
    assert.equal(leaving.length, 1);
    assert.equal(leaving[0]?.transition, 'EXITED');
  });

  test('reports dwell periodically with an accumulating duration', () => {
    const engine = new ZoneEngine([SQUARE], { dwellReportIntervalMillis: 1000 });

    engine.update('track-1', at(20, 20), utcMillis(0));
    const early = engine.update('track-1', at(20, 21), utcMillis(500));
    assert.deepEqual(early, [], 'below the report interval, stays quiet');

    const first = engine.update('track-1', at(20, 22), utcMillis(1000));
    assert.equal(first[0]?.transition, 'DWELLING');
    assert.equal(first[0]?.dwellMillis, 1000);

    const second = engine.update('track-1', at(20, 23), utcMillis(2500));
    assert.equal(second[0]?.transition, 'DWELLING');
    assert.equal(second[0]?.dwellMillis, 2500, 'dwell is measured from entry, not last report');
  });

  test('dwell resets after leaving and re-entering', () => {
    const engine = new ZoneEngine([SQUARE], { dwellReportIntervalMillis: 1000 });

    engine.update('track-1', at(20, 20), utcMillis(0));
    engine.update('track-1', at(20, 20), utcMillis(5000));
    engine.update('track-1', at(-10, 20), utcMillis(6000));
    engine.update('track-1', at(20, 20), utcMillis(7000));

    const dwell = engine.update('track-1', at(20, 20), utcMillis(8000));
    assert.equal(dwell[0]?.transition, 'DWELLING');
    assert.equal(dwell[0]?.dwellMillis, 1000, 'dwell restarts on re-entry');
  });

  test('tracks are independent of one another', () => {
    const engine = new ZoneEngine([SQUARE]);

    engine.update('track-1', at(20, 20), utcMillis(0));
    const other = engine.update('track-2', at(-10, 20), utcMillis(0));
    assert.deepEqual(other, [], 'track-2 never entered');

    assert.deepEqual(engine.zonesFor('track-1'), [SQUARE.id]);
    assert.deepEqual(engine.zonesFor('track-2'), []);
  });

  test('a tripwire fires on the movement, not on membership', () => {
    const wire = makeZone('zone-wire', { kind: 'LINE', path: [at(0, -20), at(0, 20)] });
    const engine = new ZoneEngine([wire]);

    const first = engine.update('track-1', at(-10, 0), utcMillis(0));
    assert.deepEqual(first, [], 'the first sample has nothing to cross from');

    const crossing = engine.update('track-1', at(10, 0), utcMillis(1000));
    assert.equal(crossing.length, 1);
    assert.equal(crossing[0]?.transition, 'CROSSED');
    assert.ok(crossing[0]?.crossingSign === 1 || crossing[0]?.crossingSign === -1);

    const after = engine.update('track-1', at(20, 0), utcMillis(2000));
    assert.deepEqual(after, [], 'moving on does not re-fire the wire');
  });

  test('forgetting a track releases its state', () => {
    const engine = new ZoneEngine([SQUARE]);
    engine.update('track-1', at(20, 20), utcMillis(0));
    assert.equal(engine.trackedCount, 1);

    engine.forget('track-1');
    assert.equal(engine.trackedCount, 0);
    assert.deepEqual(engine.zonesFor('track-1'), []);

    // A re-appearing track is treated as new, not as still inside.
    const reentry = engine.update('track-1', at(20, 20), utcMillis(9000));
    assert.equal(reentry[0]?.transition, 'ENTERED');
  });

  test('inactive zones are not evaluated', () => {
    const engine = new ZoneEngine([{ ...SQUARE, active: false }]);
    assert.equal(engine.zoneCount, 0);
    assert.deepEqual(engine.update('track-1', at(20, 20), utcMillis(0)), []);
  });

  test('removing a zone clears it from every track', () => {
    const engine = new ZoneEngine([SQUARE]);
    engine.update('track-1', at(20, 20), utcMillis(0));
    assert.deepEqual(engine.zonesFor('track-1'), [SQUARE.id]);

    engine.removeZone(SQUARE.id);
    assert.equal(engine.zoneCount, 0);
    assert.deepEqual(engine.zonesFor('track-1'), []);
  });

  test('a track crossing a corner produces exactly one enter and one exit', () => {
    const engine = new ZoneEngine([SQUARE]);
    const transitions: string[] = [];

    // Walk west to east along y = 20, straight through the square.
    for (let east = -10; east <= 60; east += 2) {
      const observations = engine.update('track-1', at(east, 20), utcMillis(east * 100 + 10_000));
      for (const o of observations) transitions.push(o.transition);
    }

    assert.deepEqual(
      transitions.filter((t) => t !== 'DWELLING'),
      ['ENTERED', 'EXITED'],
      'a single pass must not chatter',
    );
  });
});
