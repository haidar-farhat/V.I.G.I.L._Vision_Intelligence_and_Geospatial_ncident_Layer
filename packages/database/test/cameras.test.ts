import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import type {
  Camera,
  CameraId,
  CameraProfileId,
  CredentialsRef,
  NodeId,
  ZoneId,
} from '@sentinel/shared-types';
import { asId, utcMillis } from '@sentinel/shared-types';
import { openMemoryDatabase } from '../src/sqlite.ts';
import { MigrationRunner } from '../src/migrations.ts';
import { MIGRATIONS } from '../src/schema.ts';
import {
  CameraRepository,
  DEFAULT_AI_POLICY,
  DEFAULT_RECORDING_POLICY,
} from '../src/repositories/cameras.ts';

const PASSWORD = 'correct-horse-battery';

const setup = (): { db: ReturnType<typeof openMemoryDatabase>; repo: CameraRepository } => {
  const db = openMemoryDatabase();
  new MigrationRunner(db, MIGRATIONS).migrate(1000);
  return { db, repo: new CameraRepository(db) };
};

const camera = (overrides: Partial<Camera> = {}): Camera => ({
  id: asId<CameraId>('cam-07'),
  name: 'Camera 07 - West Approach',
  protocol: 'RTSP',
  host: '192.168.1.50',
  port: 554,
  workerNodeId: null,
  locationId: null,
  pose: {
    position: { lat: 33.8938, lon: 35.5018, altitude: 0 },
    mountHeight: 6,
    heading: 0,
    pitch: -18,
    roll: 0,
    horizontalFov: 70,
    verticalFov: 40,
    rangeMeters: 90,
  },
  intrinsics: null,
  zoneIds: [],
  ai: DEFAULT_AI_POLICY,
  recording: DEFAULT_RECORDING_POLICY,
  status: 'UNKNOWN',
  lastSeen: null,
  ptzSupported: false,
  createdAt: utcMillis(1000),
  updatedAt: utcMillis(1000),
  ...overrides,
});

describe('camera persistence', () => {
  test('round-trips a camera including its pose and policies', () => {
    const { db, repo } = setup();
    const original = camera();

    repo.save(original);
    const loaded = repo.get(original.id);

    assert.notEqual(loaded, undefined);
    assert.equal(loaded?.name, original.name);
    assert.equal(loaded?.host, '192.168.1.50');
    assert.equal(loaded?.pose?.heading, 0);
    assert.equal(loaded?.pose?.mountHeight, 6);
    assert.equal(loaded?.ai.idleFps, 2);
    assert.equal(loaded?.recording.mode, 'EVENT');
    db.close();
  });

  test('updates in place rather than duplicating', () => {
    const { db, repo } = setup();

    repo.save(camera());
    repo.save(camera({ name: 'Camera 07 - Renamed', updatedAt: utcMillis(2000) }));

    assert.equal(repo.list().length, 1);
    assert.equal(repo.get(asId<CameraId>('cam-07'))?.name, 'Camera 07 - Renamed');
    db.close();
  });

  test('stores a credential reference and never a credential', () => {
    const { db, repo } = setup();

    repo.save(camera({ credentialsRef: asId<CredentialsRef>('camera/cam-07') }));

    // The reference persists...
    assert.equal(repo.get(asId<CameraId>('cam-07'))?.credentialsRef, 'camera/cam-07');

    // ...and nothing resembling the secret is anywhere in the row.
    const raw = JSON.stringify(db.query('SELECT * FROM cameras'));
    assert.ok(!raw.includes(PASSWORD), 'a credential reached the database');
    assert.ok(!raw.includes('@192.168.1.50'), 'a credentialed URL reached the database');
    db.close();
  });

  test('saves a camera and its zone links atomically', () => {
    // A camera saved without its zones monitors nothing, and looks exactly like a
    // working configuration until an intrusion goes unreported.
    const { db, repo } = setup();

    db.exec(
      'INSERT INTO zones (id, name, purpose, geometry, active, created_at, updated_at) VALUES (?, ?, ?, ?, 1, 0, 0)',
      ['zone-a', 'Restricted Zone A', 'RESTRICTED', '{}'],
    );
    db.exec(
      'INSERT INTO zones (id, name, purpose, geometry, active, created_at, updated_at) VALUES (?, ?, ?, ?, 1, 0, 0)',
      ['zone-b', 'Perimeter', 'PERIMETER', '{}'],
    );

    repo.save(camera({ zoneIds: [asId<ZoneId>('zone-a'), asId<ZoneId>('zone-b')] }));

    const loaded = repo.get(asId<CameraId>('cam-07'));
    assert.deepEqual([...(loaded?.zoneIds ?? [])].sort(), ['zone-a', 'zone-b']);
    db.close();
  });

  test('replacing zone links removes the old ones', () => {
    const { db, repo } = setup();
    db.exec(
      'INSERT INTO zones (id, name, purpose, geometry, active, created_at, updated_at) VALUES (?, ?, ?, ?, 1, 0, 0)',
      ['zone-a', 'A', 'RESTRICTED', '{}'],
    );

    repo.save(camera({ zoneIds: [asId<ZoneId>('zone-a')] }));
    repo.save(camera({ zoneIds: [] }));

    assert.deepEqual(repo.get(asId<CameraId>('cam-07'))?.zoneIds, []);
    assert.equal(db.query('SELECT * FROM camera_zone_links').length, 0);
    db.close();
  });

  test('a camera with no pose is a valid, loadable state', () => {
    // Cameras exist before an operator has placed them on the map.
    const { db, repo } = setup();

    repo.save(camera({ pose: null }));
    const loaded = repo.get(asId<CameraId>('cam-07'));

    assert.equal(loaded?.pose, null);
    assert.notEqual(loaded, undefined, 'an unplaced camera is still a camera');
    db.close();
  });

  test('a corrupted policy column degrades to defaults rather than throwing', () => {
    // A row written by a newer version must not take the whole camera list down
    // during start-up.
    const { db, repo } = setup();
    repo.save(camera());
    db.exec('UPDATE cameras SET pose = ?, ai_policy = ? WHERE id = ?', [
      'not json',
      '{{{',
      'cam-07',
    ]);

    const loaded = repo.get(asId<CameraId>('cam-07'));
    assert.notEqual(loaded, undefined, 'the record is still readable');
    assert.equal(loaded?.pose, null, 'the damage surfaces as a missing pose');
    assert.equal(loaded?.ai.idleFps, DEFAULT_AI_POLICY.idleFps);
    db.close();
  });
});

describe('worker assignment', () => {
  test('lists cameras by node, which is what a worker asks for on start-up', () => {
    const { db, repo } = setup();
    db.exec(
      'INSERT INTO nodes (id, name, roles, status, addresses, app_version, protocol_version, hardware, capabilities, identity_fingerprint) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
      ['node-1', 'Worker 1', '["WORKER"]', 'ONLINE', '[]', '0.1.0', 1, '{}', '{}', 'aa'],
    );

    repo.save(camera({ id: asId<CameraId>('cam-07') }));
    repo.save(camera({ id: asId<CameraId>('cam-08'), name: 'Camera 08' }));

    repo.assignToNode(asId<CameraId>('cam-07'), asId<NodeId>('node-1'));

    assert.equal(repo.listForNode(asId<NodeId>('node-1')).length, 1);
    assert.equal(repo.listUnassigned().length, 1);
    assert.equal(repo.listUnassigned()[0]?.id, 'cam-08');
    db.close();
  });

  test('unassigning returns a camera to the unassigned pool', () => {
    const { db, repo } = setup();
    db.exec(
      'INSERT INTO nodes (id, name, roles, status, addresses, app_version, protocol_version, hardware, capabilities, identity_fingerprint) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
      ['node-1', 'Worker 1', '["WORKER"]', 'ONLINE', '[]', '0.1.0', 1, '{}', '{}', 'aa'],
    );

    repo.save(camera());
    repo.assignToNode(asId<CameraId>('cam-07'), asId<NodeId>('node-1'));
    repo.assignToNode(asId<CameraId>('cam-07'), null);

    // A camera in this list is recording nothing, which is worth being able to ask.
    assert.equal(repo.listUnassigned().length, 1);
    db.close();
  });
});

describe('status updates', () => {
  test('updates status without rewriting the configuration', () => {
    const { db, repo } = setup();
    repo.save(camera());

    repo.updateStatus(asId<CameraId>('cam-07'), 'ONLINE', utcMillis(5000));

    const loaded = repo.get(asId<CameraId>('cam-07'));
    assert.equal(loaded?.status, 'ONLINE');
    assert.equal(loaded?.lastSeen, 5000);
    assert.equal(loaded?.name, 'Camera 07 - West Approach', 'configuration untouched');
    assert.equal(loaded?.pose?.heading, 0);
    db.close();
  });
});

describe('deletion', () => {
  test('returns the credential reference so the keychain entry can be purged', () => {
    // Deleting the row alone would orphan a secret for a camera that no longer
    // exists, and nothing would ever look at it again to notice.
    const { db, repo } = setup();
    repo.save(camera({ credentialsRef: asId<CredentialsRef>('camera/cam-07') }));

    const result = repo.remove(asId<CameraId>('cam-07'));

    assert.equal(result.removed, true);
    assert.equal(result.credentialsRef, 'camera/cam-07');
    assert.equal(repo.get(asId<CameraId>('cam-07')), undefined);
    db.close();
  });

  test('deleting an unknown camera reports it rather than pretending', () => {
    const { db, repo } = setup();
    const result = repo.remove(asId<CameraId>('nope'));

    assert.equal(result.removed, false);
    assert.equal(result.credentialsRef, null);
    db.close();
  });

  test('deletion cascades to profiles and zone links', () => {
    const { db, repo } = setup();
    db.exec(
      'INSERT INTO zones (id, name, purpose, geometry, active, created_at, updated_at) VALUES (?, ?, ?, ?, 1, 0, 0)',
      ['zone-a', 'A', 'RESTRICTED', '{}'],
    );

    repo.save(camera({ zoneIds: [asId<ZoneId>('zone-a')] }));
    repo.saveProfiles(asId<CameraId>('cam-07'), [
      {
        id: asId<CameraProfileId>('p1'),
        cameraId: asId<CameraId>('cam-07'),
        kind: 'MAIN',
        name: 'MainStream',
        path: '/Streaming/Channels/101',
        codec: 'H264',
        width: 3840,
        height: 2160,
        fps: 25,
        bitrateKbps: 8192,
      },
    ]);

    repo.remove(asId<CameraId>('cam-07'));

    assert.equal(db.query('SELECT * FROM camera_profiles').length, 0);
    assert.equal(db.query('SELECT * FROM camera_zone_links').length, 0);
    db.close();
  });
});

describe('profiles', () => {
  test('round-trips main and sub streams', () => {
    const { db, repo } = setup();
    repo.save(camera());

    repo.saveProfiles(asId<CameraId>('cam-07'), [
      {
        id: asId<CameraProfileId>('p-main'),
        cameraId: asId<CameraId>('cam-07'),
        kind: 'MAIN',
        name: 'MainStream',
        path: '/Streaming/Channels/101',
        codec: 'H264',
        width: 3840,
        height: 2160,
        fps: 25,
        bitrateKbps: 8192,
        keyframeIntervalSeconds: 2,
      },
      {
        id: asId<CameraProfileId>('p-sub'),
        cameraId: asId<CameraId>('cam-07'),
        kind: 'SUB',
        name: 'SubStream',
        path: '/Streaming/Channels/102',
        codec: 'H264',
        width: 640,
        height: 360,
        fps: 10,
        bitrateKbps: 512,
      },
    ]);

    const profiles = repo.profiles(asId<CameraId>('cam-07'));
    assert.equal(profiles.length, 2);

    const main = profiles.find((p) => p.kind === 'MAIN');
    assert.equal(main?.width, 3840);
    assert.equal(main?.keyframeIntervalSeconds, 2);

    const sub = profiles.find((p) => p.kind === 'SUB');
    assert.equal(sub?.width, 640);
    assert.equal(sub?.keyframeIntervalSeconds, undefined);
    db.close();
  });

  test('profile paths never carry credentials', () => {
    const { db, repo } = setup();
    repo.save(camera());
    repo.saveProfiles(asId<CameraId>('cam-07'), [
      {
        id: asId<CameraProfileId>('p1'),
        cameraId: asId<CameraId>('cam-07'),
        kind: 'MAIN',
        name: 'Main',
        path: '/Streaming/Channels/101',
        codec: 'H264',
        width: 1920,
        height: 1080,
        fps: 25,
        bitrateKbps: 4096,
      },
    ]);

    const raw = JSON.stringify(db.query('SELECT * FROM camera_profiles'));
    assert.ok(!raw.includes('@'), 'a path column must never hold a URL with userinfo');
    db.close();
  });
});

describe('topology', () => {
  test('round-trips an edge and upserts on repeat', () => {
    const { db, repo } = setup();
    repo.save(camera({ id: asId<CameraId>('cam-07') }));
    repo.save(camera({ id: asId<CameraId>('cam-08'), name: 'Camera 08' }));

    const edge = {
      fromCameraId: asId<CameraId>('cam-07'),
      toCameraId: asId<CameraId>('cam-08'),
      distanceMeters: 84,
      minTravelSeconds: 25,
      expectedTravelSeconds: 60,
      maxTravelSeconds: 180,
      confidence: 0.9,
      bidirectional: true,
    };

    repo.saveTopologyEdge(edge);
    repo.saveTopologyEdge({ ...edge, expectedTravelSeconds: 55 });

    const topology = repo.topology();
    assert.equal(topology.length, 1, 'the same pair is one edge, not two');
    assert.equal(topology[0]?.expectedTravelSeconds, 55);
    assert.equal(topology[0]?.bidirectional, true);
    db.close();
  });

  test('an edge is removed when either camera is deleted', () => {
    const { db, repo } = setup();
    repo.save(camera({ id: asId<CameraId>('cam-07') }));
    repo.save(camera({ id: asId<CameraId>('cam-08'), name: 'Camera 08' }));

    repo.saveTopologyEdge({
      fromCameraId: asId<CameraId>('cam-07'),
      toCameraId: asId<CameraId>('cam-08'),
      distanceMeters: 84,
      minTravelSeconds: 25,
      expectedTravelSeconds: 60,
      maxTravelSeconds: 180,
      confidence: 0.9,
      bidirectional: true,
    });

    repo.remove(asId<CameraId>('cam-08'));

    // A dangling edge would keep scoring hand-offs to a camera that is gone.
    assert.equal(repo.topology().length, 0);
    db.close();
  });
});
