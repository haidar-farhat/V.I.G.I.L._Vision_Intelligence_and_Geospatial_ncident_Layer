import { test, describe, after } from 'node:test';
import assert from 'node:assert/strict';
import { createConnection } from 'node:net';
import type { Socket } from 'node:net';
import { asId, utcMillis } from '@sentinel/shared-types';
import type { CameraId, NodeId, UserId } from '@sentinel/shared-types';
import { secret } from '@sentinel/security';
import { MigrationRunner, MIGRATIONS, openMemoryDatabase } from '@sentinel/database';
import type { SqlDriver } from '@sentinel/database';
import { CameraRepository, MapPackageRepository } from '@sentinel/database';
import {
  MessageKind,
  Opcode,
  decodeFrame,
  encodeFrame,
  envelope,
  generateClientKey,
  parseEnvelope,
  serialiseEnvelope,
  verifyAcceptKey,
} from '@sentinel/protocol';
import { SCRYPT_COST_FOR_TESTS as COST, hashPassword } from '../src/auth.ts';
import type { StoredUser } from '../src/auth.ts';
import { startApiServer } from '../src/server.ts';
import type { RunningServer } from '../src/server.ts';
import type { AuditEntry } from '../src/router.ts';

const PASSWORD = 'correct-horse-battery-staple';
const NODE = asId<NodeId>('node-control');

const running: RunningServer[] = [];
const databases: SqlDriver[] = [];
const sockets: Socket[] = [];

after(async () => {
  for (const socket of sockets) socket.destroy();
  for (const server of running) await server.close();
  for (const database of databases) database.close();
});

type Harness = {
  readonly server: RunningServer;
  readonly database: SqlDriver;
  readonly audit: AuditEntry[];
  request(
    method: string,
    path: string,
    options?: { token?: string; body?: unknown },
  ): Promise<{ status: number; body: any; headers: Headers }>;
};

const users = (): Record<string, StoredUser> => ({
  admin: {
    id: asId<UserId>('user-admin'),
    username: 'admin',
    roles: ['ADMIN'],
    active: true,
    passwordHash: hashPassword(secret(PASSWORD), COST),
  },
  viewer: {
    id: asId<UserId>('user-viewer'),
    username: 'viewer',
    roles: ['VIEWER'],
    active: true,
    passwordHash: hashPassword(secret(PASSWORD), COST),
  },
});

const harness = async (): Promise<Harness> => {
  const database = openMemoryDatabase();
  new MigrationRunner(database, MIGRATIONS).migrate(1000);
  databases.push(database);

  const accounts = users();
  const audit: AuditEntry[] = [];

  const server = await startApiServer({
    nodeId: NODE,
    database,
    findUser: (username) => accounts[username],
    scryptCost: COST,
    onAudit: (entry) => audit.push(entry),
  });
  running.push(server);

  return {
    server,
    database,
    audit,
    request: async (method, path, options = {}) => {
      const response = await fetch(`http://127.0.0.1:${server.port}${path}`, {
        method,
        headers: {
          'Content-Type': 'application/json',
          ...(options.token === undefined ? {} : { Authorization: `Bearer ${options.token}` }),
        },
        ...(options.body === undefined ? {} : { body: JSON.stringify(options.body) }),
      });

      const text = await response.text();
      return {
        status: response.status,
        body: text === '' ? undefined : JSON.parse(text),
        headers: response.headers,
      };
    },
  };
};

const login = async (h: Harness, username = 'admin'): Promise<string> => {
  const result = await h.request('POST', '/api/auth/login', {
    body: { username, password: PASSWORD },
  });
  assert.equal(result.status, 200, `login failed: ${JSON.stringify(result.body)}`);
  return result.body.token as string;
};

describe('the server as a whole', () => {
  test('health is reachable before anyone has logged in', async () => {
    // The desktop shell polls this to decide whether the service came up at all.
    const h = await harness();
    const result = await h.request('GET', '/api/health');

    assert.equal(result.status, 200);
    assert.equal(result.body.status, 'ok');
  });

  test('binds loopback, not every interface', async () => {
    // A control plane listening on every interface should be a deliberate choice.
    const h = await harness();
    assert.equal(h.server.host, '127.0.0.1');
  });

  test('every response carries a request id', async () => {
    const h = await harness();
    const result = await h.request('GET', '/api/health');

    // The same id appears in the audit record and the log line, so "the export
    // that failed yesterday" is answerable.
    assert.ok((result.headers.get('x-request-id') ?? '').length > 0);
  });

  test('sets the headers a local API should', async () => {
    const h = await harness();
    const result = await h.request('GET', '/api/health');

    assert.equal(result.headers.get('x-content-type-options'), 'nosniff');
    assert.equal(result.headers.get('x-frame-options'), 'DENY');
    assert.equal(result.headers.get('cache-control'), 'no-store');
  });
});

describe('authentication over HTTP', () => {
  test('issues a token for correct credentials', async () => {
    const h = await harness();
    const result = await h.request('POST', '/api/auth/login', {
      body: { username: 'admin', password: PASSWORD },
    });

    assert.equal(result.status, 200);
    assert.ok(result.body.token.length > 20);
    assert.deepEqual(result.body.roles, ['ADMIN']);
    assert.ok(!JSON.stringify(result.body).includes(PASSWORD));
  });

  test('never says which half was wrong', async () => {
    const h = await harness();

    const badPassword = await h.request('POST', '/api/auth/login', {
      body: { username: 'admin', password: 'wrong' },
    });
    const badUser = await h.request('POST', '/api/auth/login', {
      body: { username: 'nobody', password: PASSWORD },
    });

    assert.equal(badPassword.status, 401);
    assert.equal(badUser.status, 401);
    assert.deepEqual(badPassword.body, badUser.body, 'the two must be indistinguishable');
    assert.match(badPassword.body.error.message, /username or password is incorrect/);
  });

  test('rejects a malformed login body', async () => {
    const h = await harness();
    const result = await h.request('POST', '/api/auth/login', { body: { username: 42 } });

    assert.equal(result.status, 400);
    assert.equal(result.body.error.code, 'BAD_REQUEST');
  });

  test('an unauthenticated call to a protected route is refused', async () => {
    const h = await harness();
    const result = await h.request('GET', '/api/cameras');

    assert.equal(result.status, 401);
    assert.equal(result.body.error.code, 'UNAUTHENTICATED');
  });

  test('a garbage token is refused, not treated as absent', async () => {
    const h = await harness();
    const result = await h.request('GET', '/api/cameras', { token: 'not-a-real-token' });
    assert.equal(result.status, 401);
  });

  test('logout revokes the session immediately', async () => {
    const h = await harness();
    const token = await login(h);

    assert.equal((await h.request('GET', '/api/cameras', { token })).status, 200);
    assert.equal((await h.request('POST', '/api/auth/logout', { token })).status, 204);
    assert.equal((await h.request('GET', '/api/cameras', { token })).status, 401);
  });
});

describe('authorization', () => {
  test('a viewer may read cameras', async () => {
    const h = await harness();
    const token = await login(h, 'viewer');

    assert.equal((await h.request('GET', '/api/cameras', { token })).status, 200);
  });

  test('a viewer may not delete a camera', async () => {
    const h = await harness();
    const token = await login(h, 'viewer');

    const result = await h.request('DELETE', '/api/cameras/cam-07', { token });
    assert.equal(result.status, 403);
    assert.equal(result.body.error.code, 'FORBIDDEN');
  });

  test('a refusal names what was required, not what the caller holds', async () => {
    // Enumerating a user's permissions to an unauthorised caller is itself a leak.
    const h = await harness();
    const token = await login(h, 'viewer');

    const result = await h.request('DELETE', '/api/cameras/cam-07', { token });
    assert.match(result.body.error.message, /camera:delete/);
    assert.ok(!result.body.error.message.includes('camera:view'));
  });

  test('an admin may delete', async () => {
    const h = await harness();
    const token = await login(h);

    const repo = new CameraRepository(h.database);
    repo.save(camera());

    const result = await h.request('DELETE', '/api/cameras/cam-07', { token });
    assert.equal(result.status, 200);
    assert.equal(result.body.removed, true);
  });
});

describe('routing', () => {
  test('an unknown path is 404', async () => {
    const h = await harness();
    assert.equal((await h.request('GET', '/api/nonexistent')).status, 404);
  });

  test('a known path under the wrong verb is 405, not 404', async () => {
    // A client that used the wrong method deserves to be told so.
    const h = await harness();
    const result = await h.request('DELETE', '/api/health');

    assert.equal(result.status, 405);
    assert.equal(result.body.error.code, 'METHOD_NOT_ALLOWED');
  });

  test('a path parameter reaches the handler decoded exactly once', async () => {
    const h = await harness();
    const token = await login(h);

    // Double-decoding is how %252e%252e becomes a traversal on the second pass.
    const result = await h.request('GET', '/api/cameras/%2e%2e%2fetc', { token });
    assert.equal(result.status, 404, 'no such camera, rather than a traversal');
  });

  test('rejects an oversized request body', async () => {
    const h = await harness();
    const response = await fetch(`http://127.0.0.1:${h.server.port}/api/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: 'a', password: 'x'.repeat(2 * 1024 * 1024) }),
    }).catch(() => null);

    // Either refused with 413 or the socket was destroyed mid-body; both are the
    // cap working, and which one happens depends on timing.
    assert.ok(response === null || response.status === 413, `got ${response?.status}`);
  });
});

describe('audit', () => {
  test('records a successful privileged call', async () => {
    const h = await harness();
    const token = await login(h);
    new CameraRepository(h.database).save(camera());

    await h.request('DELETE', '/api/cameras/cam-07', { token });

    const entry = h.audit.find((e) => e.method === 'DELETE');
    assert.notEqual(entry, undefined);
    assert.equal(entry?.username, 'admin');
    assert.equal(entry?.permission, 'camera:delete');
    assert.equal(entry?.outcome, 'SUCCESS');
  });

  test('records a refusal, not just a success', async () => {
    const h = await harness();
    const token = await login(h, 'viewer');

    await h.request('DELETE', '/api/cameras/cam-07', { token });

    const entry = h.audit.find((e) => e.method === 'DELETE');
    assert.equal(entry?.outcome, 'DENIED');
    assert.equal(entry?.status, 403);
    assert.equal(entry?.username, 'viewer');
  });

  test('records an unauthenticated attempt', async () => {
    const h = await harness();
    await h.request('DELETE', '/api/cameras/cam-07');

    const entry = h.audit.find((e) => e.method === 'DELETE');
    assert.equal(entry?.outcome, 'DENIED');
    assert.equal(entry?.username, null);
    assert.equal(entry?.status, 401);
  });

  test('carries the request id that the response returned', async () => {
    const h = await harness();
    const token = await login(h);

    const result = await h.request('GET', '/api/cameras', { token });
    const entry = h.audit.find((e) => e.path === '/api/cameras');

    assert.equal(entry?.requestId, result.headers.get('x-request-id'));
  });
});

describe('data routes', () => {
  test('lists cameras from the database', async () => {
    const h = await harness();
    const token = await login(h);
    new CameraRepository(h.database).save(camera());

    const result = await h.request('GET', '/api/cameras', { token });
    assert.equal(result.body.cameras.length, 1);
    assert.equal(result.body.cameras[0].name, 'Camera 07 - West Approach');
  });

  test('a camera response carries no credential', async () => {
    const h = await harness();
    const token = await login(h);
    new CameraRepository(h.database).save(camera({ credentialsRef: 'camera/cam-07' as never }));

    const result = await h.request('GET', '/api/cameras/cam-07', { token });
    const serialised = JSON.stringify(result.body);

    // The reference is present; nothing resembling a secret is.
    assert.match(serialised, /camera\/cam-07/);
    assert.ok(!serialised.includes(PASSWORD));
    assert.ok(!serialised.includes('@192.168'));
  });

  test('reports installed map packages and the default', async () => {
    const h = await harness();
    const token = await login(h);

    new MapPackageRepository(h.database).install({
      id: asId('map-site'),
      name: 'Site',
      region: 'Site',
      bounds: { minLat: 33, minLon: 35, maxLat: 34, maxLon: 36 },
      minZoom: 0,
      maxZoom: 16,
      tileType: 'vector (MVT)',
      sizeBytes: 1000,
      sha256: 'a'.repeat(64),
      relativePath: 'site/region.pmtiles',
      isDefault: false,
      importedAt: utcMillis(0),
    });

    const result = await h.request('GET', '/api/maps', { token });
    assert.equal(result.body.packages.length, 1);
    assert.equal(result.body.default.id, 'map-site', 'the first package becomes the default');
  });

  test('states the offline guarantee plainly', async () => {
    const h = await harness();
    const token = await login(h);

    const result = await h.request('GET', '/api/diagnostics/network', { token });
    assert.equal(result.body.wan, 'BLOCKED');
    assert.equal(result.body.internetDependency, 'NONE');
    assert.equal(result.body.telemetry, 'DISABLED');
  });
});

// -------------------------------------------------------------- realtime

/** Complete a real WebSocket handshake against the running server. */
const openSocket = async (
  server: RunningServer,
  token: string,
): Promise<{ socket: Socket; frames: { opcode: Opcode; payload: Buffer }[] }> => {
  const key = generateClientKey();
  const socket = createConnection({ port: server.port, host: '127.0.0.1' });
  sockets.push(socket);

  await new Promise<void>((resolve) => socket.once('connect', () => resolve()));

  socket.write(
    `GET /ws?token=${encodeURIComponent(token)} HTTP/1.1\r\n` +
      `Host: 127.0.0.1:${server.port}\r\n` +
      'Upgrade: websocket\r\nConnection: Upgrade\r\n' +
      `Sec-WebSocket-Key: ${key}\r\nSec-WebSocket-Version: 13\r\n` +
      'Sec-WebSocket-Protocol: sentinel.v1\r\n\r\n',
  );

  const frames: { opcode: Opcode; payload: Buffer }[] = [];
  let buffer = Buffer.alloc(0);
  let upgraded = false;
  let status = 0;

  socket.on('data', (chunk: Buffer) => {
    buffer = Buffer.concat([buffer, chunk]);

    if (!upgraded) {
      const end = buffer.indexOf('\r\n\r\n');
      if (end === -1) return;

      const head = buffer.subarray(0, end).toString('utf8');
      status = Number.parseInt(/HTTP\/1\.1 (\d+)/.exec(head)?.[1] ?? '0', 10);

      if (status === 101) {
        const accept = /Sec-WebSocket-Accept: (.+)/i.exec(head)?.[1]?.trim() ?? '';
        assert.ok(verifyAcceptKey(key, accept), 'server did not perform the transformation');
        upgraded = true;
      }
      buffer = buffer.subarray(end + 4);
      if (!upgraded) return;
    }

    for (;;) {
      const decoded = decodeFrame(buffer, false);
      if (decoded === null) return;
      buffer = buffer.subarray(decoded.consumed);
      frames.push({ opcode: decoded.frame.opcode, payload: decoded.frame.payload });
    }
  });

  socket.on('error', () => {});
  await new Promise((resolve) => setTimeout(resolve, 120));

  return { socket, frames };
};

const settle = (): Promise<void> => new Promise((resolve) => setTimeout(resolve, 120));

describe('realtime over the running server', () => {
  test('a session token completes the upgrade and receives a welcome', async () => {
    const h = await harness();
    const token = await login(h);

    const { frames } = await openSocket(h.server, token);
    const messages = frames
      .filter((f) => f.opcode === Opcode.Text)
      .map((f) => parseEnvelope(f.payload.toString('utf8')));

    assert.equal(messages[0]?.kind, MessageKind.Welcome);
    assert.equal(h.server.hub.connectionCount, 1);
  });

  test('an upgrade without a session is refused before the socket is adopted', async () => {
    // Upgrading first and authenticating afterwards leaves a window in which an
    // unauthenticated socket is attached to the hub.
    const h = await harness();

    const socket = createConnection({ port: h.server.port, host: '127.0.0.1' });
    sockets.push(socket);
    await new Promise<void>((resolve) => socket.once('connect', () => resolve()));

    let response = '';
    socket.on('data', (chunk: Buffer) => {
      response += chunk.toString('utf8');
    });
    socket.on('error', () => {});

    socket.write(
      'GET /ws HTTP/1.1\r\nHost: x\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n' +
        `Sec-WebSocket-Key: ${generateClientKey()}\r\nSec-WebSocket-Version: 13\r\n\r\n`,
    );
    await settle();

    assert.match(response, /^HTTP\/1\.1 401/);
    assert.match(response, /valid session is required/);
    assert.equal(h.server.hub.connectionCount, 0, 'the hub never saw it');
  });

  test('delivers a published event to a subscriber', async () => {
    const h = await harness();
    const token = await login(h);
    const { socket, frames } = await openSocket(h.server, token);

    socket.write(
      encodeFrame(
        Opcode.Text,
        Buffer.from(
          serialiseEnvelope(
            envelope(MessageKind.Subscribe, { channels: ['events'] }, asId<NodeId>('desktop')),
          ),
        ),
        { mask: true },
      ),
    );
    await settle();

    assert.equal(h.server.hub.publish(MessageKind.Event, { id: 'e1', severity: 'HIGH' }), 1);
    await settle();

    const events = frames
      .filter((f) => f.opcode === Opcode.Text)
      .map((f) => parseEnvelope(f.payload.toString('utf8')))
      .filter((m) => m.kind === MessageKind.Event);

    assert.equal(events.length, 1);
    assert.deepEqual(events[0]?.payload, { id: 'e1', severity: 'HIGH' });
  });

  test('logging out closes the operator live stream too', async () => {
    // A revoked session must not leave a subscription running.
    const h = await harness();
    const token = await login(h);
    await openSocket(h.server, token);

    assert.equal(h.server.hub.connectionCount, 1);
    await h.request('POST', '/api/auth/logout', { token });
    await settle();

    assert.equal(h.server.hub.connectionCount, 0);
  });

  test('diagnostics report live connection counts', async () => {
    const h = await harness();
    const token = await login(h);
    await openSocket(h.server, token);

    const result = await h.request('GET', '/api/diagnostics/realtime', { token });
    assert.equal(result.body.connections, 1);
    assert.ok(result.body.activeSessions >= 1);
  });
});

const camera = (overrides: Record<string, unknown> = {}) =>
  ({
    id: asId<CameraId>('cam-07'),
    name: 'Camera 07 - West Approach',
    protocol: 'RTSP',
    host: '192.168.1.50',
    port: 554,
    workerNodeId: null,
    locationId: null,
    pose: null,
    intrinsics: null,
    zoneIds: [],
    ai: {
      enabled: true,
      idleFps: 2,
      activeFps: 12,
      classes: ['person'],
      confidenceThreshold: 0.5,
      trackingEnabled: true,
      eventGenerationEnabled: true,
    },
    recording: {
      mode: 'EVENT',
      segmentSeconds: 60,
      retentionDays: 7,
      preEventSeconds: 10,
      postEventSeconds: 30,
    },
    status: 'UNKNOWN',
    lastSeen: null,
    ptzSupported: false,
    createdAt: utcMillis(1000),
    updatedAt: utcMillis(1000),
    ...overrides,
  }) as never;
