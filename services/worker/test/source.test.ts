import { test, describe, after } from 'node:test';
import assert from 'node:assert/strict';
import { secret } from '@sentinel/security';
import { asId } from '@sentinel/shared-types';
import type { CameraId, CameraStatus } from '@sentinel/shared-types';
import { startMockRtspServer } from '@sentinel/test-utils';
import type { MockRtspServer } from '@sentinel/test-utils';
import { VideoSource } from '../src/ingest/source.ts';

const PASSWORD = 'correct-horse-battery';
const CAMERA = asId<CameraId>('cam-07');

const servers: MockRtspServer[] = [];
const sources: VideoSource[] = [];

const spawnServer = async (behaviour = {}): Promise<MockRtspServer> => {
  const server = await startMockRtspServer({ password: PASSWORD, ...behaviour });
  servers.push(server);
  return server;
};

const spawnSource = (options: ConstructorParameters<typeof VideoSource>[0]): VideoSource => {
  const source = new VideoSource(options);
  sources.push(source);
  return source;
};

after(async () => {
  for (const source of sources) await source.stop();
  for (const server of servers) await server.close();
});

/** Poll until a predicate holds, so tests wait on state rather than on a clock. */
const until = async (predicate: () => boolean, timeoutMillis = 5000): Promise<boolean> => {
  const deadline = Date.now() + timeoutMillis;
  while (Date.now() < deadline) {
    if (predicate()) return true;
    await new Promise((resolve) => setTimeout(resolve, 20));
  }
  return predicate();
};

describe('connecting', () => {
  test('reaches ONLINE and reports the endpoint it is streaming from', async () => {
    const server = await spawnServer();
    const transitions: CameraStatus[] = [];

    const source = spawnSource({
      cameraId: CAMERA,
      host: '127.0.0.1',
      port: server.port,
      path: '/live',
      username: 'admin',
      password: secret(PASSWORD),
      onStatusChange: (status) => transitions.push(status),
    });

    await source.start();
    assert.ok(await until(() => source.state.status === 'ONLINE'), 'never reached ONLINE');

    assert.equal(source.state.running, true);
    assert.match(source.state.detail, /Streaming from rtsp:\/\/127\.0\.0\.1/);
    assert.ok(transitions.includes('ONLINE'));

    await source.stop();
  });

  test('the reported endpoint carries no credential', async () => {
    const server = await spawnServer();

    const source = spawnSource({
      cameraId: CAMERA,
      host: '127.0.0.1',
      port: server.port,
      path: '/live',
      username: 'admin',
      password: secret(PASSWORD),
    });

    await source.start();
    await until(() => source.state.status === 'ONLINE');

    // state.detail reaches the UI and the logs.
    const serialised = JSON.stringify(source.state);
    assert.ok(!serialised.includes(PASSWORD), `credential leaked into state: ${serialised}`);
    assert.ok(!serialised.includes('@127.0.0.1'));

    await source.stop();
  });

  test('exposes the negotiated session', async () => {
    const server = await spawnServer({ supportsGetParameter: true, sessionTimeoutSeconds: 30 });

    const source = spawnSource({
      cameraId: CAMERA,
      host: '127.0.0.1',
      port: server.port,
      path: '/live',
      username: 'admin',
      password: secret(PASSWORD),
    });

    await source.start();
    await until(() => source.state.status === 'ONLINE');

    assert.equal(source.session?.timeoutSeconds, 30);
    assert.equal(source.session?.encoding, 'H264');
    await source.stop();
  });

  test('start resolves after the first attempt, so cameras start in parallel', async () => {
    // A caller bringing up twenty cameras must not serialise on the slowest.
    const server = await spawnServer({ silent: true });

    const source = spawnSource({
      cameraId: CAMERA,
      host: '127.0.0.1',
      port: server.port,
      path: '/live',
      timeoutMillis: 300,
      initialBackoffMillis: 10_000,
    });

    const began = Date.now();
    await source.start();
    const elapsed = Date.now() - began;

    assert.ok(elapsed < 3000, `start() blocked for ${elapsed} ms`);
    await source.stop();
  });
});

describe('recovery', () => {
  test('retries a transient failure with backoff rather than giving up', async () => {
    // Port 1 is reliably refused. The source must keep trying: a camera may come
    // back in an hour, and a worker that stopped would need a human to notice.
    const source = spawnSource({
      cameraId: CAMERA,
      host: '127.0.0.1',
      port: 1,
      path: '/live',
      timeoutMillis: 500,
      initialBackoffMillis: 20,
      maxBackoffMillis: 60,
    });

    await source.start();
    assert.ok(await until(() => /Retrying in/.test(source.state.detail)), 'never announced a retry');

    assert.equal(source.state.status, 'OFFLINE');
    assert.equal(source.state.running, true, 'still supervising');

    // And it is still trying some time later, not silently stopped.
    await new Promise((resolve) => setTimeout(resolve, 200));
    assert.equal(source.state.running, true);

    await source.stop();
  });

  test('a wrong password stops, because retrying locks the account', async () => {
    const server = await spawnServer();

    const source = spawnSource({
      cameraId: CAMERA,
      host: '127.0.0.1',
      port: server.port,
      path: '/live',
      username: 'admin',
      password: secret('wrong-password'),
      initialBackoffMillis: 20,
    });

    await source.start();
    assert.ok(await until(() => source.state.lastErrorCode === 'AUTH'), 'never classified as auth');

    assert.equal(source.state.status, 'OFFLINE');
    assert.match(source.state.detail, /needs a configuration change/);

    // The number of authentication attempts must not keep climbing.
    const attemptsAfterFirst = server.authorizations.length;
    await new Promise((resolve) => setTimeout(resolve, 250));
    assert.equal(
      server.authorizations.length,
      attemptsAfterFirst,
      'kept retrying a credential the camera already rejected',
    );

    await source.stop();
  });

  test('an unusable stream stops rather than retrying forever', async () => {
    // No amount of reconnecting will grow a video track on an audio-only camera.
    const server = await spawnServer({
      sdp: ['v=0', 's=Audio', 'm=audio 0 RTP/AVP 97', 'a=rtpmap:97 PCMU/8000'].join('\r\n'),
    });

    const source = spawnSource({
      cameraId: CAMERA,
      host: '127.0.0.1',
      port: server.port,
      path: '/live',
      username: 'admin',
      password: secret(PASSWORD),
      initialBackoffMillis: 20,
    });

    await source.start();
    assert.ok(
      await until(() => source.state.lastErrorCode === 'RTSP_NO_VIDEO_TRACK'),
      'never classified the failure',
    );
    assert.match(source.state.detail, /needs a change on the camera/);

    await source.stop();
  });

  test('reports the failure that is actually blocking it', async () => {
    const source = spawnSource({
      cameraId: CAMERA,
      host: '127.0.0.1',
      port: 1,
      path: '/live',
      timeoutMillis: 400,
      initialBackoffMillis: 20,
    });

    await source.start();
    await until(() => source.state.lastErrorCode !== null);

    // Not a generic "disconnected": the code names the stage that failed.
    assert.equal(source.state.lastErrorCode, 'RTSP_CONNECT_FAILED');
    await source.stop();
  });
});

describe('honest state', () => {
  test('a flapping camera is DEGRADED, not ONLINE', async () => {
    // At any given instant it is connected, so a naive implementation shows a
    // green dot while the camera delivers nothing usable.
    const server = await spawnServer();

    const source = spawnSource({
      cameraId: CAMERA,
      host: '127.0.0.1',
      port: server.port,
      path: '/live',
      username: 'admin',
      password: secret(PASSWORD),
      flappingThreshold: 2,
      flappingWindowMillis: 60_000,
      initialBackoffMillis: 5,
      maxBackoffMillis: 20,
    });

    await source.start();
    await until(() => source.state.status === 'ONLINE');

    // Force repeated reconnections by cycling the source.
    for (let i = 0; i < 3; i += 1) {
      await source.stop();
      await source.start();
      await until(() => source.state.status !== 'UNKNOWN');
    }

    assert.ok(
      await until(() => source.state.status === 'DEGRADED'),
      `expected DEGRADED after repeated reconnects, got ${source.state.status}`,
    );
    assert.match(source.state.detail, /reconnected \d+ times/);

    await source.stop();
  });

  test('stop is clean and idempotent', async () => {
    const server = await spawnServer();

    const source = spawnSource({
      cameraId: CAMERA,
      host: '127.0.0.1',
      port: server.port,
      path: '/live',
      username: 'admin',
      password: secret(PASSWORD),
    });

    await source.start();
    await until(() => source.state.status === 'ONLINE');

    await source.stop();
    await source.stop();

    assert.equal(source.state.running, false);
    assert.equal(source.state.status, 'UNKNOWN');
  });

  test('stopping does not wait out a long backoff', async () => {
    const source = spawnSource({
      cameraId: CAMERA,
      host: '127.0.0.1',
      port: 1,
      path: '/live',
      timeoutMillis: 300,
      initialBackoffMillis: 30_000,
    });

    await source.start();
    await until(() => /Retrying in/.test(source.state.detail));

    const began = Date.now();
    await source.stop();
    const elapsed = Date.now() - began;

    assert.ok(elapsed < 2000, `stop() waited ${elapsed} ms for the backoff to expire`);
  });
});

describe('health reporting', () => {
  test('reports a health snapshot carrying the blocking error code', async () => {
    const source = spawnSource({
      cameraId: CAMERA,
      host: '127.0.0.1',
      port: 1,
      path: '/live',
      timeoutMillis: 300,
      initialBackoffMillis: 20,
    });

    await source.start();
    await until(() => source.state.lastErrorCode !== null);

    const health = source.health();
    assert.equal(health.cameraId, CAMERA);
    assert.equal(health.status, 'OFFLINE');
    assert.equal(health.lastErrorCode, 'RTSP_CONNECT_FAILED');
    assert.ok(health.observedAt > 0);

    await source.stop();
  });

  test('health carries no credential', async () => {
    const server = await spawnServer();

    const source = spawnSource({
      cameraId: CAMERA,
      host: '127.0.0.1',
      port: server.port,
      path: '/live',
      username: 'admin',
      password: secret(PASSWORD),
    });

    await source.start();
    await until(() => source.state.status === 'ONLINE');

    // Health is sent to the control node in every heartbeat.
    assert.ok(!JSON.stringify(source.health()).includes(PASSWORD));
    await source.stop();
  });
});

describe('decoder boundary', () => {
  test('starts the decoder once a session is established', async () => {
    const server = await spawnServer();

    let started = 0;
    let stopped = 0;
    const decoder = {
      start: async (): Promise<void> => {
        started += 1;
      },
      stop: async (): Promise<void> => {
        stopped += 1;
      },
      framesDecoded: 0,
      decodeErrors: 0,
    };

    const source = spawnSource({
      cameraId: CAMERA,
      host: '127.0.0.1',
      port: server.port,
      path: '/live',
      username: 'admin',
      password: secret(PASSWORD),
      decoder,
      onFrame: () => {},
    });

    await source.start();
    await until(() => source.state.status === 'ONLINE');

    assert.equal(started, 1, 'the decoder starts only after RTSP negotiated a session');

    await source.stop();
    assert.equal(stopped, 1);
  });

  test('the decoder is never started when the session fails', async () => {
    // Spawning a decoder against a stream that does not exist wastes a process
    // and produces a confusing second failure.
    let started = 0;
    const decoder = {
      start: async (): Promise<void> => {
        started += 1;
      },
      stop: async (): Promise<void> => {},
      framesDecoded: 0,
      decodeErrors: 0,
    };

    const source = spawnSource({
      cameraId: CAMERA,
      host: '127.0.0.1',
      port: 1,
      path: '/live',
      timeoutMillis: 300,
      initialBackoffMillis: 20,
      decoder,
      onFrame: () => {},
    });

    await source.start();
    await until(() => source.state.lastErrorCode !== null);

    assert.equal(started, 0);
    await source.stop();
  });

  test('decode errors surface in health', async () => {
    const server = await spawnServer();

    const decoder = {
      start: async (): Promise<void> => {},
      stop: async (): Promise<void> => {},
      framesDecoded: 120,
      decodeErrors: 7,
    };

    const source = spawnSource({
      cameraId: CAMERA,
      host: '127.0.0.1',
      port: server.port,
      path: '/live',
      username: 'admin',
      password: secret(PASSWORD),
      decoder,
      onFrame: () => {},
    });

    await source.start();
    await until(() => source.state.status === 'ONLINE');

    assert.equal(source.health().decodeErrors, 7);
    await source.stop();
  });
});
