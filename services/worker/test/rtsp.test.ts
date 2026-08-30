import { test, describe, after } from 'node:test';
import assert from 'node:assert/strict';
import { secret } from '@sentinel/security';
import { DEFAULT_SDP, startMockRtspServer } from '@sentinel/test-utils';
import type { MockRtspServer } from '@sentinel/test-utils';
import { parseSdp, resolveControlUrl, selectVideoTrack } from '../src/rtsp/sdp.ts';
import {
  DigestSession,
  buildBasicHeader,
  buildDigestHeader,
  parseAuthChallenges,
} from '../src/rtsp/auth.ts';
import { RtspClient, RtspError, parseResponse } from '../src/rtsp/client.ts';

/** The fixture credential. It must never appear in anything the client emits. */
const PASSWORD = 'correct-horse-battery';

const servers: MockRtspServer[] = [];
const spawnServer = async (behaviour = {}): Promise<MockRtspServer> => {
  const server = await startMockRtspServer(behaviour);
  servers.push(server);
  return server;
};

after(async () => {
  for (const server of servers) await server.close();
});

// ---------------------------------------------------------------------- SDP

describe('SDP parsing', () => {
  test('extracts the video track from a typical camera description', () => {
    const sdp = parseSdp(DEFAULT_SDP);

    assert.equal(sdp.sessionName, 'Sentinel Test Camera');
    assert.equal(sdp.control, '*');
    assert.equal(sdp.media.length, 2);

    const video = selectVideoTrack(sdp);
    assert.notEqual(video, null);
    assert.equal(video?.encoding, 'H264');
    assert.equal(video?.clockRate, 90000);
    assert.equal(video?.control, 'trackID=1');
    assert.equal(video?.formatParameters['packetization-mode'], '1');
    assert.ok(video?.formatParameters['sprop-parameter-sets'] !== undefined);
  });

  test('prefers a codec the decoder supports over one it does not', () => {
    const sdp = parseSdp(
      [
        'v=0',
        'm=video 0 RTP/AVP 26',
        'a=rtpmap:26 VP8/90000',
        'a=control:trackID=1',
        'm=video 0 RTP/AVP 96',
        'a=rtpmap:96 H264/90000',
        'a=control:trackID=2',
      ].join('\r\n'),
    );

    assert.equal(selectVideoTrack(sdp)?.control, 'trackID=2', 'H264 must win over VP8');
  });

  test('reports no usable video rather than guessing', () => {
    const audioOnly = parseSdp(['v=0', 'm=audio 0 RTP/AVP 97', 'a=rtpmap:97 PCMU/8000'].join('\r\n'));
    assert.equal(selectVideoTrack(audioOnly), null);
  });

  test('a video track with no rtpmap is still attempted', () => {
    // Static payload types are implied by the RTP profile, and older cameras
    // rely on that. Refusing them would exclude working hardware.
    const sdp = parseSdp(['v=0', 'm=video 0 RTP/AVP 26', 'a=control:trackID=1'].join('\r\n'));
    assert.equal(selectVideoTrack(sdp)?.control, 'trackID=1');
  });

  test('survives malformed input and says what it could not read', () => {
    const sdp = parseSdp(
      [
        'v=0',
        'this line has no equals sign',
        'q=unknown type',
        'm=video 0 RTP/AVP 96',
        'a=rtpmap:garbage',
        'a=control:trackID=1',
      ].join('\r\n'),
    );

    // One bad attribute must not take a working camera offline.
    assert.equal(sdp.media.length, 1);
    assert.equal(sdp.media[0]?.control, 'trackID=1');
    assert.ok(sdp.warnings.length >= 2, 'what could not be parsed is reported, not discarded');
  });

  test('caps input so a hostile device cannot exhaust memory', () => {
    const huge = `v=0\r\n${'a=x:y\r\n'.repeat(5000)}`;
    const sdp = parseSdp(huge);
    assert.ok(sdp.warnings.some((w) => /more than \d+ lines/.test(w)));

    const enormous = `v=0\r\na=x:${'y'.repeat(200_000)}`;
    assert.ok(parseSdp(enormous).warnings.some((w) => /truncated/.test(w)));
  });

  test('handles an empty description without throwing', () => {
    const sdp = parseSdp('');
    assert.equal(sdp.media.length, 0);
    assert.equal(selectVideoTrack(sdp), null);
  });
});

describe('control URL resolution', () => {
  const base = 'rtsp://192.168.1.50:554/live';

  test('resolves a relative control attribute', () => {
    assert.equal(resolveControlUrl(base, 'trackID=1'), 'rtsp://192.168.1.50:554/live/trackID=1');
  });

  test('an absolute control attribute wins outright', () => {
    assert.equal(
      resolveControlUrl(base, 'rtsp://192.168.1.50:554/other/track1'),
      'rtsp://192.168.1.50:554/other/track1',
    );
  });

  test('"*" and absence both mean the session URL', () => {
    assert.equal(resolveControlUrl(base, '*'), base);
    assert.equal(resolveControlUrl(base, null), base);
    assert.equal(resolveControlUrl(base, ''), base);
  });

  test('does not double a slash', () => {
    assert.equal(resolveControlUrl('rtsp://h/live/', '/trackID=1'), 'rtsp://h/live/trackID=1');
  });
});

// --------------------------------------------------------------------- auth

describe('authentication', () => {
  test('parses a Digest challenge', () => {
    const challenges = parseAuthChallenges(
      'Digest realm="Camera", nonce="abc123", qop="auth", algorithm=MD5, opaque="xyz"',
    );

    assert.equal(challenges.length, 1);
    const challenge = challenges[0];
    assert.equal(challenge?.scheme, 'Digest');
    if (challenge?.scheme !== 'Digest') return;

    assert.equal(challenge.realm, 'Camera');
    assert.equal(challenge.nonce, 'abc123');
    assert.deepEqual(challenge.qop, ['auth']);
    assert.equal(challenge.algorithm, 'MD5');
    assert.equal(challenge.opaque, 'xyz');
  });

  test('prefers Digest when a camera offers both', () => {
    const challenges = parseAuthChallenges(
      'Basic realm="Camera", Digest realm="Camera", nonce="abc123"',
    );
    assert.equal(challenges.length, 2);
    assert.equal(challenges[0]?.scheme, 'Digest', 'Digest must be offered first');
  });

  test('ignores a Digest challenge with no nonce', () => {
    // Treating it as valid would produce an unauthenticated request that looks
    // authenticated.
    assert.deepEqual(parseAuthChallenges('Digest realm="Camera"'), []);
  });

  test('refuses an algorithm this client does not implement', () => {
    // Silently falling back to MD5 would fail in a way indistinguishable from a
    // wrong password.
    assert.deepEqual(parseAuthChallenges('Digest realm="C", nonce="n", algorithm=SHA-512'), []);
  });

  test('accepts SHA-256 challenges', () => {
    const challenges = parseAuthChallenges('Digest realm="C", nonce="n", algorithm=SHA-256');
    assert.equal(challenges[0]?.scheme === 'Digest' ? challenges[0].algorithm : null, 'SHA-256');
  });

  test('the Digest header carries a hash, never the password', () => {
    const challenge = parseAuthChallenges(
      'Digest realm="Camera", nonce="abc123", qop="auth", algorithm=MD5',
    )[0];
    assert.ok(challenge !== undefined && challenge.scheme === 'Digest');

    const header = buildDigestHeader(
      challenge,
      new DigestSession('deadbeef'),
      'DESCRIBE',
      'rtsp://192.168.1.50:554/live',
      'admin',
      secret(PASSWORD),
    );

    assert.ok(!header.includes(PASSWORD), `the password leaked into the header: ${header}`);
    assert.match(header, /response="[0-9a-f]{32}"/);
    assert.match(header, /nc=00000001/);
    assert.match(header, /cnonce="deadbeef"/);
  });

  test('the Digest response is reproducible for the same inputs', () => {
    const challenge = parseAuthChallenges('Digest realm="C", nonce="n", algorithm=MD5')[0];
    assert.ok(challenge !== undefined && challenge.scheme === 'Digest');

    const build = (): string =>
      buildDigestHeader(challenge, new DigestSession('fixed'), 'PLAY', 'rtsp://h/s', 'u', secret('p'));

    assert.equal(build(), build());
  });

  test('a credential containing a quote cannot break the header', () => {
    const challenge = parseAuthChallenges('Digest realm="C", nonce="n", algorithm=MD5')[0];
    assert.ok(challenge !== undefined && challenge.scheme === 'Digest');

    const header = buildDigestHeader(
      challenge,
      new DigestSession('x'),
      'PLAY',
      'rtsp://h/s',
      'ad"min',
      secret('p'),
    );
    assert.match(header, /username="ad\\"min"/);
  });

  test('Basic encodes reversibly, which is exactly why it is opt-in', () => {
    const header = buildBasicHeader('admin', secret(PASSWORD));
    assert.match(header, /^Basic /);
    assert.equal(
      Buffer.from(header.slice(6), 'base64').toString('utf8'),
      `admin:${PASSWORD}`,
      'Basic is reversible by design; the test documents why it is gated',
    );
  });

  test('the nonce count advances per request', () => {
    const session = new DigestSession('c');
    assert.equal(session.nextNonceCount(), '00000001');
    assert.equal(session.nextNonceCount(), '00000002');
  });
});

// ----------------------------------------------------------- response parsing

describe('response parsing', () => {
  test('parses a complete response', () => {
    const raw = 'RTSP/1.0 200 OK\r\nCSeq: 1\r\nPublic: OPTIONS, PLAY\r\n\r\n';
    const parsed = parseResponse(raw);

    assert.notEqual(parsed, null);
    assert.equal(parsed?.response.statusCode, 200);
    assert.equal(parsed?.response.headers['public'], 'OPTIONS, PLAY');
    assert.equal(parsed?.consumed, raw.length);
  });

  test('normalises header case, because cameras do not', () => {
    const parsed = parseResponse('RTSP/1.0 200 OK\r\ncSeQ: 3\r\nSESSION: abc\r\n\r\n');
    assert.equal(parsed?.response.headers['cseq'], '3');
    assert.equal(parsed?.response.headers['session'], 'abc');
  });

  test('returns null until the body has fully arrived', () => {
    const headers = 'RTSP/1.0 200 OK\r\nContent-Length: 10\r\n\r\n';
    assert.equal(parseResponse(headers), null, 'body not yet present');
    assert.equal(parseResponse(`${headers}12345`), null, 'body incomplete');
    assert.notEqual(parseResponse(`${headers}1234567890`), null);
  });

  test('reports consumed bytes so a pipelined response is not lost', () => {
    const first = 'RTSP/1.0 200 OK\r\nCSeq: 1\r\n\r\n';
    const second = 'RTSP/1.0 200 OK\r\nCSeq: 2\r\n\r\n';
    const parsed = parseResponse(first + second);
    assert.equal(parsed?.consumed, first.length);
  });

  test('refuses a non-RTSP response instead of misreading it', () => {
    assert.throws(
      () => parseResponse('HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n'),
      (error: unknown) => {
        assert.ok(error instanceof RtspError);
        assert.equal(error.code, 'RTSP_MALFORMED');
        assert.equal(error.recoverable, false);
        return true;
      },
    );
  });

  test('refuses an implausibly large declared body', () => {
    assert.throws(
      () => parseResponse('RTSP/1.0 200 OK\r\nContent-Length: 99999999\r\n\r\n'),
      (error: unknown) => {
        assert.ok(error instanceof RtspError);
        assert.equal(error.code, 'RTSP_BODY_TOO_LARGE');
        return true;
      },
    );
  });
});

// ------------------------------------------------------------ live handshake

describe('handshake against a mock camera', () => {
  test('completes the full sequence with Digest authentication', async () => {
    const server = await spawnServer({ supportsGetParameter: true });

    const client = new RtspClient({
      host: '127.0.0.1',
      port: server.port,
      path: '/live',
      username: 'admin',
      password: secret(PASSWORD),
    });

    const session = await client.open();

    assert.equal(session.sessionId, '12345678');
    assert.equal(session.timeoutSeconds, 60);
    assert.equal(session.encoding, 'H264');
    assert.equal(session.supportsGetParameter, true);
    assert.match(session.videoTrackUrl, /trackID=1$/);

    const methods = server.requests.map((r) => r.split(' ')[0]);
    assert.deepEqual(
      methods.filter((m, i) => methods.indexOf(m) === i),
      ['OPTIONS', 'DESCRIBE', 'SETUP', 'PLAY'],
      'the handshake must run in the order every device expects',
    );

    await client.teardown();
    client.close();
  });

  test('never puts the credential in the request URL', async () => {
    const server = await spawnServer();

    const client = new RtspClient({
      host: '127.0.0.1',
      port: server.port,
      path: '/live',
      username: 'admin',
      password: secret(PASSWORD),
    });
    await client.open();

    // The URL is what gets logged, shown in diagnostics and put in error
    // messages. If a credential can reach it, it reaches all three.
    assert.ok(!client.url.includes(PASSWORD));
    assert.ok(!client.url.includes('@'));

    for (const line of server.requests) {
      assert.ok(!line.includes(PASSWORD), `credential appeared in a request line: ${line}`);
      assert.ok(!line.includes('@'), `userinfo appeared in a request line: ${line}`);
    }

    // And the Authorization headers carried hashes, not the password.
    assert.ok(server.authorizations.length > 0, 'authentication actually happened');
    for (const header of server.authorizations) {
      assert.ok(!header.includes(PASSWORD), `credential leaked in Authorization: ${header}`);
    }

    client.close();
  });

  test('authenticates exactly once per request, not repeatedly', async () => {
    const server = await spawnServer();

    const client = new RtspClient({
      host: '127.0.0.1',
      port: server.port,
      path: '/live',
      username: 'admin',
      password: secret(PASSWORD),
    });
    await client.open();

    // Retrying a rejected credential in a loop is how an account gets locked out
    // and how a device log fills with failed-auth entries somebody must explain.
    const optionsRequests = server.requests.filter((r) => r.startsWith('OPTIONS'));
    assert.equal(optionsRequests.length, 2, 'one unauthenticated probe, then one authenticated');

    client.close();
  });

  test('rejects a wrong password without revealing which part was wrong', async () => {
    const server = await spawnServer();

    const client = new RtspClient({
      host: '127.0.0.1',
      port: server.port,
      path: '/live',
      username: 'admin',
      password: secret('wrong-password'),
    });

    await assert.rejects(
      () => client.open(),
      (error: unknown) => {
        const message = String((error as Error).message);
        assert.match(message, /rejected the supplied credentials/);
        // Distinguishing "no such user" from "wrong password" is an enumeration
        // oracle, so the message must do neither.
        assert.ok(!/unknown user|no such user|user not found/i.test(message));
        assert.ok(!message.includes('wrong-password'));
        return true;
      },
    );
    client.close();
  });

  test('connects to an unsecured camera with no credentials at all', async () => {
    const server = await spawnServer({ noAuth: true });

    const client = new RtspClient({ host: '127.0.0.1', port: server.port, path: '/live' });
    const session = await client.open();

    assert.equal(session.encoding, 'H264');
    assert.equal(server.authorizations.length, 0);
    client.close();
  });

  test('refuses Basic-only cameras unless Basic is explicitly enabled', async () => {
    const server = await spawnServer({ authSchemes: ['Basic'] });

    const strict = new RtspClient({
      host: '127.0.0.1',
      port: server.port,
      path: '/live',
      username: 'admin',
      password: secret(PASSWORD),
    });

    await assert.rejects(() => strict.open(), /does not support/);
    strict.close();

    const permissive = new RtspClient({
      host: '127.0.0.1',
      port: server.port,
      path: '/live',
      username: 'admin',
      password: secret(PASSWORD),
      allowBasicAuth: true,
    });

    const session = await permissive.open();
    assert.equal(session.sessionId, '12345678');
    permissive.close();
  });

  test('handles a response split across TCP segments at awkward boundaries', async () => {
    // Segments that end mid-status-line, mid-header and mid-body. A client that
    // assumes one response per chunk passes every naive test and then fails
    // against a real camera on a congested link.
    const server = await spawnServer({ dribble: true });

    const client = new RtspClient({
      host: '127.0.0.1',
      port: server.port,
      path: '/live',
      username: 'admin',
      password: secret(PASSWORD),
    });

    const session = await client.open();
    assert.equal(session.encoding, 'H264');
    client.close();
  });

  test('reports a camera with no decodable video as exactly that', async () => {
    const server = await spawnServer({
      sdp: ['v=0', 's=Audio Only', 'm=audio 0 RTP/AVP 97', 'a=rtpmap:97 PCMU/8000'].join('\r\n'),
    });

    const client = new RtspClient({
      host: '127.0.0.1',
      port: server.port,
      path: '/live',
      username: 'admin',
      password: secret(PASSWORD),
    });

    await assert.rejects(
      () => client.open(),
      (error: unknown) => {
        assert.ok(error instanceof RtspError);
        assert.equal(error.code, 'RTSP_NO_VIDEO_TRACK');
        assert.equal(error.recoverable, false, 'retrying will not grow a video track');
        // The message must say what the camera did offer, or an integrator has
        // nothing to work with.
        assert.match(error.message, /audio/);
        return true;
      },
    );
    client.close();
  });

  test('surfaces a SETUP rejection with the status the camera gave', async () => {
    const server = await spawnServer({ setupStatus: 461 });

    const client = new RtspClient({
      host: '127.0.0.1',
      port: server.port,
      path: '/live',
      username: 'admin',
      password: secret(PASSWORD),
    });

    await assert.rejects(
      () => client.open(),
      (error: unknown) => {
        assert.ok(error instanceof RtspError);
        assert.equal(error.code, 'RTSP_SETUP_FAILED');
        assert.equal(error.statusCode, 461);
        return true;
      },
    );
    client.close();
  });

  test('refuses a device that answers with something other than RTSP', async () => {
    const server = await spawnServer({ speakGarbage: true });

    const client = new RtspClient({ host: '127.0.0.1', port: server.port, path: '/live' });

    await assert.rejects(
      () => client.open(),
      (error: unknown) => {
        assert.ok(error instanceof RtspError);
        assert.equal(error.code, 'RTSP_MALFORMED');
        return true;
      },
    );
    client.close();
  });

  test('times out rather than hanging on a silent device', async () => {
    const server = await spawnServer({ silent: true });

    const client = new RtspClient({
      host: '127.0.0.1',
      port: server.port,
      path: '/live',
      timeoutMillis: 300,
    });

    await assert.rejects(
      () => client.open(),
      (error: unknown) => {
        assert.ok(error instanceof RtspError);
        assert.equal(error.code, 'RTSP_TIMEOUT');
        assert.equal(error.recoverable, true, 'a hung camera is worth retrying');
        return true;
      },
    );
    client.close();
  });

  test('reports an unreachable host without leaking the credential', async () => {
    const client = new RtspClient({
      // Port 1 on loopback: reliably refused, and reliably fast.
      host: '127.0.0.1',
      port: 1,
      path: '/live',
      username: 'admin',
      password: secret(PASSWORD),
      timeoutMillis: 2000,
    });

    await assert.rejects(
      () => client.open(),
      (error: unknown) => {
        const message = String((error as Error).message);
        assert.ok(!message.includes(PASSWORD), `credential leaked into an error: ${message}`);
        assert.match(message, /127\.0\.0\.1:1/, 'the endpoint is still useful to report');
        return true;
      },
    );
    client.close();
  });

  test('honours the session timeout the camera advertises', async () => {
    const server = await spawnServer({ sessionTimeoutSeconds: 15 });

    const client = new RtspClient({
      host: '127.0.0.1',
      port: server.port,
      path: '/live',
      username: 'admin',
      password: secret(PASSWORD),
    });

    const session = await client.open();
    assert.equal(session.timeoutSeconds, 15, 'keep-alive must be driven by the camera, not a guess');
    client.close();
  });

  test('keep-alive uses GET_PARAMETER when the camera supports it', async () => {
    const server = await spawnServer({ supportsGetParameter: true });

    const client = new RtspClient({
      host: '127.0.0.1',
      port: server.port,
      path: '/live',
      username: 'admin',
      password: secret(PASSWORD),
    });

    const session = await client.open();
    await client.keepAlive(session.supportsGetParameter);

    assert.ok(
      server.requests.some((r) => r.startsWith('GET_PARAMETER')),
      'GET_PARAMETER is session-scoped; OPTIONS would succeed even on a dropped session',
    );
    client.close();
  });

  test('keep-alive falls back to OPTIONS when GET_PARAMETER is absent', async () => {
    const server = await spawnServer({ supportsGetParameter: false });

    const client = new RtspClient({
      host: '127.0.0.1',
      port: server.port,
      path: '/live',
      username: 'admin',
      password: secret(PASSWORD),
    });

    const session = await client.open();
    assert.equal(session.supportsGetParameter, false);

    const before = server.requests.filter((r) => r.startsWith('OPTIONS')).length;
    await client.keepAlive(session.supportsGetParameter);
    const afterCount = server.requests.filter((r) => r.startsWith('OPTIONS')).length;

    assert.equal(afterCount, before + 1);
    client.close();
  });
});
