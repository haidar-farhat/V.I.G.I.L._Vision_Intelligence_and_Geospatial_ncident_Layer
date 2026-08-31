import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { asId, utcMillis } from '@sentinel/shared-types';
import type { NodeId } from '@sentinel/shared-types';
import {
  PROTOCOL_VERSION,
  PayloadReader,
  ProtocolError,
  envelope,
  newRequestId,
  parseEnvelope,
  serialiseEnvelope,
} from '../src/envelope.ts';
import { ALL_CHANNELS, Channel, MessageKind, channelForKind, isChannel } from '../src/channels.ts';
import {
  CloseCode,
  MAX_CONTROL_FRAME_BYTES,
  MAX_FRAME_BYTES,
  MessageAssembler,
  Opcode,
  WebSocketProtocolError,
  decodeFrame,
  encodeClose,
  encodeFrame,
  encodeText,
  parseClose,
} from '../src/websocket/frame.ts';
import {
  WEBSOCKET_VERSION,
  acceptKey,
  generateClientKey,
  rejectionResponse,
  upgradeResponse,
  validateHandshake,
  verifyAcceptKey,
} from '../src/websocket/handshake.ts';

const NODE = asId<NodeId>('node-control');

// ------------------------------------------------------------------ envelopes

describe('envelopes', () => {
  test('round-trips through the wire', () => {
    const original = envelope('event', { id: 'e1', severity: 'HIGH' }, NODE);
    const parsed = parseEnvelope(serialiseEnvelope(original));

    assert.equal(parsed.v, PROTOCOL_VERSION);
    assert.equal(parsed.kind, 'event');
    assert.equal(parsed.nodeId, NODE);
    assert.equal(parsed.requestId, original.requestId);
    assert.deepEqual(parsed.payload, { id: 'e1', severity: 'HIGH' });
  });

  test('every message carries a unique request id', () => {
    // The field exists for the moment something has gone wrong and a trace must
    // be correlated across three processes.
    const ids = new Set([newRequestId(), newRequestId(), newRequestId()]);
    assert.equal(ids.size, 3);
  });

  test('refuses an unsupported version explicitly, and says which side to upgrade', () => {
    // A system that accepts a message it does not fully understand will one day
    // act on a field that moved.
    const future = JSON.stringify({
      v: 99,
      requestId: 'r1',
      timestamp: 1000,
      nodeId: 'node-1',
      kind: 'event',
      payload: {},
    });

    assert.throws(
      () => parseEnvelope(future),
      (error: unknown) => {
        assert.ok(error instanceof ProtocolError);
        assert.equal(error.code, 'PROTOCOL_UNSUPPORTED_VERSION');
        assert.match(error.message, /Upgrade whichever side is older/);
        assert.match(error.message, /not backward compatible by design/);
        return true;
      },
    );
  });

  test('checks the version before any other field', () => {
    // A message from an unknown version may have moved every other field, so
    // validating them against this version's expectations would mislead.
    const future = JSON.stringify({ v: 99 });

    assert.throws(
      () => parseEnvelope(future),
      (error: unknown) => {
        assert.equal((error as ProtocolError).code, 'PROTOCOL_UNSUPPORTED_VERSION');
        return true;
      },
    );
  });

  test('names the missing field rather than saying "malformed"', () => {
    const cases: readonly [string, string][] = [
      [JSON.stringify({ v: 1, timestamp: 1, nodeId: 'n', kind: 'k' }), 'requestId'],
      [JSON.stringify({ v: 1, requestId: 'r', nodeId: 'n', kind: 'k' }), 'timestamp'],
      [JSON.stringify({ v: 1, requestId: 'r', timestamp: 1, kind: 'k' }), 'nodeId'],
      [JSON.stringify({ v: 1, requestId: 'r', timestamp: 1, nodeId: 'n' }), 'kind'],
    ];

    for (const [raw, field] of cases) {
      assert.throws(
        () => parseEnvelope(raw),
        (error: unknown) => {
          assert.equal((error as ProtocolError).code, 'PROTOCOL_MISSING_FIELD');
          assert.match((error as Error).message, new RegExp(field));
          return true;
        },
        `should have named "${field}"`,
      );
    }
  });

  test('refuses non-JSON and non-objects', () => {
    for (const raw of ['not json', '[]', '"string"', 'null', '42']) {
      assert.throws(() => parseEnvelope(raw), ProtocolError, `accepted: ${raw}`);
    }
  });

  test('refuses an oversized message before parsing it', () => {
    const huge = `{"v":1,"padding":"${'x'.repeat(5 * 1024 * 1024)}"}`;

    assert.throws(
      () => parseEnvelope(huge),
      (error: unknown) => {
        assert.equal((error as ProtocolError).code, 'PROTOCOL_TOO_LARGE');
        return true;
      },
    );
  });

  test('refuses an absurdly long kind', () => {
    const raw = JSON.stringify({
      v: 1,
      requestId: 'r',
      timestamp: 1,
      nodeId: 'n',
      kind: 'k'.repeat(200),
    });

    assert.throws(
      () => parseEnvelope(raw),
      (error: unknown) => {
        assert.equal((error as ProtocolError).code, 'PROTOCOL_BAD_FIELD');
        return true;
      },
    );
  });

  test('accepts bytes as readily as a string', () => {
    const original = envelope('ping', {}, NODE);
    const bytes = new TextEncoder().encode(serialiseEnvelope(original));

    assert.equal(parseEnvelope(bytes).kind, 'ping');
  });
});

describe('payload reading', () => {
  const reader = (payload: unknown): PayloadReader => new PayloadReader(payload, 'test.kind');

  test('reads typed fields', () => {
    const r = reader({ name: 'cam-07', fps: 12.5, enabled: true, zones: ['a', 'b'] });

    assert.equal(r.string('name'), 'cam-07');
    assert.equal(r.number('fps'), 12.5);
    assert.equal(r.boolean('enabled'), true);
    assert.deepEqual(r.stringArray('zones'), ['a', 'b']);
    assert.equal(r.has('name'), true);
    assert.equal(r.has('absent'), false);
  });

  test('names the kind and the field when a type is wrong', () => {
    // A generic validator's path expression tells whoever is debugging a version
    // skew far less than this does.
    assert.throws(() => reader({ name: 42 }).string('name'), /"test\.kind" requires a string "name"/);
    assert.throws(() => reader({ fps: 'fast' }).number('fps'), /requires a finite number "fps"/);
    assert.throws(() => reader({ on: 'yes' }).boolean('on'), /requires a boolean "on"/);
  });

  test('refuses a non-finite number', () => {
    // JSON cannot carry NaN, but a hand-built payload can.
    assert.throws(() => reader({ fps: Number.POSITIVE_INFINITY }).number('fps'), ProtocolError);
  });

  test('optional fields tolerate absence but not the wrong type', () => {
    assert.equal(reader({}).optionalString('note'), undefined);
    assert.equal(reader({ note: null }).optionalString('note'), undefined);
    assert.equal(reader({ note: 'x' }).optionalString('note'), 'x');
    assert.throws(() => reader({ note: 5 }).optionalString('note'), ProtocolError);
  });

  test('caps array length so a peer cannot force unbounded work', () => {
    const huge = { zones: Array.from({ length: 5000 }, (_, i) => `z${i}`) };

    assert.throws(
      () => reader(huge).stringArray('zones', 100),
      (error: unknown) => {
        assert.equal((error as ProtocolError).code, 'PROTOCOL_TOO_LARGE');
        return true;
      },
    );
  });

  test('rejects a non-object payload up front', () => {
    assert.throws(() => new PayloadReader('string', 'k'), /is not an object/);
    assert.throws(() => new PayloadReader(null, 'k'), ProtocolError);
  });
});

describe('channels', () => {
  test('routes server messages to their channel', () => {
    assert.equal(channelForKind(MessageKind.Event), Channel.Events);
    assert.equal(channelForKind(MessageKind.Track), Channel.Tracks);
    assert.equal(channelForKind(MessageKind.IncidentOpened), Channel.Incidents);
    assert.equal(channelForKind(MessageKind.IncidentUpdated), Channel.Incidents);
    assert.equal(channelForKind(MessageKind.NodeHeartbeat), Channel.Nodes);
  });

  test('a message with no channel is not broadcast anywhere', () => {
    // Welcome, pong and error are replies to one client, not subscriptions.
    assert.equal(channelForKind(MessageKind.Welcome), null);
    assert.equal(channelForKind(MessageKind.Pong), null);
    assert.equal(channelForKind('made.up'), null);
  });

  test('validates channel names from the wire', () => {
    assert.ok(isChannel('events'));
    assert.ok(!isChannel('everything'));
    assert.equal(ALL_CHANNELS.length, 6);
  });
});

// ------------------------------------------------------------------- frames

describe('WebSocket frames', () => {
  /** Masked as a client must, so the server-side decoder accepts it. */
  const clientFrame = (opcode: Opcode, payload: Buffer, fin = true): Buffer =>
    encodeFrame(opcode, payload, { fin, mask: true });

  test('round-trips a small text frame', () => {
    const encoded = clientFrame(Opcode.Text, Buffer.from('hello'));
    const decoded = decodeFrame(encoded, true);

    assert.notEqual(decoded, null);
    assert.equal(decoded?.frame.opcode, Opcode.Text);
    assert.equal(decoded?.frame.fin, true);
    assert.equal(decoded?.frame.payload.toString('utf8'), 'hello');
    assert.equal(decoded?.consumed, encoded.length);
  });

  test('round-trips each payload length encoding', () => {
    // 7-bit, 16-bit and 64-bit length fields are three separate code paths.
    for (const size of [0, 1, 125, 126, 65535, 65536, 200_000]) {
      const payload = Buffer.alloc(size, 0x61);
      const decoded = decodeFrame(clientFrame(Opcode.Binary, payload), true);

      assert.equal(decoded?.frame.payload.length, size, `failed at ${size} bytes`);
    }
  });

  test('unmasks correctly, including across the four-byte key boundary', () => {
    const payload = Buffer.from('the quick brown fox jumps over the lazy dog');
    const decoded = decodeFrame(clientFrame(Opcode.Text, payload), true);

    assert.equal(decoded?.frame.payload.toString('utf8'), payload.toString('utf8'));
  });

  test('masking actually changes the bytes on the wire', () => {
    const payload = Buffer.from('aaaaaaaa');
    const masked = encodeFrame(Opcode.Text, payload, { mask: true });

    // Header is 2 bytes plus a 4-byte key; the body must not be the plaintext.
    assert.notEqual(masked.subarray(6).toString('utf8'), 'aaaaaaaa');
  });

  test('returns null until the whole frame has arrived', () => {
    const encoded = clientFrame(Opcode.Text, Buffer.from('a longer message here'));

    for (let cut = 1; cut < encoded.length; cut += 1) {
      assert.equal(decodeFrame(encoded.subarray(0, cut), true), null, `at ${cut} bytes`);
    }
    assert.notEqual(decodeFrame(encoded, true), null);
  });

  test('reports consumed bytes so a pipelined frame is not lost', () => {
    const first = clientFrame(Opcode.Text, Buffer.from('one'));
    const second = clientFrame(Opcode.Text, Buffer.from('two'));

    const decoded = decodeFrame(Buffer.concat([first, second]), true);
    assert.equal(decoded?.consumed, first.length);

    const rest = Buffer.concat([first, second]).subarray(decoded?.consumed ?? 0);
    assert.equal(decodeFrame(rest, true)?.frame.payload.toString('utf8'), 'two');
  });

  test('refuses an unmasked client frame', () => {
    // Masking is not confidentiality - the key travels with the frame. It exists
    // so a frame cannot be crafted to look like an HTTP request to a proxy, which
    // is why RFC 6455 requires failing the connection rather than accepting it.
    const unmasked = encodeFrame(Opcode.Text, Buffer.from('hi'), { mask: false });

    assert.throws(
      () => decodeFrame(unmasked, true),
      (error: unknown) => {
        assert.ok(error instanceof WebSocketProtocolError);
        assert.equal(error.closeCode, CloseCode.ProtocolError);
        assert.match(error.message, /must be masked/);
        return true;
      },
    );
  });

  test('refuses a masked server frame', () => {
    assert.throws(
      () => decodeFrame(encodeFrame(Opcode.Text, Buffer.from('hi'), { mask: true }), false),
      /must not be masked/,
    );
  });

  test('refuses reserved bits when no extension was negotiated', () => {
    const frame = clientFrame(Opcode.Text, Buffer.from('x'));
    frame[0] = (frame[0] ?? 0) | 0x40; // RSV1

    assert.throws(() => decodeFrame(frame, true), /Reserved bits are set/);
  });

  test('refuses an unknown opcode', () => {
    const frame = clientFrame(Opcode.Text, Buffer.from('x'));
    frame[0] = 0x80 | 0x5; // reserved non-control opcode

    assert.throws(() => decodeFrame(frame, true), /Unknown opcode/);
  });

  test('refuses a fragmented control frame', () => {
    const frame = clientFrame(Opcode.Ping, Buffer.from('x'), false);
    assert.throws(() => decodeFrame(frame, true), /may not be fragmented/);
  });

  test('refuses an oversized control frame', () => {
    assert.throws(
      () => encodeFrame(Opcode.Ping, Buffer.alloc(MAX_CONTROL_FRAME_BYTES + 1), { mask: true }),
      /beyond the 125-byte limit/,
    );
  });

  test('refuses a declared length beyond the limit without allocating it', () => {
    // A peer must not be able to claim 2 GB and have the buffer reserved on its
    // say-so. The header alone is enough to refuse.
    const header = Buffer.alloc(14);
    header[0] = 0x82; // fin + binary
    header[1] = 0xff; // masked + 64-bit length
    header.writeBigUInt64BE(BigInt(2 ** 31), 2);

    assert.throws(
      () => decodeFrame(header, true),
      (error: unknown) => {
        assert.ok(error instanceof WebSocketProtocolError);
        assert.equal(error.closeCode, CloseCode.MessageTooBig);
        return true;
      },
    );
  });

  test('a 64-bit length above 2^53 cannot be rounded into acceptability', () => {
    const header = Buffer.alloc(14);
    header[0] = 0x82;
    header[1] = 0xff;
    header.writeBigUInt64BE(BigInt('18446744073709551615'), 2);

    assert.throws(() => decodeFrame(header, true), WebSocketProtocolError);
  });

  test('encodes and parses a close frame', () => {
    const encoded = encodeClose(CloseCode.PolicyViolation, 'subscription refused');
    const decoded = decodeFrame(encoded, false);

    assert.equal(decoded?.frame.opcode, Opcode.Close);

    const parsed = parseClose(decoded?.frame.payload ?? Buffer.alloc(0));
    assert.equal(parsed.code, CloseCode.PolicyViolation);
    assert.equal(parsed.reason, 'subscription refused');
  });

  test('truncates a long close reason rather than failing to send one', () => {
    // Failing to send a close because the explanation was long leaves the peer
    // with no explanation at all.
    const encoded = encodeClose(CloseCode.Normal, 'x'.repeat(500));
    const decoded = decodeFrame(encoded, false);

    assert.ok((decoded?.frame.payload.length ?? 0) <= MAX_CONTROL_FRAME_BYTES);
    assert.equal(parseClose(decoded?.frame.payload ?? Buffer.alloc(0)).code, CloseCode.Normal);
  });

  test('an empty close payload is a normal close', () => {
    assert.deepEqual(parseClose(Buffer.alloc(0)), { code: CloseCode.Normal, reason: '' });
  });

  test('encodeText is a masked or unmasked text frame as asked', () => {
    assert.equal(decodeFrame(encodeText('hi', true), true)?.frame.payload.toString(), 'hi');
    assert.equal(decodeFrame(encodeText('hi', false), false)?.frame.payload.toString(), 'hi');
  });
});

describe('message assembly', () => {
  test('reassembles a fragmented message', () => {
    const assembler = new MessageAssembler();

    assert.equal(
      assembler.accept({ fin: false, opcode: Opcode.Text, payload: Buffer.from('hello ') }),
      null,
    );
    assert.equal(assembler.inProgress, true);

    const done = assembler.accept({
      fin: true,
      opcode: Opcode.Continuation,
      payload: Buffer.from('world'),
    });

    assert.equal(done?.payload.toString('utf8'), 'hello world');
    assert.equal(done?.opcode, Opcode.Text);
    assert.equal(assembler.inProgress, false);
  });

  test('a single unfragmented frame completes immediately', () => {
    const assembler = new MessageAssembler();
    const done = assembler.accept({ fin: true, opcode: Opcode.Text, payload: Buffer.from('x') });

    assert.equal(done?.payload.toString('utf8'), 'x');
  });

  test('refuses a continuation with nothing in progress', () => {
    const assembler = new MessageAssembler();

    assert.throws(
      () => assembler.accept({ fin: true, opcode: Opcode.Continuation, payload: Buffer.alloc(0) }),
      /no message in progress/,
    );
  });

  test('refuses a new data frame interleaved into a fragmented message', () => {
    const assembler = new MessageAssembler();
    assembler.accept({ fin: false, opcode: Opcode.Text, payload: Buffer.from('a') });

    assert.throws(
      () => assembler.accept({ fin: true, opcode: Opcode.Text, payload: Buffer.from('b') }),
      /still in progress/,
    );
  });

  test('fragmentation cannot smuggle a message past the size limit', () => {
    const assembler = new MessageAssembler(1000);

    assembler.accept({ fin: false, opcode: Opcode.Binary, payload: Buffer.alloc(600) });

    assert.throws(
      () => assembler.accept({ fin: true, opcode: Opcode.Continuation, payload: Buffer.alloc(600) }),
      (error: unknown) => {
        assert.ok(error instanceof WebSocketProtocolError);
        assert.equal(error.closeCode, CloseCode.MessageTooBig);
        return true;
      },
    );

    // And the partial message is discarded rather than left to accumulate.
    assert.equal(assembler.inProgress, false);
  });

  test('the default limit matches the frame limit', () => {
    assert.equal(MAX_FRAME_BYTES, 4 * 1024 * 1024);
  });
});

// ---------------------------------------------------------------- handshake

describe('WebSocket handshake', () => {
  const request = (overrides: Record<string, string> = {}, method = 'GET') => ({
    method,
    path: '/ws',
    headers: {
      upgrade: 'websocket',
      connection: 'Upgrade',
      'sec-websocket-version': '13',
      'sec-websocket-key': 'dGhlIHNhbXBsZSBub25jZQ==',
      ...overrides,
    },
  });

  test('accepts a well-formed upgrade', () => {
    const result = validateHandshake(request());

    assert.equal(result.ok, true);
    if (!result.ok) return;
    // The value from RFC 6455 section 1.3, which is what makes this checkable.
    assert.equal(result.acceptKey, 's3pPLMBiTxaQ9kYGzzhZRbK+xOo=');
  });

  test('the accept key proves the server understood rather than reflected', () => {
    const key = generateClientKey();
    assert.ok(verifyAcceptKey(key, acceptKey(key)));
    assert.ok(!verifyAcceptKey(key, key), 'a reflected key must not verify');
    assert.ok(!verifyAcceptKey(key, 'short'), 'a length mismatch must not throw');
  });

  test('refuses a non-GET upgrade', () => {
    const result = validateHandshake(request({}, 'POST'));
    assert.equal(result.ok, false);
    if (result.ok) return;
    assert.equal(result.rejection.status, 405);
  });

  test('refuses a missing or wrong Upgrade header', () => {
    for (const headers of [{ upgrade: '' }, { upgrade: 'h2c' }]) {
      const result = validateHandshake(request(headers));
      assert.equal(result.ok, false);
    }
  });

  test('accepts a Connection header carrying other tokens', () => {
    // Proxies routinely add to this list; it is comma-separated for a reason.
    assert.equal(validateHandshake(request({ connection: 'keep-alive, Upgrade' })).ok, true);
    assert.equal(validateHandshake(request({ connection: 'keep-alive' })).ok, false);
  });

  test('refuses an unsupported version and advertises the supported one', () => {
    const result = validateHandshake(request({ 'sec-websocket-version': '8' }));

    assert.equal(result.ok, false);
    if (result.ok) return;
    assert.equal(result.rejection.status, 426);
    assert.match(rejectionResponse(result.rejection), new RegExp(`Sec-WebSocket-Version: ${WEBSOCKET_VERSION}`));
  });

  test('refuses a malformed client key', () => {
    // A key of the wrong shape means the peer is not a WebSocket client, and
    // completing the handshake would connect to something that is not one.
    for (const key of ['', 'short', 'not base64 at all!!', 'dGhlIHNhbXBsZSBub25jZQ']) {
      assert.equal(validateHandshake(request({ 'sec-websocket-key': key })).ok, false, key);
    }
  });

  test('selects a supported subprotocol', () => {
    const result = validateHandshake(request({ 'sec-websocket-protocol': 'sentinel.v1, other' }), {
      supportedProtocols: ['sentinel.v1'],
    });

    assert.equal(result.ok, true);
    if (!result.ok) return;
    assert.equal(result.protocol, 'sentinel.v1');
    assert.match(upgradeResponse(result.acceptKey, result.protocol), /Sec-WebSocket-Protocol: sentinel\.v1/);
  });

  test('refuses when none of the offered subprotocols is supported', () => {
    const result = validateHandshake(request({ 'sec-websocket-protocol': 'chat' }), {
      supportedProtocols: ['sentinel.v1'],
    });

    assert.equal(result.ok, false);
    if (result.ok) return;
    assert.match(result.rejection.reason, /chat/);
    assert.match(result.rejection.reason, /sentinel\.v1/);
  });

  test('a rejection is a real HTTP response, not a dropped socket', () => {
    // An upgrade that fails silently is miserable to debug: the client sees a
    // closed socket and the server logs nothing.
    const result = validateHandshake(request({}, 'DELETE'));
    assert.equal(result.ok, false);
    if (result.ok) return;

    const response = rejectionResponse(result.rejection);
    assert.match(response, /^HTTP\/1\.1 405 Method Not Allowed/);
    assert.match(response, /Content-Length: \d+/);
    assert.match(response, /must use GET/);
  });

  test('the 101 response carries what the client checks', () => {
    const response = upgradeResponse('s3pPLMBiTxaQ9kYGzzhZRbK+xOo=', null);

    assert.match(response, /^HTTP\/1\.1 101 Switching Protocols/);
    assert.match(response, /Upgrade: websocket/);
    assert.match(response, /Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK\+xOo=/);
    assert.ok(!response.includes('Sec-WebSocket-Protocol'), 'no protocol means no header');
    assert.ok(response.endsWith('\r\n\r\n'));
  });
});
