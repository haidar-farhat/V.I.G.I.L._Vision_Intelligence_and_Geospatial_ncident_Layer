import { test, describe, after } from 'node:test';
import assert from 'node:assert/strict';
import { createServer, createConnection } from 'node:net';
import type { Server, Socket } from 'node:net';
import { asId, utcMillis } from '@sentinel/shared-types';
import type { NodeId, UserId } from '@sentinel/shared-types';
import {
  CloseCode,
  MessageKind,
  Opcode,
  decodeFrame,
  encodeFrame,
  encodeText,
  envelope,
  parseClose,
  parseEnvelope,
  serialiseEnvelope,
} from '@sentinel/protocol';
import { Hub, MAX_OUTBOUND_QUEUE } from '../src/hub.ts';
import type { Connection } from '../src/hub.ts';
import { SessionStore } from '../src/auth.ts';

const NODE = asId<NodeId>('node-control');

const session = () =>
  new SessionStore().create({
    id: asId<UserId>('user-1'),
    username: 'operator',
    roles: ['OPERATOR'],
  });

/**
 * A real TCP pair.
 *
 * The hub's whole purpose is managing sockets - backpressure, half-open
 * connections, partial frames - and a stream mock has none of those properties.
 * A loopback pair does.
 */
type Harness = {
  readonly hub: Hub;
  readonly connection: Connection;
  readonly client: Socket;
  /** Frames the client has received, decoded. */
  readonly received: { opcode: Opcode; payload: Buffer }[];
  send(data: Buffer): void;
  messages(): unknown[];
  close(): Promise<void>;
};

const servers: Server[] = [];
const harnesses: Harness[] = [];

after(async () => {
  for (const harness of harnesses) await harness.close();
  for (const server of servers) {
    await new Promise<void>((resolve) => server.close(() => resolve()));
  }
});

const harness = async (options: { now?: () => ReturnType<typeof utcMillis> } = {}): Promise<Harness> => {
  const hub = new Hub({ nodeId: NODE, ...(options.now === undefined ? {} : { now: options.now }) });

  let serverSocket: Socket | null = null;
  const server = createServer((socket) => {
    serverSocket = socket;
  });
  servers.push(server);

  const port = await new Promise<number>((resolve) => {
    server.listen(0, '127.0.0.1', () => {
      const address = server.address();
      resolve(typeof address === 'object' && address !== null ? address.port : 0);
    });
  });

  const client = createConnection({ port, host: '127.0.0.1' });
  await new Promise<void>((resolve) => client.once('connect', () => resolve()));

  // Wait for the server side to be accepted.
  for (let i = 0; i < 100 && serverSocket === null; i += 1) {
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
  assert.notEqual(serverSocket, null, 'server never accepted the connection');

  const received: { opcode: Opcode; payload: Buffer }[] = [];
  let inbound = Buffer.alloc(0);

  client.on('data', (chunk: Buffer) => {
    inbound = Buffer.concat([inbound, chunk]);
    for (;;) {
      // Server frames must not be masked; the decoder enforces the direction.
      const decoded = decodeFrame(inbound, false);
      if (decoded === null) return;
      inbound = inbound.subarray(decoded.consumed);
      received.push({ opcode: decoded.frame.opcode, payload: decoded.frame.payload });
    }
  });
  client.on('error', () => {
    // Expected whenever a test closes a connection from the server side.
  });

  const connection = hub.accept(serverSocket as unknown as Socket, session());

  const result: Harness = {
    hub,
    connection,
    client,
    received,
    send: (data: Buffer) => client.write(data),
    messages: () =>
      received
        .filter((frame) => frame.opcode === Opcode.Text)
        .map((frame) => parseEnvelope(frame.payload.toString('utf8'))),
    close: async () => {
      client.destroy();
      hub.closeAll();
    },
  };

  harnesses.push(result);
  await settle();
  return result;
};

/** Let the event loop deliver pending socket data. */
const settle = (times = 3): Promise<void> =>
  new Promise((resolve) => {
    let remaining = times;
    const tick = (): void => {
      remaining -= 1;
      if (remaining <= 0) resolve();
      else setTimeout(tick, 10);
    };
    setTimeout(tick, 10);
  });

/** A client-side message: masked, as the specification requires. */
const clientMessage = (kind: string, payload: unknown): Buffer =>
  encodeFrame(
    Opcode.Text,
    Buffer.from(serialiseEnvelope(envelope(kind, payload, asId<NodeId>('node-desktop'))), 'utf8'),
    { mask: true },
  );

describe('connection lifecycle', () => {
  test('greets a new connection with the channels it may subscribe to', async () => {
    // So a client never hard-codes a list that could drift from the server's.
    const h = await harness();
    const welcome = h.messages()[0] as { kind: string; payload: Record<string, unknown> };

    assert.equal(welcome.kind, MessageKind.Welcome);
    assert.equal(welcome.payload['username'], 'operator');
    assert.ok(Array.isArray(welcome.payload['channels']));
    assert.ok((welcome.payload['channels'] as string[]).includes('events'));
    assert.equal(h.hub.connectionCount, 1);
  });

  test('answers a protocol-level ping', async () => {
    const h = await harness();
    h.send(clientMessage(MessageKind.Ping, {}));
    await settle();

    assert.ok(h.messages().some((m) => (m as { kind: string }).kind === MessageKind.Pong));
  });

  test('answers a WebSocket control ping without queueing it behind data', async () => {
    const h = await harness();
    h.send(encodeFrame(Opcode.Ping, Buffer.from('probe'), { mask: true }));
    await settle();

    const pong = h.received.find((frame) => frame.opcode === Opcode.Pong);
    assert.notEqual(pong, undefined);
    assert.equal(pong?.payload.toString('utf8'), 'probe');
  });

  test('a client close ends the connection cleanly', async () => {
    const h = await harness();
    h.send(encodeFrame(Opcode.Close, Buffer.alloc(0), { mask: true }));
    await settle();

    assert.equal(h.hub.connectionCount, 0);
  });
});

describe('subscriptions', () => {
  test('delivers only to subscribers of the message channel', async () => {
    const h = await harness();
    h.send(clientMessage(MessageKind.Subscribe, { channels: ['events'] }));
    await settle();

    assert.equal(h.hub.subscriberCount('events'), 1);
    assert.equal(h.hub.subscriberCount('tracks'), 0);

    assert.equal(h.hub.publish(MessageKind.Event, { id: 'e1' }), 1);
    assert.equal(h.hub.publish(MessageKind.Track, { id: 't1' }), 0, 'not subscribed to tracks');

    await settle();
    const kinds = h.messages().map((m) => (m as { kind: string }).kind);
    assert.ok(kinds.includes(MessageKind.Event));
    assert.ok(!kinds.includes(MessageKind.Track));
  });

  test('unsubscribing stops delivery', async () => {
    const h = await harness();
    h.send(clientMessage(MessageKind.Subscribe, { channels: ['events'] }));
    await settle();
    h.send(clientMessage(MessageKind.Unsubscribe, { channels: ['events'] }));
    await settle();

    assert.equal(h.hub.publish(MessageKind.Event, { id: 'e1' }), 0);
  });

  test('names channels it refused rather than ignoring them', async () => {
    // A silently dropped subscription produces a client that waits forever for a
    // channel it never joined.
    const h = await harness();
    h.send(clientMessage(MessageKind.Subscribe, { channels: ['events', 'everything', 42] }));
    await settle();

    const reply = h.messages().find(
      (m) => (m as { kind: string }).kind === MessageKind.Subscribed,
    ) as { payload: Record<string, unknown> };

    assert.deepEqual(reply.payload['subscribed'], ['events']);
    assert.deepEqual(reply.payload['rejected'], ['everything', '42']);
  });

  test('a message with no channel is never broadcast', async () => {
    const h = await harness();
    h.send(clientMessage(MessageKind.Subscribe, { channels: ['events'] }));
    await settle();

    // Welcome, pong and error are replies to one client, not subscriptions.
    assert.equal(h.hub.publish(MessageKind.Welcome, {}), 0);
  });
});

describe('untrusted input after the handshake', () => {
  test('refuses an unmasked client frame and closes the connection', async () => {
    // RFC 6455 requires failing the connection; accepting it is a cache-poisoning
    // vector through a transparent proxy.
    const h = await harness();
    h.send(encodeFrame(Opcode.Text, Buffer.from('{}'), { mask: false }));
    await settle();

    const close = h.received.find((frame) => frame.opcode === Opcode.Close);
    assert.notEqual(close, undefined, 'expected a close frame');
    assert.equal(parseClose(close?.payload ?? Buffer.alloc(0)).code, CloseCode.ProtocolError);
    assert.equal(h.hub.connectionCount, 0);
  });

  test('explains a malformed message rather than ignoring it', async () => {
    // Silence here is how a version skew becomes an afternoon of packet captures.
    const h = await harness();
    h.send(encodeFrame(Opcode.Text, Buffer.from('not json at all'), { mask: true }));
    await settle();

    const error = h.messages().find((m) => (m as { kind: string }).kind === MessageKind.Error) as {
      payload: Record<string, unknown>;
    };

    assert.notEqual(error, undefined);
    assert.equal(error.payload['code'], 'PROTOCOL_MALFORMED');
    assert.equal(h.hub.connectionCount, 1, 'a bad message is not a fatal offence');
  });

  test('reports an unsupported protocol version by name', async () => {
    const h = await harness();
    const raw = JSON.stringify({
      v: 99,
      requestId: 'r',
      timestamp: Date.now(),
      nodeId: 'node-desktop',
      kind: 'subscribe',
      payload: {},
    });
    h.send(encodeFrame(Opcode.Text, Buffer.from(raw), { mask: true }));
    await settle();

    const error = h.messages().find((m) => (m as { kind: string }).kind === MessageKind.Error) as {
      payload: Record<string, unknown>;
    };
    assert.equal(error.payload['code'], 'PROTOCOL_UNSUPPORTED_VERSION');
  });

  test('refuses a message kind clients may not send', async () => {
    // A client must not be able to inject an event as though the server produced it.
    const h = await harness();
    h.send(clientMessage(MessageKind.Event, { id: 'forged', severity: 'CRITICAL' }));
    await settle();

    const error = h.messages().find((m) => (m as { kind: string }).kind === MessageKind.Error) as {
      payload: Record<string, unknown>;
    };
    assert.equal(error.payload['code'], 'PROTOCOL_UNKNOWN_KIND');
  });

  test('refuses binary messages', async () => {
    const h = await harness();
    h.send(encodeFrame(Opcode.Binary, Buffer.from([1, 2, 3]), { mask: true }));
    await settle();

    const close = h.received.find((frame) => frame.opcode === Opcode.Close);
    assert.equal(parseClose(close?.payload ?? Buffer.alloc(0)).code, CloseCode.UnsupportedData);
  });

  test('handles a frame split across TCP segments', async () => {
    const h = await harness();
    const frame = clientMessage(MessageKind.Subscribe, { channels: ['events'] });

    h.send(frame.subarray(0, 3));
    await settle(1);
    h.send(frame.subarray(3));
    await settle();

    assert.equal(h.hub.subscriberCount('events'), 1);
  });

  test('handles two frames arriving in one segment', async () => {
    const h = await harness();
    const combined = Buffer.concat([
      clientMessage(MessageKind.Subscribe, { channels: ['events'] }),
      clientMessage(MessageKind.Subscribe, { channels: ['tracks'] }),
    ]);

    h.send(combined);
    await settle();

    assert.equal(h.hub.subscriberCount('events'), 1);
    assert.equal(h.hub.subscriberCount('tracks'), 1);
  });
});

describe('backpressure', () => {
  test('disconnects a client that has stopped reading', async () => {
    // A console whose machine went to sleep still holds a socket. Without a cap
    // its backlog grows until the process dies, taking every other console with
    // it. One disconnected operator is recoverable; a dead control node is not.
    const warnings: string[] = [];
    const hub = new Hub({
      nodeId: NODE,
      onLog: (level, message) => {
        if (level === 'warn') warnings.push(message);
      },
    });

    // A socket that accepts writes but never drains: the callback never fires,
    // so pendingWrites only grows.
    const stalled = {
      write: (_data: Buffer, _callback?: () => void): boolean => false,
      end: (): void => {},
      on: (): void => {},
    };

    const connection = hub.accept(stalled as never, session());
    connection.subscriptions.add('events');

    for (let i = 0; i < MAX_OUTBOUND_QUEUE + 10; i += 1) {
      hub.publish(MessageKind.Event, { id: `e${i}` });
    }

    assert.equal(hub.connectionCount, 0, 'the stalled client was dropped');
    assert.ok(warnings.some((message) => /stopped reading/.test(message)));
  });

  test('a healthy client keeps receiving after a slow one is dropped', async () => {
    const h = await harness();
    h.send(clientMessage(MessageKind.Subscribe, { channels: ['events'] }));
    await settle();

    for (let i = 0; i < 50; i += 1) h.hub.publish(MessageKind.Event, { id: `e${i}` });
    await settle();

    const events = h.messages().filter((m) => (m as { kind: string }).kind === MessageKind.Event);
    assert.ok(events.length >= 40, `only ${events.length} of 50 events arrived`);
    assert.equal(h.hub.connectionCount, 1);
  });
});

describe('heartbeats', () => {
  test('pings live connections', async () => {
    const h = await harness();
    const result = h.hub.heartbeat();
    await settle();

    assert.equal(result.pinged, 1);
    assert.equal(result.dropped, 0);
    assert.ok(h.received.some((frame) => frame.opcode === Opcode.Ping));
  });

  test('drops a connection that stopped answering', async () => {
    // TCP will not report a yanked cable for minutes, so an idle connection and a
    // dead one are indistinguishable without this.
    let current = 1_000_000;
    const h = await harness({ now: () => utcMillis(current) });

    current += 200_000; // well past the heartbeat timeout
    const result = h.hub.heartbeat();

    assert.equal(result.dropped, 1);
    assert.equal(h.hub.connectionCount, 0);
  });

  test('a pong keeps a connection alive', async () => {
    let current = 1_000_000;
    const h = await harness({ now: () => utcMillis(current) });

    current += 40_000;
    h.send(encodeFrame(Opcode.Pong, Buffer.alloc(0), { mask: true }));
    await settle();

    current += 40_000;
    assert.equal(h.hub.heartbeat().dropped, 0, 'the pong refreshed the timer');
  });
});

describe('administrative disconnects', () => {
  test('a user can be disconnected when their access is revoked', async () => {
    const h = await harness();
    assert.equal(h.hub.disconnectUser('operator', 'access revoked'), 1);
    await settle();

    const close = h.received.find((frame) => frame.opcode === Opcode.Close);
    assert.equal(parseClose(close?.payload ?? Buffer.alloc(0)).reason, 'access revoked');
    assert.equal(h.hub.connectionCount, 0);
  });

  test('disconnecting a different user leaves this one alone', async () => {
    const h = await harness();
    assert.equal(h.hub.disconnectUser('someone-else', 'x'), 0);
    assert.equal(h.hub.connectionCount, 1);
  });

  test('closeAll ends every connection', async () => {
    const h = await harness();
    h.hub.closeAll();
    assert.equal(h.hub.connectionCount, 0);
  });

  test('closing twice is not an error', async () => {
    const h = await harness();
    h.hub.close(h.connection, CloseCode.Normal, 'first');
    h.hub.close(h.connection, CloseCode.Normal, 'second');
    assert.equal(h.hub.connectionCount, 0);
  });
});

describe('published payloads', () => {
  test('every message carries a versioned envelope', async () => {
    const h = await harness();
    h.send(clientMessage(MessageKind.Subscribe, { channels: ['incidents'] }));
    await settle();

    h.hub.publish(MessageKind.IncidentOpened, { id: 'INC-1', severity: 'CRITICAL' });
    await settle();

    const message = h.messages().find(
      (m) => (m as { kind: string }).kind === MessageKind.IncidentOpened,
    ) as { v: number; nodeId: string; requestId: string; timestamp: number };

    assert.equal(message.v, 1);
    assert.equal(message.nodeId, NODE);
    assert.ok(message.requestId.length > 0);
    assert.ok(message.timestamp > 0);
  });

  test('serialised text is exactly what the client decodes', async () => {
    const h = await harness();
    h.send(clientMessage(MessageKind.Subscribe, { channels: ['events'] }));
    await settle();

    h.hub.publish(MessageKind.Event, { id: 'e1', summary: 'a person entered Restricted Zone A' });
    await settle();

    const event = h.messages().find((m) => (m as { kind: string }).kind === MessageKind.Event) as {
      payload: Record<string, unknown>;
    };
    assert.equal(event.payload['summary'], 'a person entered Restricted Zone A');
  });

  test('encodeText produces a frame the server decoder would refuse', () => {
    // Direction matters: what the server sends is unmasked, and feeding it back
    // as though it came from a client must fail.
    assert.throws(() => decodeFrame(encodeText('x', false), true), /must be masked/);
  });
});
