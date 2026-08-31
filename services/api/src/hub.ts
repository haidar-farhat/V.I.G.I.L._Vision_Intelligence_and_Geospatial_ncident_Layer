import type { Duplex } from 'node:stream';
import type { NodeId, UtcMillis } from '@sentinel/shared-types';
import { utcMillis } from '@sentinel/shared-types';
import type { Channel, Envelope, Frame } from '@sentinel/protocol';
import {
  ALL_CHANNELS,
  CloseCode,
  MessageAssembler,
  MessageKind,
  Opcode,
  ProtocolError,
  WebSocketProtocolError,
  channelForKind,
  decodeFrame,
  encodeClose,
  encodeFrame,
  encodeText,
  envelope,
  isChannel,
  parseEnvelope,
  serialiseEnvelope,
} from '@sentinel/protocol';
import type { Session } from './auth.ts';

/**
 * The realtime hub.
 *
 * Holds every open operator connection and pushes to the channels each has asked
 * for. Three problems make this harder than a broadcast loop, and each is the
 * reason for a specific piece of machinery below.
 *
 * **A slow client must not slow the system.** An operator on a laptop that has
 * gone to sleep still holds a socket. Writing to it blocks in the kernel, and a
 * naive broadcast loop then stalls every other client behind it. So each
 * connection has a bounded outbound queue and is disconnected when it overflows -
 * losing one console is recoverable, stalling the event stream is not.
 *
 * **A dead connection looks exactly like an idle one.** TCP will not notice a
 * yanked cable for minutes. Heartbeats make the difference observable, so a
 * disconnected console is reported rather than silently receiving nothing.
 *
 * **The client is not trusted after the handshake.** Authentication happens
 * once; message validation happens every time.
 */

export const HEARTBEAT_INTERVAL_MILLIS = 30_000;
export const HEARTBEAT_TIMEOUT_MILLIS = 75_000;

/**
 * Outbound queue depth per connection.
 *
 * Roughly ten seconds of a busy site's traffic. Deep enough to ride out a garbage
 * collection pause or a slow render; shallow enough that a genuinely stalled
 * client is disconnected long before its backlog matters.
 */
export const MAX_OUTBOUND_QUEUE = 256;

export type Connection = {
  readonly id: string;
  readonly session: Session;
  readonly socket: Duplex;
  readonly subscriptions: Set<Channel>;
  readonly connectedAt: UtcMillis;
  lastSeenAt: UtcMillis;
  /** Frames written but not yet flushed. Bounded; overflow closes the socket. */
  pendingWrites: number;
  closed: boolean;
  buffer: Buffer;
  readonly assembler: MessageAssembler;
};

export type HubOptions = {
  readonly nodeId: NodeId;
  readonly now?: () => UtcMillis;
  readonly onLog?: (level: 'info' | 'warn', message: string, detail?: unknown) => void;
};

export class Hub {
  readonly #connections = new Map<string, Connection>();
  readonly #nodeId: NodeId;
  readonly #now: () => UtcMillis;
  readonly #log: (level: 'info' | 'warn', message: string, detail?: unknown) => void;
  #sequence = 0;

  constructor(options: HubOptions) {
    this.#nodeId = options.nodeId;
    this.#now = options.now ?? (() => utcMillis(Date.now()));
    this.#log = options.onLog ?? (() => {});
  }

  get connectionCount(): number {
    return this.#connections.size;
  }

  /** Connections currently subscribed to a channel, for diagnostics. */
  subscriberCount(channel: Channel): number {
    let count = 0;
    for (const connection of this.#connections.values()) {
      if (connection.subscriptions.has(channel)) count += 1;
    }
    return count;
  }

  /** Adopt a socket whose handshake has completed and whose session is known. */
  accept(socket: Duplex, session: Session): Connection {
    this.#sequence += 1;
    const now = this.#now();

    const connection: Connection = {
      id: `conn-${this.#sequence}`,
      session,
      socket,
      subscriptions: new Set<Channel>(),
      connectedAt: now,
      lastSeenAt: now,
      pendingWrites: 0,
      closed: false,
      buffer: Buffer.alloc(0),
      assembler: new MessageAssembler(),
    };

    this.#connections.set(connection.id, connection);

    socket.on('data', (chunk: Buffer) => this.#onData(connection, chunk));
    socket.on('error', () => this.#drop(connection, 'socket error'));
    socket.on('close', () => this.#drop(connection, 'socket closed'));

    // The welcome names the channels available, so a client never has to hard-code
    // a list that could drift from the server's.
    this.#send(
      connection,
      envelope(
        MessageKind.Welcome,
        {
          connectionId: connection.id,
          username: session.username,
          roles: session.roles,
          channels: ALL_CHANNELS,
          heartbeatIntervalMillis: HEARTBEAT_INTERVAL_MILLIS,
        },
        this.#nodeId,
        { now: this.#now },
      ),
    );

    return connection;
  }

  #onData(connection: Connection, chunk: Buffer): void {
    if (connection.closed) return;

    connection.buffer = Buffer.concat([connection.buffer, chunk]);
    connection.lastSeenAt = this.#now();

    for (;;) {
      let decoded: { frame: Frame; consumed: number } | null;
      try {
        // Client frames must be masked; the codec enforces the direction rule.
        decoded = decodeFrame(connection.buffer, true);
      } catch (error) {
        const code =
          error instanceof WebSocketProtocolError ? error.closeCode : CloseCode.ProtocolError;
        this.close(connection, code, error instanceof Error ? error.message : 'protocol error');
        return;
      }

      if (decoded === null) return;
      connection.buffer = connection.buffer.subarray(decoded.consumed);

      try {
        this.#onFrame(connection, decoded.frame);
      } catch (error) {
        const code =
          error instanceof WebSocketProtocolError ? error.closeCode : CloseCode.ProtocolError;
        this.close(connection, code, error instanceof Error ? error.message : 'protocol error');
        return;
      }

      if (connection.closed) return;
    }
  }

  #onFrame(connection: Connection, frame: Frame): void {
    switch (frame.opcode) {
      case Opcode.Ping:
        // Control frames are answered immediately and never queued behind data.
        this.#write(connection, encodeFrame(Opcode.Pong, frame.payload));
        return;

      case Opcode.Pong:
        connection.lastSeenAt = this.#now();
        return;

      case Opcode.Close:
        this.close(connection, CloseCode.Normal, 'client closed');
        return;

      case Opcode.Text:
      case Opcode.Continuation:
      case Opcode.Binary: {
        const message = connection.assembler.accept(frame);
        if (message === null) return;

        if (message.opcode === Opcode.Binary) {
          // Every message this protocol defines is JSON text. A binary message
          // means the peer is speaking something else.
          this.close(connection, CloseCode.UnsupportedData, 'binary messages are not accepted');
          return;
        }

        this.#onMessage(connection, message.payload.toString('utf8'));
        return;
      }

      default:
        this.close(connection, CloseCode.ProtocolError, 'unexpected opcode');
    }
  }

  #onMessage(connection: Connection, raw: string): void {
    let message: Envelope;
    try {
      message = parseEnvelope(raw);
    } catch (error) {
      // A malformed message is refused with an explanation rather than ignored.
      // Silence here is how a version skew becomes an afternoon of packet
      // captures.
      this.#send(
        connection,
        envelope(
          MessageKind.Error,
          {
            code: error instanceof ProtocolError ? error.code : 'PROTOCOL_MALFORMED',
            message: error instanceof Error ? error.message : 'unparseable message',
          },
          this.#nodeId,
          { now: this.#now },
        ),
      );
      return;
    }

    switch (message.kind) {
      case MessageKind.Ping:
        this.#send(connection, envelope(MessageKind.Pong, {}, this.#nodeId, { now: this.#now }));
        return;

      case MessageKind.Subscribe:
      case MessageKind.Unsubscribe: {
        const requested = Array.isArray((message.payload as { channels?: unknown })?.channels)
          ? ((message.payload as { channels: unknown[] }).channels)
          : [];

        const accepted: Channel[] = [];
        const rejected: string[] = [];

        for (const entry of requested) {
          if (typeof entry !== 'string' || !isChannel(entry)) {
            rejected.push(String(entry));
            continue;
          }
          if (message.kind === MessageKind.Subscribe) connection.subscriptions.add(entry);
          else connection.subscriptions.delete(entry);
          accepted.push(entry);
        }

        this.#send(
          connection,
          envelope(
            MessageKind.Subscribed,
            {
              subscribed: [...connection.subscriptions],
              accepted,
              // Named explicitly: a silently ignored subscription produces a
              // client that waits forever for a channel it never joined.
              rejected,
            },
            this.#nodeId,
            { now: this.#now },
          ),
        );
        return;
      }

      default:
        this.#send(
          connection,
          envelope(
            MessageKind.Error,
            {
              code: 'PROTOCOL_UNKNOWN_KIND',
              message: `This server does not accept "${message.kind}" from a client.`,
            },
            this.#nodeId,
            { now: this.#now },
          ),
        );
    }
  }

  /**
   * Publish to every connection subscribed to the message's channel.
   *
   * Returns how many connections received it, which is what the diagnostics
   * screen shows: an event generated but delivered to nobody is a different
   * problem from one never generated.
   */
  publish<P>(kind: string, payload: P): number {
    const channel = channelForKind(kind);
    if (channel === null) return 0;

    const message = envelope(kind, payload, this.#nodeId, { now: this.#now });
    const encoded = encodeText(serialiseEnvelope(message));

    let delivered = 0;
    for (const connection of this.#connections.values()) {
      if (connection.closed || !connection.subscriptions.has(channel)) continue;
      if (this.#write(connection, encoded)) delivered += 1;
    }
    return delivered;
  }

  /** Send to one connection regardless of its subscriptions. */
  #send(connection: Connection, message: Envelope): void {
    this.#write(connection, encodeText(serialiseEnvelope(message)));
  }

  /**
   * Write, refusing to buffer without limit.
   *
   * A console whose machine has gone to sleep still holds a socket. Without a cap
   * its backlog grows until the process dies, taking every other console with it.
   * One disconnected operator is recoverable; a dead control node is not.
   */
  #write(connection: Connection, data: Buffer): boolean {
    if (connection.closed) return false;

    if (connection.pendingWrites >= MAX_OUTBOUND_QUEUE) {
      this.#log('warn', 'disconnecting a client that stopped reading', {
        connectionId: connection.id,
        username: connection.session.username,
        pending: connection.pendingWrites,
      });
      this.close(connection, CloseCode.PolicyViolation, 'client is not consuming its stream');
      return false;
    }

    connection.pendingWrites += 1;
    connection.socket.write(data, () => {
      connection.pendingWrites = Math.max(0, connection.pendingWrites - 1);
    });

    return true;
  }

  /**
   * Heartbeat sweep.
   *
   * TCP will not report a yanked cable for minutes, so an idle connection and a
   * dead one are indistinguishable without this. Called on a timer by the server.
   */
  heartbeat(): { readonly pinged: number; readonly dropped: number } {
    const now = this.#now();
    let pinged = 0;
    let dropped = 0;

    for (const connection of [...this.#connections.values()]) {
      if (connection.closed) continue;

      if (now - connection.lastSeenAt > HEARTBEAT_TIMEOUT_MILLIS) {
        this.close(connection, CloseCode.GoingAway, 'no response to heartbeat');
        dropped += 1;
        continue;
      }

      this.#write(connection, encodeFrame(Opcode.Ping, Buffer.alloc(0)));
      pinged += 1;
    }

    return { pinged, dropped };
  }

  close(connection: Connection, code: CloseCode, reason: string): void {
    if (connection.closed) return;
    connection.closed = true;

    try {
      connection.socket.write(encodeClose(code, reason));
    } catch {
      // The socket is already gone, which is the case this is cleaning up after.
    }

    connection.socket.end();
    this.#connections.delete(connection.id);
  }

  #drop(connection: Connection, reason: string): void {
    if (connection.closed) {
      this.#connections.delete(connection.id);
      return;
    }
    connection.closed = true;
    this.#connections.delete(connection.id);
    this.#log('info', 'connection ended', { connectionId: connection.id, reason });
  }

  /** Close every connection, e.g. on shutdown. */
  closeAll(reason = 'server shutting down'): void {
    for (const connection of [...this.#connections.values()]) {
      this.close(connection, CloseCode.GoingAway, reason);
    }
  }

  /** Revoke a user's connections when their session is revoked or role changes. */
  disconnectUser(username: string, reason: string): number {
    let closed = 0;
    for (const connection of [...this.#connections.values()]) {
      if (connection.session.username !== username) continue;
      this.close(connection, CloseCode.PolicyViolation, reason);
      closed += 1;
    }
    return closed;
  }
}
