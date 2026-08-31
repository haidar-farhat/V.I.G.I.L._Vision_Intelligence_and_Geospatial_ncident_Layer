import { randomBytes } from 'node:crypto';

/**
 * RFC 6455 frame codec.
 *
 * Written rather than imported for the same reason the rest of the core is: a
 * security appliance's dependency list is part of its attack surface, and this is
 * a well-specified format of a few hundred lines. Writing it also means the size
 * limits, the fragmentation rules and the close semantics are decisions made here
 * rather than inherited from a library's defaults.
 *
 * Everything this parses arrives from the network. The rules that follow from
 * that, all of which the specification requires and implementations routinely
 * miss:
 *
 *  - A client-to-server frame **must** be masked, and a server-to-client frame
 *    must not be. An unmasked client frame is either a broken client or a
 *    cache-poisoning attempt through a transparent proxy, and RFC 6455 requires
 *    the connection be failed rather than the frame accepted.
 *  - Payload length is checked **before** allocating, so a peer cannot claim a
 *    2 GB frame and have the buffer reserved on its say-so.
 *  - Control frames may not be fragmented and may not exceed 125 bytes.
 *  - Reserved bits must be zero when no extension has been negotiated, and this
 *    build negotiates none.
 */

export const Opcode = {
  Continuation: 0x0,
  Text: 0x1,
  Binary: 0x2,
  Close: 0x8,
  Ping: 0x9,
  Pong: 0xa,
} as const;
export type Opcode = (typeof Opcode)[keyof typeof Opcode];

/** RFC 6455 section 7.4.1, plus the ones this implementation actually sends. */
export const CloseCode = {
  Normal: 1000,
  GoingAway: 1001,
  ProtocolError: 1002,
  UnsupportedData: 1003,
  InvalidPayload: 1007,
  PolicyViolation: 1008,
  MessageTooBig: 1009,
  InternalError: 1011,
} as const;
export type CloseCode = (typeof CloseCode)[keyof typeof CloseCode];

export const MAX_FRAME_BYTES = 4 * 1024 * 1024;
export const MAX_CONTROL_FRAME_BYTES = 125;

export class WebSocketProtocolError extends Error {
  readonly closeCode: CloseCode;

  constructor(message: string, closeCode: CloseCode = CloseCode.ProtocolError) {
    super(message);
    this.name = 'WebSocketProtocolError';
    this.closeCode = closeCode;
  }
}

export type Frame = {
  readonly fin: boolean;
  readonly opcode: Opcode;
  readonly payload: Buffer;
};

export type DecodeResult = {
  readonly frame: Frame;
  /** Bytes consumed, so the caller can advance its buffer. */
  readonly consumed: number;
};

const isControlOpcode = (opcode: number): boolean => (opcode & 0x08) !== 0;

const KNOWN_OPCODES = new Set<number>([
  Opcode.Continuation,
  Opcode.Text,
  Opcode.Binary,
  Opcode.Close,
  Opcode.Ping,
  Opcode.Pong,
]);

/**
 * Decode one frame.
 *
 * Returns null when more bytes are needed, so a caller feeding a socket can keep
 * reading. Throws only on frames that are actually invalid - a partial frame is a
 * normal condition on a stream, not an error.
 */
export const decodeFrame = (buffer: Buffer, expectMasked: boolean): DecodeResult | null => {
  if (buffer.length < 2) return null;

  const first = buffer[0] ?? 0;
  const second = buffer[1] ?? 0;

  const fin = (first & 0x80) !== 0;
  const reserved = first & 0x70;
  const opcode = first & 0x0f;
  const masked = (second & 0x80) !== 0;
  let length = second & 0x7f;

  // No extensions are negotiated, so a set reserved bit means the peer believes
  // something was agreed that was not.
  if (reserved !== 0) {
    throw new WebSocketProtocolError('Reserved bits are set but no extension was negotiated.');
  }

  if (!KNOWN_OPCODES.has(opcode)) {
    throw new WebSocketProtocolError(`Unknown opcode 0x${opcode.toString(16)}.`);
  }

  if (isControlOpcode(opcode)) {
    if (!fin) {
      throw new WebSocketProtocolError('Control frames may not be fragmented.');
    }
    if (length > MAX_CONTROL_FRAME_BYTES) {
      throw new WebSocketProtocolError(
        `Control frame payload is ${length} bytes, beyond the ${MAX_CONTROL_FRAME_BYTES}-byte limit.`,
      );
    }
  }

  // Masking is not a confidentiality measure - the key travels with the frame.
  // It exists so a frame cannot be crafted to look like a valid HTTP request to
  // an intermediary, which is why the direction rule is absolute.
  if (masked !== expectMasked) {
    throw new WebSocketProtocolError(
      expectMasked
        ? 'Client frames must be masked. Refusing an unmasked frame.'
        : 'Server frames must not be masked. Refusing a masked frame.',
    );
  }

  let offset = 2;

  if (length === 126) {
    if (buffer.length < offset + 2) return null;
    length = buffer.readUInt16BE(offset);
    offset += 2;
  } else if (length === 127) {
    if (buffer.length < offset + 8) return null;

    const extended = buffer.readBigUInt64BE(offset);
    // Checked before the comparison below so a length above 2^53 cannot be
    // silently rounded into an acceptable-looking number.
    if (extended > BigInt(MAX_FRAME_BYTES)) {
      throw new WebSocketProtocolError(
        `Frame declares ${extended} bytes, beyond the ${MAX_FRAME_BYTES}-byte limit.`,
        CloseCode.MessageTooBig,
      );
    }
    length = Number(extended);
    offset += 8;
  }

  if (length > MAX_FRAME_BYTES) {
    throw new WebSocketProtocolError(
      `Frame declares ${length} bytes, beyond the ${MAX_FRAME_BYTES}-byte limit.`,
      CloseCode.MessageTooBig,
    );
  }

  const maskLength = masked ? 4 : 0;
  // The whole frame must be present before anything is copied. Nothing is
  // allocated on the strength of a declared length alone.
  if (buffer.length < offset + maskLength + length) return null;

  const mask = masked ? buffer.subarray(offset, offset + 4) : null;
  offset += maskLength;

  const payload = Buffer.allocUnsafe(length);
  buffer.copy(payload, 0, offset, offset + length);

  if (mask !== null) {
    for (let i = 0; i < length; i += 1) {
      payload[i] = (payload[i] ?? 0) ^ (mask[i % 4] ?? 0);
    }
  }

  return {
    frame: { fin, opcode: opcode as Opcode, payload },
    consumed: offset + length,
  };
};

/**
 * Encode a frame.
 *
 * `mask` is required for a client and forbidden for a server; the caller states
 * which it is rather than this guessing from context.
 */
export const encodeFrame = (
  opcode: Opcode,
  payload: Buffer,
  options: { readonly fin?: boolean; readonly mask?: boolean } = {},
): Buffer => {
  const fin = options.fin ?? true;
  const mask = options.mask ?? false;
  const length = payload.length;

  if (isControlOpcode(opcode) && length > MAX_CONTROL_FRAME_BYTES) {
    throw new WebSocketProtocolError(
      `Control frame payload is ${length} bytes, beyond the ${MAX_CONTROL_FRAME_BYTES}-byte limit.`,
    );
  }

  const lengthBytes = length < 126 ? 0 : length < 65536 ? 2 : 8;
  const header = Buffer.allocUnsafe(2 + lengthBytes + (mask ? 4 : 0));

  header[0] = (fin ? 0x80 : 0) | opcode;

  if (lengthBytes === 0) {
    header[1] = (mask ? 0x80 : 0) | length;
  } else if (lengthBytes === 2) {
    header[1] = (mask ? 0x80 : 0) | 126;
    header.writeUInt16BE(length, 2);
  } else {
    header[1] = (mask ? 0x80 : 0) | 127;
    header.writeBigUInt64BE(BigInt(length), 2);
  }

  if (!mask) return Buffer.concat([header, payload]);

  const maskKey = randomBytes(4);
  maskKey.copy(header, 2 + lengthBytes);

  const masked = Buffer.allocUnsafe(length);
  for (let i = 0; i < length; i += 1) {
    masked[i] = (payload[i] ?? 0) ^ (maskKey[i % 4] ?? 0);
  }

  return Buffer.concat([header, masked]);
};

export const encodeText = (text: string, mask = false): Buffer =>
  encodeFrame(Opcode.Text, Buffer.from(text, 'utf8'), { mask });

export const encodeClose = (code: CloseCode, reason = '', mask = false): Buffer => {
  const reasonBytes = Buffer.from(reason, 'utf8');
  // The reason is truncated rather than the frame refused: failing to send a
  // close because the explanation was long would leave the peer with no
  // explanation at all.
  const truncated = reasonBytes.subarray(0, MAX_CONTROL_FRAME_BYTES - 2);

  const payload = Buffer.allocUnsafe(2 + truncated.length);
  payload.writeUInt16BE(code, 0);
  truncated.copy(payload, 2);

  return encodeFrame(Opcode.Close, payload, { mask });
};

export const parseClose = (payload: Buffer): { code: number; reason: string } => {
  if (payload.length < 2) return { code: CloseCode.Normal, reason: '' };
  return {
    code: payload.readUInt16BE(0),
    reason: payload.subarray(2).toString('utf8'),
  };
};

/**
 * Reassembles fragmented messages.
 *
 * A large message may arrive as a text frame followed by continuations, with
 * control frames interleaved - a ping must be answerable mid-message. This
 * enforces the ordering rules and caps the total so fragmentation cannot be used
 * to smuggle a frame past the per-frame limit.
 */
export class MessageAssembler {
  readonly #maxBytes: number;
  #opcode: Opcode | null = null;
  #chunks: Buffer[] = [];
  #size = 0;

  constructor(maxBytes: number = MAX_FRAME_BYTES) {
    this.#maxBytes = maxBytes;
  }

  /**
   * Feed a data frame. Returns the complete message, or null while incomplete.
   * Control frames must be handled by the caller and never passed here.
   */
  accept(frame: Frame): { readonly opcode: Opcode; readonly payload: Buffer } | null {
    if (frame.opcode === Opcode.Continuation) {
      if (this.#opcode === null) {
        throw new WebSocketProtocolError('Continuation frame with no message in progress.');
      }
    } else {
      if (this.#opcode !== null) {
        throw new WebSocketProtocolError(
          'New data frame arrived while a fragmented message was still in progress.',
        );
      }
      this.#opcode = frame.opcode;
    }

    this.#size += frame.payload.length;
    if (this.#size > this.#maxBytes) {
      this.reset();
      throw new WebSocketProtocolError(
        `Fragmented message exceeds the ${this.#maxBytes}-byte limit.`,
        CloseCode.MessageTooBig,
      );
    }

    this.#chunks.push(frame.payload);

    if (!frame.fin) return null;

    const opcode = this.#opcode;
    const payload = this.#chunks.length === 1 ? this.#chunks[0]! : Buffer.concat(this.#chunks);
    this.reset();

    return { opcode: opcode ?? Opcode.Binary, payload };
  }

  reset(): void {
    this.#opcode = null;
    this.#chunks = [];
    this.#size = 0;
  }

  get inProgress(): boolean {
    return this.#opcode !== null;
  }
}
