import { randomUUID } from 'node:crypto';
import type { NodeId, RequestId, UtcMillis } from '@sentinel/shared-types';
import { asId, utcMillis } from '@sentinel/shared-types';

/**
 * The message envelope.
 *
 * Every message on every channel carries the same four fields, and the rule that
 * makes them worth having: **a version mismatch is a hard, explicit failure**,
 * never a best-effort parse. A system that accepts a message it does not fully
 * understand is a system that will one day act on a field that moved.
 *
 * The other three fields exist for things that only matter once something has
 * gone wrong: `requestId` correlates a trace across three processes, `timestamp`
 * lets a replay be refused, and `nodeId` says which machine is talking.
 */

/**
 * Current protocol version.
 *
 * Increment on **any** wire-visible change. There is no minor version: a field
 * added is a field some peer will not send, and pretending otherwise is how a
 * distributed system develops a fault nobody can reproduce.
 */
export const PROTOCOL_VERSION = 1;

/** Versions this build can talk to. Widened only when compatibility is proven. */
export const SUPPORTED_VERSIONS: readonly number[] = Object.freeze([1]);

export type Envelope<K extends string = string, P = unknown> = {
  readonly v: number;
  readonly requestId: RequestId;
  readonly timestamp: UtcMillis;
  readonly nodeId: NodeId;
  readonly kind: K;
  readonly payload: P;
};

/** Hard limits. A peer must not be able to force unbounded work or allocation. */
export const MAX_MESSAGE_BYTES = 4 * 1024 * 1024;
export const MAX_KIND_LENGTH = 64;

export const newRequestId = (): RequestId => asId<RequestId>(randomUUID());

export const envelope = <K extends string, P>(
  kind: K,
  payload: P,
  nodeId: NodeId,
  options: { readonly requestId?: RequestId; readonly now?: () => UtcMillis } = {},
): Envelope<K, P> => ({
  v: PROTOCOL_VERSION,
  requestId: options.requestId ?? newRequestId(),
  timestamp: (options.now ?? (() => utcMillis(Date.now())))(),
  nodeId,
  kind,
  payload,
});

// --------------------------------------------------------------------- errors

export const ProtocolErrorCode = {
  Malformed: 'PROTOCOL_MALFORMED',
  UnsupportedVersion: 'PROTOCOL_UNSUPPORTED_VERSION',
  TooLarge: 'PROTOCOL_TOO_LARGE',
  MissingField: 'PROTOCOL_MISSING_FIELD',
  BadField: 'PROTOCOL_BAD_FIELD',
  UnknownKind: 'PROTOCOL_UNKNOWN_KIND',
} as const;
export type ProtocolErrorCode = (typeof ProtocolErrorCode)[keyof typeof ProtocolErrorCode];

export class ProtocolError extends Error {
  readonly code: ProtocolErrorCode;
  readonly recoverable: boolean;

  constructor(message: string, code: ProtocolErrorCode, recoverable = false) {
    super(message);
    this.name = 'ProtocolError';
    this.code = code;
    this.recoverable = recoverable;
  }
}

// ----------------------------------------------------------------- validation

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

/**
 * Parse and validate an envelope from the wire.
 *
 * Everything arriving here is untrusted, including from a node that was paired
 * yesterday: pairing proves identity, not that firmware has not been replaced
 * since. So the shape is checked field by field and a failure names the field,
 * because "malformed message" tells whoever is debugging a version skew nothing
 * at all.
 */
export const parseEnvelope = (raw: string | Uint8Array): Envelope => {
  const text = typeof raw === 'string' ? raw : new TextDecoder().decode(raw);

  if (text.length > MAX_MESSAGE_BYTES) {
    throw new ProtocolError(
      `Message is ${text.length} bytes, beyond the ${MAX_MESSAGE_BYTES}-byte limit.`,
      ProtocolErrorCode.TooLarge,
    );
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    throw new ProtocolError('Message is not valid JSON.', ProtocolErrorCode.Malformed);
  }

  if (!isRecord(parsed)) {
    throw new ProtocolError('Message is not a JSON object.', ProtocolErrorCode.Malformed);
  }

  // Version is checked before anything else. A message from a version this build
  // does not know may have moved every other field, so validating them against
  // this version's expectations would produce a misleading error.
  const version = parsed['v'];
  if (typeof version !== 'number' || !Number.isInteger(version)) {
    throw new ProtocolError(
      'Message has no integer protocol version.',
      ProtocolErrorCode.MissingField,
    );
  }
  if (!SUPPORTED_VERSIONS.includes(version)) {
    throw new ProtocolError(
      `Message uses protocol version ${version}; this build speaks ` +
        `${SUPPORTED_VERSIONS.join(', ')}. Upgrade whichever side is older - the ` +
        'protocol is not backward compatible by design.',
      ProtocolErrorCode.UnsupportedVersion,
    );
  }

  const requestId = parsed['requestId'];
  if (typeof requestId !== 'string' || requestId === '') {
    throw new ProtocolError('Message has no requestId.', ProtocolErrorCode.MissingField);
  }

  const timestamp = parsed['timestamp'];
  if (typeof timestamp !== 'number' || !Number.isFinite(timestamp)) {
    throw new ProtocolError('Message has no numeric timestamp.', ProtocolErrorCode.MissingField);
  }

  const nodeId = parsed['nodeId'];
  if (typeof nodeId !== 'string' || nodeId === '') {
    throw new ProtocolError('Message has no nodeId.', ProtocolErrorCode.MissingField);
  }

  const kind = parsed['kind'];
  if (typeof kind !== 'string' || kind === '') {
    throw new ProtocolError('Message has no kind.', ProtocolErrorCode.MissingField);
  }
  if (kind.length > MAX_KIND_LENGTH) {
    throw new ProtocolError(
      `Message kind is longer than ${MAX_KIND_LENGTH} characters.`,
      ProtocolErrorCode.BadField,
    );
  }

  return {
    v: version,
    requestId: asId<RequestId>(requestId),
    timestamp: utcMillis(timestamp),
    nodeId: asId<NodeId>(nodeId),
    kind,
    payload: parsed['payload'],
  };
};

export const serialiseEnvelope = (message: Envelope): string => JSON.stringify(message);

/**
 * A typed reader for a payload field.
 *
 * Validation here is deliberately manual rather than schema-driven. A schema
 * library would be a runtime dependency in a package that has none, and the field
 * count is small enough that hand-written checks stay readable - while producing
 * far better messages than a generic validator's path expressions.
 */
export class PayloadReader {
  readonly #payload: Record<string, unknown>;
  readonly #kind: string;

  constructor(payload: unknown, kind: string) {
    if (!isRecord(payload)) {
      throw new ProtocolError(
        `Payload of "${kind}" is not an object.`,
        ProtocolErrorCode.BadField,
      );
    }
    this.#payload = payload;
    this.#kind = kind;
  }

  string(field: string): string {
    const value = this.#payload[field];
    if (typeof value !== 'string') {
      throw new ProtocolError(
        `"${this.#kind}" requires a string "${field}".`,
        ProtocolErrorCode.BadField,
      );
    }
    return value;
  }

  optionalString(field: string): string | undefined {
    const value = this.#payload[field];
    if (value === undefined || value === null) return undefined;
    if (typeof value !== 'string') {
      throw new ProtocolError(
        `"${this.#kind}" expects "${field}" to be a string when present.`,
        ProtocolErrorCode.BadField,
      );
    }
    return value;
  }

  number(field: string): number {
    const value = this.#payload[field];
    if (typeof value !== 'number' || !Number.isFinite(value)) {
      throw new ProtocolError(
        `"${this.#kind}" requires a finite number "${field}".`,
        ProtocolErrorCode.BadField,
      );
    }
    return value;
  }

  boolean(field: string): boolean {
    const value = this.#payload[field];
    if (typeof value !== 'boolean') {
      throw new ProtocolError(
        `"${this.#kind}" requires a boolean "${field}".`,
        ProtocolErrorCode.BadField,
      );
    }
    return value;
  }

  stringArray(field: string, maxLength = 1000): readonly string[] {
    const value = this.#payload[field];
    if (!Array.isArray(value)) {
      throw new ProtocolError(
        `"${this.#kind}" requires an array "${field}".`,
        ProtocolErrorCode.BadField,
      );
    }
    if (value.length > maxLength) {
      throw new ProtocolError(
        `"${this.#kind}" field "${field}" has ${value.length} entries, beyond the ${maxLength} limit.`,
        ProtocolErrorCode.TooLarge,
      );
    }
    for (const entry of value) {
      if (typeof entry !== 'string') {
        throw new ProtocolError(
          `"${this.#kind}" field "${field}" must contain only strings.`,
          ProtocolErrorCode.BadField,
        );
      }
    }
    return value as readonly string[];
  }

  /** The raw value, for payloads a caller validates itself. */
  raw(field: string): unknown {
    return this.#payload[field];
  }

  has(field: string): boolean {
    return this.#payload[field] !== undefined;
  }
}
