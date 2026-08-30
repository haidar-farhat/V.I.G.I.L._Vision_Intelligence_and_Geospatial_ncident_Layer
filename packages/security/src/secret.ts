/**
 * Secret handling.
 *
 * A camera password that reaches a log file, an error payload, a UI event or a
 * crash report is a breach - and in a system that logs structured objects
 * liberally, the default behaviour of JavaScript makes that leak the *easy* path.
 * `Secret<T>` inverts that: the value is inaccessible unless a caller explicitly
 * asks for it, and every path by which a value is normally rendered - string
 * coercion, `JSON.stringify`, template literals, console inspection - yields
 * `[redacted]` instead.
 *
 * The invariant this enforces: a credential can only escape by someone typing
 * `.expose()`, which is greppable, reviewable, and rare.
 */

const REDACTED = '[redacted]';

/** Node's console/util.inspect hook. Referenced without importing `node:util`. */
const INSPECT = Symbol.for('nodejs.util.inspect.custom');

export class Secret<T = string> {
  readonly #value: T;
  /** Non-secret label for logs: "camera cam-07 password", never the password. */
  readonly #label: string;

  constructor(value: T, label = 'secret') {
    this.#value = value;
    this.#label = label;
  }

  /**
   * Retrieve the underlying value.
   *
   * Deliberately verbose and deliberately greppable. Every call site is an
   * auditable decision to move a secret into ordinary memory; a review can find
   * all of them with a single search.
   */
  expose(): T {
    return this.#value;
  }

  get label(): string {
    return this.#label;
  }

  /** Constant-time-ish comparison that never reveals the value. */
  equals(other: Secret<T>): boolean {
    const a = String(this.#value);
    const b = String(other.#value);
    if (a.length !== b.length) return false;

    let diff = 0;
    for (let i = 0; i < a.length; i += 1) {
      diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
    }
    return diff === 0;
  }

  toString(): string {
    return REDACTED;
  }

  toJSON(): string {
    return REDACTED;
  }

  [INSPECT](): string {
    return `Secret(${this.#label}) ${REDACTED}`;
  }
}

export const secret = <T>(value: T, label?: string): Secret<T> => new Secret(value, label);

export const isSecret = (value: unknown): value is Secret<unknown> => value instanceof Secret;

// ---------------------------------------------------------------- redaction

/**
 * Field names whose values are removed from any structure before it is logged,
 * returned in an API payload, or written to an audit record.
 *
 * A denylist is a backstop, not the primary control - `Secret<T>` is - but
 * structures arrive from ONVIF responses, imported configuration and third-party
 * payloads where the typed wrapper was never applied.
 */
export const SENSITIVE_KEYS: readonly string[] = Object.freeze([
  'password',
  'passwd',
  'pass',
  'secret',
  'token',
  'accesstoken',
  'refreshtoken',
  'apikey',
  'api_key',
  'authorization',
  'auth',
  'credential',
  'credentials',
  'privatekey',
  'private_key',
  'passphrase',
  'sessionkey',
  'cookie',
  'set-cookie',
]);

const isSensitiveKey = (key: string): boolean => {
  const normalised = key.toLowerCase().replace(/[-_\s]/g, '');
  return SENSITIVE_KEYS.some((candidate) => normalised === candidate.replace(/[-_]/g, ''));
};

/**
 * Strip credentials embedded in a URL's userinfo section.
 *
 * `rtsp://admin:hunter2@192.168.1.50/stream` is how essentially every camera
 * integration leaks a password: the URL is built once and then logged, thrown in
 * an error, or shown in a diagnostic. The host and path are operationally useful,
 * so they are kept; the userinfo is not.
 */
export const redactUrl = (value: string): string =>
  value.replace(
    /\b([a-zA-Z][a-zA-Z0-9+.-]*:\/\/)([^/\s@]+)@/g,
    (_match, scheme: string, userinfo: string) => {
      const user = userinfo.split(':')[0] ?? '';
      return `${scheme}${user}:${REDACTED}@`;
    },
  );

/**
 * Recursively redact a value for logging or transport.
 *
 * Handles cycles, so a redaction call can never be the thing that crashes the
 * logger. Depth is bounded for the same reason.
 */
export const redact = (value: unknown, maxDepth = 12): unknown => {
  const seen = new WeakSet<object>();

  const walk = (current: unknown, depth: number): unknown => {
    if (depth > maxDepth) return '[truncated]';
    if (current === null || current === undefined) return current;
    if (isSecret(current)) return REDACTED;

    if (typeof current === 'string') return redactUrl(current);
    if (typeof current !== 'object') return current;

    if (seen.has(current)) return '[circular]';
    seen.add(current);

    if (Array.isArray(current)) return current.map((item) => walk(item, depth + 1));

    if (current instanceof Error) {
      return {
        name: current.name,
        message: redactUrl(current.message),
      };
    }

    const result: Record<string, unknown> = {};
    for (const [key, entry] of Object.entries(current)) {
      result[key] = isSensitiveKey(key) ? REDACTED : walk(entry, depth + 1);
    }
    return result;
  };

  return walk(value, 0);
};

/**
 * Build an RTSP URL with credentials, kept inside a `Secret` so the assembled
 * string cannot be logged by accident.
 *
 * This is the single place in the codebase permitted to join a credential to a
 * URL. Everything else passes the `Secret` around and lets the transport expose
 * it at the moment of connection.
 */
export const buildRtspUrl = (
  host: string,
  port: number,
  path: string,
  username?: string,
  password?: Secret<string>,
): Secret<string> => {
  const normalisedPath = path.startsWith('/') ? path : `/${path}`;

  if (username === undefined || password === undefined) {
    return secret(`rtsp://${host}:${port}${normalisedPath}`, `rtsp ${host}`);
  }

  const userinfo = `${encodeURIComponent(username)}:${encodeURIComponent(password.expose())}`;
  return secret(`rtsp://${userinfo}@${host}:${port}${normalisedPath}`, `rtsp ${host}`);
};

/** A safe, loggable description of a stream endpoint. Never carries credentials. */
export const describeStream = (host: string, port: number, path: string): string =>
  `rtsp://${host}:${port}${path.startsWith('/') ? path : `/${path}`}`;
