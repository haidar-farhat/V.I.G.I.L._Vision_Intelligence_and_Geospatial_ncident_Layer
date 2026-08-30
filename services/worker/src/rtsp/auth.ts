import { createHash, randomBytes } from 'node:crypto';
import type { Secret } from '@sentinel/security';

/**
 * RTSP authentication (RFC 7616 Digest, RFC 7617 Basic).
 *
 * Every function here takes the password as a `Secret<string>` and exposes it for
 * exactly the length of one hash computation. The credential is never returned,
 * never stored on the challenge object, and never appears in the header this
 * produces - Digest sends a hash, which is precisely why it is preferred.
 *
 * Basic is implemented because a depressing number of cameras still offer nothing
 * else, but it transmits the password in reversible form and is therefore gated
 * behind an explicit opt-in with a stated reason.
 */

/** Digest algorithms this client implements. Anything else is refused, not guessed. */
export type DigestAlgorithm = 'MD5' | 'MD5-sess' | 'SHA-256' | 'SHA-256-sess';

export type AuthChallenge =
  | {
      readonly scheme: 'Digest';
      readonly realm: string;
      readonly nonce: string;
      readonly opaque: string | null;
      readonly algorithm: DigestAlgorithm;
      readonly qop: readonly string[];
      readonly stale: boolean;
    }
  | { readonly scheme: 'Basic'; readonly realm: string };

/**
 * Parse a WWW-Authenticate header.
 *
 * Returns every challenge offered, strongest first, so a caller can prefer Digest
 * when a device offers both - which many do, listing Basic first.
 */
export const parseAuthChallenges = (header: string): readonly AuthChallenge[] => {
  const challenges: AuthChallenge[] = [];

  // Split on scheme boundaries. A single header may carry several challenges.
  const parts = header.split(/,\s*(?=(?:Digest|Basic)\s)/i);

  for (const part of parts) {
    const trimmed = part.trim();

    if (/^Digest\s/i.test(trimmed)) {
      const parameters = parseAuthParameters(trimmed.slice(6));
      const realm = parameters['realm'] ?? '';
      const nonce = parameters['nonce'] ?? '';
      // A Digest challenge without a nonce cannot be answered; treating it as
      // valid would produce an unauthenticated request that looks authenticated.
      if (nonce === '') continue;

      const algorithm = normaliseAlgorithm(parameters['algorithm']);
      if (algorithm === null) continue;

      challenges.push({
        scheme: 'Digest',
        realm,
        nonce,
        opaque: parameters['opaque'] ?? null,
        algorithm,
        qop:
          parameters['qop'] === undefined
            ? []
            : parameters['qop'].split(',').map((q) => q.trim()).filter((q) => q !== ''),
        stale: (parameters['stale'] ?? '').toLowerCase() === 'true',
      });
    } else if (/^Basic\s/i.test(trimmed)) {
      const parameters = parseAuthParameters(trimmed.slice(5));
      challenges.push({ scheme: 'Basic', realm: parameters['realm'] ?? '' });
    }
  }

  // Digest before Basic, always.
  return challenges.sort((a, b) => (a.scheme === b.scheme ? 0 : a.scheme === 'Digest' ? -1 : 1));
};

const normaliseAlgorithm = (raw: string | undefined): DigestAlgorithm | null => {
  // Absent means MD5 per RFC 7616, which is what most cameras rely on.
  const value = (raw ?? 'MD5').toUpperCase();
  switch (value) {
    case 'MD5':
      return 'MD5';
    case 'MD5-SESS':
      return 'MD5-sess';
    case 'SHA-256':
      return 'SHA-256';
    case 'SHA-256-SESS':
      return 'SHA-256-sess';
    default:
      // An algorithm this code does not implement must not silently fall back to
      // MD5 - that would answer a SHA-256 challenge with an MD5 hash and fail in
      // a way indistinguishable from a wrong password, sending an integrator off
      // to re-check credentials that were correct all along.
      return null;
  }
};

/** Parse `key="value", key2=value2` into a map, tolerating unquoted values. */
const parseAuthParameters = (raw: string): Record<string, string> => {
  const parameters: Record<string, string> = {};
  const pattern = /([a-zA-Z][a-zA-Z0-9_-]*)\s*=\s*(?:"([^"]*)"|([^,\s]+))/g;

  for (const match of raw.matchAll(pattern)) {
    const key = match[1];
    if (key === undefined) continue;
    parameters[key.toLowerCase()] = match[2] ?? match[3] ?? '';
  }
  return parameters;
};

const digestOf = (algorithm: string, input: string): string => {
  const hash = algorithm.startsWith('SHA-256') ? 'sha256' : 'md5';
  return createHash(hash).update(input, 'utf8').digest('hex');
};

/** Per-nonce request counter, required by qop=auth. */
export class DigestSession {
  #nonceCount = 0;
  readonly #cnonce: string;

  constructor(cnonce?: string) {
    // A client nonce must be unpredictable: it is what stops a hostile server
    // from choosing both sides of the hash input and precomputing a table.
    this.#cnonce = cnonce ?? randomBytes(16).toString('hex');
  }

  get cnonce(): string {
    return this.#cnonce;
  }

  nextNonceCount(): string {
    this.#nonceCount += 1;
    return this.#nonceCount.toString(16).padStart(8, '0');
  }
}

/**
 * Build an Authorization header answering a Digest challenge.
 *
 * The password is exposed for exactly the duration of the HA1 computation and is
 * not retained anywhere afterwards. What leaves this function is a hash.
 */
export const buildDigestHeader = (
  challenge: Extract<AuthChallenge, { scheme: 'Digest' }>,
  session: DigestSession,
  method: string,
  uri: string,
  username: string,
  password: Secret<string>,
): string => {
  const { algorithm, realm, nonce, opaque, qop } = challenge;

  let ha1 = digestOf(algorithm, `${username}:${realm}:${password.expose()}`);

  if (algorithm.endsWith('-sess')) {
    ha1 = digestOf(algorithm, `${ha1}:${nonce}:${session.cnonce}`);
  }

  const ha2 = digestOf(algorithm, `${method}:${uri}`);

  const useQop = qop.includes('auth');
  const nc = useQop ? session.nextNonceCount() : null;

  const response = useQop
    ? digestOf(algorithm, `${ha1}:${nonce}:${nc}:${session.cnonce}:auth:${ha2}`)
    : digestOf(algorithm, `${ha1}:${nonce}:${ha2}`);

  const fields = [
    `username="${escapeQuoted(username)}"`,
    `realm="${escapeQuoted(realm)}"`,
    `nonce="${escapeQuoted(nonce)}"`,
    `uri="${escapeQuoted(uri)}"`,
    `response="${response}"`,
    `algorithm=${algorithm}`,
  ];

  if (useQop) {
    fields.push('qop=auth', `nc=${nc ?? '00000001'}`, `cnonce="${session.cnonce}"`);
  }
  if (opaque !== null) fields.push(`opaque="${escapeQuoted(opaque)}"`);

  return `Digest ${fields.join(', ')}`;
};

/**
 * Build a Basic Authorization header.
 *
 * Basic sends the password reversibly encoded. It exists only because some
 * cameras offer nothing else, and callers must opt in explicitly.
 */
export const buildBasicHeader = (username: string, password: Secret<string>): string =>
  `Basic ${Buffer.from(`${username}:${password.expose()}`, 'utf8').toString('base64')}`;

/** Quoted-string escaping, so a credential containing a quote cannot break the header. */
const escapeQuoted = (value: string): string => value.replace(/(["\\])/g, '\\$1');

export class AuthenticationError extends Error {
  readonly code = 'RTSP_AUTH_FAILED';
  readonly recoverable = true;

  constructor(message: string) {
    // Never includes the credential, and never distinguishes "wrong password"
    // from "unknown user" - that distinction is an enumeration oracle.
    super(message);
    this.name = 'AuthenticationError';
  }
}

export class UnsupportedAuthError extends Error {
  readonly code = 'RTSP_AUTH_UNSUPPORTED';
  readonly recoverable = false;

  constructor(offered: readonly string[]) {
    super(
      `The camera offered only authentication schemes this client does not support ` +
        `(${offered.join(', ') || 'none'}). Digest is required unless Basic is explicitly enabled.`,
    );
    this.name = 'UnsupportedAuthError';
  }
}
