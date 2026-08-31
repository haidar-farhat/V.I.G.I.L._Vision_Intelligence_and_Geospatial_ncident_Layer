import { randomBytes, scryptSync, timingSafeEqual } from 'node:crypto';
import type { Role, User, UserId, UtcMillis } from '@sentinel/shared-types';
import { asId, utcMillis } from '@sentinel/shared-types';
import { RateLimiter, Secret } from '@sentinel/security';

/**
 * Local authentication.
 *
 * Standalone mode has no directory to defer to, so the system holds its own
 * accounts. Three decisions shape this file, and each exists because the obvious
 * implementation is wrong in a way that only shows up under attack.
 *
 * **scrypt, not a bare hash.** A password database is stolen intact or not at
 * all, so the only defence that survives theft is making each guess expensive.
 * The parameters are stored alongside the hash so they can be raised later
 * without invalidating existing accounts.
 *
 * **Constant-time comparison, and constant-time failure.** An unknown username
 * performs the same work as a known one. Returning early on "no such user" turns
 * response latency into a user-enumeration oracle, which is how an attacker
 * learns which accounts are worth attacking.
 *
 * **Sessions are opaque random tokens, not signed claims.** A JWT cannot be
 * revoked before it expires; a token that indexes server-side state can be, and
 * revoking a compromised operator session immediately is worth more here than
 * statelessness.
 */

const SCRYPT_KEY_BYTES = 64;
const SCRYPT_SALT_BYTES = 16;

/**
 * Cost parameters. N=2^15 is roughly 100 ms on current hardware, which is
 * unnoticeable at a login prompt and ruinous across a stolen database.
 */
export const SCRYPT_PARAMS = { N: 32768, r: 8, p: 1, maxmem: 96 * 1024 * 1024 } as const;

export type ScryptCost = {
  readonly N: number;
  readonly r: number;
  readonly p: number;
  readonly maxmem: number;
};

/**
 * A deliberately cheap cost, for tests only.
 *
 * The production cost is around 100 ms per hash, which is the entire point and
 * also enough to add seconds to a suite that hashes in almost every case. Because
 * the parameters travel with the hash, a test using this exercises exactly the
 * same code path - whereas mocking the hash would test nothing at all.
 *
 * Nothing outside a test ever passes it.
 */
export const SCRYPT_COST_FOR_TESTS: ScryptCost = Object.freeze({
  N: 1024,
  r: 8,
  p: 1,
  maxmem: 32 * 1024 * 1024,
});

/**
 * Encode a hash with the parameters that produced it, so raising the cost later
 * does not lock out every existing account.
 */
export const hashPassword = (
  password: Secret<string>,
  cost: ScryptCost = SCRYPT_PARAMS,
): string => {
  const salt = randomBytes(SCRYPT_SALT_BYTES);
  const derived = scryptSync(password.expose(), salt, SCRYPT_KEY_BYTES, cost);

  return [
    'scrypt',
    cost.N,
    cost.r,
    cost.p,
    salt.toString('base64'),
    derived.toString('base64'),
  ].join('$');
};

/**
 * A hash of a value nobody knows, used to equalise work on an unknown username.
 *
 * Cached per cost setting, because it must be verified with the *same*
 * parameters real accounts use. A cheap dummy checked against expensive real
 * hashes would reintroduce the timing difference it exists to hide.
 */
const dummyHashes = new Map<string, string>();

const dummyHashFor = (cost: ScryptCost): string => {
  const key = `${cost.N}:${cost.r}:${cost.p}`;
  const existing = dummyHashes.get(key);
  if (existing !== undefined) return existing;

  const created = hashPassword(new Secret(randomBytes(32).toString('hex')), cost);
  dummyHashes.set(key, created);
  return created;
};

/** Read the cost out of an encoded hash, so the dummy can be made to match it. */
const costOf = (encoded: string): ScryptCost => {
  const parts = encoded.split('$');
  const N = Number.parseInt(parts[1] ?? '', 10);
  const r = Number.parseInt(parts[2] ?? '', 10);
  const p = Number.parseInt(parts[3] ?? '', 10);

  return Number.isInteger(N) && Number.isInteger(r) && Number.isInteger(p)
    ? { N, r, p, maxmem: SCRYPT_PARAMS.maxmem }
    : SCRYPT_PARAMS;
};

export const verifyPassword = (password: Secret<string>, encoded: string): boolean => {
  const parts = encoded.split('$');
  if (parts.length !== 6 || parts[0] !== 'scrypt') return false;

  const N = Number.parseInt(parts[1] ?? '', 10);
  const r = Number.parseInt(parts[2] ?? '', 10);
  const p = Number.parseInt(parts[3] ?? '', 10);
  if (!Number.isInteger(N) || !Number.isInteger(r) || !Number.isInteger(p)) return false;

  let salt: Buffer;
  let expected: Buffer;
  try {
    salt = Buffer.from(parts[4] ?? '', 'base64');
    expected = Buffer.from(parts[5] ?? '', 'base64');
  } catch {
    return false;
  }
  if (salt.length === 0 || expected.length === 0) return false;

  let derived: Buffer;
  try {
    derived = scryptSync(password.expose(), salt, expected.length, {
      N,
      r,
      p,
      maxmem: SCRYPT_PARAMS.maxmem,
    });
  } catch {
    // Parameters beyond maxmem, which a tampered record could specify.
    return false;
  }

  return derived.length === expected.length && timingSafeEqual(derived, expected);
};

// ------------------------------------------------------------------ sessions

export type Session = {
  readonly token: string;
  readonly userId: UserId;
  readonly username: string;
  readonly roles: readonly Role[];
  readonly createdAt: UtcMillis;
  readonly expiresAt: UtcMillis;
  readonly lastSeenAt: UtcMillis;
};

export type SessionStoreOptions = {
  /** Absolute lifetime. A session cannot be extended past this. */
  readonly maxLifetimeMillis?: number;
  /** Idle timeout. An unattended console must not stay authenticated all night. */
  readonly idleTimeoutMillis?: number;
  readonly now?: () => UtcMillis;
};

const DEFAULTS = {
  maxLifetimeMillis: 12 * 60 * 60 * 1000,
  idleTimeoutMillis: 60 * 60 * 1000,
} as const;

export class SessionStore {
  readonly #sessions = new Map<string, Session>();
  readonly #maxLifetime: number;
  readonly #idleTimeout: number;
  readonly #now: () => UtcMillis;

  constructor(options: SessionStoreOptions = {}) {
    this.#maxLifetime = options.maxLifetimeMillis ?? DEFAULTS.maxLifetimeMillis;
    this.#idleTimeout = options.idleTimeoutMillis ?? DEFAULTS.idleTimeoutMillis;
    this.#now = options.now ?? (() => utcMillis(Date.now()));
  }

  create(user: Pick<User, 'id' | 'username' | 'roles'>): Session {
    const now = this.#now();
    // 32 bytes of entropy. Guessing is not a viable attack on this.
    const token = randomBytes(32).toString('base64url');

    const session: Session = {
      token,
      userId: user.id,
      username: user.username,
      roles: user.roles,
      createdAt: now,
      expiresAt: utcMillis(now + this.#maxLifetime),
      lastSeenAt: now,
    };

    this.#sessions.set(token, session);
    return session;
  }

  /**
   * Look up a session and refresh its idle timer.
   *
   * Returns undefined for absent, expired or idle-timed-out sessions alike - the
   * caller has no use for the distinction and reporting it would tell an attacker
   * whether a token was ever valid.
   */
  get(token: string): Session | undefined {
    const session = this.#sessions.get(token);
    if (session === undefined) return undefined;

    const now = this.#now();

    if (now >= session.expiresAt || now - session.lastSeenAt > this.#idleTimeout) {
      this.#sessions.delete(token);
      return undefined;
    }

    const refreshed: Session = { ...session, lastSeenAt: now };
    this.#sessions.set(token, refreshed);
    return refreshed;
  }

  revoke(token: string): boolean {
    return this.#sessions.delete(token);
  }

  /** Revoke every session for a user, e.g. on deactivation or a role change. */
  revokeAllFor(userId: UserId): number {
    let removed = 0;
    for (const [token, session] of this.#sessions) {
      if (session.userId === userId) {
        this.#sessions.delete(token);
        removed += 1;
      }
    }
    return removed;
  }

  /** Drop expired sessions. Bounded memory under a login-churning attacker. */
  prune(): number {
    const now = this.#now();
    let removed = 0;

    for (const [token, session] of this.#sessions) {
      if (now >= session.expiresAt || now - session.lastSeenAt > this.#idleTimeout) {
        this.#sessions.delete(token);
        removed += 1;
      }
    }
    return removed;
  }

  get activeCount(): number {
    return this.#sessions.size;
  }
}

// ---------------------------------------------------------------- login flow

export type LoginOutcome =
  | { readonly ok: true; readonly session: Session }
  | { readonly ok: false; readonly reason: LoginFailure; readonly retryAfterMillis?: number };

export const LoginFailure = {
  /** Wrong username or wrong password. Deliberately not distinguished. */
  InvalidCredentials: 'INVALID_CREDENTIALS',
  AccountDisabled: 'ACCOUNT_DISABLED',
  RateLimited: 'RATE_LIMITED',
} as const;
export type LoginFailure = (typeof LoginFailure)[keyof typeof LoginFailure];

export type StoredUser = Pick<User, 'id' | 'username' | 'roles' | 'active'> & {
  readonly passwordHash: string;
};

/**
 * The login flow.
 *
 * Rate limiting is keyed on username **and** source address. Keying on username
 * alone lets anyone lock out a known account by failing to log in as them; keying
 * on address alone lets a botnet spread attempts across hosts. Both are needed,
 * and the account lock is deliberately the shorter of the two.
 */
export class Authenticator {
  readonly #sessions: SessionStore;
  readonly #byUsername: RateLimiter;
  readonly #byAddress: RateLimiter;
  readonly #now: () => UtcMillis;
  readonly #cost: ScryptCost;

  constructor(
    sessions: SessionStore,
    options: {
      readonly now?: () => UtcMillis;
      readonly attemptsPerUsername?: number;
      readonly attemptsPerAddress?: number;
      readonly windowMillis?: number;
      /** Cost used for the equal-work dummy. Tests lower it; nothing else does. */
      readonly cost?: ScryptCost;
    } = {},
  ) {
    this.#sessions = sessions;
    this.#now = options.now ?? (() => utcMillis(Date.now()));
    this.#cost = options.cost ?? SCRYPT_PARAMS;

    const window = options.windowMillis ?? 15 * 60 * 1000;
    this.#byUsername = new RateLimiter(options.attemptsPerUsername ?? 5, window);
    this.#byAddress = new RateLimiter(options.attemptsPerAddress ?? 30, window);
  }

  login(
    username: string,
    password: Secret<string>,
    sourceAddress: string,
    lookup: (username: string) => StoredUser | undefined,
  ): LoginOutcome {
    const now = this.#now();

    const perUser = this.#byUsername.attempt(`user:${username.toLowerCase()}`, now);
    const perAddress = this.#byAddress.attempt(`addr:${sourceAddress}`, now);

    if (!perUser.allowed || !perAddress.allowed) {
      const resetAt = Math.max(perUser.resetAt, perAddress.resetAt);
      return {
        ok: false,
        reason: LoginFailure.RateLimited,
        retryAfterMillis: Math.max(0, resetAt - now),
      };
    }

    const user = lookup(username);

    // An unknown username performs the same scrypt work as a known one.
    // Returning early here would turn response latency into an enumeration
    // oracle - the attacker learns which accounts exist without ever logging in.
    const hash =
      user === undefined ? dummyHashFor(this.#cost) : user.passwordHash;
    // When the account exists the dummy is not used, but the cache is warmed at
    // the account's own cost so a later unknown-username lookup matches it.
    if (user !== undefined) dummyHashFor(costOf(user.passwordHash));
    const correct = verifyPassword(password, hash);

    if (user === undefined || !correct) {
      return { ok: false, reason: LoginFailure.InvalidCredentials };
    }

    if (!user.active) {
      // Reported distinctly only because the credential was already proven
      // correct, so this reveals nothing the caller did not just demonstrate.
      return { ok: false, reason: LoginFailure.AccountDisabled };
    }

    // A successful login clears the account's own limiter but not the address's:
    // one valid credential should not reset the budget for guessing others.
    this.#byUsername.reset(`user:${username.toLowerCase()}`);

    return { ok: true, session: this.#sessions.create(user) };
  }
}

/** The bootstrap administrator created by the first-run wizard. */
export const createInitialAdmin = (
  username: string,
  password: Secret<string>,
  now: UtcMillis,
  cost: ScryptCost = SCRYPT_PARAMS,
): StoredUser & { readonly createdAt: UtcMillis } => ({
  id: asId<UserId>(`user-${randomBytes(8).toString('hex')}`),
  username,
  roles: ['ADMIN'],
  active: true,
  passwordHash: hashPassword(password, cost),
  createdAt: now,
});

/**
 * Password strength.
 *
 * Length over composition rules. Mandating a symbol produces "Password1!" on
 * every console in the building; requiring twelve characters produces something
 * an attacker has to work for. The check refuses rather than warns, because a
 * warning on an account that can delete evidence is not enough.
 */
export const checkPasswordStrength = (
  password: Secret<string>,
): { readonly acceptable: boolean; readonly reason?: string } => {
  const value = password.expose();

  if (value.length < 12) {
    return {
      acceptable: false,
      reason: 'Use at least 12 characters. Length matters more than symbols.',
    };
  }
  if (/^(.)\1+$/.test(value)) {
    return { acceptable: false, reason: 'This is a single repeated character.' };
  }
  if (COMMON_PASSWORDS.has(value.toLowerCase())) {
    return { acceptable: false, reason: 'This is one of the most commonly used passwords.' };
  }
  return { acceptable: true };
};

/**
 * A deliberately short list of the passwords that actually appear on deployed
 * security appliances. A full breach corpus belongs in an imported data file, not
 * compiled into a package that has no dependencies.
 */
const COMMON_PASSWORDS = new Set([
  'password1234',
  'administrator',
  'sentinelvision',
  'securitycamera',
  '123456789012',
  'qwertyuiop12',
  'changemeplease',
  'letmein12345',
]);
