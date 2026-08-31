import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { asId, utcMillis } from '@sentinel/shared-types';
import type { UserId, UtcMillis } from '@sentinel/shared-types';
import { secret } from '@sentinel/security';
import {
  Authenticator,
  SessionStore,
  checkPasswordStrength,
  createInitialAdmin,
  SCRYPT_COST_FOR_TESTS as COST,
  hashPassword as hashAtCost,
  verifyPassword,
} from '../src/auth.ts';
import type { StoredUser } from '../src/auth.ts';

const PASSWORD = 'correct-horse-battery-staple';

/*
 * Hash at the reduced test cost.
 *
 * The parameters travel with the hash, so this exercises exactly the same code
 * path production does, only cheaper. Mocking the hash would test nothing, and
 * hashing at production cost adds seconds to almost every case in this file.
 */
const hashPassword = (password: Parameters<typeof hashAtCost>[0]): string =>
  hashAtCost(password, COST);

/** A clock the tests advance by hand, so timeouts are exact rather than slept. */
const clock = (start = 1_000_000) => {
  let current = start;
  return {
    now: (): UtcMillis => utcMillis(current),
    advance: (millis: number): void => {
      current += millis;
    },
  };
};

describe('password hashing', () => {
  test('verifies a correct password', () => {
    const hash = hashPassword(secret(PASSWORD));
    assert.ok(verifyPassword(secret(PASSWORD), hash));
  });

  test('rejects a wrong password', () => {
    const hash = hashPassword(secret(PASSWORD));
    assert.ok(!verifyPassword(secret('wrong'), hash));
    assert.ok(!verifyPassword(secret(`${PASSWORD} `), hash), 'a trailing space is a wrong password');
  });

  test('the hash contains no trace of the password', () => {
    const hash = hashPassword(secret(PASSWORD));
    assert.ok(!hash.includes(PASSWORD));
    assert.ok(!hash.toLowerCase().includes('horse'));
  });

  test('the same password hashes differently every time', () => {
    // A shared salt would let one rainbow table crack every account at once.
    assert.notEqual(hashPassword(secret(PASSWORD)), hashPassword(secret(PASSWORD)));
  });

  test('the cost parameters travel with the hash', () => {
    // So they can be raised later without invalidating existing accounts.
    // And a hash made at one cost still verifies after the default changes.
    assert.match(hashAtCost(secret(PASSWORD), COST), /^scrypt\$1024\$8\$1\$/);
    assert.match(hashAtCost(secret(PASSWORD)), /^scrypt\$32768\$8\$1\$/);
    assert.ok(verifyPassword(secret(PASSWORD), hashAtCost(secret(PASSWORD), COST)));
  });

  test('a malformed or tampered record fails closed', () => {
    for (const bad of [
      '',
      'not-a-hash',
      'scrypt$abc$8$1$c2FsdA==$aGFzaA==',
      'bcrypt$32768$8$1$c2FsdA==$aGFzaA==',
      'scrypt$32768$8$1$$',
      'scrypt$32768$8$1$c2FsdA==',
    ]) {
      assert.equal(verifyPassword(secret(PASSWORD), bad), false, `accepted: ${bad}`);
    }
  });

  test('absurd stored parameters are refused rather than exhausting memory', () => {
    // A tampered record could otherwise ask for gigabytes during verification.
    const hostile = 'scrypt$1073741824$8$1$c2FsdA==$aGFzaA==';
    assert.equal(verifyPassword(secret(PASSWORD), hostile), false);
  });
});

describe('password strength', () => {
  test('requires length over composition', () => {
    // Mandating a symbol produces "Password1!" on every console in the building.
    assert.equal(checkPasswordStrength(secret('Sh0rt!')).acceptable, false);
    assert.equal(checkPasswordStrength(secret('the north gate is blue')).acceptable, true);
  });

  test('rejects a single repeated character and known-common choices', () => {
    assert.equal(checkPasswordStrength(secret('aaaaaaaaaaaaaaa')).acceptable, false);
    assert.equal(checkPasswordStrength(secret('administrator')).acceptable, false);
    assert.equal(checkPasswordStrength(secret('SentinelVision')).acceptable, false);
  });

  test('explains the refusal', () => {
    const result = checkPasswordStrength(secret('short'));
    assert.match(result.reason ?? '', /at least 12 characters/);
  });
});

describe('sessions', () => {
  const user = { id: asId<UserId>('user-1'), username: 'operator', roles: ['OPERATOR'] as const };

  test('issues an opaque high-entropy token', () => {
    const store = new SessionStore();
    const session = store.create(user);

    // Opaque rather than a signed claim, so it can be revoked before expiry.
    assert.ok(session.token.length >= 40);
    assert.ok(!session.token.includes('operator'), 'the token encodes nothing');
    assert.ok(!session.token.includes('.'), 'not a JWT');
  });

  test('resolves an active session and refreshes its idle timer', () => {
    const time = clock();
    const store = new SessionStore({ now: time.now, idleTimeoutMillis: 1000 });
    const session = store.create(user);

    time.advance(800);
    assert.equal(store.get(session.token)?.username, 'operator');

    // The lookup refreshed lastSeen, so another 800 ms is still inside the window.
    time.advance(800);
    assert.notEqual(store.get(session.token), undefined);
  });

  test('expires an idle session, because an unattended console must not stay open', () => {
    const time = clock();
    const store = new SessionStore({ now: time.now, idleTimeoutMillis: 1000 });
    const session = store.create(user);

    time.advance(1500);
    assert.equal(store.get(session.token), undefined);
  });

  test('enforces an absolute lifetime regardless of activity', () => {
    const time = clock();
    const store = new SessionStore({
      now: time.now,
      maxLifetimeMillis: 5000,
      idleTimeoutMillis: 10_000,
    });
    const session = store.create(user);

    // Kept active throughout, but the absolute cap still applies.
    for (let i = 0; i < 5; i += 1) {
      time.advance(1000);
      store.get(session.token);
    }
    time.advance(100);

    assert.equal(store.get(session.token), undefined);
  });

  test('an unknown token is indistinguishable from an expired one', () => {
    // Reporting the difference tells an attacker whether a token was ever valid.
    const store = new SessionStore();
    assert.equal(store.get('never-existed'), undefined);
  });

  test('revocation is immediate', () => {
    const store = new SessionStore();
    const session = store.create(user);

    assert.equal(store.revoke(session.token), true);
    assert.equal(store.get(session.token), undefined);
    assert.equal(store.revoke(session.token), false, 'revoking twice is not an error');
  });

  test('every session for a user can be revoked at once', () => {
    // What happens when an account is deactivated or its roles change.
    const store = new SessionStore();
    store.create(user);
    store.create(user);
    store.create({ ...user, id: asId<UserId>('user-2'), username: 'other' });

    assert.equal(store.revokeAllFor(asId<UserId>('user-1')), 2);
    assert.equal(store.activeCount, 1);
  });

  test('pruning bounds memory under a login-churning attacker', () => {
    const time = clock();
    const store = new SessionStore({ now: time.now, idleTimeoutMillis: 1000 });

    for (let i = 0; i < 100; i += 1) store.create(user);
    assert.equal(store.activeCount, 100);

    time.advance(2000);
    assert.equal(store.prune(), 100);
    assert.equal(store.activeCount, 0);
  });
});

describe('login', () => {
  const stored = (overrides: Partial<StoredUser> = {}): StoredUser => ({
    id: asId<UserId>('user-1'),
    username: 'operator',
    roles: ['OPERATOR'],
    active: true,
    passwordHash: hashPassword(secret(PASSWORD)),
    ...overrides,
  });

  const lookupFor = (user: StoredUser | undefined) => (username: string) =>
    user !== undefined && username === user.username ? user : undefined;

  test('accepts correct credentials and issues a session', () => {
    const store = new SessionStore();
    const auth = new Authenticator(store, { cost: COST });
    const user = stored();

    const result = auth.login('operator', secret(PASSWORD), '127.0.0.1', lookupFor(user));

    assert.equal(result.ok, true);
    if (!result.ok) return;
    assert.equal(result.session.username, 'operator');
    assert.deepEqual(result.session.roles, ['OPERATOR']);
  });

  test('does not distinguish a wrong password from an unknown user', () => {
    // The distinction is a user-enumeration oracle.
    const auth = new Authenticator(new SessionStore(), { cost: COST });

    const wrongPassword = auth.login('operator', secret('nope'), '127.0.0.1', lookupFor(stored()));
    const unknownUser = auth.login('ghost', secret(PASSWORD), '127.0.0.1', lookupFor(stored()));

    assert.equal(wrongPassword.ok, false);
    assert.equal(unknownUser.ok, false);
    if (wrongPassword.ok || unknownUser.ok) return;
    assert.equal(wrongPassword.reason, unknownUser.reason);
    assert.equal(wrongPassword.reason, 'INVALID_CREDENTIALS');
  });

  test('an unknown user performs the same work as a known one', () => {
    // Returning early would turn response latency into the same oracle the
    // identical reason string was chosen to avoid.
    const auth = new Authenticator(new SessionStore(), { cost: COST });
    const user = stored();

    const timeOf = (username: string): number => {
      const began = process.hrtime.bigint();
      auth.login(username, secret('a-wrong-password'), `10.0.0.${username.length}`, lookupFor(user));
      return Number(process.hrtime.bigint() - began) / 1e6;
    };

    // Warm up, then compare. scrypt dominates both paths, so the ratio should be
    // close to 1 - a fast path for unknown users would show up as a large gap.
    timeOf('operator');
    timeOf('ghost');

    const known = timeOf('operator');
    const unknown = timeOf('ghost');
    const ratio = Math.max(known, unknown) / Math.max(1, Math.min(known, unknown));

    assert.ok(
      ratio < 4,
      `known ${known.toFixed(2)}ms vs unknown ${unknown.toFixed(2)}ms (ratio ${ratio.toFixed(2)})`,
    );
  });

  test('a disabled account is refused even with the right password', () => {
    const auth = new Authenticator(new SessionStore(), { cost: COST });
    const result = auth.login(
      'operator',
      secret(PASSWORD),
      '127.0.0.1',
      lookupFor(stored({ active: false })),
    );

    assert.equal(result.ok, false);
    if (result.ok) return;
    // Distinct only because the credential was already proven correct, so this
    // reveals nothing the caller did not just demonstrate.
    assert.equal(result.reason, 'ACCOUNT_DISABLED');
  });

  test('rate limits attempts per username', () => {
    const time = clock();
    const auth = new Authenticator(new SessionStore({ now: time.now }), {
      now: time.now,
      attemptsPerUsername: 3,
      windowMillis: 60_000,
      cost: COST,
    });
    const user = stored();

    for (let i = 0; i < 3; i += 1) {
      auth.login('operator', secret('wrong'), '127.0.0.1', lookupFor(user));
    }

    const blocked = auth.login('operator', secret(PASSWORD), '127.0.0.1', lookupFor(user));
    assert.equal(blocked.ok, false);
    if (blocked.ok) return;
    assert.equal(blocked.reason, 'RATE_LIMITED');
    assert.ok((blocked.retryAfterMillis ?? 0) > 0, 'says when to try again');
  });

  test('rate limits per source address too, so a botnet cannot spread attempts', () => {
    const time = clock();
    const auth = new Authenticator(new SessionStore({ now: time.now }), {
      now: time.now,
      attemptsPerUsername: 100,
      attemptsPerAddress: 4,
      windowMillis: 60_000,
      cost: COST,
    });

    // Four different usernames from one address.
    for (let i = 0; i < 4; i += 1) {
      auth.login(`user${i}`, secret('wrong'), '10.0.0.9', lookupFor(undefined));
    }

    const blocked = auth.login('user9', secret('wrong'), '10.0.0.9', lookupFor(undefined));
    assert.equal(blocked.ok, false);
    if (blocked.ok) return;
    assert.equal(blocked.reason, 'RATE_LIMITED');
  });

  test('one account being attacked does not lock out another', () => {
    const time = clock();
    const auth = new Authenticator(new SessionStore({ now: time.now }), {
      now: time.now,
      attemptsPerUsername: 2,
      attemptsPerAddress: 100,
      windowMillis: 60_000,
      cost: COST,
    });
    const victim = stored({ username: 'victim' });

    for (let i = 0; i < 5; i += 1) {
      auth.login('victim', secret('wrong'), '10.0.0.1', lookupFor(victim));
    }

    const other = auth.login('other', secret('wrong'), '10.0.0.2', lookupFor(undefined));
    assert.equal(other.ok, false);
    if (other.ok) return;
    assert.equal(other.reason, 'INVALID_CREDENTIALS', 'not rate limited by someone else attack');
  });

  test('a successful login clears the account limiter but not the address budget', () => {
    // One valid credential should not reset the budget for guessing others.
    const time = clock();
    const auth = new Authenticator(new SessionStore({ now: time.now }), {
      now: time.now,
      attemptsPerUsername: 5,
      attemptsPerAddress: 6,
      windowMillis: 60_000,
      cost: COST,
    });
    const user = stored();

    for (let i = 0; i < 4; i += 1) {
      auth.login('operator', secret('wrong'), '10.0.0.5', lookupFor(user));
    }

    assert.equal(auth.login('operator', secret(PASSWORD), '10.0.0.5', lookupFor(user)).ok, true);

    // The account limiter was reset, but the address has spent 5 of 6.
    auth.login('someone', secret('wrong'), '10.0.0.5', lookupFor(undefined));
    const next = auth.login('someone', secret('wrong'), '10.0.0.5', lookupFor(undefined));

    assert.equal(next.ok, false);
    if (next.ok) return;
    assert.equal(next.reason, 'RATE_LIMITED');
  });
});

describe('first-run administrator', () => {
  test('creates an active admin whose password is stored hashed', () => {
    const admin = createInitialAdmin('admin', secret(PASSWORD), utcMillis(1000), COST);

    assert.deepEqual(admin.roles, ['ADMIN']);
    assert.equal(admin.active, true);
    assert.ok(!JSON.stringify(admin).includes(PASSWORD));
    assert.ok(verifyPassword(secret(PASSWORD), admin.passwordHash));
  });
});
