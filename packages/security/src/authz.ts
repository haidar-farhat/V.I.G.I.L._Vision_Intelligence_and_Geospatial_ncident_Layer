import type { Permission, Role, User } from '@sentinel/shared-types';
import { HIGH_RISK_PERMISSIONS, ROLE_PERMISSIONS } from '@sentinel/shared-types';

/**
 * Authorization.
 *
 * Checks are always against a **permission**, never a role name. Adding a role,
 * or widening an existing one, then cannot accidentally open a door somewhere
 * else in the codebase: the door is defined by the permission it requires.
 */

export class AuthorizationError extends Error {
  readonly code = 'FORBIDDEN';
  readonly permission: Permission;
  readonly recoverable = false;

  constructor(permission: Permission) {
    // Deliberately says what was required, never what the user does have -
    // enumerating a user's permissions to an unauthorised caller is itself a leak.
    super(`Permission denied: "${permission}" is required for this action.`);
    this.name = 'AuthorizationError';
    this.permission = permission;
  }
}

/** Every permission granted by a set of roles. */
export const permissionsFor = (roles: readonly Role[]): ReadonlySet<Permission> => {
  const permissions = new Set<Permission>();
  for (const role of roles) {
    for (const permission of ROLE_PERMISSIONS[role] ?? []) permissions.add(permission);
  }
  return permissions;
};

export const can = (user: Pick<User, 'roles' | 'active'>, permission: Permission): boolean => {
  // An inactive account holds no permissions at all, regardless of its roles.
  if (!user.active) return false;
  return permissionsFor(user.roles).has(permission);
};

/** Assert a permission, throwing `AuthorizationError` when absent. */
export const require_ = (
  user: Pick<User, 'roles' | 'active'>,
  permission: Permission,
): void => {
  if (!can(user, permission)) throw new AuthorizationError(permission);
};

/**
 * Whether an action needs explicit operator confirmation on top of the
 * permission. Destructive, irreversible or physically consequential actions do:
 * holding the right is not the same as intending to use it right now.
 */
export const requiresConfirmation = (permission: Permission): boolean =>
  HIGH_RISK_PERMISSIONS.includes(permission);

/**
 * Fixed-window rate limiter.
 *
 * Applied to logins, node pairing, API routes, WebSocket messages, file imports
 * and export generation. Deterministic and clock-injectable so the limits can be
 * tested without waiting in real time.
 */
export class RateLimiter {
  readonly #limit: number;
  readonly #windowMillis: number;
  readonly #buckets = new Map<string, { count: number; resetAt: number }>();

  constructor(limit: number, windowMillis: number) {
    this.#limit = limit;
    this.#windowMillis = windowMillis;
  }

  /**
   * Record an attempt. Returns whether it is allowed, and when the window resets.
   */
  attempt(key: string, now: number): { allowed: boolean; remaining: number; resetAt: number } {
    const bucket = this.#buckets.get(key);

    if (bucket === undefined || now >= bucket.resetAt) {
      const resetAt = now + this.#windowMillis;
      this.#buckets.set(key, { count: 1, resetAt });
      return { allowed: true, remaining: this.#limit - 1, resetAt };
    }

    if (bucket.count >= this.#limit) {
      return { allowed: false, remaining: 0, resetAt: bucket.resetAt };
    }

    bucket.count += 1;
    return { allowed: true, remaining: this.#limit - bucket.count, resetAt: bucket.resetAt };
  }

  /** Drop expired buckets so memory stays bounded under key churn. */
  prune(now: number): void {
    for (const [key, bucket] of this.#buckets) {
      if (now >= bucket.resetAt) this.#buckets.delete(key);
    }
  }

  reset(key: string): void {
    this.#buckets.delete(key);
  }

  get trackedKeys(): number {
    return this.#buckets.size;
  }
}

/**
 * Replay protection for signed LAN messages.
 *
 * A captured, valid message replayed later must be rejected. Two independent
 * conditions guard that: the timestamp must sit inside a narrow window, and the
 * request id must not have been seen before within it. The window bounds how much
 * state has to be retained, which is what keeps this from becoming a memory leak
 * an attacker can drive.
 */
export class ReplayGuard {
  readonly #windowMillis: number;
  readonly #seen = new Map<string, number>();

  constructor(windowMillis = 60_000) {
    this.#windowMillis = windowMillis;
  }

  /**
   * @returns null when acceptable, or a reason string when the message must be
   * refused.
   */
  check(requestId: string, timestamp: number, now: number): string | null {
    const skew = timestamp - now;

    if (skew > this.#windowMillis) {
      return `message timestamp is ${Math.round(skew / 1000)}s in the future`;
    }
    if (-skew > this.#windowMillis) {
      return `message timestamp is ${Math.round(-skew / 1000)}s old`;
    }
    if (this.#seen.has(requestId)) {
      return 'request id has already been used';
    }

    this.#seen.set(requestId, now);
    this.#prune(now);
    return null;
  }

  #prune(now: number): void {
    const cutoff = now - this.#windowMillis;
    for (const [id, at] of this.#seen) {
      if (at < cutoff) this.#seen.delete(id);
    }
  }

  get trackedIds(): number {
    return this.#seen.size;
  }
}
