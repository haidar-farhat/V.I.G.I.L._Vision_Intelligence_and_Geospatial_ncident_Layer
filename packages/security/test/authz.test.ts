import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import type { Role } from '@sentinel/shared-types';
import { Permission } from '@sentinel/shared-types';
import {
  AuthorizationError,
  RateLimiter,
  ReplayGuard,
  can,
  permissionsFor,
  require_,
  requiresConfirmation,
} from '../src/authz.ts';

const user = (roles: readonly Role[], active = true): { roles: readonly Role[]; active: boolean } =>
  ({ roles, active });

describe('permissions', () => {
  test('an admin holds every permission', () => {
    const admin = permissionsFor(['ADMIN']);
    for (const permission of Object.values(Permission)) {
      assert.ok(admin.has(permission), `admin should hold ${permission}`);
    }
  });

  test('a viewer can look but not touch', () => {
    const viewer = user(['VIEWER']);
    assert.ok(can(viewer, Permission.CameraView));
    assert.ok(can(viewer, Permission.IncidentView));

    assert.ok(!can(viewer, Permission.CameraDelete));
    assert.ok(!can(viewer, Permission.CameraUpdate));
    assert.ok(!can(viewer, Permission.IncidentAcknowledge));
    assert.ok(!can(viewer, Permission.RuleEdit));
    assert.ok(!can(viewer, Permission.SettingsEdit));
    assert.ok(!can(viewer, Permission.CameraPtz));
  });

  test('an operator can run the shift but not reconfigure the system', () => {
    const operator = user(['OPERATOR']);
    assert.ok(can(operator, Permission.IncidentAcknowledge));
    assert.ok(can(operator, Permission.IncidentResolve));
    assert.ok(can(operator, Permission.CameraPtz));
    assert.ok(can(operator, Permission.ZoneEdit));

    assert.ok(!can(operator, Permission.UserEdit), 'operators do not manage accounts');
    assert.ok(!can(operator, Permission.CameraDelete), 'nor delete cameras');
    assert.ok(!can(operator, Permission.EvidenceDelete), 'nor destroy evidence');
    assert.ok(!can(operator, Permission.RetentionEdit));
    assert.ok(!can(operator, Permission.SettingsEdit));
  });

  test('an analyst can review and export but not operate cameras', () => {
    const analyst = user(['ANALYST']);
    assert.ok(can(analyst, Permission.IncidentExport));
    assert.ok(can(analyst, Permission.AuditView));

    assert.ok(!can(analyst, Permission.CameraPtz), 'analysts do not move cameras');
    assert.ok(!can(analyst, Permission.ZoneEdit));
    assert.ok(!can(analyst, Permission.EvidenceDelete));
  });

  test('an inactive account holds nothing, whatever its roles', () => {
    const suspendedAdmin = user(['ADMIN'], false);
    assert.ok(!can(suspendedAdmin, Permission.CameraView));
    assert.ok(!can(suspendedAdmin, Permission.SettingsEdit));
  });

  test('multiple roles union their permissions', () => {
    const both = user(['VIEWER', 'ANALYST']);
    assert.ok(can(both, Permission.AuditView), 'from ANALYST');
    assert.ok(can(both, Permission.CameraView), 'from VIEWER');
  });

  test('require_ throws a forbidden error that does not enumerate held rights', () => {
    assert.throws(
      () => require_(user(['VIEWER']), Permission.CameraDelete),
      (error: unknown) => {
        assert.ok(error instanceof AuthorizationError);
        assert.equal(error.code, 'FORBIDDEN');
        assert.equal(error.permission, Permission.CameraDelete);
        assert.match(error.message, /camera:delete/);
        // Telling an unauthorised caller what they *do* hold is itself a leak.
        assert.ok(!error.message.includes('camera:view'));
        return true;
      },
    );

    assert.doesNotThrow(() => require_(user(['ADMIN']), Permission.CameraDelete));
  });
});

describe('high-risk actions', () => {
  test('destructive and physical actions need confirmation', () => {
    assert.ok(requiresConfirmation(Permission.CameraDelete));
    assert.ok(requiresConfirmation(Permission.EvidenceDelete));
    assert.ok(requiresConfirmation(Permission.IncidentExport));
    assert.ok(requiresConfirmation(Permission.RetentionEdit));
    assert.ok(requiresConfirmation(Permission.CameraPtz), 'PTZ moves a physical device');
  });

  test('reading does not', () => {
    assert.ok(!requiresConfirmation(Permission.CameraView));
    assert.ok(!requiresConfirmation(Permission.IncidentView));
  });
});

describe('RateLimiter', () => {
  test('allows up to the limit then refuses', () => {
    const limiter = new RateLimiter(3, 60_000);

    assert.equal(limiter.attempt('user-a', 0).allowed, true);
    assert.equal(limiter.attempt('user-a', 10).allowed, true);
    assert.equal(limiter.attempt('user-a', 20).allowed, true);

    const blocked = limiter.attempt('user-a', 30);
    assert.equal(blocked.allowed, false);
    assert.equal(blocked.remaining, 0);
  });

  test('keys are independent, so one user cannot lock out another', () => {
    const limiter = new RateLimiter(1, 60_000);
    assert.equal(limiter.attempt('attacker', 0).allowed, true);
    assert.equal(limiter.attempt('attacker', 1).allowed, false);
    assert.equal(limiter.attempt('victim', 1).allowed, true);
  });

  test('the window reopens once it has elapsed', () => {
    const limiter = new RateLimiter(1, 1000);
    assert.equal(limiter.attempt('k', 0).allowed, true);
    assert.equal(limiter.attempt('k', 500).allowed, false);
    assert.equal(limiter.attempt('k', 1000).allowed, true, 'window reset');
  });

  test('pruning bounds memory under key churn', () => {
    const limiter = new RateLimiter(5, 1000);
    for (let i = 0; i < 100; i += 1) limiter.attempt(`key-${i}`, 0);
    assert.equal(limiter.trackedKeys, 100);

    limiter.prune(2000);
    assert.equal(limiter.trackedKeys, 0, 'expired buckets are released');
  });
});

describe('ReplayGuard', () => {
  test('accepts a fresh, unique message', () => {
    const guard = new ReplayGuard(60_000);
    assert.equal(guard.check('req-1', 1000, 1000), null);
  });

  test('refuses a replayed request id', () => {
    const guard = new ReplayGuard(60_000);
    assert.equal(guard.check('req-1', 1000, 1000), null);

    const replayed = guard.check('req-1', 1000, 1500);
    assert.ok(replayed !== null);
    assert.match(replayed, /already been used/);
  });

  test('refuses a message from too far in the past', () => {
    const guard = new ReplayGuard(60_000);
    const stale = guard.check('req-old', 0, 120_000);
    assert.ok(stale !== null);
    assert.match(stale, /old/);
  });

  test('refuses a message from the future', () => {
    const guard = new ReplayGuard(60_000);
    const ahead = guard.check('req-future', 200_000, 0);
    assert.ok(ahead !== null);
    assert.match(ahead, /future/);
  });

  test('tolerates clock skew inside the window', () => {
    const guard = new ReplayGuard(60_000);
    assert.equal(guard.check('a', 30_000, 0), null, '30s ahead is within tolerance');
    assert.equal(guard.check('b', 0, 30_000), null, '30s behind is within tolerance');
  });

  test('bounds retained ids to the window', () => {
    const guard = new ReplayGuard(1000);
    for (let i = 0; i < 50; i += 1) guard.check(`req-${i}`, i, i);
    assert.ok(guard.trackedIds > 0);

    guard.check('later', 10_000, 10_000);
    assert.equal(guard.trackedIds, 1, 'old ids are dropped once they cannot be replayed');
  });
});
