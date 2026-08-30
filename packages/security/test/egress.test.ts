import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import {
  AddressScope,
  EgressBlockedError,
  EgressGuard,
  classifyHost,
  isLocalScope,
  networkIsolationReport,
} from '../src/egress.ts';

describe('address classification', () => {
  test('recognises loopback', () => {
    assert.equal(classifyHost('127.0.0.1'), AddressScope.Loopback);
    assert.equal(classifyHost('127.1.2.3'), AddressScope.Loopback);
    assert.equal(classifyHost('::1'), AddressScope.Loopback);
    assert.equal(classifyHost('localhost'), AddressScope.Loopback);
    assert.equal(classifyHost('LOCALHOST'), AddressScope.Loopback, 'case-insensitive');
  });

  test('recognises every RFC1918 range', () => {
    assert.equal(classifyHost('10.0.0.1'), AddressScope.Private);
    assert.equal(classifyHost('10.255.255.254'), AddressScope.Private);
    assert.equal(classifyHost('172.16.0.1'), AddressScope.Private);
    assert.equal(classifyHost('172.31.255.254'), AddressScope.Private);
    assert.equal(classifyHost('192.168.1.50'), AddressScope.Private);
  });

  test('does not mistake neighbouring public ranges for private ones', () => {
    // The classic off-by-one: 172.15 and 172.32 are public, 172.16-172.31 are not.
    assert.equal(classifyHost('172.15.0.1'), AddressScope.Public);
    assert.equal(classifyHost('172.32.0.1'), AddressScope.Public);
    assert.equal(classifyHost('192.167.1.1'), AddressScope.Public);
    assert.equal(classifyHost('11.0.0.1'), AddressScope.Public);
  });

  test('recognises link-local and multicast', () => {
    assert.equal(classifyHost('169.254.1.1'), AddressScope.LinkLocal);
    assert.equal(classifyHost('224.0.0.251'), AddressScope.Multicast, 'the mDNS group');
    assert.equal(classifyHost('fe80::1'), AddressScope.LinkLocal);
    assert.equal(classifyHost('ff02::fb'), AddressScope.Multicast);
  });

  test('recognises IPv6 unique local addresses', () => {
    assert.equal(classifyHost('fd00::1'), AddressScope.Private);
    assert.equal(classifyHost('fc00::1'), AddressScope.Private);
  });

  test('classifies IPv4-mapped IPv6 by the embedded address', () => {
    assert.equal(classifyHost('::ffff:192.168.1.1'), AddressScope.Private);
    assert.equal(classifyHost('::ffff:8.8.8.8'), AddressScope.Public);
  });

  test('classifies public IPv4 and IPv6 as public', () => {
    assert.equal(classifyHost('8.8.8.8'), AddressScope.Public);
    assert.equal(classifyHost('1.1.1.1'), AddressScope.Public);
    assert.equal(classifyHost('2606:4700::1111'), AddressScope.Public);
  });

  test('treats .local names as reachable on the link', () => {
    assert.equal(classifyHost('camera-07.local'), AddressScope.Private);
    assert.equal(classifyHost('control.home.arpa'), AddressScope.Private);
  });

  test('cannot prove a public DNS name is local, so reports it unresolved', () => {
    // Distinguished from Public so callers can tell "definitely external" from
    // "would need an Internet resolver to find out". Both are refused.
    assert.equal(classifyHost('api.example.com'), AddressScope.Unresolved);
    assert.equal(classifyHost('registry.npmjs.org'), AddressScope.Unresolved);
    assert.equal(classifyHost(''), AddressScope.Unresolved);
  });

  test('only local scopes count as reachable offline', () => {
    assert.ok(isLocalScope(AddressScope.Loopback));
    assert.ok(isLocalScope(AddressScope.Private));
    assert.ok(isLocalScope(AddressScope.LinkLocal));
    assert.ok(isLocalScope(AddressScope.Multicast));
    assert.ok(!isLocalScope(AddressScope.Public));
    assert.ok(!isLocalScope(AddressScope.Unresolved));
  });
});

describe('EgressGuard', () => {
  test('permits LAN addresses', () => {
    const guard = new EgressGuard();
    for (const host of ['192.168.1.50', '10.0.0.1', '127.0.0.1', 'cam.local', 'fd00::1']) {
      assert.doesNotThrow(() => guard.check(host), `${host} must be permitted`);
    }
    assert.equal(guard.allowedCount, 5);
  });

  test('blocks public addresses with an explanatory error', () => {
    const guard = new EgressGuard();

    assert.throws(
      () => guard.check('8.8.8.8'),
      (error: unknown) => {
        assert.ok(error instanceof EgressBlockedError);
        assert.equal(error.code, 'EGRESS_BLOCKED');
        assert.equal(error.scope, AddressScope.Public);
        // The message must explain the design decision, not just say "denied":
        // whoever hits this needs to know it is deliberate.
        assert.match(error.message, /without\s+Internet access by design/);
        assert.match(error.message, /never falls back/);
        return true;
      },
    );

    assert.equal(guard.blockedCount, 1);
  });

  test('blocks a public hostname as firmly as a public IP', () => {
    const guard = new EgressGuard();
    assert.throws(() => guard.check('maps.example.com'), EgressBlockedError);
    assert.throws(() => guard.check('api.openai.com'), EgressBlockedError);
  });

  test('blocks a URL by its host', () => {
    const guard = new EgressGuard();
    assert.throws(() => guard.checkUrl('https://tiles.example.com/style.json'), EgressBlockedError);
    assert.doesNotThrow(() => guard.checkUrl('http://192.168.1.50:8080/onvif'));
  });

  test('rejects a malformed URL rather than letting it through', () => {
    const guard = new EgressGuard();
    assert.throws(() => guard.checkUrl('not a url'), EgressBlockedError);
  });

  test('honours an explicit operator allow-list', () => {
    const guard = new EgressGuard({ allowHosts: ['vendor-nvr.example.com'] });
    assert.doesNotThrow(() => guard.check('vendor-nvr.example.com'));
    assert.throws(() => guard.check('other.example.com'), EgressBlockedError);
  });

  test('in report-only mode it counts without blocking', () => {
    const guard = new EgressGuard({ enforce: false });
    assert.doesNotThrow(() => guard.check('8.8.8.8'));
    assert.equal(guard.blockedCount, 1, 'the attempt is still recorded');
    assert.equal(guard.enforcing, false);
  });

  test('permits() answers without throwing', () => {
    const guard = new EgressGuard();
    assert.equal(guard.permits('192.168.1.1'), true);
    assert.equal(guard.permits('8.8.8.8'), false);
    assert.equal(guard.blockedCount, 0, 'permits() is a query, not an attempt');
  });
});

describe('network isolation report', () => {
  test('states the guarantee plainly for the diagnostic screen', () => {
    const guard = new EgressGuard();
    guard.check('192.168.1.50');
    try {
      guard.check('8.8.8.8');
    } catch {
      // expected
    }

    const report = networkIsolationReport(guard, true);

    assert.equal(report.wan, 'BLOCKED');
    assert.equal(report.lan, 'ACTIVE');
    assert.equal(report.internetDependency, 'NONE');
    assert.equal(report.cloudServices, 'DISABLED');
    assert.equal(report.telemetry, 'DISABLED');
    assert.equal(report.blockedAttempts, 1);
    assert.equal(report.allowedConnections, 1);
  });

  test('reports WAN as not required when enforcement is off', () => {
    const report = networkIsolationReport(new EgressGuard({ enforce: false }), false);
    assert.equal(report.wan, 'NOT_REQUIRED');
    assert.equal(report.lan, 'INACTIVE');
    assert.equal(report.internetDependency, 'NONE', 'never depends on the Internet either way');
  });
});
