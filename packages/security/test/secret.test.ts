import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { inspect } from 'node:util';
import {
  Secret,
  buildRtspUrl,
  describeStream,
  redact,
  redactUrl,
  secret,
} from '../src/secret.ts';

/**
 * The password used throughout this suite. Nothing in any assertion output, any
 * serialisation, or any log line produced by the code under test may contain it.
 */
const PASSWORD = 'hunter2-correct-horse';

describe('Secret', () => {
  test('does not reveal its value when coerced to a string', () => {
    const s = secret(PASSWORD, 'camera password');
    assert.equal(String(s), '[redacted]');
    assert.equal(`${s}`, '[redacted]');
    assert.equal(s + '', '[redacted]');
    assert.ok(!String(s).includes(PASSWORD));
  });

  test('does not reveal its value through JSON serialisation', () => {
    const s = secret(PASSWORD);
    const payload = JSON.stringify({ username: 'admin', password: s });

    assert.ok(!payload.includes(PASSWORD), `leaked through JSON: ${payload}`);
    assert.ok(payload.includes('[redacted]'));
  });

  test('does not reveal its value through console inspection', () => {
    const s = secret(PASSWORD, 'camera password');
    const rendered = inspect({ credential: s }, { depth: 5 });

    assert.ok(!rendered.includes(PASSWORD), `leaked through inspect: ${rendered}`);
    assert.ok(rendered.includes('[redacted]'));
    assert.ok(rendered.includes('camera password'), 'the label is safe and useful');
  });

  test('does not reveal its value when nested inside an error', () => {
    const s = secret(PASSWORD);
    const error = new Error(`connect failed for ${String(s)}`);

    assert.ok(!error.message.includes(PASSWORD));
    assert.ok(!inspect(error).includes(PASSWORD));
  });

  test('exposes the value only on an explicit call', () => {
    const s = secret(PASSWORD);
    assert.equal(s.expose(), PASSWORD);
  });

  test('compares without revealing', () => {
    assert.ok(secret(PASSWORD).equals(secret(PASSWORD)));
    assert.ok(!secret(PASSWORD).equals(secret('different')));
    assert.ok(!secret('abc').equals(secret('abcd')), 'differing lengths are unequal');
  });

  test('is recognisable at runtime', () => {
    assert.ok(secret('x') instanceof Secret);
  });
});

describe('redactUrl', () => {
  test('strips the password from a credentialed RTSP URL', () => {
    const url = `rtsp://admin:${PASSWORD}@192.168.1.50:554/Streaming/Channels/101`;
    const safe = redactUrl(url);

    assert.ok(!safe.includes(PASSWORD), `leaked: ${safe}`);
    assert.ok(safe.includes('admin'), 'the username stays, it is operationally useful');
    assert.ok(safe.includes('192.168.1.50'), 'the host stays');
    assert.ok(safe.includes('/Streaming/Channels/101'), 'the path stays');
  });

  test('handles several URLs in one string', () => {
    const message =
      `failed rtsp://a:${PASSWORD}@10.0.0.1/s1 and ` + `rtsp://b:${PASSWORD}@10.0.0.2/s2`;
    const safe = redactUrl(message);
    assert.ok(!safe.includes(PASSWORD), `leaked: ${safe}`);
  });

  test('leaves a URL without credentials untouched', () => {
    const url = 'rtsp://192.168.1.50:554/stream';
    assert.equal(redactUrl(url), url);
  });

  test('covers other schemes too', () => {
    const safe = redactUrl(`http://user:${PASSWORD}@camera.local/onvif`);
    assert.ok(!safe.includes(PASSWORD));
  });
});

describe('redact', () => {
  test('removes values under sensitive keys', () => {
    const input = {
      username: 'admin',
      password: PASSWORD,
      apiKey: PASSWORD,
      api_key: PASSWORD,
      Authorization: `Bearer ${PASSWORD}`,
      privateKey: PASSWORD,
      host: '192.168.1.50',
    };

    const output = JSON.stringify(redact(input));
    assert.ok(!output.includes(PASSWORD), `leaked: ${output}`);
    assert.ok(output.includes('admin'), 'non-sensitive fields survive');
    assert.ok(output.includes('192.168.1.50'));
  });

  test('recurses into nested structures and arrays', () => {
    const input = {
      cameras: [
        { name: 'cam-07', credentials: { username: 'admin', password: PASSWORD } },
        { name: 'cam-08', credentials: { username: 'admin', password: PASSWORD } },
      ],
    };

    const output = JSON.stringify(redact(input));
    assert.ok(!output.includes(PASSWORD), `leaked: ${output}`);
    assert.ok(output.includes('cam-07'));
  });

  test('redacts Secret instances wherever they appear', () => {
    const output = JSON.stringify(redact({ nested: { deep: secret(PASSWORD) } }));
    assert.ok(!output.includes(PASSWORD));
  });

  test('redacts credentials embedded in string values', () => {
    const input = { streamUrl: `rtsp://admin:${PASSWORD}@10.0.0.5/live` };
    const output = JSON.stringify(redact(input));
    assert.ok(!output.includes(PASSWORD), `leaked: ${output}`);
  });

  test('survives circular references rather than crashing the logger', () => {
    const input: Record<string, unknown> = { name: 'node', password: PASSWORD };
    input['self'] = input;

    const output = JSON.stringify(redact(input));
    assert.ok(output.includes('[circular]'));
    assert.ok(!output.includes(PASSWORD));
  });

  test('bounds recursion depth', () => {
    let deep: Record<string, unknown> = { password: PASSWORD };
    for (let i = 0; i < 40; i += 1) deep = { nested: deep };

    const output = JSON.stringify(redact(deep));
    assert.ok(output.includes('[truncated]'));
    assert.ok(!output.includes(PASSWORD));
  });

  test('redacts an Error without losing its diagnostic value', () => {
    const error = new Error(`connect ECONNREFUSED rtsp://admin:${PASSWORD}@10.0.0.5/live`);
    const output = JSON.stringify(redact({ error }));

    assert.ok(!output.includes(PASSWORD), `leaked: ${output}`);
    assert.ok(output.includes('ECONNREFUSED'), 'the useful part of the message survives');
  });

  test('passes through primitives unchanged', () => {
    assert.equal(redact(42), 42);
    assert.equal(redact(true), true);
    assert.equal(redact(null), null);
    assert.equal(redact(undefined), undefined);
  });
});

describe('buildRtspUrl', () => {
  test('returns the assembled URL inside a Secret, never as a bare string', () => {
    const url = buildRtspUrl('192.168.1.50', 554, '/live', 'admin', secret(PASSWORD));

    assert.ok(url instanceof Secret);
    assert.equal(String(url), '[redacted]');
    assert.ok(!JSON.stringify({ url }).includes(PASSWORD));

    const exposed = url.expose();
    assert.ok(exposed.includes(PASSWORD), 'the real URL is available on demand');
    assert.ok(exposed.startsWith('rtsp://admin:'));
  });

  test('percent-encodes credentials that contain URL metacharacters', () => {
    const awkward = 'p@ss:w/rd?';
    const url = buildRtspUrl('10.0.0.5', 554, 'live', 'ad min', secret(awkward)).expose();

    assert.ok(url.includes(encodeURIComponent(awkward)));
    assert.ok(!url.includes('p@ss:w/rd?'), 'raw metacharacters would corrupt the URL');
  });

  test('omits the userinfo section entirely when there are no credentials', () => {
    const url = buildRtspUrl('10.0.0.5', 554, '/live').expose();
    assert.equal(url, 'rtsp://10.0.0.5:554/live');
    assert.ok(!url.includes('@'));
  });

  test('normalises a path that is missing its leading slash', () => {
    assert.equal(buildRtspUrl('10.0.0.5', 554, 'live').expose(), 'rtsp://10.0.0.5:554/live');
  });
});

describe('describeStream', () => {
  test('produces a loggable endpoint description with no credential slot', () => {
    const description = describeStream('192.168.1.50', 554, '/live');
    assert.equal(description, 'rtsp://192.168.1.50:554/live');
    assert.ok(!description.includes('@'));
  });
});
