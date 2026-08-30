import { test, describe, after } from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { secret } from '@sentinel/security';
import { probeMatchXml, startMockOnvifDevice } from '@sentinel/test-utils';
import type { MockOnvifDevice } from '@sentinel/test-utils';
import {
  SoapError,
  assertSafeXml,
  buildEnvelope,
  buildSecurityHeader,
  escapeXml,
  extractAttribute,
  extractBlocks,
  extractValue,
  extractValues,
  faultFrom,
} from '../src/onvif/soap.ts';
import { parseProbeMatch, parseScopes } from '../src/onvif/discovery.ts';
import {
  OnvifDevice,
  selectInferenceProfile,
  selectRecordingProfile,
} from '../src/onvif/device.ts';

const PASSWORD = 'correct-horse-battery';

const devices: MockOnvifDevice[] = [];
const spawnDevice = async (behaviour = {}): Promise<MockOnvifDevice> => {
  const device = await startMockOnvifDevice(behaviour);
  devices.push(device);
  return device;
};

after(async () => {
  for (const device of devices) await device.close();
});

// --------------------------------------------------------------------- SOAP

describe('WS-Security', () => {
  test('sends a digest, never the password', () => {
    const header = buildSecurityHeader(
      'admin',
      secret(PASSWORD),
      new Date('2024-06-12T02:14:00.000Z'),
      Buffer.from('0123456789abcdef'),
    );

    assert.ok(!header.includes(PASSWORD), `the password leaked into the header: ${header}`);
    assert.match(header, /<wsse:Username>admin<\/wsse:Username>/);
    assert.match(header, /Type="[^"]*PasswordDigest"/);
    assert.match(header, /<wsu:Created>2024-06-12T02:14:00\.000Z<\/wsu:Created>/);
  });

  test('the digest is exactly Base64(SHA1(nonce + created + password))', () => {
    // Getting this wrong produces a token every device rejects, which is
    // indistinguishable from a wrong password until somebody reads a capture.
    const nonce = Buffer.from('0123456789abcdef');
    const created = '2024-06-12T02:14:00.000Z';

    const header = buildSecurityHeader('admin', secret(PASSWORD), new Date(created), nonce);
    const digest = /<wsse:Password[^>]*>([^<]+)</.exec(header)?.[1];

    const expected = createHash('sha1')
      .update(Buffer.concat([nonce, Buffer.from(created), Buffer.from(PASSWORD)]))
      .digest('base64');

    assert.equal(digest, expected);
  });

  test('a fresh nonce each time, so a captured token cannot be replayed', () => {
    const a = buildSecurityHeader('admin', secret(PASSWORD));
    const b = buildSecurityHeader('admin', secret(PASSWORD));

    const nonceOf = (header: string): string | undefined => /<wsse:Nonce[^>]*>([^<]+)</.exec(header)?.[1];
    assert.notEqual(nonceOf(a), nonceOf(b));
  });

  test('a username containing markup cannot break the envelope', () => {
    const header = buildSecurityHeader('ad<min&"', secret(PASSWORD));
    assert.match(header, /<wsse:Username>ad&lt;min&amp;&quot;<\/wsse:Username>/);
  });

  test('an envelope without credentials carries no security header', () => {
    const envelope = buildEnvelope('<x/>');
    assert.ok(!envelope.includes('Security'));
    assert.match(envelope, /<s:Body><x\/><\/s:Body>/);
  });
});

describe('XML extraction', () => {
  test('matches on the local name, because vendors pick their own prefixes', () => {
    for (const xml of [
      '<tds:Manufacturer>Acme</tds:Manufacturer>',
      '<s0:Manufacturer>Acme</s0:Manufacturer>',
      '<Manufacturer>Acme</Manufacturer>',
      '<ns2:Manufacturer xmlns:ns2="x">Acme</ns2:Manufacturer>',
    ]) {
      assert.equal(extractValue(xml, 'Manufacturer'), 'Acme', `failed for: ${xml}`);
    }
  });

  test('decodes the predefined entities', () => {
    assert.equal(extractValue('<Name>Front &amp; Rear &lt;Gate&gt;</Name>', 'Name'), 'Front & Rear <Gate>');
  });

  test('returns null rather than guessing when an element is absent', () => {
    assert.equal(extractValue('<a>1</a>', 'Missing'), null);
  });

  test('extracts repeated elements and nested blocks', () => {
    const xml = '<Root><Item>a</Item><Item>b</Item></Root>';
    assert.deepEqual(extractValues(xml, 'Item'), ['a', 'b']);
    assert.equal(extractBlocks(xml, 'Item').length, 2);
  });

  test('extracts attributes', () => {
    assert.equal(
      extractAttribute('<trt:Profiles token="MainProfile" fixed="true">x</trt:Profiles>', 'Profiles', 'token'),
      'MainProfile',
    );
  });

  test('refuses a DTD or entity declaration outright', () => {
    // No legitimate ONVIF device sends one, and both are the entry point for
    // entity-expansion attacks. Refusing is safer than sanitising.
    for (const hostile of [
      '<!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><foo>&xxe;</foo>',
      '<!ENTITY lol "lollol">',
    ]) {
      assert.throws(
        () => assertSafeXml(hostile),
        (error: unknown) => {
          assert.ok(error instanceof SoapError);
          assert.equal(error.code, 'SOAP_UNSAFE_XML');
          assert.equal(error.recoverable, false);
          return true;
        },
      );
    }
  });

  test('refuses an oversized response', () => {
    assert.throws(() => assertSafeXml('x'.repeat(600_000)), /exceeded/);
  });

  test('escaping is symmetric with decoding', () => {
    const original = 'Front & Rear <Gate> "main" \'sub\'';
    assert.equal(extractValue(`<N>${escapeXml(original)}</N>`, 'N'), original);
  });
});

describe('SOAP faults', () => {
  test('digs the reason out of the layer the vendor chose', () => {
    const xml =
      '<s:Fault><s:Code><s:Value>s:Sender</s:Value>' +
      '<s:Subcode><s:Value>ter:NotAuthorized</s:Value></s:Subcode></s:Code>' +
      '<s:Reason><s:Text xml:lang="en">Sender not authorized</s:Text></s:Reason></s:Fault>';

    assert.equal(faultFrom(xml), 'Sender not authorized');
  });

  test('falls back through the layers rather than reporting nothing', () => {
    const noText =
      '<s:Fault><s:Code><s:Subcode><s:Value>ter:InvalidArgVal</s:Value></s:Subcode></s:Code></s:Fault>';
    assert.equal(faultFrom(noText), 'ter:InvalidArgVal');

    const bare = '<s:Fault><s:Code/></s:Fault>';
    assert.match(faultFrom(bare) ?? '', /no description/);
  });

  test('a successful response is not a fault', () => {
    assert.equal(faultFrom('<GetDeviceInformationResponse><Model>X</Model></GetDeviceInformationResponse>'), null);
  });
});

// ---------------------------------------------------------------- discovery

describe('WS-Discovery parsing', () => {
  test('reads a ProbeMatch from a typical camera', () => {
    const match = parseProbeMatch(probeMatchXml(), '192.168.1.50');

    assert.notEqual(match, null);
    assert.equal(match?.address, 'urn:uuid:11223344-5566-7788-99aa-bbccddeeff00');
    assert.deepEqual(match?.serviceUrls, ['http://192.168.1.50/onvif/device_service']);
  });

  test('reads the scope vocabulary, which is all an operator has pre-auth', () => {
    const match = parseProbeMatch(probeMatchXml(), '192.168.1.50');
    const scope = parseScopes(match?.scopes ?? []);

    assert.equal(scope.name, 'Front Door', 'percent-escapes are decoded');
    assert.equal(scope.hardware, 'AX-4000');
    assert.equal(scope.location, 'Gate');
  });

  test('discards a device advertising an address off the local network', () => {
    // A hostile device on the subnet can claim any address it likes. Following
    // one would turn discovery into an outbound request to somewhere else.
    const match = parseProbeMatch(
      probeMatchXml({ xaddrs: 'http://198.51.100.7/onvif/device_service' }),
      '192.168.1.50',
    );
    assert.equal(match, null);
  });

  test('keeps only the private endpoints when a device advertises several', () => {
    const match = parseProbeMatch(
      probeMatchXml({
        xaddrs: 'http://8.8.8.8/onvif/device_service http://192.168.1.50/onvif/device_service',
      }),
      '192.168.1.50',
    );

    assert.deepEqual(match?.serviceUrls, ['http://192.168.1.50/onvif/device_service']);
  });

  test('ignores unrelated traffic on the discovery group', () => {
    // Printers and media servers share this multicast group. One unparsable
    // datagram must not abort a scan.
    assert.equal(parseProbeMatch('<Hello>printer</Hello>', '192.168.1.9'), null);
    assert.equal(parseProbeMatch('not xml at all', '192.168.1.9'), null);
    assert.equal(parseProbeMatch('', '192.168.1.9'), null);
  });

  test('refuses a ProbeMatch carrying a DTD', () => {
    const hostile = `<!DOCTYPE x [<!ENTITY e "v">]>${probeMatchXml()}`;
    assert.equal(parseProbeMatch(hostile, '192.168.1.50'), null);
  });

  test('a malformed percent-escape in a scope does not throw', () => {
    const scope = parseScopes(['onvif://www.onvif.org/name/Front%ZZDoor']);
    assert.equal(scope.name, 'Front%ZZDoor', 'reported verbatim rather than crashing the scan');
  });
});

// ------------------------------------------------------------- device client

describe('ONVIF device client', () => {
  test('reads device information with digest authentication', async () => {
    const device = await spawnDevice();

    const client = new OnvifDevice({
      host: '127.0.0.1',
      port: device.port,
      username: 'admin',
      password: secret(PASSWORD),
    });

    const info = await client.getDeviceInformation();
    assert.equal(info.manufacturer, 'Acme Optics');
    assert.equal(info.model, 'AX-4000');
    assert.equal(info.serialNumber, 'SN-0424-7781');
  });

  test('works across vendor namespace prefixes', async () => {
    const device = await spawnDevice({ prefix: 'ns7' });

    const client = new OnvifDevice({
      host: '127.0.0.1',
      port: device.port,
      username: 'admin',
      password: secret(PASSWORD),
    });

    assert.equal((await client.getDeviceInformation()).model, 'AX-4000');
  });

  test('maps a bad credential to an auth error, not a generic failure', async () => {
    // ONVIF signals this as a SOAP fault with HTTP 500. A client that only reads
    // the status code tells the operator nothing actionable.
    const device = await spawnDevice();

    const client = new OnvifDevice({
      host: '127.0.0.1',
      port: device.port,
      username: 'admin',
      password: secret('wrong-password'),
    });

    await assert.rejects(
      () => client.getDeviceInformation(),
      (error: unknown) => {
        assert.ok(error instanceof SoapError);
        assert.equal(error.code, 'ONVIF_AUTH_FAILED');
        assert.equal(error.recoverable, false, 'retrying a wrong password will not help');
        assert.match(error.message, /rejected the supplied credentials/);
        assert.ok(!error.message.includes('wrong-password'));
        return true;
      },
    );
  });

  test('reads both media profiles with their real encoder settings', async () => {
    const device = await spawnDevice();

    const client = new OnvifDevice({
      host: '127.0.0.1',
      port: device.port,
      username: 'admin',
      password: secret(PASSWORD),
    });

    const profiles = await client.getProfiles();
    assert.equal(profiles.length, 2);

    const main = profiles.find((p) => p.token === 'MainProfile');
    assert.equal(main?.width, 3840);
    assert.equal(main?.height, 2160);
    assert.equal(main?.encoding, 'H264');
    assert.equal(main?.frameRateLimit, 25);
    assert.equal(main?.gopLength, 50);

    const sub = profiles.find((p) => p.token === 'SubProfile');
    assert.equal(sub?.width, 640);
    assert.equal(sub?.frameRateLimit, 10);
  });

  test('reports capabilities', async () => {
    const device = await spawnDevice();
    const client = new OnvifDevice({
      host: '127.0.0.1',
      port: device.port,
      username: 'admin',
      password: secret(PASSWORD),
    });

    const capabilities = await client.getCapabilities();
    assert.equal(capabilities.media, true);
    assert.equal(capabilities.events, true);
    assert.equal(capabilities.ptz, false);
  });

  test('decomposes the stream URI instead of keeping a URL', async () => {
    const device = await spawnDevice();
    const client = new OnvifDevice({
      host: '127.0.0.1',
      port: device.port,
      username: 'admin',
      password: secret(PASSWORD),
    });

    const endpoint = await client.getStreamUri('SubProfile');
    assert.equal(endpoint.host, '127.0.0.1');
    assert.equal(endpoint.port, 554);
    assert.equal(endpoint.path, '/Streaming/Channels/102');
    assert.equal(endpoint.profileToken, 'SubProfile');
  });

  test('strips credentials the camera embedded in the stream URI', async () => {
    // Real firmware does return rtsp://user:pass@host/path, and an integration
    // that passes that string around is how camera passwords reach log files.
    const device = await spawnDevice({ credentialsInStreamUri: true });

    const client = new OnvifDevice({
      host: '127.0.0.1',
      port: device.port,
      username: 'admin',
      password: secret(PASSWORD),
    });

    const endpoint = await client.getStreamUri('MainProfile');
    const serialised = JSON.stringify(endpoint);

    assert.ok(!serialised.includes(PASSWORD), `credential survived into the endpoint: ${serialised}`);
    assert.ok(!serialised.includes('admin'));
    assert.ok(!serialised.includes('@'));
    assert.equal(endpoint.host, '127.0.0.1');
    assert.equal(endpoint.path, '/Streaming/Channels/101');
  });

  test('prefers the reachable host over one only the camera can see', () => {
    // A NATed camera reports its internal address; a hostile one could report
    // somebody else's. Either way the address we reached it on is the truth.
    const client = new OnvifDevice({ host: '192.168.1.50', port: 80 });
    const endpoint = client.parseStreamUri('rtsp://10.99.0.1:554/live', 'P1');

    assert.equal(endpoint.host, '192.168.1.50');
    assert.equal(endpoint.path, '/live');
  });

  test('refuses a stream URI that is not RTSP', () => {
    const client = new OnvifDevice({ host: '192.168.1.50', port: 80 });

    assert.throws(
      () => client.parseStreamUri('http://192.168.1.50/snapshot.jpg', 'P1'),
      (error: unknown) => {
        assert.ok(error instanceof SoapError);
        assert.equal(error.code, 'ONVIF_BAD_STREAM_URI');
        return true;
      },
    );
    assert.throws(() => client.parseStreamUri('not a url', 'P1'), SoapError);
  });

  test('refuses to contact a device outside the local network', () => {
    // The egress guard applies here too: a discovery response cannot lure the
    // client into an outbound request.
    assert.throws(() => new OnvifDevice({ host: '8.8.8.8', port: 80 }), /Refused to contact/);
  });

  test('refuses a response containing a DTD', async () => {
    const device = await spawnDevice({ injectDoctype: true });
    const client = new OnvifDevice({
      host: '127.0.0.1',
      port: device.port,
      username: 'admin',
      password: secret(PASSWORD),
    });

    await assert.rejects(() => client.getDeviceInformation(), /DTD or entity declaration/);
  });

  test('surfaces a device fault with the text the device supplied', async () => {
    const device = await spawnDevice({ faultWith: 'Optional Action Not Implemented' });
    const client = new OnvifDevice({
      host: '127.0.0.1',
      port: device.port,
      username: 'admin',
      password: secret(PASSWORD),
    });

    await assert.rejects(
      () => client.getDeviceInformation(),
      (error: unknown) => {
        assert.ok(error instanceof SoapError);
        assert.equal(error.code, 'ONVIF_FAULT');
        assert.match(error.message, /Optional Action Not Implemented/);
        return true;
      },
    );
  });

  test('reports an unreachable device without hanging', async () => {
    const client = new OnvifDevice({ host: '127.0.0.1', port: 1, timeoutMillis: 2000 });
    await assert.rejects(() => client.getDeviceInformation(), /Could not reach/);
  });
});

describe('profile selection', () => {
  const profiles = [
    { token: 'main', name: 'Main', encoding: 'H264', width: 3840, height: 2160, frameRateLimit: 25, bitrateLimit: 8192, gopLength: 50 },
    { token: 'sub', name: 'Sub', encoding: 'H264', width: 640, height: 360, frameRateLimit: 10, bitrateLimit: 512, gopLength: 20 },
    { token: 'tiny', name: 'Tiny', encoding: 'H264', width: 320, height: 180, frameRateLimit: 5, bitrateLimit: 128, gopLength: 10 },
  ];

  test('inference takes the smallest stream still big enough to detect a person', () => {
    // Inference cost scales with pixels while detection quality plateaus well
    // below 4K. Running a detector on a main stream is the most common way a
    // deployment needs four times the hardware it actually requires.
    assert.equal(selectInferenceProfile(profiles)?.token, 'sub');
  });

  test('recording takes the highest resolution, because that is the evidence', () => {
    assert.equal(selectRecordingProfile(profiles)?.token, 'main');
  });

  test('falls back sensibly when every stream is tiny', () => {
    const tinyOnly = [profiles[2]!];
    assert.equal(selectInferenceProfile(tinyOnly)?.token, 'tiny');
  });

  test('a camera with no profiles yields null rather than a guess', () => {
    assert.equal(selectInferenceProfile([]), null);
    assert.equal(selectRecordingProfile([]), null);
  });
});
