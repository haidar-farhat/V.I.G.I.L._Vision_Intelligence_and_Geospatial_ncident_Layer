import { test, describe, after } from 'node:test';
import assert from 'node:assert/strict';
import { secret } from '@sentinel/security';
import { startMockOnvifDevice, startMockRtspServer } from '@sentinel/test-utils';
import type { MockOnvifDevice, MockRtspServer } from '@sentinel/test-utils';
import { CheckStatus, summariseReport, testCameraConnection } from '../src/ingest/connection-test.ts';
import type { ConnectionTestReport } from '../src/ingest/connection-test.ts';

const PASSWORD = 'correct-horse-battery';

const onvifDevices: MockOnvifDevice[] = [];
const rtspServers: MockRtspServer[] = [];

after(async () => {
  for (const device of onvifDevices) await device.close();
  for (const server of rtspServers) await server.close();
});

/** A camera with both an ONVIF service and a working RTSP stream. */
const spawnCamera = async (
  onvifBehaviour = {},
  rtspBehaviour = {},
): Promise<{ onvifPort: number; rtspPort: number }> => {
  const rtsp = await startMockRtspServer({ password: PASSWORD, ...rtspBehaviour });
  rtspServers.push(rtsp);

  const onvif = await startMockOnvifDevice({
    password: PASSWORD,
    rtspPort: rtsp.port,
    ...onvifBehaviour,
  });
  onvifDevices.push(onvif);

  return { onvifPort: onvif.port, rtspPort: rtsp.port };
};

const named = (report: ConnectionTestReport, name: string) =>
  report.checks.find((c) => c.name === name);

describe('camera onboarding: the happy path', () => {
  test('discovers the device, its profiles and a working stream', async () => {
    const camera = await spawnCamera();

    const report = await testCameraConnection({
      host: '127.0.0.1',
      onvifPort: camera.onvifPort,
      username: 'admin',
      password: secret(PASSWORD),
    });

    assert.equal(report.reachable, true);
    assert.equal(report.authenticated, true);
    assert.equal(report.streaming, true);

    assert.equal(report.device?.manufacturer, 'Acme Optics');
    assert.equal(report.device?.model, 'AX-4000');
    assert.equal(report.profiles.length, 2);

    for (const check of report.checks) {
      assert.notEqual(check.status, CheckStatus.Failed, `${check.name}: ${check.detail}`);
    }
    assert.equal(summariseReport(report), 'Ready to add.');
  });

  test('assigns the sub stream to inference and the main stream to recording', async () => {
    // This is the sizing decision that decides whether a deployment needs one
    // GPU or four, so the wizard makes it explicitly rather than by default.
    const camera = await spawnCamera();

    const report = await testCameraConnection({
      host: '127.0.0.1',
      onvifPort: camera.onvifPort,
      username: 'admin',
      password: secret(PASSWORD),
    });

    assert.equal(report.inferenceStream?.path, '/Streaming/Channels/102');
    assert.equal(report.recordingStream?.path, '/Streaming/Channels/101');
    assert.match(named(report, 'Stream endpoints')?.detail ?? '', /SubStream/);
  });

  test('never exposes the credential anywhere in the report', async () => {
    // The report is rendered in the UI, written to diagnostics and pasted into
    // support tickets. Every one of those is a leak path.
    const camera = await spawnCamera({ credentialsInStreamUri: true });

    const report = await testCameraConnection({
      host: '127.0.0.1',
      onvifPort: camera.onvifPort,
      username: 'admin',
      password: secret(PASSWORD),
    });

    const serialised = JSON.stringify(report);
    assert.ok(!serialised.includes(PASSWORD), 'the credential reached the report');
    assert.ok(!serialised.includes('@127.0.0.1'), 'a credentialed URL reached the report');
  });

  test('reports PTZ availability, which gates a permissioned action', async () => {
    const camera = await spawnCamera();

    const report = await testCameraConnection({
      host: '127.0.0.1',
      onvifPort: camera.onvifPort,
      username: 'admin',
      password: secret(PASSWORD),
    });

    assert.equal(report.ptzSupported, false, 'the mock advertises no PTZ');
    assert.match(named(report, 'Capabilities')?.detail ?? '', /PTZ no/);
  });
});

describe('camera onboarding: distinguishing failures', () => {
  test('a wrong password says so, and says why it might be wrong', async () => {
    const camera = await spawnCamera();

    const report = await testCameraConnection({
      host: '127.0.0.1',
      onvifPort: camera.onvifPort,
      username: 'admin',
      password: secret('wrong-password'),
    });

    const check = named(report, 'ONVIF device');
    assert.equal(check?.status, CheckStatus.Failed);
    assert.match(check?.detail ?? '', /rejected the credentials/);

    // The remedy is the actual point. "Authentication failed" sends somebody up
    // a ladder; naming the separate ONVIF account saves the trip.
    assert.match(check?.remedy ?? '', /separate ONVIF user/i);
    assert.equal(report.authenticated, false);
  });

  test('an unreachable host is not confused with a wrong password', async () => {
    const report = await testCameraConnection({
      host: '127.0.0.1',
      onvifPort: 1,
      username: 'admin',
      password: secret(PASSWORD),
      timeoutMillis: 1500,
    });

    const check = named(report, 'ONVIF device');
    assert.equal(check?.status, CheckStatus.Failed);
    assert.match(check?.remedy ?? '', /powered and on this VLAN/i);
    assert.equal(report.reachable, false);
  });

  test('a public address is refused with the design reason, not a network error', async () => {
    const report = await testCameraConnection({
      host: '8.8.8.8',
      username: 'admin',
      password: secret(PASSWORD),
    });

    const check = named(report, 'Address');
    assert.equal(check?.status, CheckStatus.Failed);
    assert.match(check?.remedy ?? '', /does not reach the\s+Internet, by design/);
  });

  test('checks after a failure are skipped, not failed', async () => {
    // Reporting "RTSP failed" when the host was never reachable buries the one
    // line that mattered under three that did not.
    const report = await testCameraConnection({
      host: '8.8.8.8',
      username: 'admin',
      password: secret(PASSWORD),
    });

    assert.equal(report.checks.filter((c) => c.status === CheckStatus.Failed).length, 1);
    assert.ok(!report.checks.some((c) => c.name === 'RTSP' && c.status === CheckStatus.Failed));
  });

  test('a camera with no media profiles is told what to configure', async () => {
    const camera = await spawnCamera({ noProfiles: true });

    const report = await testCameraConnection({
      host: '127.0.0.1',
      onvifPort: camera.onvifPort,
      username: 'admin',
      password: secret(PASSWORD),
    });

    const check = named(report, 'Media profiles');
    assert.equal(check?.status, CheckStatus.Failed);
    assert.match(check?.remedy ?? '', /Create at least one video profile/);
    assert.equal(named(report, 'Stream')?.status, CheckStatus.Skipped);
  });

  test('an unsupported codec is a specific, fixable problem', async () => {
    const camera = await spawnCamera(
      {},
      {
        sdp: [
          'v=0',
          's=VP9 Camera',
          'a=control:*',
          'm=video 0 RTP/AVP 96',
          'a=rtpmap:96 VP9/90000',
          'a=control:trackID=1',
        ].join('\r\n'),
      },
    );

    const report = await testCameraConnection({
      host: '127.0.0.1',
      onvifPort: camera.onvifPort,
      username: 'admin',
      password: secret(PASSWORD),
    });

    // A camera streaming VP9 fails at RTSP rather than reaching the codec check,
    // because no track could be selected. The remedy must still name the codec
    // problem rather than telling somebody to enable a video profile they have.
    const check = named(report, 'RTSP');
    assert.equal(check?.status, CheckStatus.Failed);
    assert.match(check?.detail ?? '', /VP9/);
    assert.match(check?.remedy ?? '', /not in a codec this system decodes/);
    assert.match(check?.remedy ?? '', /H264/);
  });

  test('a device sending a DTD is flagged as suspect, not as a parse error', async () => {
    const camera = await spawnCamera({ injectDoctype: true });

    const report = await testCameraConnection({
      host: '127.0.0.1',
      onvifPort: camera.onvifPort,
      username: 'admin',
      password: secret(PASSWORD),
    });

    const check = named(report, 'ONVIF device');
    assert.equal(check?.status, CheckStatus.Failed);
    assert.match(check?.remedy ?? '', /Treat this device as suspect/);
  });
});

describe('camera onboarding: cameras without ONVIF', () => {
  test('an RTSP-only camera can be added by path', async () => {
    // Plenty of working cameras expose RTSP and no usable ONVIF. Requiring ONVIF
    // would exclude hardware that is otherwise perfectly serviceable.
    const rtsp = await startMockRtspServer({ password: PASSWORD });
    rtspServers.push(rtsp);

    const report = await testCameraConnection({
      host: '127.0.0.1',
      rtspPort: rtsp.port,
      rtspPath: '/live',
      username: 'admin',
      password: secret(PASSWORD),
    });

    assert.equal(report.streaming, true);
    assert.equal(named(report, 'RTSP')?.status, CheckStatus.Passed);
    assert.equal(named(report, 'Codec')?.status, CheckStatus.Passed);

    // No ONVIF was attempted, so nothing about it is reported either way.
    assert.equal(named(report, 'ONVIF device'), undefined);
    assert.equal(report.device, null);
  });

  test('an unsecured RTSP camera works with no credentials', async () => {
    const rtsp = await startMockRtspServer({ noAuth: true });
    rtspServers.push(rtsp);

    const report = await testCameraConnection({
      host: '127.0.0.1',
      rtspPort: rtsp.port,
      rtspPath: '/live',
    });

    assert.equal(report.streaming, true);
  });

  test('a wrong RTSP path is reported as such', async () => {
    const rtsp = await startMockRtspServer({
      password: PASSWORD,
      sdp: ['v=0', 's=Audio', 'm=audio 0 RTP/AVP 97', 'a=rtpmap:97 PCMU/8000'].join('\r\n'),
    });
    rtspServers.push(rtsp);

    const report = await testCameraConnection({
      host: '127.0.0.1',
      rtspPort: rtsp.port,
      rtspPath: '/wrong',
      username: 'admin',
      password: secret(PASSWORD),
    });

    const check = named(report, 'RTSP');
    assert.equal(check?.status, CheckStatus.Failed);
    assert.match(check?.remedy ?? '', /Enable a video profile/);
  });
});

describe('the report as a whole', () => {
  test('every check carries a duration, so a slow camera is visible', async () => {
    const camera = await spawnCamera();

    const report = await testCameraConnection({
      host: '127.0.0.1',
      onvifPort: camera.onvifPort,
      username: 'admin',
      password: secret(PASSWORD),
    });

    for (const check of report.checks) {
      assert.ok(check.durationMillis >= 0, `${check.name} has no duration`);
    }
    assert.ok(report.durationMillis > 0);
  });

  test('the summary leads with the failure an operator must act on', async () => {
    const camera = await spawnCamera();

    const failing = await testCameraConnection({
      host: '127.0.0.1',
      onvifPort: camera.onvifPort,
      username: 'admin',
      password: secret('wrong-password'),
    });

    assert.match(summariseReport(failing), /^Cannot use this camera yet:/);
    assert.match(summariseReport(failing), /rejected the credentials/);
  });

  test('every failed check offers a remedy', async () => {
    // A diagnostic that names a problem without naming an action is only half a
    // diagnostic.
    const reports = await Promise.all([
      testCameraConnection({ host: '8.8.8.8' }),
      testCameraConnection({ host: '127.0.0.1', onvifPort: 1, timeoutMillis: 1200 }),
    ]);

    for (const report of reports) {
      for (const check of report.checks) {
        if (check.status !== CheckStatus.Failed) continue;
        assert.ok(
          (check.remedy ?? '').length > 0,
          `"${check.name}" failed with no remedy: ${check.detail}`,
        );
      }
    }
  });
});
