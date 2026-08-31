import type { Secret } from '@sentinel/security';
import { EgressGuard } from '@sentinel/security';
import type { UtcMillis } from '@sentinel/shared-types';
import { utcMillis } from '@sentinel/shared-types';
import { OnvifDevice, selectInferenceProfile, selectRecordingProfile } from '../onvif/device.ts';
import type { DeviceInformation, MediaProfile, StreamEndpoint } from '../onvif/device.ts';
import { SoapError } from '../onvif/soap.ts';
import { RtspClient, RtspError } from '../rtsp/client.ts';
import { DECODABLE_CODECS } from '../rtsp/sdp.ts';
import { AuthenticationError, UnsupportedAuthError } from '../rtsp/auth.ts';

/**
 * The "TEST CONNECTION" step of camera onboarding.
 *
 * Adding a camera fails for a dozen different reasons and they are not
 * interchangeable: a typo'd password, a blocked port, a camera that only speaks
 * H.265, and a VLAN that does not route are four completely different jobs for
 * whoever is holding the ladder. A test that reports "connection failed" makes
 * all four look the same and sends an integrator hunting.
 *
 * So this runs the checks in dependency order and reports each one individually
 * with a remedy. Later checks are skipped rather than failed when a prerequisite
 * did not pass - reporting "RTSP failed" when the host was unreachable is noise
 * that buries the one line that mattered.
 */

export const CheckStatus = {
  Passed: 'PASSED',
  Failed: 'FAILED',
  /** A prerequisite did not pass, so this was never attempted. */
  Skipped: 'SKIPPED',
  /** Worked, but something is worth knowing before this goes into service. */
  Warning: 'WARNING',
} as const;
export type CheckStatus = (typeof CheckStatus)[keyof typeof CheckStatus];

export type CheckResult = {
  readonly name: string;
  readonly status: CheckStatus;
  readonly detail: string;
  /** What to do about it. Absent only when there is nothing to do. */
  readonly remedy?: string;
  readonly durationMillis: number;
};

export type ConnectionTestReport = {
  readonly reachable: boolean;
  readonly authenticated: boolean;
  readonly streaming: boolean;
  readonly checks: readonly CheckResult[];
  readonly device: DeviceInformation | null;
  readonly profiles: readonly MediaProfile[];
  /** Endpoint chosen for inference: the smallest usable stream. */
  readonly inferenceStream: StreamEndpoint | null;
  /** Endpoint chosen for recording: the highest resolution available. */
  readonly recordingStream: StreamEndpoint | null;
  readonly ptzSupported: boolean;
  readonly startedAt: UtcMillis;
  readonly durationMillis: number;
};

export type ConnectionTestOptions = {
  readonly host: string;
  readonly onvifPort?: number;
  readonly rtspPort?: number;
  readonly username?: string;
  readonly password?: Secret<string>;
  /** Skip ONVIF and test this RTSP path directly. For devices with no ONVIF. */
  readonly rtspPath?: string;
  readonly timeoutMillis?: number;
  readonly allowBasicAuth?: boolean;
  readonly egressGuard?: EgressGuard;
  readonly now?: () => UtcMillis;
};

export const testCameraConnection = async (
  options: ConnectionTestOptions,
): Promise<ConnectionTestReport> => {
  const now = options.now ?? (() => utcMillis(Date.now()));
  const startedAt = now();
  const checks: CheckResult[] = [];

  /*
   * Results live on an object rather than in local `let`s.
   *
   * Every one of these is assigned inside a callback passed to `run`, and the
   * compiler cannot see that the callback has executed - so a plain `let x: T |
   * null = null` stays narrowed to `null` at every later read. Holding them as
   * properties keeps the declared type, which is the honest one here.
   */
  const state: {
    device: DeviceInformation | null;
    profiles: readonly MediaProfile[];
    inferenceStream: StreamEndpoint | null;
    recordingStream: StreamEndpoint | null;
    ptzSupported: boolean;
    reachable: boolean;
    authenticated: boolean;
    streaming: boolean;
  } = {
    device: null,
    profiles: [],
    inferenceStream: null,
    recordingStream: null,
    ptzSupported: false,
    reachable: false,
    authenticated: false,
    streaming: false,
  };

  const run = async (
    name: string,
    body: () => Promise<Omit<CheckResult, 'name' | 'durationMillis'>>,
  ): Promise<CheckResult> => {
    const began = Date.now();
    try {
      const outcome = await body();
      const result = { name, ...outcome, durationMillis: Date.now() - began };
      checks.push(result);
      return result;
    } catch (error) {
      const result: CheckResult = {
        name,
        ...describeFailure(error),
        durationMillis: Date.now() - began,
      };
      checks.push(result);
      return result;
    }
  };

  const skip = (name: string, detail: string): void => {
    checks.push({ name, status: CheckStatus.Skipped, detail, durationMillis: 0 });
  };

  // ----------------------------------------------------------- 1. addressing
  const guard = options.egressGuard ?? new EgressGuard();
  const addressCheck = await run('Address', async () => {
    guard.check(options.host);
    return {
      status: CheckStatus.Passed,
      detail: `${options.host} is on the local network.`,
    };
  });

  if (addressCheck.status !== CheckStatus.Passed) {
    return finish();
  }

  // ------------------------------------------------------------- 2. ONVIF
  const useOnvif = options.rtspPath === undefined;

  if (useOnvif) {
    const onvif = new OnvifDevice({
      host: options.host,
      port: options.onvifPort ?? 80,
      ...(options.username === undefined ? {} : { username: options.username }),
      ...(options.password === undefined ? {} : { password: options.password }),
      ...(options.timeoutMillis === undefined ? {} : { timeoutMillis: options.timeoutMillis }),
      egressGuard: guard,
    });

    const deviceCheck = await run('ONVIF device', async () => {
      state.device = await onvif.getDeviceInformation();
      state.reachable = true;
      state.authenticated = true;

      const described = [state.device.manufacturer, state.device.model].filter((v) => v !== null).join(' ');
      return {
        status: CheckStatus.Passed,
        detail:
          described === ''
            ? 'The device answered but reported no make or model.'
            : `${described}, firmware ${state.device.firmwareVersion ?? 'unknown'}.`,
      };
    });

    if (deviceCheck.status !== CheckStatus.Passed) {
      skip('Media profiles', 'Not attempted: the device did not answer.');
      skip('Stream', 'Not attempted: the device did not answer.');
      skip('Codec', 'Not attempted: no stream was established.');
      return finish();
    }

    await run('Capabilities', async () => {
      const capabilities = await onvif.getCapabilities();
      state.ptzSupported = capabilities.ptz;
      return {
        status: CheckStatus.Passed,
        detail:
          `media ${capabilities.media ? 'yes' : 'no'}, ` +
          `events ${capabilities.events ? 'yes' : 'no'}, ` +
          `PTZ ${capabilities.ptz ? 'yes' : 'no'}.`,
      };
    });

    const profileCheck = await run('Media profiles', async () => {
      state.profiles = await onvif.getProfiles();

      if (state.profiles.length === 0) {
        return {
          status: CheckStatus.Failed,
          detail: 'The camera reports no media profiles.',
          remedy:
            'Create at least one video profile in the camera web interface, then test again.',
        };
      }

      const summary = state.profiles
        .map((p) => `${p.name ?? p.token} ${p.width ?? '?'}x${p.height ?? '?'} ${p.encoding ?? '?'}`)
        .join(', ');
      return { status: CheckStatus.Passed, detail: `${state.profiles.length} profile(s): ${summary}.` };
    });

    if (profileCheck.status !== CheckStatus.Passed) {
      skip('Stream', 'Not attempted: no usable media profile.');
      skip('Codec', 'Not attempted: no stream was established.');
      return finish();
    }

    const inference = selectInferenceProfile(state.profiles);
    const recording = selectRecordingProfile(state.profiles);

    await run('Stream endpoints', async () => {
      if (inference !== null) state.inferenceStream = await onvif.getStreamUri(inference.token);
      if (recording !== null && recording.token !== inference?.token) {
        state.recordingStream = await onvif.getStreamUri(recording.token);
      } else {
        state.recordingStream = state.inferenceStream;
      }

      const sameStream = inference?.token === recording?.token;
      return {
        status: sameStream ? CheckStatus.Warning : CheckStatus.Passed,
        detail: sameStream
          ? `Only one profile is available, so inference and recording share it ` +
            `(${inference?.width ?? '?'}x${inference?.height ?? '?'}).`
          : `Inference will use ${inference?.name ?? inference?.token} ` +
            `(${inference?.width}x${inference?.height}), recording ` +
            `${recording?.name ?? recording?.token} (${recording?.width}x${recording?.height}).`,
        ...(sameStream
          ? {
              remedy:
                'Enable a sub stream on the camera. Running inference on a full-resolution ' +
                'stream costs several times the compute for no detection benefit.',
            }
          : {}),
      };
    });
  }

  // -------------------------------------------------------------- 3. RTSP
  const rtspPath = options.rtspPath ?? state.inferenceStream?.path;
  const rtspPort = options.rtspPort ?? state.inferenceStream?.port ?? 554;

  if (rtspPath === undefined) {
    skip('RTSP', 'Not attempted: no stream path is known.');
    skip('Codec', 'Not attempted: no stream was established.');
    return finish();
  }

  const client = new RtspClient({
    host: options.host,
    port: rtspPort,
    path: rtspPath,
    ...(options.username === undefined ? {} : { username: options.username }),
    ...(options.password === undefined ? {} : { password: options.password }),
    ...(options.timeoutMillis === undefined ? {} : { timeoutMillis: options.timeoutMillis }),
    ...(options.allowBasicAuth === undefined ? {} : { allowBasicAuth: options.allowBasicAuth }),
  });

  let encoding: string | null = null;

  const rtspCheck = await run('RTSP', async () => {
    const session = await client.open();
    state.reachable = true;
    state.authenticated = true;
    state.streaming = true;
    encoding = session.encoding;

    return {
      status: CheckStatus.Passed,
      detail:
        `Session established on ${client.url}. ` +
        `Keep-alive every ${session.timeoutSeconds}s via ` +
        `${session.supportsGetParameter ? 'GET_PARAMETER' : 'OPTIONS'}.`,
      ...(session.sdp.warnings.length > 0
        ? { remedy: `The camera's stream description had issues: ${session.sdp.warnings.join('; ')}` }
        : {}),
    };
  });

  if (rtspCheck.status === CheckStatus.Passed) {
    await run('Codec', async () => {
      if (encoding === null) {
        return {
          status: CheckStatus.Warning,
          detail: 'The camera did not declare a codec.',
          remedy: 'The stream may still decode. Confirm with a live view before relying on it.',
        };
      }

      const supported = DECODABLE_CODECS.includes(encoding);
      return supported
        ? { status: CheckStatus.Passed, detail: `${encoding}, which this system decodes.` }
        : {
            status: CheckStatus.Failed,
            detail: `The stream is ${encoding}, which this system does not decode.`,
            remedy:
              `Set the profile to one of ${DECODABLE_CODECS.join(', ')} in the camera ` +
              'web interface. H.264 is the safest choice for compatibility.',
          };
    });
  } else {
    skip('Codec', 'Not attempted: no stream was established.');
  }

  await client.teardown();
  client.close();

  return finish();

  function finish(): ConnectionTestReport {
    return {
      reachable: state.reachable,
      authenticated: state.authenticated,
      streaming: state.streaming,
      checks,
      device: state.device,
      profiles: state.profiles,
      inferenceStream: state.inferenceStream,
      recordingStream: state.recordingStream,
      ptzSupported: state.ptzSupported,
      startedAt,
      durationMillis: Date.now() - startedAt,
    };
  }
};

/**
 * Turn an error into a diagnosis and a remedy.
 *
 * This is the function that decides whether an integrator on a ladder gets "check
 * the password" or "connection failed". Every branch names a distinct physical
 * cause, because they require distinct physical actions.
 */
const describeFailure = (error: unknown): { status: CheckStatus; detail: string; remedy?: string } => {
  if (error instanceof AuthenticationError || (error instanceof SoapError && error.code === 'ONVIF_AUTH_FAILED')) {
    return {
      status: CheckStatus.Failed,
      detail: 'The camera rejected the credentials.',
      remedy:
        'Check the username and password. Many cameras use a separate ONVIF user ' +
        'account from the web-interface login, and some require the ONVIF user to be ' +
        'created explicitly before it will authenticate.',
    };
  }

  if (error instanceof UnsupportedAuthError) {
    return {
      status: CheckStatus.Failed,
      detail: error.message,
      remedy: 'Enable Digest authentication on the camera, or allow Basic explicitly if the device supports nothing else.',
    };
  }

  if (error instanceof RtspError) {
    switch (error.code) {
      case 'RTSP_CONNECT_FAILED':
        return {
          status: CheckStatus.Failed,
          detail: error.message,
          remedy:
            'Check that the camera is powered and on this VLAN, and that the port is ' +
            'not blocked. A camera that pings but refuses the port is usually a firewall ' +
            'rule or a changed service port rather than a broken device.',
        };
      case 'RTSP_CONNECT_TIMEOUT':
      case 'RTSP_TIMEOUT':
        return {
          status: CheckStatus.Failed,
          detail: error.message,
          remedy:
            'The camera accepted the connection but did not answer. This usually means ' +
            'the port belongs to a different service, or the camera is overloaded by ' +
            'existing stream clients.',
        };
      case 'RTSP_NO_VIDEO_TRACK':
        return {
          status: CheckStatus.Failed,
          detail: error.message,
          remedy: 'Enable a video profile on the camera. An audio-only stream cannot be used.',
        };
      case 'RTSP_UNSUPPORTED_CODEC':
        return {
          status: CheckStatus.Failed,
          detail: error.message,
          remedy:
            `The camera streams video, but not in a codec this system decodes. Set the ` +
            `profile to one of ${DECODABLE_CODECS.join(', ')} in the camera web interface.`,
        };
      case 'RTSP_MALFORMED':
        return {
          status: CheckStatus.Failed,
          detail: error.message,
          remedy:
            'The endpoint answered, but not in RTSP. Check the port: 80 and 8000 are ' +
            'usually the web interface, not the stream.',
        };
      case 'RTSP_SETUP_FAILED':
        return {
          status: CheckStatus.Failed,
          detail: error.message,
          remedy:
            'The camera refused the transport. Some devices allow only a fixed number of ' +
            'concurrent streams; disconnect other clients and try again.',
        };
      default:
        return { status: CheckStatus.Failed, detail: error.message };
    }
  }

  if (error instanceof SoapError) {
    switch (error.code) {
      case 'ONVIF_UNREACHABLE':
        return {
          status: CheckStatus.Failed,
          detail: error.message,
          remedy:
            'Check that the camera is powered and on this VLAN, and that the ONVIF ' +
            'service port is not blocked. Many cameras ship with ONVIF disabled and ' +
            'need it enabled in the web interface before this will answer.',
        };
      case 'ONVIF_TIMEOUT':
        return {
          status: CheckStatus.Failed,
          detail: error.message,
          remedy:
            'The device accepted the connection but did not answer. Check that the ONVIF ' +
            'port is right: 80 and 8000 are often the web interface rather than the service.',
        };
      case 'ONVIF_HTTP_ERROR':
        return {
          status: CheckStatus.Failed,
          detail: error.message,
          remedy:
            'The endpoint answered, but not as an ONVIF service. Check the ONVIF service ' +
            'path, which is usually /onvif/device_service.',
        };
      case 'SOAP_UNSAFE_XML':
        return {
          status: CheckStatus.Failed,
          detail: error.message,
          remedy:
            'The device sent XML containing a document type declaration, which no ' +
            'legitimate ONVIF camera does. Treat this device as suspect.',
        };
      case 'ONVIF_NO_STREAM_URI':
      case 'ONVIF_BAD_STREAM_URI':
        return {
          status: CheckStatus.Failed,
          detail: error.message,
          remedy:
            'The camera did not return a usable RTSP address for this profile. Enter the ' +
            'stream path manually to bypass ONVIF.',
        };
      default:
        return {
          status: CheckStatus.Failed,
          detail: error.message,
          remedy:
            'The camera reported a fault. Its own message is quoted above; check the ' +
            'device log if it is not self-explanatory.',
        };
    }
  }

  const message = error instanceof Error ? error.message : String(error);

  if (/Refused to contact/.test(message)) {
    return {
      status: CheckStatus.Failed,
      detail: message,
      remedy:
        'Cameras must be on a private network. Sentinel Vision does not reach the ' +
        'Internet, by design.',
    };
  }

  return { status: CheckStatus.Failed, detail: message };
};

/** A one-line verdict for the wizard's header. */
export const summariseReport = (report: ConnectionTestReport): string => {
  const failed = report.checks.filter((c) => c.status === CheckStatus.Failed);
  const warned = report.checks.filter((c) => c.status === CheckStatus.Warning);

  if (failed.length > 0) {
    return `Cannot use this camera yet: ${failed[0]?.detail ?? 'a check failed'}`;
  }
  if (warned.length > 0) {
    return `Ready, with ${warned.length} thing(s) worth reviewing.`;
  }
  return 'Ready to add.';
};
