import { request as httpRequest } from 'node:http';
import type { Secret } from '@sentinel/security';
import { EgressGuard } from '@sentinel/security';
import type { SoapCredentials } from './soap.ts';
import {
  ONVIF_NAMESPACES,
  SoapError,
  assertSafeXml,
  buildEnvelope,
  extractAttribute,
  extractBlocks,
  extractValue,
  faultFrom,
} from './soap.ts';

/**
 * ONVIF device and media service client.
 *
 * Covers the four calls needed to turn a discovered address into a configured
 * camera: device information, capabilities, media profiles, and the stream URI
 * for a chosen profile.
 *
 * The stream URI is the interesting one. ONVIF devices return it as a complete
 * `rtsp://host:554/path` URL, and a great many integrations then splice
 * credentials into that string and hand the result around. That single habit is
 * responsible for most camera passwords that end up in logs. Here the URL is
 * split into host, port and path immediately, and the credential is never joined
 * to it - the RTSP client authenticates in a header instead.
 */

const DEFAULT_TIMEOUT_MILLIS = 8000;

export type OnvifDeviceOptions = {
  readonly host: string;
  readonly port?: number;
  readonly username?: string;
  readonly password?: Secret<string>;
  readonly timeoutMillis?: number;
  /** Service path. Almost universally /onvif/device_service. */
  readonly path?: string;
  readonly egressGuard?: EgressGuard;
};

export type DeviceInformation = {
  readonly manufacturer: string | null;
  readonly model: string | null;
  readonly firmwareVersion: string | null;
  readonly serialNumber: string | null;
  readonly hardwareId: string | null;
};

export type MediaProfile = {
  /** Opaque profile token, required by every later media call. */
  readonly token: string;
  readonly name: string | null;
  readonly encoding: string | null;
  readonly width: number | null;
  readonly height: number | null;
  readonly frameRateLimit: number | null;
  readonly bitrateLimit: number | null;
  readonly gopLength: number | null;
};

/**
 * A stream endpoint, decomposed.
 *
 * Never a single URL string, and never carries a credential. Keeping the parts
 * separate is what makes it impossible to accidentally log a credentialed URL,
 * because no credentialed URL is ever constructed.
 */
export type StreamEndpoint = {
  readonly host: string;
  readonly port: number;
  readonly path: string;
  /** The profile this endpoint belongs to. */
  readonly profileToken: string;
};

export class OnvifDevice {
  readonly #options: Required<Omit<OnvifDeviceOptions, 'username' | 'password' | 'egressGuard'>> &
    Pick<OnvifDeviceOptions, 'username' | 'password'>;
  readonly #guard: EgressGuard;

  constructor(options: OnvifDeviceOptions) {
    this.#options = {
      host: options.host,
      port: options.port ?? 80,
      path: options.path ?? '/onvif/device_service',
      timeoutMillis: options.timeoutMillis ?? DEFAULT_TIMEOUT_MILLIS,
      ...(options.username === undefined ? {} : { username: options.username }),
      ...(options.password === undefined ? {} : { password: options.password }),
    };

    // A device that redirects discovery to a public address must not be followed.
    this.#guard = options.egressGuard ?? new EgressGuard();
    this.#guard.check(this.#options.host);
  }

  get endpoint(): string {
    return `http://${this.#options.host}:${this.#options.port}${this.#options.path}`;
  }

  #credentials(): SoapCredentials | undefined {
    const { username, password } = this.#options;
    return username === undefined || password === undefined ? undefined : { username, password };
  }

  /** Issue one SOAP call and return the response body. */
  async call(action: string, body: string): Promise<string> {
    const envelope = buildEnvelope(body, this.#credentials());

    const response = await new Promise<{ status: number; body: string }>((resolve, reject) => {
      const req = httpRequest(
        {
          host: this.#options.host,
          port: this.#options.port,
          path: this.#options.path,
          method: 'POST',
          headers: {
            'Content-Type': `application/soap+xml; charset=utf-8; action="${action}"`,
            'Content-Length': Buffer.byteLength(envelope, 'utf8'),
          },
          timeout: this.#options.timeoutMillis,
        },
        (res) => {
          const chunks: Buffer[] = [];
          let total = 0;

          res.on('data', (chunk: Buffer) => {
            total += chunk.length;
            // Stop reading rather than buffering whatever a hostile device sends.
            if (total > 512 * 1024) {
              res.destroy();
              reject(new SoapError('The device sent an oversized response.', 'SOAP_RESPONSE_TOO_LARGE', false));
              return;
            }
            chunks.push(chunk);
          });

          res.on('end', () =>
            resolve({ status: res.statusCode ?? 0, body: Buffer.concat(chunks).toString('utf8') }),
          );
        },
      );

      req.on('timeout', () => {
        req.destroy();
        reject(
          new SoapError(
            `${this.endpoint} did not respond within ${this.#options.timeoutMillis} ms.`,
            'ONVIF_TIMEOUT',
          ),
        );
      });

      req.on('error', (error: NodeJS.ErrnoException) => {
        reject(
          new SoapError(`Could not reach ${this.endpoint}: ${error.code ?? error.message}`, 'ONVIF_UNREACHABLE'),
        );
      });

      req.write(envelope, 'utf8');
      req.end();
    });

    assertSafeXml(response.body);

    const fault = faultFrom(response.body);
    if (fault !== null) {
      // ONVIF signals a bad credential as a SOAP fault with HTTP 500, and the
      // fault text is the only place the reason appears. Mapping it to a specific
      // error is the difference between "check the password" and "it broke".
      const unauthorised = /not\s*authoriz|unauthoriz|authentication|sender not authorized/i.test(fault);
      throw new SoapError(
        unauthorised
          ? `The camera at ${this.#options.host} rejected the supplied credentials.`
          : `The camera at ${this.#options.host} reported: ${fault}`,
        unauthorised ? 'ONVIF_AUTH_FAILED' : 'ONVIF_FAULT',
        !unauthorised,
        fault,
      );
    }

    if (response.status !== 200) {
      throw new SoapError(
        `${this.endpoint} returned HTTP ${response.status}.`,
        'ONVIF_HTTP_ERROR',
      );
    }

    return response.body;
  }

  async getDeviceInformation(): Promise<DeviceInformation> {
    const xml = await this.call(
      `${ONVIF_NAMESPACES.device}/GetDeviceInformation`,
      `<tds:GetDeviceInformation xmlns:tds="${ONVIF_NAMESPACES.device}"/>`,
    );

    return {
      manufacturer: extractValue(xml, 'Manufacturer'),
      model: extractValue(xml, 'Model'),
      firmwareVersion: extractValue(xml, 'FirmwareVersion'),
      serialNumber: extractValue(xml, 'SerialNumber'),
      hardwareId: extractValue(xml, 'HardwareId'),
    };
  }

  /**
   * Media profiles.
   *
   * A camera typically exposes a main stream and a sub stream. The sub stream is
   * what this platform runs inference on - a 640x360 stream at 10 fps costs a
   * fraction of what 4K does and detects the same people - while the main stream
   * is recorded as evidence. Reading both configurations is what lets the
   * operator make that choice knowingly.
   */
  async getProfiles(): Promise<readonly MediaProfile[]> {
    const xml = await this.call(
      `${ONVIF_NAMESPACES.media}/GetProfiles`,
      `<trt:GetProfiles xmlns:trt="${ONVIF_NAMESPACES.media}"/>`,
    );

    const profiles: MediaProfile[] = [];

    for (const block of extractBlocks(xml, 'Profiles')) {
      const token = extractAttribute(block, 'Profiles', 'token');
      if (token === null) continue;

      const videoEncoder = extractBlocks(block, 'VideoEncoderConfiguration')[0] ?? '';
      const resolution = extractBlocks(videoEncoder, 'Resolution')[0] ?? '';
      const rateControl = extractBlocks(videoEncoder, 'RateControl')[0] ?? '';

      profiles.push({
        token,
        name: extractValue(block, 'Name'),
        encoding: extractValue(videoEncoder, 'Encoding'),
        width: numberOrNull(extractValue(resolution, 'Width')),
        height: numberOrNull(extractValue(resolution, 'Height')),
        frameRateLimit: numberOrNull(extractValue(rateControl, 'FrameRateLimit')),
        bitrateLimit: numberOrNull(extractValue(rateControl, 'BitrateLimit')),
        gopLength: numberOrNull(extractValue(videoEncoder, 'GovLength')),
      });
    }

    return profiles;
  }

  /**
   * Stream endpoint for a profile.
   *
   * The device returns a full RTSP URL, which is decomposed here and never
   * reassembled with a credential. A returned URL whose host differs from the
   * device's own is treated as suspect and the known-good host is used instead -
   * cameras behind NAT routinely report an address only they can see, and a
   * hostile device could report one belonging to somebody else entirely.
   */
  async getStreamUri(profileToken: string): Promise<StreamEndpoint> {
    const xml = await this.call(
      `${ONVIF_NAMESPACES.media}/GetStreamUri`,
      `<trt:GetStreamUri xmlns:trt="${ONVIF_NAMESPACES.media}">` +
        '<trt:StreamSetup>' +
        '<tt:Stream xmlns:tt="http://www.onvif.org/ver10/schema">RTP-Unicast</tt:Stream>' +
        '<tt:Transport xmlns:tt="http://www.onvif.org/ver10/schema"><tt:Protocol>RTSP</tt:Protocol></tt:Transport>' +
        '</trt:StreamSetup>' +
        `<trt:ProfileToken>${profileToken}</trt:ProfileToken>` +
        '</trt:GetStreamUri>',
    );

    const uri = extractValue(xml, 'Uri');
    if (uri === null) {
      throw new SoapError(
        `The camera returned no stream URI for profile "${profileToken}".`,
        'ONVIF_NO_STREAM_URI',
        false,
      );
    }

    return this.parseStreamUri(uri, profileToken);
  }

  /** Exposed for testing: URL decomposition is where credentials would leak. */
  parseStreamUri(uri: string, profileToken: string): StreamEndpoint {
    let parsed: URL;
    try {
      parsed = new URL(uri);
    } catch {
      throw new SoapError(
        `The camera returned a stream URI that is not a URL.`,
        'ONVIF_BAD_STREAM_URI',
        false,
      );
    }

    if (parsed.protocol !== 'rtsp:' && parsed.protocol !== 'rtsps:') {
      throw new SoapError(
        `The camera returned a "${parsed.protocol}" stream URI; only RTSP is supported.`,
        'ONVIF_BAD_STREAM_URI',
        false,
      );
    }

    // Some firmware embeds credentials in the URI it returns. They are discarded
    // here and never propagated: the RTSP client authenticates in a header, and a
    // credential that never enters a URL cannot be logged from one.
    const host = parsed.hostname === this.#options.host ? parsed.hostname : this.#options.host;
    const port = parsed.port === '' ? 554 : Number.parseInt(parsed.port, 10);

    return {
      host,
      port: Number.isNaN(port) ? 554 : port,
      path: `${parsed.pathname}${parsed.search}`,
      profileToken,
    };
  }

  /** Service capabilities, used to decide whether PTZ and events are available. */
  async getCapabilities(): Promise<{
    readonly media: boolean;
    readonly ptz: boolean;
    readonly events: boolean;
    readonly imaging: boolean;
  }> {
    const xml = await this.call(
      `${ONVIF_NAMESPACES.device}/GetCapabilities`,
      `<tds:GetCapabilities xmlns:tds="${ONVIF_NAMESPACES.device}"><tds:Category>All</tds:Category></tds:GetCapabilities>`,
    );

    return {
      media: extractBlocks(xml, 'Media').length > 0,
      ptz: extractBlocks(xml, 'PTZ').length > 0,
      events: extractBlocks(xml, 'Events').length > 0,
      imaging: extractBlocks(xml, 'Imaging').length > 0,
    };
  }
}

const numberOrNull = (value: string | null): number | null => {
  if (value === null) return null;
  const parsed = Number.parseFloat(value);
  return Number.isNaN(parsed) ? null : parsed;
};

/**
 * Choose the profile to run inference on.
 *
 * Prefers the smallest stream that is still large enough to detect a person at
 * range - roughly 640x360 - because inference cost scales with pixels while
 * detection quality plateaus well below 4K. Running a detector on a main stream
 * is the single most common way a deployment ends up needing four times the
 * hardware it actually requires.
 */
export const selectInferenceProfile = (
  profiles: readonly MediaProfile[],
): MediaProfile | null => {
  if (profiles.length === 0) return null;

  const usable = profiles.filter(
    (p) => p.width !== null && p.height !== null && p.width >= 480 && p.height >= 270,
  );
  if (usable.length === 0) return profiles[0] ?? null;

  return usable.reduce((best, candidate) =>
    (candidate.width ?? 0) * (candidate.height ?? 0) < (best.width ?? 0) * (best.height ?? 0)
      ? candidate
      : best,
  );
};

/** Choose the profile to record. The highest resolution available is evidence. */
export const selectRecordingProfile = (
  profiles: readonly MediaProfile[],
): MediaProfile | null => {
  if (profiles.length === 0) return null;

  return profiles.reduce((best, candidate) =>
    (candidate.width ?? 0) * (candidate.height ?? 0) > (best.width ?? 0) * (best.height ?? 0)
      ? candidate
      : best,
  );
};
