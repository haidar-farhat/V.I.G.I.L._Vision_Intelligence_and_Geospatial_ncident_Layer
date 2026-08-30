import { createConnection } from 'node:net';
import type { Socket } from 'node:net';
import type { Secret } from '@sentinel/security';
import { describeStream } from '@sentinel/security';
import type { AuthChallenge } from './auth.ts';
import {
  AuthenticationError,
  DigestSession,
  UnsupportedAuthError,
  buildBasicHeader,
  buildDigestHeader,
  parseAuthChallenges,
} from './auth.ts';
import type { SdpSession } from './sdp.ts';
import { parseSdp, resolveControlUrl, selectVideoTrack } from './sdp.ts';

/**
 * RTSP 1.0 (RFC 2326) control client.
 *
 * Implements the control plane only - OPTIONS, DESCRIBE, SETUP, PLAY, TEARDOWN
 * and keep-alive - which is where essentially all of the protocol complexity,
 * the interoperability pain and the security surface live. Media transport and
 * decoding happen downstream (see ingest/decoder.ts).
 *
 * Two things drive the design:
 *
 * **Everything on the wire is untrusted.** Response sizes are capped, header
 * counts are capped, and the parser cannot be made to allocate unboundedly by a
 * device that sends a Content-Length it never satisfies.
 *
 * **Credentials never appear in a URL.** The request target is always the
 * credential-free stream URL; authentication happens in a header. That is what
 * makes it safe for every error, log line and diagnostic in this file to include
 * the URL verbatim.
 */

const MAX_RESPONSE_BYTES = 256 * 1024;
const MAX_HEADERS = 100;
const DEFAULT_TIMEOUT_MILLIS = 10_000;

export type RtspTransport = 'TCP_INTERLEAVED' | 'UDP';

export type RtspOptions = {
  readonly host: string;
  readonly port?: number;
  /** Stream path. Never a full URL with credentials. */
  readonly path: string;
  readonly username?: string;
  readonly password?: Secret<string>;
  readonly timeoutMillis?: number;
  /**
   * Interleaved TCP is the default because it traverses the firewalls and NAT
   * that a security LAN invariably has, and because a dropped UDP stream is
   * indistinguishable from a dead camera until the RTCP timeout expires.
   */
  readonly transport?: RtspTransport;
  /**
   * Permit Basic authentication. Off by default: Basic transmits the password
   * reversibly, and enabling it should be a decision someone made on purpose.
   */
  readonly allowBasicAuth?: boolean;
  readonly userAgent?: string;
};

export type RtspResponse = {
  readonly statusCode: number;
  readonly statusText: string;
  readonly headers: Readonly<Record<string, string>>;
  readonly body: string;
};

export type RtspSessionInfo = {
  readonly sessionId: string;
  readonly timeoutSeconds: number;
  readonly transport: string;
  readonly sdp: SdpSession;
  readonly videoTrackUrl: string;
  readonly encoding: string | null;
  /**
   * Whether the camera advertised GET_PARAMETER. It is the preferred keep-alive
   * because, unlike OPTIONS, it is session-scoped - a camera that has silently
   * dropped the session answers OPTIONS happily and the client never notices.
   */
  readonly supportsGetParameter: boolean;
  readonly methods: readonly string[];
};

export class RtspError extends Error {
  readonly code: string;
  readonly statusCode: number | null;
  readonly recoverable: boolean;

  constructor(message: string, code: string, statusCode: number | null, recoverable = true) {
    super(message);
    this.name = 'RtspError';
    this.code = code;
    this.statusCode = statusCode;
    this.recoverable = recoverable;
  }
}

/**
 * Parse an RTSP response.
 *
 * Returns null when more bytes are needed, so the caller can keep reading. This
 * is separated from the socket so it can be tested directly against the
 * malformed input real cameras produce.
 */
export const parseResponse = (
  raw: string,
): { response: RtspResponse; consumed: number } | null => {
  const headerEnd = raw.indexOf('\r\n\r\n');
  if (headerEnd === -1) return null;

  const headerBlock = raw.slice(0, headerEnd);
  const lines = headerBlock.split('\r\n');

  const statusLine = lines[0] ?? '';
  const statusMatch = /^RTSP\/\d\.\d\s+(\d{3})\s*(.*)$/.exec(statusLine);
  if (statusMatch === null) {
    throw new RtspError(
      `The camera sent a response that is not RTSP: "${statusLine.slice(0, 60)}"`,
      'RTSP_MALFORMED',
      null,
      false,
    );
  }

  const headers: Record<string, string> = {};
  for (const line of lines.slice(1, 1 + MAX_HEADERS)) {
    const colon = line.indexOf(':');
    if (colon === -1) continue;
    // Header names are case-insensitive, and cameras are gloriously inconsistent
    // about them. Normalising here means the rest of the file need not care.
    headers[line.slice(0, colon).trim().toLowerCase()] = line.slice(colon + 1).trim();
  }

  const contentLength = Number.parseInt(headers['content-length'] ?? '0', 10);
  const bodyLength = Number.isNaN(contentLength) ? 0 : Math.max(0, contentLength);

  if (bodyLength > MAX_RESPONSE_BYTES) {
    throw new RtspError(
      `The camera declared a ${bodyLength} byte body, beyond the ${MAX_RESPONSE_BYTES} byte limit.`,
      'RTSP_BODY_TOO_LARGE',
      null,
      false,
    );
  }

  const bodyStart = headerEnd + 4;
  if (raw.length < bodyStart + bodyLength) return null;

  return {
    response: {
      statusCode: Number.parseInt(statusMatch[1] ?? '0', 10),
      statusText: statusMatch[2] ?? '',
      headers,
      body: raw.slice(bodyStart, bodyStart + bodyLength),
    },
    consumed: bodyStart + bodyLength,
  };
};

export class RtspClient {
  readonly #options: Required<Omit<RtspOptions, 'username' | 'password'>> &
    Pick<RtspOptions, 'username' | 'password'>;
  readonly #url: string;

  #socket: Socket | null = null;
  #buffer = '';
  #sequence = 0;
  #sessionId: string | null = null;
  #digest: DigestSession | null = null;
  #challenge: AuthChallenge | null = null;
  #pending: {
    resolve: (response: RtspResponse) => void;
    reject: (error: Error) => void;
    timer: NodeJS.Timeout;
  } | null = null;

  constructor(options: RtspOptions) {
    this.#options = {
      host: options.host,
      port: options.port ?? 554,
      path: options.path.startsWith('/') ? options.path : `/${options.path}`,
      timeoutMillis: options.timeoutMillis ?? DEFAULT_TIMEOUT_MILLIS,
      transport: options.transport ?? 'TCP_INTERLEAVED',
      allowBasicAuth: options.allowBasicAuth ?? false,
      userAgent: options.userAgent ?? 'SentinelVision/0.1',
      ...(options.username === undefined ? {} : { username: options.username }),
      ...(options.password === undefined ? {} : { password: options.password }),
    };

    // Credential-free by construction. Safe to log, safe to show an operator.
    this.#url = describeStream(this.#options.host, this.#options.port, this.#options.path);
  }

  /** The stream URL, without credentials. */
  get url(): string {
    return this.#url;
  }

  get sessionId(): string | null {
    return this.#sessionId;
  }

  async connect(): Promise<void> {
    if (this.#socket !== null) return;

    await new Promise<void>((resolve, reject) => {
      const socket = createConnection({
        host: this.#options.host,
        port: this.#options.port,
        // No Nagle: RTSP is a request/response protocol where a 40 ms delay per
        // exchange turns a six-step handshake into a quarter of a second.
        noDelay: true,
      });

      const timer = setTimeout(() => {
        socket.destroy();
        reject(
          new RtspError(
            `No response from ${this.#url} within ${this.#options.timeoutMillis} ms.`,
            'RTSP_CONNECT_TIMEOUT',
            null,
          ),
        );
      }, this.#options.timeoutMillis);

      socket.once('connect', () => {
        clearTimeout(timer);
        this.#socket = socket;
        socket.setEncoding('utf8');
        socket.on('data', (chunk: string) => this.#onData(chunk));
        socket.on('error', (error) => this.#onError(error));
        socket.on('close', () => this.#onClose());
        resolve();
      });

      socket.once('error', (error: NodeJS.ErrnoException) => {
        clearTimeout(timer);
        reject(
          new RtspError(
            `Could not reach ${this.#url}: ${error.code ?? error.message}`,
            'RTSP_CONNECT_FAILED',
            null,
          ),
        );
      });
    });
  }

  #onData(chunk: string): void {
    this.#buffer += chunk;

    if (this.#buffer.length > MAX_RESPONSE_BYTES) {
      const error = new RtspError(
        `The camera sent more than ${MAX_RESPONSE_BYTES} bytes without a complete response.`,
        'RTSP_RESPONSE_TOO_LARGE',
        null,
        false,
      );
      this.#buffer = '';
      this.#settle(null, error);
      this.close();
      return;
    }

    for (;;) {
      let parsed: { response: RtspResponse; consumed: number } | null;
      try {
        parsed = parseResponse(this.#buffer);
      } catch (error) {
        this.#buffer = '';
        this.#settle(null, error instanceof Error ? error : new Error(String(error)));
        this.close();
        return;
      }

      if (parsed === null) return;

      this.#buffer = this.#buffer.slice(parsed.consumed);
      this.#settle(parsed.response, null);
    }
  }

  #onError(error: Error): void {
    this.#settle(null, new RtspError(`Connection to ${this.#url} failed: ${error.message}`, 'RTSP_SOCKET_ERROR', null));
  }

  #onClose(): void {
    this.#socket = null;
    this.#settle(
      null,
      new RtspError(`The camera closed the connection to ${this.#url}.`, 'RTSP_CLOSED', null),
    );
  }

  #settle(response: RtspResponse | null, error: Error | null): void {
    const pending = this.#pending;
    if (pending === null) return;
    this.#pending = null;
    clearTimeout(pending.timer);

    if (error !== null) pending.reject(error);
    else if (response !== null) pending.resolve(response);
  }

  /**
   * Send one request and await its response.
   *
   * On a 401 the request is retried once with an Authorization header built from
   * the challenge. Exactly once: retrying repeatedly against a camera that is
   * rejecting valid credentials is how an account gets locked out, and how a
   * misconfigured system generates thousands of failed-auth entries in a device
   * log that somebody later has to explain.
   */
  async request(
    method: string,
    url: string = this.#url,
    extraHeaders: Readonly<Record<string, string>> = {},
    allowRetry = true,
  ): Promise<RtspResponse> {
    if (this.#socket === null) {
      throw new RtspError('Not connected.', 'RTSP_NOT_CONNECTED', null);
    }
    if (this.#pending !== null) {
      throw new RtspError(
        'An RTSP request is already in flight; this client is single-flight by design.',
        'RTSP_BUSY',
        null,
        false,
      );
    }

    this.#sequence += 1;

    const headers: Record<string, string> = {
      CSeq: String(this.#sequence),
      'User-Agent': this.#options.userAgent,
      ...extraHeaders,
    };

    if (this.#sessionId !== null) headers['Session'] = this.#sessionId;

    const authorization = this.#authorizationFor(method, url);
    if (authorization !== null) headers['Authorization'] = authorization;

    const request =
      `${method} ${url} RTSP/1.0\r\n` +
      Object.entries(headers)
        .map(([key, value]) => `${key}: ${value}`)
        .join('\r\n') +
      '\r\n\r\n';

    const response = await new Promise<RtspResponse>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.#pending = null;
        reject(
          new RtspError(
            `${method} to ${this.#url} timed out after ${this.#options.timeoutMillis} ms.`,
            'RTSP_TIMEOUT',
            null,
          ),
        );
      }, this.#options.timeoutMillis);

      this.#pending = { resolve, reject, timer };
      this.#socket?.write(request, 'utf8');
    });

    if (response.statusCode === 401 && allowRetry) {
      this.#acceptChallenge(response);
      return this.request(method, url, extraHeaders, false);
    }

    if (response.statusCode === 401) {
      throw new AuthenticationError(
        `The camera at ${this.#url} rejected the supplied credentials.`,
      );
    }

    return response;
  }

  #acceptChallenge(response: RtspResponse): void {
    const header = response.headers['www-authenticate'];
    if (header === undefined) {
      throw new AuthenticationError(
        `The camera at ${this.#url} requires authentication but offered no challenge.`,
      );
    }

    const challenges = parseAuthChallenges(header);
    const digest = challenges.find((c) => c.scheme === 'Digest');
    const basic = challenges.find((c) => c.scheme === 'Basic');

    if (digest !== undefined) {
      this.#challenge = digest;
      // A fresh session per challenge: the nonce count is scoped to the nonce,
      // and reusing a count across nonces is a protocol violation some cameras
      // reject outright.
      this.#digest = new DigestSession();
      return;
    }

    if (basic !== undefined && this.#options.allowBasicAuth) {
      this.#challenge = basic;
      return;
    }

    throw new UnsupportedAuthError(challenges.map((c) => c.scheme));
  }

  #authorizationFor(method: string, url: string): string | null {
    const challenge = this.#challenge;
    const username = this.#options.username;
    const password = this.#options.password;

    if (challenge === null || username === undefined || password === undefined) return null;

    if (challenge.scheme === 'Digest') {
      const session = this.#digest ?? new DigestSession();
      this.#digest = session;
      return buildDigestHeader(challenge, session, method, url, username, password);
    }

    return buildBasicHeader(username, password);
  }

  // ------------------------------------------------------------- handshake

  async options(): Promise<readonly string[]> {
    const response = await this.request('OPTIONS');
    if (response.statusCode !== 200) {
      throw new RtspError(
        `OPTIONS failed: ${response.statusCode} ${response.statusText}`,
        'RTSP_OPTIONS_FAILED',
        response.statusCode,
      );
    }
    return (response.headers['public'] ?? '')
      .split(',')
      .map((m) => m.trim().toUpperCase())
      .filter((m) => m !== '');
  }

  async describe(): Promise<SdpSession> {
    const response = await this.request('DESCRIBE', this.#url, { Accept: 'application/sdp' });

    if (response.statusCode !== 200) {
      throw new RtspError(
        `DESCRIBE failed for ${this.#url}: ${response.statusCode} ${response.statusText}`,
        'RTSP_DESCRIBE_FAILED',
        response.statusCode,
      );
    }
    return parseSdp(response.body);
  }

  async setup(trackUrl: string): Promise<{ sessionId: string; timeoutSeconds: number; transport: string }> {
    const transportHeader =
      this.#options.transport === 'TCP_INTERLEAVED'
        ? 'RTP/AVP/TCP;unicast;interleaved=0-1'
        : 'RTP/AVP;unicast;client_port=0-1';

    const response = await this.request('SETUP', trackUrl, { Transport: transportHeader });

    if (response.statusCode !== 200) {
      throw new RtspError(
        `SETUP failed for ${trackUrl}: ${response.statusCode} ${response.statusText}`,
        'RTSP_SETUP_FAILED',
        response.statusCode,
      );
    }

    const sessionHeader = response.headers['session'] ?? '';
    const sessionId = sessionHeader.split(';')[0]?.trim() ?? '';
    if (sessionId === '') {
      throw new RtspError(
        `The camera accepted SETUP but returned no session identifier.`,
        'RTSP_NO_SESSION',
        response.statusCode,
      );
    }

    // The session timeout governs how often a keep-alive is required. Cameras
    // that omit it conventionally mean 60 seconds.
    const timeoutMatch = /timeout=(\d+)/i.exec(sessionHeader);
    const timeoutSeconds = timeoutMatch?.[1] === undefined ? 60 : Number.parseInt(timeoutMatch[1], 10);

    this.#sessionId = sessionId;
    return {
      sessionId,
      timeoutSeconds: Number.isNaN(timeoutSeconds) ? 60 : timeoutSeconds,
      transport: response.headers['transport'] ?? transportHeader,
    };
  }

  async play(): Promise<void> {
    const response = await this.request('PLAY', this.#url, { Range: 'npt=0.000-' });
    if (response.statusCode !== 200) {
      throw new RtspError(
        `PLAY failed for ${this.#url}: ${response.statusCode} ${response.statusText}`,
        'RTSP_PLAY_FAILED',
        response.statusCode,
      );
    }
  }

  /** Keep-alive. GET_PARAMETER when supported, OPTIONS otherwise. */
  async keepAlive(supportsGetParameter: boolean): Promise<void> {
    if (supportsGetParameter) {
      await this.request('GET_PARAMETER');
      return;
    }
    await this.request('OPTIONS');
  }

  async teardown(): Promise<void> {
    if (this.#sessionId === null || this.#socket === null) return;
    try {
      await this.request('TEARDOWN');
    } catch {
      // A camera that has already dropped the session will refuse or ignore
      // TEARDOWN. The socket is being closed regardless, so this is not worth
      // surfacing as an error.
    } finally {
      this.#sessionId = null;
    }
  }

  close(): void {
    const socket = this.#socket;
    this.#socket = null;
    this.#sessionId = null;
    this.#buffer = '';
    socket?.destroy();
  }

  /**
   * The full handshake, in the order every RTSP device expects it.
   *
   * OPTIONS is not strictly required before DESCRIBE, but it is how the client
   * learns whether GET_PARAMETER is available for keep-alive, and it surfaces an
   * unreachable or non-RTSP endpoint before credentials are ever sent.
   */
  async open(): Promise<RtspSessionInfo> {
    await this.connect();

    const methods = await this.options();
    const sdp = await this.describe();

    const track = selectVideoTrack(sdp);
    if (track === null) {
      throw new RtspError(
        `The camera at ${this.#url} offers no video track this system can decode. ` +
          `It described: ${sdp.media.map((m) => `${m.kind}/${m.encoding ?? 'unknown'}`).join(', ') || 'nothing'}.`,
        'RTSP_NO_VIDEO_TRACK',
        null,
        false,
      );
    }

    const base = resolveControlUrl(this.#url, sdp.control);
    const trackUrl = resolveControlUrl(base, track.control);

    const session = await this.setup(trackUrl);
    await this.play();

    return {
      sessionId: session.sessionId,
      timeoutSeconds: session.timeoutSeconds,
      transport: session.transport,
      sdp,
      videoTrackUrl: trackUrl,
      encoding: track.encoding,
      supportsGetParameter: methods.includes('GET_PARAMETER'),
      methods,
    };
  }
}
