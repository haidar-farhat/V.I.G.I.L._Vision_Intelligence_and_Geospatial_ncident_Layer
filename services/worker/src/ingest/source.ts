import type { CameraHealth, CameraId, CameraStatus, UtcMillis } from '@sentinel/shared-types';
import { utcMillis } from '@sentinel/shared-types';
import type { Secret } from '@sentinel/security';
import { RtspClient, RtspError } from '../rtsp/client.ts';
import type { RtspSessionInfo } from '../rtsp/client.ts';
import { AuthenticationError, UnsupportedAuthError } from '../rtsp/auth.ts';
import { Backoff, delay } from './backoff.ts';

/**
 * Supervised video source.
 *
 * A camera that has been up for six months will still drop: a PoE switch reboots,
 * a firmware update restarts the device, a cable gets nudged. The system's job is
 * to notice, reconnect without hammering the network, and be honest about what
 * happened - a camera that silently stopped recording an hour ago is worse than
 * one that is visibly offline, because nobody goes looking for the footage until
 * they need it.
 *
 * Three behaviours matter:
 *
 * **Bounded reconnection.** Exponential backoff with jitter, forever. Never a
 * tight loop, and never giving up - a camera may come back in an hour and a
 * worker that stopped trying needs a human to notice.
 *
 * **Distinguishing what will and will not fix itself.** A wrong password is not
 * a transient fault. Retrying it every thirty seconds for a week locks the
 * account and fills the device log; the source stops and says why.
 *
 * **Honest state.** DEGRADED means "connected but not delivering what it
 * promised". OFFLINE means "not connected". A camera that reconnects every ten
 * seconds is DEGRADED, not ONLINE, even though at any given instant it is
 * connected.
 */

export type FrameSink = (frame: {
  readonly cameraId: CameraId;
  readonly capturedAt: UtcMillis;
  readonly sequence: number;
}) => void;

/**
 * Decoder boundary.
 *
 * Implemented by an FFmpeg subprocess in production. Injected rather than
 * constructed here so the supervision logic can be tested without a decoder, and
 * so a hardware decoder can replace the software one without touching this file.
 */
export type Decoder = {
  /** Begin decoding. Resolves when the first frame has been produced. */
  start(session: RtspSessionInfo, sink: FrameSink): Promise<void>;
  stop(): Promise<void>;
  readonly framesDecoded: number;
  readonly decodeErrors: number;
};

export type VideoSourceOptions = {
  readonly cameraId: CameraId;
  readonly host: string;
  readonly port?: number;
  readonly path: string;
  readonly username?: string;
  readonly password?: Secret<string>;
  readonly allowBasicAuth?: boolean;
  readonly timeoutMillis?: number;
  /** Absent means the session is established but no frames are decoded. */
  readonly decoder?: Decoder;
  readonly onFrame?: FrameSink;
  /** Called on every state change, so health reaches the UI without polling. */
  readonly onStatusChange?: (status: CameraStatus, detail: string) => void;
  readonly initialBackoffMillis?: number;
  readonly maxBackoffMillis?: number;
  /**
   * Reconnects within this window before the camera is called DEGRADED rather
   * than ONLINE. A flapping camera is not a healthy one.
   */
  readonly flappingWindowMillis?: number;
  readonly flappingThreshold?: number;
  readonly now?: () => UtcMillis;
};

export type VideoSourceState = {
  readonly status: CameraStatus;
  readonly detail: string;
  readonly reconnectCount: number;
  readonly lastConnectedAt: UtcMillis | null;
  readonly lastErrorCode: string | null;
  readonly running: boolean;
};

export class VideoSource {
  readonly cameraId: CameraId;
  readonly #options: VideoSourceOptions;
  readonly #now: () => UtcMillis;

  #client: RtspClient | null = null;
  #session: RtspSessionInfo | null = null;
  #controller: AbortController | null = null;
  #keepAliveTimer: NodeJS.Timeout | null = null;
  #loop: Promise<void> | null = null;

  #status: CameraStatus = 'UNKNOWN';
  #detail = 'Not started.';
  #reconnectCount = 0;
  #lastConnectedAt: UtcMillis | null = null;
  #lastErrorCode: string | null = null;
  /** Timestamps of recent reconnects, for flapping detection. */
  #recentReconnects: number[] = [];

  constructor(options: VideoSourceOptions) {
    this.cameraId = options.cameraId;
    this.#options = options;
    this.#now = options.now ?? (() => utcMillis(Date.now()));
  }

  get state(): VideoSourceState {
    return {
      status: this.#status,
      detail: this.#detail,
      reconnectCount: this.#reconnectCount,
      lastConnectedAt: this.#lastConnectedAt,
      lastErrorCode: this.#lastErrorCode,
      running: this.#controller !== null,
    };
  }

  #setStatus(status: CameraStatus, detail: string): void {
    if (this.#status === status && this.#detail === detail) return;
    this.#status = status;
    this.#detail = detail;
    this.#options.onStatusChange?.(status, detail);
  }

  /**
   * Start, and keep running until stopped.
   *
   * Returns once the first connection attempt has been made, so a caller starting
   * twenty cameras does not serialise on the slowest. The supervision loop
   * continues in the background.
   */
  async start(): Promise<void> {
    if (this.#controller !== null) return;

    const controller = new AbortController();
    this.#controller = controller;

    let firstAttemptDone: () => void = () => {};
    const firstAttempt = new Promise<void>((resolve) => {
      firstAttemptDone = resolve;
    });

    this.#loop = this.#supervise(controller.signal, firstAttemptDone);
    await firstAttempt;
  }

  async #supervise(signal: AbortSignal, firstAttemptDone: () => void): Promise<void> {
    const backoff = new Backoff({
      ...(this.#options.initialBackoffMillis === undefined
        ? {}
        : { initialMillis: this.#options.initialBackoffMillis }),
      ...(this.#options.maxBackoffMillis === undefined
        ? {}
        : { maxMillis: this.#options.maxBackoffMillis }),
      jitter: 'full',
    });

    let announcedFirstAttempt = false;
    const announce = (): void => {
      if (announcedFirstAttempt) return;
      announcedFirstAttempt = true;
      firstAttemptDone();
    };

    while (!signal.aborted) {
      try {
        await this.#connectOnce();
        backoff.reset();
        announce();

        // Hold the connection open until it drops or we are told to stop.
        await this.#awaitDisconnect(signal);
      } catch (error) {
        announce();

        if (signal.aborted) break;

        // A credential problem will not resolve itself. Retrying it every thirty
        // seconds for a week locks the account and fills the device log with
        // failures somebody will later have to explain.
        if (error instanceof AuthenticationError || error instanceof UnsupportedAuthError) {
          this.#lastErrorCode = 'AUTH';
          this.#setStatus(
            'OFFLINE',
            `${error.message} Not retrying: this needs a configuration change.`,
          );
          break;
        }

        if (error instanceof RtspError && error.recoverable === false) {
          this.#lastErrorCode = error.code;
          this.#setStatus('OFFLINE', `${error.message} Not retrying: this needs a change on the camera.`);
          break;
        }

        this.#lastErrorCode = error instanceof RtspError ? error.code : 'UNKNOWN';

        const wait = backoff.nextDelay() ?? 0;
        this.#setStatus(
          'OFFLINE',
          `${error instanceof Error ? error.message : String(error)} Retrying in ${Math.round(wait / 1000)}s.`,
        );

        try {
          await delay(wait, signal);
        } catch {
          break; // Aborted during backoff.
        }
      }
    }

    this.#teardown();
  }

  async #connectOnce(): Promise<void> {
    this.#teardown();

    const client = new RtspClient({
      host: this.#options.host,
      path: this.#options.path,
      ...(this.#options.port === undefined ? {} : { port: this.#options.port }),
      ...(this.#options.username === undefined ? {} : { username: this.#options.username }),
      ...(this.#options.password === undefined ? {} : { password: this.#options.password }),
      ...(this.#options.timeoutMillis === undefined
        ? {}
        : { timeoutMillis: this.#options.timeoutMillis }),
      ...(this.#options.allowBasicAuth === undefined
        ? {}
        : { allowBasicAuth: this.#options.allowBasicAuth }),
    });

    const session = await client.open();

    this.#client = client;
    this.#session = session;
    this.#lastConnectedAt = this.#now();
    this.#lastErrorCode = null;

    if (this.#options.decoder !== undefined && this.#options.onFrame !== undefined) {
      await this.#options.decoder.start(session, this.#options.onFrame);
    }

    this.#startKeepAlive(session);
    this.#recordConnection();
  }

  /**
   * Record a connection and decide whether the camera is healthy or flapping.
   *
   * A camera reconnecting every ten seconds is connected at any given instant but
   * is not delivering usable video, and calling it ONLINE would hide a real fault
   * behind a green dot.
   */
  #recordConnection(): void {
    const now = this.#now();
    const windowMillis = this.#options.flappingWindowMillis ?? 120_000;
    const threshold = this.#options.flappingThreshold ?? 3;

    this.#recentReconnects.push(now);
    this.#recentReconnects = this.#recentReconnects.filter((at) => now - at <= windowMillis);

    if (this.#recentReconnects.length > threshold) {
      this.#setStatus(
        'DEGRADED',
        `Connected, but has reconnected ${this.#recentReconnects.length} times in the last ` +
          `${Math.round(windowMillis / 1000)}s. Check the link or the camera's stream limit.`,
      );
      return;
    }

    this.#setStatus('ONLINE', `Streaming from ${this.#client?.url ?? this.#options.host}.`);
  }

  /**
   * Keep-alive on the camera's own schedule.
   *
   * Sent at half the advertised session timeout, because a single dropped
   * keep-alive on a lossy link should not end the session. A camera that stops
   * answering is treated as disconnected, which is the whole point of sending it.
   */
  #startKeepAlive(session: RtspSessionInfo): void {
    const intervalMillis = Math.max(5000, (session.timeoutSeconds * 1000) / 2);

    this.#keepAliveTimer = setInterval(() => {
      const client = this.#client;
      if (client === null) return;

      client.keepAlive(session.supportsGetParameter).catch(() => {
        // The camera stopped answering. Dropping the client makes the supervision
        // loop treat this as a disconnect and reconnect on the normal schedule.
        this.#reconnectCount += 1;
        this.#client = null;
        client.close();
      });
    }, intervalMillis);

    this.#keepAliveTimer.unref?.();
  }

  /** Resolve when the connection drops, or when told to stop. */
  async #awaitDisconnect(signal: AbortSignal): Promise<void> {
    await new Promise<void>((resolve) => {
      const check = setInterval(() => {
        if (signal.aborted || this.#client === null) {
          clearInterval(check);
          resolve();
        }
      }, 250);
      check.unref?.();

      signal.addEventListener(
        'abort',
        () => {
          clearInterval(check);
          resolve();
        },
        { once: true },
      );
    });
  }

  #teardown(): void {
    if (this.#keepAliveTimer !== null) {
      clearInterval(this.#keepAliveTimer);
      this.#keepAliveTimer = null;
    }
    this.#client?.close();
    this.#client = null;
    this.#session = null;
  }

  async stop(): Promise<void> {
    const controller = this.#controller;
    this.#controller = null;
    controller?.abort();

    await this.#options.decoder?.stop();
    this.#teardown();

    if (this.#loop !== null) {
      await this.#loop.catch(() => {
        // The loop unwinding on abort is the expected path, not a failure.
      });
      this.#loop = null;
    }

    this.#setStatus('UNKNOWN', 'Stopped.');
  }

  /** Health snapshot for the camera detail page and node heartbeat. */
  health(): CameraHealth {
    const decoder = this.#options.decoder;

    return {
      cameraId: this.cameraId,
      status: this.#status,
      observedAt: this.#now(),
      fps: 0,
      targetFps: 0,
      inferenceFps: 0,
      droppedFrames: 0,
      decodeErrors: decoder?.decodeErrors ?? 0,
      reconnectCount: this.#reconnectCount,
      bitrateKbps: 0,
      latencyMs: 0,
      pingMs: null,
      ...(this.#lastErrorCode === null ? {} : { lastErrorCode: this.#lastErrorCode }),
    };
  }

  get session(): RtspSessionInfo | null {
    return this.#session;
  }
}
