/**
 * Bounded exponential backoff with jitter.
 *
 * The failure this prevents: a camera goes offline, the worker retries in a tight
 * loop, and one dead camera becomes a broadcast storm and a pegged core. Worse,
 * when a *switch* fails and forty cameras drop at once, every worker retries in
 * lockstep and the network is unusable for the whole period the recovery would
 * otherwise take.
 *
 * Jitter is what fixes the second problem, and it is the part most
 * implementations omit. Without it, synchronised clients stay synchronised
 * forever: they all fail together, all wait the same interval, and all retry
 * together, indefinitely. Full jitter - a uniform draw over the whole interval,
 * not a small perturbation of it - de-correlates them after a single round.
 */

export type BackoffOptions = {
  /** First delay, in milliseconds. */
  readonly initialMillis?: number;
  /** Ceiling. Never wait longer than this, however many attempts have failed. */
  readonly maxMillis?: number;
  readonly multiplier?: number;
  /**
   * Jitter strategy. `full` draws uniformly over [0, interval]; `equal` draws over
   * [interval/2, interval], keeping a floor while still de-correlating.
   */
  readonly jitter?: 'full' | 'equal' | 'none';
  /**
   * Give up after this many attempts. Undefined means never give up, which is the
   * right default for a camera: it may come back in an hour, and a worker that
   * stopped trying would need a human to notice.
   */
  readonly maxAttempts?: number;
  /** Injectable for tests. Defaults to Math.random. */
  readonly random?: () => number;
};

const DEFAULTS = {
  initialMillis: 500,
  maxMillis: 60_000,
  multiplier: 2,
  jitter: 'full',
} as const;

export class Backoff {
  readonly #initial: number;
  readonly #max: number;
  readonly #multiplier: number;
  readonly #jitter: 'full' | 'equal' | 'none';
  readonly #maxAttempts: number | undefined;
  readonly #random: () => number;

  #attempts = 0;

  constructor(options: BackoffOptions = {}) {
    this.#initial = options.initialMillis ?? DEFAULTS.initialMillis;
    this.#max = options.maxMillis ?? DEFAULTS.maxMillis;
    this.#multiplier = options.multiplier ?? DEFAULTS.multiplier;
    this.#jitter = options.jitter ?? DEFAULTS.jitter;
    this.#maxAttempts = options.maxAttempts;
    this.#random = options.random ?? Math.random;
  }

  get attempts(): number {
    return this.#attempts;
  }

  get exhausted(): boolean {
    return this.#maxAttempts !== undefined && this.#attempts >= this.#maxAttempts;
  }

  /** The un-jittered interval for the current attempt count. */
  get currentInterval(): number {
    const raw = this.#initial * this.#multiplier ** this.#attempts;
    return Math.min(this.#max, raw);
  }

  /**
   * Record a failure and return how long to wait, or null when exhausted.
   */
  nextDelay(): number | null {
    if (this.exhausted) return null;

    const interval = this.currentInterval;
    this.#attempts += 1;

    switch (this.#jitter) {
      case 'none':
        return Math.round(interval);
      case 'equal':
        return Math.round(interval / 2 + this.#random() * (interval / 2));
      case 'full':
        return Math.round(this.#random() * interval);
    }
  }

  /** Call after a success. The next failure starts from the initial interval. */
  reset(): void {
    this.#attempts = 0;
  }
}

/**
 * Sleep that can be cancelled by an AbortSignal.
 *
 * A worker being shut down must not sit in a 60-second backoff before noticing.
 * Rejecting on abort lets the supervisor's loop unwind immediately.
 */
export const delay = (millis: number, signal?: AbortSignal): Promise<void> =>
  new Promise((resolve, reject) => {
    if (signal?.aborted === true) {
      reject(new Error('aborted'));
      return;
    }

    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort);
      resolve();
    }, millis);

    // Do not hold the process open purely for a retry timer.
    if (typeof timer.unref === 'function') timer.unref();

    function onAbort(): void {
      clearTimeout(timer);
      reject(new Error('aborted'));
    }

    signal?.addEventListener('abort', onAbort, { once: true });
  });

/**
 * Retry an operation with backoff until it succeeds, is aborted, or exhausts.
 *
 * `onRetry` exists so the caller can record the attempt in camera health. A
 * reconnect that nobody can see is indistinguishable from a camera that never
 * dropped, and the difference matters when diagnosing a flaky link.
 */
export const retry = async <T>(
  operation: () => Promise<T>,
  options: BackoffOptions & {
    readonly signal?: AbortSignal;
    readonly onRetry?: (attempt: number, delayMillis: number, error: unknown) => void;
  } = {},
): Promise<T> => {
  const backoff = new Backoff(options);

  for (;;) {
    try {
      return await operation();
    } catch (error) {
      if (options.signal?.aborted === true) throw error;

      const wait = backoff.nextDelay();
      if (wait === null) throw error;

      options.onRetry?.(backoff.attempts, wait, error);
      await delay(wait, options.signal);
    }
  }
};
