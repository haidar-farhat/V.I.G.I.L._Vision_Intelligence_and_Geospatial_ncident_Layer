/**
 * Seeded pseudo-random number generation.
 *
 * Every source of randomness in the simulator flows through here. A scenario run
 * with the same seed produces byte-identical detections, tracks, events and
 * incidents - which is what makes "the demo scenario always ends in one HIGH
 * incident" a testable assertion rather than a hope.
 *
 * mulberry32: small, fast, and good enough for detector jitter and dropout. It is
 * emphatically not cryptographic and is never used for anything security-bearing.
 */

export class Rng {
  #state: number;

  constructor(seed: number) {
    // Any 32-bit state works; normalise so a seed of 0 is not degenerate.
    this.#state = (seed | 0) === 0 ? 0x9e3779b9 : seed | 0;
  }

  /** Uniform in [0, 1). */
  next(): number {
    this.#state = (this.#state + 0x6d2b79f5) | 0;
    let t = this.#state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  /** Uniform in [min, max). */
  between(min: number, max: number): number {
    return min + this.next() * (max - min);
  }

  /** Approximately normal, via the central limit theorem. Mean 0, sd 1. */
  gaussian(): number {
    let sum = 0;
    for (let i = 0; i < 6; i += 1) sum += this.next();
    return (sum - 3) / Math.sqrt(0.5);
  }

  /** True with the given probability. */
  chance(probability: number): boolean {
    return this.next() < probability;
  }

  pick<T>(items: readonly T[]): T | undefined {
    if (items.length === 0) return undefined;
    return items[Math.floor(this.next() * items.length)];
  }

  /** A derived generator, so adding a new noise source never shifts existing ones. */
  fork(label: string): Rng {
    let hash = 0;
    for (let i = 0; i < label.length; i += 1) {
      hash = (Math.imul(hash, 31) + label.charCodeAt(i)) | 0;
    }
    return new Rng((this.#state ^ hash) | 0);
  }
}
