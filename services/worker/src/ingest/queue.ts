/**
 * Bounded queues and backpressure.
 *
 * Invariant I5: every queue is bounded and every retry is backed off. This is the
 * file that makes that true rather than aspirational.
 *
 * The failure this prevents is specific and common. A camera delivers 25 frames a
 * second whether or not anything is reading them. If the model behind the queue
 * slows down - a thermal throttle, a competing process, a larger model swapped in
 * - an unbounded queue grows until the worker is killed by the OS, taking
 * recording down with it. That is the worst possible outcome: the one moment the
 * site most needs video is the moment the process dies.
 *
 * So the queue is bounded and drops. Which end it drops from is the interesting
 * decision, and it differs by what is queued:
 *
 *   frames        DROP_OLDEST  - a stale frame is worthless; the newest is what
 *                               matters, and losing old ones is visible as a
 *                               counter rather than as a crash
 *   events        REJECT       - events are never dropped. If this queue fills,
 *                               something is badly wrong upstream and the caller
 *                               must handle it rather than lose evidence
 */

export const DropPolicy = {
  /** Discard the oldest item to make room. Correct for frames. */
  DropOldest: 'DROP_OLDEST',
  /** Discard the incoming item. Correct when order matters more than recency. */
  DropNewest: 'DROP_NEWEST',
  /** Refuse the push and tell the caller. Correct for anything that is evidence. */
  Reject: 'REJECT',
} as const;
export type DropPolicy = (typeof DropPolicy)[keyof typeof DropPolicy];

export type QueueStats = {
  readonly depth: number;
  readonly capacity: number;
  readonly pushed: number;
  readonly popped: number;
  readonly dropped: number;
  /** High-water mark, for sizing decisions after the fact. */
  readonly peakDepth: number;
};

export type PushResult =
  | { readonly accepted: true; readonly droppedItem: undefined }
  | { readonly accepted: true; readonly droppedItem: unknown }
  | { readonly accepted: false; readonly droppedItem: undefined };

/**
 * A fixed-capacity ring buffer.
 *
 * Uses a ring rather than an array with `shift()`, because `shift()` is O(n) and
 * this sits on the hot path of every camera at frame rate. At 25 fps across 16
 * cameras that is 400 operations a second whose cost must not scale with depth.
 */
export class BoundedQueue<T> {
  readonly #items: (T | undefined)[];
  readonly #capacity: number;
  readonly #policy: DropPolicy;

  #head = 0;
  #tail = 0;
  #depth = 0;

  #pushed = 0;
  #popped = 0;
  #dropped = 0;
  #peakDepth = 0;

  constructor(capacity: number, policy: DropPolicy = DropPolicy.DropOldest) {
    if (!Number.isInteger(capacity) || capacity < 1) {
      throw new RangeError(`queue capacity must be a positive integer, got ${capacity}`);
    }
    this.#capacity = capacity;
    this.#policy = policy;
    this.#items = new Array<T | undefined>(capacity).fill(undefined);
  }

  get depth(): number {
    return this.#depth;
  }

  get capacity(): number {
    return this.#capacity;
  }

  get policy(): DropPolicy {
    return this.#policy;
  }

  get isFull(): boolean {
    return this.#depth === this.#capacity;
  }

  get isEmpty(): boolean {
    return this.#depth === 0;
  }

  /**
   * Add an item.
   *
   * Returns whether it was accepted and, when the policy displaced something, the
   * item that was displaced - so a caller that needs to release a buffer or count
   * a specific loss can do so rather than having it vanish silently.
   */
  push(item: T): PushResult {
    if (this.#depth < this.#capacity) {
      this.#items[this.#tail] = item;
      this.#tail = (this.#tail + 1) % this.#capacity;
      this.#depth += 1;
      this.#pushed += 1;
      if (this.#depth > this.#peakDepth) this.#peakDepth = this.#depth;
      return { accepted: true, droppedItem: undefined };
    }

    switch (this.#policy) {
      case DropPolicy.DropOldest: {
        const displaced = this.#items[this.#head];
        this.#items[this.#head] = item;
        // Overwriting the head advances both pointers: the slot just written is
        // now the newest, and the next-oldest becomes the head.
        this.#head = (this.#head + 1) % this.#capacity;
        this.#tail = this.#head;
        this.#pushed += 1;
        this.#dropped += 1;
        return { accepted: true, droppedItem: displaced };
      }

      case DropPolicy.DropNewest:
        this.#dropped += 1;
        return { accepted: true, droppedItem: item };

      case DropPolicy.Reject:
        this.#dropped += 1;
        return { accepted: false, droppedItem: undefined };
    }
  }

  pop(): T | undefined {
    if (this.#depth === 0) return undefined;

    const item = this.#items[this.#head];
    this.#items[this.#head] = undefined;
    this.#head = (this.#head + 1) % this.#capacity;
    this.#depth -= 1;
    this.#popped += 1;
    return item;
  }

  /** Look at the oldest item without removing it. */
  peek(): T | undefined {
    return this.#depth === 0 ? undefined : this.#items[this.#head];
  }

  /**
   * Take everything currently queued, in order.
   *
   * Used by the inference stage, which prefers to process the newest frame and
   * discard the rest rather than fall further behind on stale ones.
   */
  drain(): T[] {
    const items: T[] = [];
    let item = this.pop();
    while (item !== undefined) {
      items.push(item);
      item = this.pop();
    }
    return items;
  }

  clear(): void {
    while (this.#depth > 0) this.pop();
  }

  get stats(): QueueStats {
    return {
      depth: this.#depth,
      capacity: this.#capacity,
      pushed: this.#pushed,
      popped: this.#popped,
      dropped: this.#dropped,
      peakDepth: this.#peakDepth,
    };
  }

  /**
   * Fraction of pushes that were dropped, 0..1.
   *
   * Surfaced in camera health. A non-zero value is not necessarily a fault - a
   * camera deliberately sampled below its frame rate will show one - but a rising
   * value on a camera that used to keep up is the earliest signal that a node is
   * over-subscribed.
   */
  get dropRate(): number {
    return this.#pushed === 0 ? 0 : this.#dropped / this.#pushed;
  }
}
