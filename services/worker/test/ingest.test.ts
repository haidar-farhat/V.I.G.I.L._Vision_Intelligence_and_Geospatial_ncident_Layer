import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { BoundedQueue, DropPolicy } from '../src/ingest/queue.ts';
import { Backoff, delay, retry } from '../src/ingest/backoff.ts';

describe('BoundedQueue', () => {
  test('behaves as a FIFO below capacity', () => {
    const queue = new BoundedQueue<number>(4);
    queue.push(1);
    queue.push(2);
    queue.push(3);

    assert.equal(queue.depth, 3);
    assert.equal(queue.pop(), 1);
    assert.equal(queue.pop(), 2);
    assert.equal(queue.pop(), 3);
    assert.equal(queue.pop(), undefined);
    assert.ok(queue.isEmpty);
  });

  test('DROP_OLDEST keeps the newest frames, which are the only useful ones', () => {
    const queue = new BoundedQueue<number>(3, DropPolicy.DropOldest);
    for (const n of [1, 2, 3, 4, 5]) queue.push(n);

    assert.equal(queue.depth, 3);
    assert.deepEqual(queue.drain(), [3, 4, 5], 'the stale frames were discarded, not the fresh ones');
    assert.equal(queue.stats.dropped, 2);
  });

  test('DROP_OLDEST hands back what it displaced', () => {
    // The caller may need to release a buffer or count a specific loss; a
    // displaced item must not simply vanish.
    const queue = new BoundedQueue<string>(2, DropPolicy.DropOldest);
    queue.push('a');
    queue.push('b');

    const result = queue.push('c');
    assert.equal(result.accepted, true);
    assert.equal(result.droppedItem, 'a');
  });

  test('REJECT refuses rather than losing anything', () => {
    // Correct for events: an event queue that silently drops is a system that
    // loses evidence and cannot tell you it did.
    const queue = new BoundedQueue<number>(2, DropPolicy.Reject);
    assert.equal(queue.push(1).accepted, true);
    assert.equal(queue.push(2).accepted, true);

    const rejected = queue.push(3);
    assert.equal(rejected.accepted, false);
    assert.equal(queue.depth, 2);
    assert.deepEqual(queue.drain(), [1, 2], 'nothing already queued was disturbed');
  });

  test('DROP_NEWEST discards the incoming item', () => {
    const queue = new BoundedQueue<number>(2, DropPolicy.DropNewest);
    queue.push(1);
    queue.push(2);

    const result = queue.push(3);
    assert.equal(result.droppedItem, 3);
    assert.deepEqual(queue.drain(), [1, 2]);
  });

  test('the ring wraps correctly under sustained churn', () => {
    // The bug this catches is an off-by-one in pointer arithmetic that only
    // appears after the buffer has wrapped several times.
    const queue = new BoundedQueue<number>(3, DropPolicy.DropOldest);

    for (let i = 0; i < 100; i += 1) {
      queue.push(i);
      if (i % 3 === 0) queue.pop();
    }

    assert.ok(queue.depth <= 3, `depth ${queue.depth} exceeded capacity`);
    for (const item of queue.drain()) {
      assert.ok(typeof item === 'number', 'no undefined slots leaked out of the ring');
    }
  });

  test('never exceeds capacity, whatever the policy', () => {
    for (const policy of [DropPolicy.DropOldest, DropPolicy.DropNewest, DropPolicy.Reject]) {
      const queue = new BoundedQueue<number>(5, policy);
      for (let i = 0; i < 1000; i += 1) queue.push(i);

      assert.ok(queue.depth <= 5, `${policy} grew to ${queue.depth}`);
      assert.ok(queue.stats.peakDepth <= 5);
    }
  });

  test('reports a drop rate for camera health', () => {
    const queue = new BoundedQueue<number>(2, DropPolicy.DropOldest);
    for (let i = 0; i < 10; i += 1) queue.push(i);

    // 2 accepted into free slots, 8 displacing. A rising drop rate on a camera
    // that used to keep up is the earliest signal a node is over-subscribed.
    assert.equal(queue.stats.pushed, 10);
    assert.equal(queue.stats.dropped, 8);
    assert.equal(queue.dropRate, 0.8);
  });

  test('an empty queue reports a zero drop rate rather than NaN', () => {
    assert.equal(new BoundedQueue<number>(4).dropRate, 0);
  });

  test('peek does not consume', () => {
    const queue = new BoundedQueue<string>(2);
    queue.push('a');
    assert.equal(queue.peek(), 'a');
    assert.equal(queue.depth, 1);
  });

  test('clear empties without disturbing the counters', () => {
    const queue = new BoundedQueue<number>(4);
    queue.push(1);
    queue.push(2);
    queue.clear();

    assert.ok(queue.isEmpty);
    assert.equal(queue.stats.pushed, 2);
  });

  test('rejects a nonsensical capacity at construction', () => {
    assert.throws(() => new BoundedQueue<number>(0), RangeError);
    assert.throws(() => new BoundedQueue<number>(-1), RangeError);
    assert.throws(() => new BoundedQueue<number>(1.5), RangeError);
  });
});

describe('Backoff', () => {
  /** Fixed source so intervals are exact rather than statistical. */
  const fixedRandom = (value: number) => (): number => value;

  test('grows exponentially up to the ceiling', () => {
    const backoff = new Backoff({
      initialMillis: 100,
      multiplier: 2,
      maxMillis: 1000,
      jitter: 'none',
    });

    assert.equal(backoff.nextDelay(), 100);
    assert.equal(backoff.nextDelay(), 200);
    assert.equal(backoff.nextDelay(), 400);
    assert.equal(backoff.nextDelay(), 800);
    assert.equal(backoff.nextDelay(), 1000, 'capped');
    assert.equal(backoff.nextDelay(), 1000, 'stays capped forever');
  });

  test('full jitter spreads across the whole interval', () => {
    // This is what de-correlates forty cameras that dropped together when a
    // switch failed. A small perturbation would leave them synchronised.
    const low = new Backoff({ initialMillis: 1000, jitter: 'full', random: fixedRandom(0) });
    const high = new Backoff({ initialMillis: 1000, jitter: 'full', random: fixedRandom(0.999) });

    assert.equal(low.nextDelay(), 0);
    assert.equal(high.nextDelay(), 999);
  });

  test('equal jitter keeps a floor while still de-correlating', () => {
    const low = new Backoff({ initialMillis: 1000, jitter: 'equal', random: fixedRandom(0) });
    const high = new Backoff({ initialMillis: 1000, jitter: 'equal', random: fixedRandom(1) });

    assert.equal(low.nextDelay(), 500, 'never below half the interval');
    assert.equal(high.nextDelay(), 1000);
  });

  test('a delay is never longer than the ceiling, with any jitter', () => {
    const backoff = new Backoff({ initialMillis: 100, maxMillis: 5000, jitter: 'full' });
    for (let i = 0; i < 200; i += 1) {
      const wait = backoff.nextDelay();
      assert.ok(wait !== null && wait >= 0 && wait <= 5000, `delay ${wait} out of range`);
    }
  });

  test('retries forever by default, because a camera may come back in an hour', () => {
    const backoff = new Backoff({ initialMillis: 1, jitter: 'none' });
    for (let i = 0; i < 500; i += 1) assert.notEqual(backoff.nextDelay(), null);
    assert.equal(backoff.exhausted, false);
  });

  test('gives up when a maximum attempt count is set', () => {
    const backoff = new Backoff({ initialMillis: 1, jitter: 'none', maxAttempts: 3 });

    assert.notEqual(backoff.nextDelay(), null);
    assert.notEqual(backoff.nextDelay(), null);
    assert.notEqual(backoff.nextDelay(), null);
    assert.equal(backoff.nextDelay(), null, 'exhausted');
    assert.equal(backoff.exhausted, true);
  });

  test('reset returns to the initial interval after a success', () => {
    const backoff = new Backoff({ initialMillis: 100, multiplier: 2, jitter: 'none' });
    backoff.nextDelay();
    backoff.nextDelay();
    assert.equal(backoff.attempts, 2);

    backoff.reset();
    assert.equal(backoff.attempts, 0);
    assert.equal(backoff.nextDelay(), 100);
  });
});

describe('delay', () => {
  test('resolves after the interval', async () => {
    const started = Date.now();
    await delay(30);
    assert.ok(Date.now() - started >= 25);
  });

  test('rejects immediately when the signal is already aborted', async () => {
    const controller = new AbortController();
    controller.abort();
    await assert.rejects(() => delay(10_000, controller.signal), /aborted/);
  });

  test('a shutdown does not have to wait out a long backoff', async () => {
    const controller = new AbortController();
    const pending = delay(60_000, controller.signal);

    setTimeout(() => controller.abort(), 10);
    await assert.rejects(() => pending, /aborted/);
  });
});

describe('retry', () => {
  test('returns the first success without waiting', async () => {
    let calls = 0;
    const result = await retry(async () => {
      calls += 1;
      return 'ok';
    });

    assert.equal(result, 'ok');
    assert.equal(calls, 1);
  });

  test('retries until the operation succeeds', async () => {
    let calls = 0;
    const attempts: number[] = [];

    const result = await retry(
      async () => {
        calls += 1;
        if (calls < 3) throw new Error('camera not ready');
        return calls;
      },
      { initialMillis: 1, jitter: 'none', onRetry: (attempt) => attempts.push(attempt) },
    );

    assert.equal(result, 3);
    // A reconnect nobody can see is indistinguishable from a camera that never
    // dropped, and the difference matters when diagnosing a flaky link.
    assert.deepEqual(attempts, [1, 2]);
  });

  test('rethrows the last error once attempts are exhausted', async () => {
    await assert.rejects(
      () =>
        retry(
          async () => {
            throw new Error('camera unreachable');
          },
          { initialMillis: 1, jitter: 'none', maxAttempts: 2 },
        ),
      /camera unreachable/,
    );
  });

  test('stops promptly when aborted mid-backoff', async () => {
    const controller = new AbortController();
    let calls = 0;

    const pending = retry(
      async () => {
        calls += 1;
        throw new Error('still down');
      },
      { initialMillis: 5000, jitter: 'none', signal: controller.signal },
    );

    setTimeout(() => controller.abort(), 15);
    await assert.rejects(() => pending);
    assert.ok(calls <= 2, `should not have kept retrying after abort, called ${calls} times`);
  });
});
