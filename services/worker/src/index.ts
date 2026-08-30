/**
 * `@sentinel/worker` - the headless edge node.
 *
 * Decodes, infers, tracks and evaluates zones. Runs without the desktop UI and
 * keeps running when the control node is unreachable.
 */

export * from './pipeline.ts';
export * from './ingest/queue.ts';
export * from './ingest/backoff.ts';
export * from './rtsp/sdp.ts';
export * from './rtsp/auth.ts';
export * from './rtsp/client.ts';
