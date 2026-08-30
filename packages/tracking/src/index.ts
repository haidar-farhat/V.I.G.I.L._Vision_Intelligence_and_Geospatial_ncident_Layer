/**
 * `@sentinel/tracking` - object identity over time and across cameras.
 *
 * Within a camera, identity is asserted (a track is one object). Across cameras
 * it is only ever scored, never asserted.
 */

export * from './tracker.ts';
export * from './association.ts';
