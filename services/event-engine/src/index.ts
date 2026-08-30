/**
 * `@sentinel/event-engine` - observations become events, events become incidents.
 *
 * The layer that decides what an operator is actually shown. Its measure of
 * success is how few incidents it raises, not how many detections it makes.
 */

export * from './rules.ts';
export * from './risk.ts';
export * from './correlation.ts';
