/**
 * `@sentinel/test-utils` - deterministic fixtures and protocol test doubles.
 *
 * The mock devices here exist because real IP cameras are the least
 * standards-compliant hardware most engineers ever integrate with, and a client
 * written purely from a specification discovers that on a customer's site rather
 * than in CI.
 */

export * from './mock-rtsp.ts';
