/**
 * `@sentinel/geometry` - deterministic spatial mathematics.
 *
 * Pure and seedable: given the same inputs these functions always produce the
 * same outputs, which is what makes zone behaviour reproducible in tests and
 * defensible in an incident review.
 */

export * from './vec.ts';
export * from './geodesy.ts';
export * from './polygon.ts';
export * from './projection.ts';
export * from './zones.ts';
export * from './coverage.ts';
