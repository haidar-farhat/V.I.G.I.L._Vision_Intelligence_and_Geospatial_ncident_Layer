/**
 * `@sentinel/shared-types` - the domain model.
 *
 * Every service, the desktop UI and the database schema agree here and nowhere
 * else. This package has no runtime dependencies and, apart from a handful of
 * frozen lookup tables, no runtime behaviour.
 */

export * from './ids.ts';
export * from './enums.ts';
export * from './geo.ts';
export * from './camera.ts';
export * from './zone.ts';
export * from './track.ts';
export * from './event.ts';
export * from './incident.ts';
export * from './node.ts';
export * from './model.ts';
export * from './access.ts';
