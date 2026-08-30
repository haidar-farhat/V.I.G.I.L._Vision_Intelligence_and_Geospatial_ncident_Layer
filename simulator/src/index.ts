/**
 * `@sentinel/simulator` - synthetic cameras, actors and scenarios.
 *
 * Replaces only the camera and the detector. Everything downstream of a detection
 * is the production pipeline, so a scenario run is a real test of the system
 * rather than a test of a parallel mock.
 */

export * from './rng.ts';
export * from './world.ts';
export * from './scenario.ts';
export * from './run.ts';
