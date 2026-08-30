/**
 * `@sentinel/ai` - model abstraction and the evidence-bound analyst contract.
 *
 * Models are replaceable; the event engine never sees their output directly. The
 * analyst may summarise and explain, but never identify, judge, or command.
 */

export * from './detector.ts';
export * from './analyst.ts';
export * from './deterministic-analyst.ts';
export * from './registry.ts';
