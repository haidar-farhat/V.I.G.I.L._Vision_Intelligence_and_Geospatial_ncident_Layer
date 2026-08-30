import type { ModelId, ModelKind, RegisteredModel, UtcMillis } from '@sentinel/shared-types';

/**
 * Model registry.
 *
 * Every event records the model that produced its detections. Six months later,
 * "why did the system flag this?" must be answerable down to the exact artifact,
 * its version and its hash - otherwise the detection is an assertion nobody can
 * check, which is the opposite of what this platform is for.
 *
 * Models are only ever imported from local files. Nothing here downloads
 * anything, and there is no code path that could.
 */

export type ModelValidationIssue = {
  readonly field: string;
  readonly detail: string;
};

/**
 * Validate a model manifest before installation.
 *
 * The hash is required, not optional: without it there is no way to prove later
 * that the file backing a six-month-old detection is the file that produced it.
 */
export const validateModel = (
  model: Partial<RegisteredModel>,
): readonly ModelValidationIssue[] => {
  const issues: ModelValidationIssue[] = [];

  const required: readonly (keyof RegisteredModel)[] = [
    'id',
    'name',
    'version',
    'kind',
    'format',
    'runtime',
    'sha256',
    'relativePath',
  ];

  for (const field of required) {
    const value = model[field];
    if (value === undefined || value === null || value === '') {
      issues.push({ field: String(field), detail: 'is required' });
    }
  }

  if (model.sha256 !== undefined && !/^[a-f0-9]{64}$/i.test(model.sha256)) {
    issues.push({ field: 'sha256', detail: 'must be a 64-character hex SHA-256 digest' });
  }

  if (model.license === undefined || model.license === '') {
    // Redistribution and deployment both depend on this being known.
    issues.push({ field: 'license', detail: 'is required so redistribution terms are auditable' });
  }

  if (model.kind === 'DETECTOR' && (model.classes === undefined || model.classes.length === 0)) {
    issues.push({ field: 'classes', detail: 'a detector must declare the classes it emits' });
  }

  if (model.relativePath !== undefined) {
    // The path is joined onto the models root, so it must not be able to escape.
    if (
      model.relativePath.includes('..') ||
      model.relativePath.startsWith('/') ||
      model.relativePath.startsWith('\\') ||
      /^[a-zA-Z]:/.test(model.relativePath)
    ) {
      issues.push({
        field: 'relativePath',
        detail: 'must be relative and must not traverse outside the models directory',
      });
    }
  }

  return issues;
};

export class ModelValidationError extends Error {
  readonly code = 'MODEL_INVALID';
  readonly issues: readonly ModelValidationIssue[];
  readonly recoverable = true;

  constructor(issues: readonly ModelValidationIssue[]) {
    super(`Model rejected: ${issues.map((i) => `${i.field} ${i.detail}`).join('; ')}`);
    this.name = 'ModelValidationError';
    this.issues = issues;
  }
}

export class ModelRegistry {
  readonly #models = new Map<ModelId, RegisteredModel>();

  constructor(models: readonly RegisteredModel[] = []) {
    for (const model of models) this.#models.set(model.id, model);
  }

  /** Install a validated model. Throws rather than storing something unverifiable. */
  install(model: RegisteredModel): void {
    const issues = validateModel(model);
    if (issues.length > 0) throw new ModelValidationError(issues);
    this.#models.set(model.id, model);
  }

  get(modelId: ModelId): RegisteredModel | undefined {
    return this.#models.get(modelId);
  }

  /** Whether a model id is known. Used to reject events citing a phantom model. */
  has(modelId: ModelId): boolean {
    return this.#models.has(modelId);
  }

  list(kind?: ModelKind): readonly RegisteredModel[] {
    const all = [...this.#models.values()];
    return kind === undefined ? all : all.filter((m) => m.kind === kind);
  }

  enabled(kind: ModelKind): readonly RegisteredModel[] {
    return this.list(kind).filter((m) => m.enabled);
  }

  remove(modelId: ModelId): boolean {
    return this.#models.delete(modelId);
  }

  get size(): number {
    return this.#models.size;
  }

  /**
   * A provenance record suitable for an evidence manifest: enough to prove which
   * artifacts were involved, with no filesystem detail that would not survive
   * being moved to another machine.
   */
  provenance(): readonly {
    id: ModelId;
    name: string;
    version: string;
    sha256: string;
    license: string;
  }[] {
    return this.list().map((m) => ({
      id: m.id,
      name: m.name,
      version: m.version,
      sha256: m.sha256,
      license: m.license,
    }));
  }
}

/** Descriptor for the built-in simulated detector, so simulated runs are traceable too. */
export const simulatedDetectorModel = (installedAt: UtcMillis): RegisteredModel => ({
  id: 'builtin:simulated-detector' as ModelId,
  name: 'Simulated Detector',
  version: '1.0.0',
  kind: 'DETECTOR',
  format: 'simulated',
  classes: ['person', 'vehicle', 'car', 'truck', 'bicycle', 'animal'],
  inputWidth: 640,
  inputHeight: 640,
  runtime: 'builtin',
  backend: 'SIMULATED',
  precision: 'FP32',
  // Identifies the built-in rather than a file on disk; there is no artifact to hash.
  sha256: '0'.repeat(64),
  sizeBytes: 0,
  license: 'internal',
  relativePath: 'builtin/simulated-detector',
  installedAt,
  enabled: true,
});
