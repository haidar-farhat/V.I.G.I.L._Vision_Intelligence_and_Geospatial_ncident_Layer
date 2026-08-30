import type { ModelId, UtcMillis } from './ids.ts';
import type { ComputeBackend, DetectedClass, ModelKind, Precision } from './enums.ts';

/**
 * A registered model artifact.
 *
 * Every event records the `ModelId` that produced its detections. Six months
 * later, "why did the system flag this?" is answerable down to the exact file,
 * its hash and its version - which is the difference between an auditable system
 * and a black box.
 */
export type RegisteredModel = {
  readonly id: ModelId;
  readonly name: string;
  readonly version: string;
  readonly kind: ModelKind;
  /** onnx, gguf, torchscript, tensorrt, simulated, ... */
  readonly format: string;
  readonly classes: readonly DetectedClass[];
  readonly inputWidth: number;
  readonly inputHeight: number;
  readonly runtime: string;
  readonly backend: ComputeBackend;
  readonly precision: Precision;
  readonly sha256: string;
  readonly sizeBytes: number;
  /** SPDX identifier or free text. Tracked because redistribution depends on it. */
  readonly license: string;
  readonly relativePath: string;
  readonly installedAt: UtcMillis;
  readonly enabled: boolean;
};

/** Result of a benchmark run, used by the model comparison screen. */
export type ModelBenchmark = {
  readonly modelId: ModelId;
  readonly ranAt: UtcMillis;
  readonly backend: ComputeBackend;
  readonly precision: Precision;
  readonly inputWidth: number;
  readonly inputHeight: number;
  readonly fps: number;
  readonly latencyP50Ms: number;
  readonly latencyP95Ms: number;
  readonly gpuPercent: number | null;
  readonly vramMb: number | null;
  readonly cpuPercent: number;
};

/** Configuration for a local LLM or VLM endpoint. Never a cloud provider. */
export type LocalInferenceEndpoint = {
  readonly name: string;
  /** Must resolve to a loopback or private-range address; enforced at runtime. */
  readonly endpoint: string;
  readonly model: string;
  readonly contextTokens: number;
  /** Defaults to 0 for incident analysis: reports must be reproducible. */
  readonly temperature: number;
  readonly maxTokens: number;
};

/** A versioned prompt. Prompts are configuration, and reports cite the version. */
export type PromptDefinition = {
  readonly id: string;
  readonly version: string;
  readonly purpose: string;
  readonly template: string;
  readonly createdAt: UtcMillis;
};
