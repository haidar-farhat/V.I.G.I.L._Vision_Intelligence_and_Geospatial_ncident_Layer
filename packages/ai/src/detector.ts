import type {
  BoundingBox,
  CameraId,
  ComputeBackend,
  ComputeDevice,
  DetectedClass,
  Detection,
  ModelId,
  Precision,
  UtcMillis,
} from '@sentinel/shared-types';

/**
 * Model abstraction.
 *
 * The event engine consumes typed observations, never model output, so a
 * detector can be swapped - YOLO for RT-DETR, ONNX for TensorRT, GPU for CPU -
 * without a line changing anywhere downstream. That indirection is what makes the
 * model registry meaningful: the system can honestly record *which* artifact
 * produced a detection because nothing else in the pipeline assumes what it was.
 */

/** A decoded frame handed to a model. Pixel data stays opaque to this layer. */
export type Frame = {
  readonly cameraId: CameraId;
  readonly capturedAt: UtcMillis;
  readonly width: number;
  readonly height: number;
  /**
   * Raw pixels, when a real decoder produced them. Null for synthetic sources
   * that carry ground truth instead - the simulator has no pixels and needs none.
   */
  readonly pixels: Uint8Array | null;
  /** Monotonic index within the stream, for ordering and drop accounting. */
  readonly sequence: number;
};

/** A model's own report of what it found, before tracking. */
export type DetectionResult = {
  readonly detections: readonly Detection[];
  readonly inferenceMillis: number;
  readonly modelId: ModelId;
};

export type Detector = {
  readonly modelId: ModelId;
  readonly classes: readonly DetectedClass[];
  readonly backend: ComputeBackend;
  detect(frame: Frame): Promise<DetectionResult>;
  close(): Promise<void>;
};

export type Classifier = {
  readonly modelId: ModelId;
  classify(frame: Frame, box: BoundingBox): Promise<{ label: string; confidence: number }>;
};

/** Appearance embeddings for cross-camera association. Not biometric identity. */
export type EmbeddingModel = {
  readonly modelId: ModelId;
  readonly dimensions: number;
  embed(frame: Frame, box: BoundingBox): Promise<readonly number[]>;
};

/**
 * Vision-language model, invoked on selected frames of candidate events only.
 *
 * Running a VLM over every frame of every camera is the difference between a
 * system that fits on one GPU and one that cannot exist. The pipeline calls this
 * for a handful of representative frames per event, never for the stream.
 */
export type Vlm = {
  readonly modelId: ModelId;
  describe(frames: readonly Frame[], question: string): Promise<string>;
};

// ------------------------------------------------------------ device discovery

/**
 * What the host can actually run.
 *
 * Reported rather than assumed: the UI shows the operator what was found, and the
 * scheduler sizes workloads from it. A machine with no accelerator is a supported
 * configuration, not an error - it runs a smaller model at a lower frame rate and
 * keeps recording either way.
 */
export type DeviceReport = {
  readonly devices: readonly ComputeDevice[];
  readonly selected: ComputeDevice;
  readonly reason: string;
};

/** Always-present fallback. Every host can do this. */
export const CPU_DEVICE: ComputeDevice = Object.freeze({
  name: 'CPU',
  backend: 'CPU' satisfies ComputeBackend,
  vramMb: null,
  driverVersion: null,
  supportedPrecisions: Object.freeze<readonly Precision[]>(['FP32']),
  recommendedConcurrentStreams: 2,
});

/**
 * Choose a device.
 *
 * Prefers the accelerator with the most VRAM, because stream count is bounded by
 * memory long before it is bounded by compute. Falls back to CPU with a stated
 * reason rather than failing - losing the GPU must degrade the system, not stop
 * it (see the failure matrix in ARCHITECTURE.md).
 */
export const selectDevice = (devices: readonly ComputeDevice[]): DeviceReport => {
  const accelerators = devices.filter((d) => d.backend !== 'CPU' && d.backend !== 'SIMULATED');

  if (accelerators.length === 0) {
    return {
      devices: devices.length === 0 ? [CPU_DEVICE] : devices,
      selected: CPU_DEVICE,
      reason: 'No accelerator detected. Running on CPU at a reduced frame rate.',
    };
  }

  let best = accelerators[0] as ComputeDevice;
  for (const device of accelerators) {
    if ((device.vramMb ?? 0) > (best.vramMb ?? 0)) best = device;
  }

  return {
    devices,
    selected: best,
    reason: `Selected ${best.name} (${best.backend}${
      best.vramMb === null ? '' : `, ${best.vramMb} MB VRAM`
    }).`,
  };
};

// ------------------------------------------------------------------ sampling

/**
 * Adaptive inference rate.
 *
 * A camera watching an empty car park does not deserve the same compute as one
 * with someone climbing a fence. Idle cameras run at a trickle and step up the
 * moment something happens, which is what allows a single machine to cover far
 * more cameras than a fixed-rate design.
 */
export class AdaptiveSampler {
  readonly #idleFps: number;
  readonly #activeFps: number;
  readonly #activeHoldMillis: number;
  #lastSampleAt: UtcMillis | null = null;
  #activeUntil = 0;

  constructor(idleFps: number, activeFps: number, activeHoldMillis = 10_000) {
    this.#idleFps = Math.max(0.1, idleFps);
    this.#activeFps = Math.max(this.#idleFps, activeFps);
    this.#activeHoldMillis = activeHoldMillis;
  }

  /** Signal that something is happening, raising the rate for the hold period. */
  markActive(at: UtcMillis): void {
    this.#activeUntil = Math.max(this.#activeUntil, at + this.#activeHoldMillis);
  }

  targetFps(at: UtcMillis): number {
    return at < this.#activeUntil ? this.#activeFps : this.#idleFps;
  }

  /** Whether this frame should be run through the model. */
  shouldSample(at: UtcMillis): boolean {
    const interval = 1000 / this.targetFps(at);
    if (this.#lastSampleAt === null || at - this.#lastSampleAt >= interval) {
      this.#lastSampleAt = at;
      return true;
    }
    return false;
  }

  reset(): void {
    this.#lastSampleAt = null;
    this.#activeUntil = 0;
  }
}
