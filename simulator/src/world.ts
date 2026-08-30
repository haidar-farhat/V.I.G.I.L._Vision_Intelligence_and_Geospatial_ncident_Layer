import type {
  CameraId,
  CameraPose,
  DetectedClass,
  Detection,
  LatLon,
  ModelId,
  UtcMillis,
} from '@sentinel/shared-types';
import { utcMillis } from '@sentinel/shared-types';
import { boxForObject, destinationPoint, haversineDistance, projectToImage } from '@sentinel/geometry';
import { Rng } from './rng.ts';

/**
 * The simulated world.
 *
 * Actors move through real geographic space along waypoint routes. Simulated
 * cameras observe them by projecting their positions into image space with the
 * production geometry, then adding the imperfections a real detector has:
 * localisation jitter, confidence variation, occasional dropped detections and
 * occasional false positives.
 *
 * The important property is that nothing downstream knows this is a simulation.
 * The tracker, zone engine, rule engine and correlator receive exactly the shape
 * of data a real camera produces, so the simulator exercises the real pipeline
 * rather than a mock of it.
 */

/** Physical dimensions used to synthesise plausible detection boxes. */
const CLASS_SIZES: Readonly<Record<string, { height: number; width: number }>> = Object.freeze({
  person: { height: 1.75, width: 0.55 },
  car: { height: 1.5, width: 1.8 },
  truck: { height: 3.2, width: 2.4 },
  bus: { height: 3.0, width: 2.5 },
  motorcycle: { height: 1.6, width: 0.8 },
  bicycle: { height: 1.7, width: 0.6 },
  animal: { height: 0.6, width: 0.9 },
  bag: { height: 0.4, width: 0.35 },
});

const DEFAULT_SIZE = { height: 1.7, width: 0.6 };

export type Waypoint = {
  readonly at: LatLon;
  /** Seconds from scenario start at which the actor reaches this point. */
  readonly timeSeconds: number;
};

export type Actor = {
  readonly id: string;
  readonly objectClass: DetectedClass;
  readonly route: readonly Waypoint[];
};

export type SimulatedCamera = {
  readonly id: CameraId;
  readonly name: string;
  readonly pose: CameraPose;
  /** Detector reliability on this camera, 0..1. */
  readonly detectionRate: number;
  /** False positives per minute. */
  readonly falsePositiveRate: number;
};

export type DetectorProfile = {
  /** Standard deviation of box-centre jitter, in normalised image units. */
  readonly jitter: number;
  readonly confidenceMean: number;
  readonly confidenceSpread: number;
  readonly modelId: ModelId;
};

export const DEFAULT_DETECTOR_PROFILE: DetectorProfile = Object.freeze({
  jitter: 0.004,
  confidenceMean: 0.86,
  confidenceSpread: 0.07,
  modelId: 'builtin:simulated-detector' as ModelId,
});

/**
 * Where an actor is at a given time.
 *
 * Linear interpolation between waypoints. Before the first waypoint and after the
 * last, the actor is not present at all - it has not arrived, or it has left -
 * rather than being clamped to the endpoint, which would leave phantom stationary
 * objects sitting in zones forever.
 */
export const actorPositionAt = (actor: Actor, elapsedSeconds: number): LatLon | null => {
  const route = actor.route;
  const first = route[0];
  const last = route[route.length - 1];
  if (first === undefined || last === undefined) return null;
  if (elapsedSeconds < first.timeSeconds || elapsedSeconds > last.timeSeconds) return null;

  for (let i = 0; i < route.length - 1; i += 1) {
    const a = route[i];
    const b = route[i + 1];
    if (a === undefined || b === undefined) continue;
    if (elapsedSeconds < a.timeSeconds || elapsedSeconds > b.timeSeconds) continue;

    const span = b.timeSeconds - a.timeSeconds;
    const t = span <= 0 ? 0 : (elapsedSeconds - a.timeSeconds) / span;

    return {
      lat: a.at.lat + (b.at.lat - a.at.lat) * t,
      lon: a.at.lon + (b.at.lon - a.at.lon) * t,
    };
  }

  return last.at;
};

/** Ground-truth record, kept so tests can measure the pipeline's real error. */
export type GroundTruth = {
  readonly actorId: string;
  readonly cameraId: CameraId;
  readonly at: UtcMillis;
  readonly position: LatLon;
  readonly objectClass: DetectedClass;
};

export type FrameObservation = {
  readonly cameraId: CameraId;
  readonly at: UtcMillis;
  readonly detections: readonly Detection[];
  readonly truth: readonly GroundTruth[];
};

/**
 * A camera that turns world state into detections.
 *
 * Holds its own RNG stream so that adding a camera to a scenario does not perturb
 * the noise of the existing ones - which would otherwise make every scenario
 * expectation brittle to unrelated edits.
 */
export class SimulatedSensor {
  readonly camera: SimulatedCamera;
  readonly #rng: Rng;
  readonly #profile: DetectorProfile;

  constructor(camera: SimulatedCamera, seed: number, profile: DetectorProfile = DEFAULT_DETECTOR_PROFILE) {
    this.camera = camera;
    this.#rng = new Rng(seed).fork(String(camera.id));
    this.#profile = profile;
  }

  /**
   * Observe the world at one instant.
   *
   * Actors outside the field of view, beyond range, or occluded by the frame edge
   * simply do not appear - exactly as they would not on a real camera.
   */
  observe(
    actors: readonly Actor[],
    elapsedSeconds: number,
    startedAt: UtcMillis,
    frameIntervalSeconds: number,
  ): FrameObservation {
    const at = utcMillis(startedAt + Math.round(elapsedSeconds * 1000));
    const detections: Detection[] = [];
    const truth: GroundTruth[] = [];

    for (const actor of actors) {
      const position = actorPositionAt(actor, elapsedSeconds);
      if (position === null) continue;

      const image = projectToImage(this.camera.pose, position, 0);
      if (image === null) continue;

      truth.push({
        actorId: actor.id,
        cameraId: this.camera.id,
        at,
        position,
        objectClass: actor.objectClass,
      });

      // Detector dropout. A real model misses frames, and a tracker that cannot
      // survive that is useless, so the simulator must produce the gaps.
      if (!this.#rng.chance(this.camera.detectionRate)) continue;

      const size = CLASS_SIZES[actor.objectClass] ?? DEFAULT_SIZE;
      const box = boxForObject(this.camera.pose, position, size.height, size.width);
      if (box === null) continue;

      const jitter = this.#profile.jitter;
      const confidence = Math.max(
        0.05,
        Math.min(
          0.99,
          this.#profile.confidenceMean + this.#rng.gaussian() * this.#profile.confidenceSpread,
        ),
      );

      detections.push({
        cameraId: this.camera.id,
        frameAt: at,
        objectClass: actor.objectClass,
        confidence,
        box: {
          x: box.x + this.#rng.gaussian() * jitter,
          y: box.y + this.#rng.gaussian() * jitter,
          w: Math.max(0.005, box.w * (1 + this.#rng.gaussian() * 0.05)),
          h: Math.max(0.005, box.h * (1 + this.#rng.gaussian() * 0.05)),
        },
        modelId: this.#profile.modelId,
      });
    }

    // False positives, at the configured rate per minute.
    const falsePositiveChance = (this.camera.falsePositiveRate * frameIntervalSeconds) / 60;
    if (falsePositiveChance > 0 && this.#rng.chance(falsePositiveChance)) {
      detections.push({
        cameraId: this.camera.id,
        frameAt: at,
        objectClass: 'person',
        confidence: this.#rng.between(0.3, 0.55),
        box: {
          x: this.#rng.between(0.05, 0.85),
          y: this.#rng.between(0.05, 0.75),
          w: 0.05,
          h: 0.12,
        },
        modelId: this.#profile.modelId,
      });
    }

    return { cameraId: this.camera.id, at, detections, truth };
  }
}

/** Build a straight-line route between two points at a walking or driving speed. */
export const straightRoute = (
  from: LatLon,
  to: LatLon,
  startSeconds: number,
  speedMps: number,
): readonly Waypoint[] => {
  const distance = haversineDistance(from, to);
  const duration = speedMps <= 0 ? 0 : distance / speedMps;
  return [
    { at: from, timeSeconds: startSeconds },
    { at: to, timeSeconds: startSeconds + duration },
  ];
};

/** Offset a point by metres east and north. Convenient for laying out a site. */
export const offset = (origin: LatLon, east: number, north: number): LatLon =>
  destinationPoint(destinationPoint(origin, 0, north), 90, east);
