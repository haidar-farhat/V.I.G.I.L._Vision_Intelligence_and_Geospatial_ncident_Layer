import type {
  BoundingBox,
  CameraId,
  CameraPose,
  Detection,
  DetectedClass,
  LatLon,
  PositionEstimate,
  PositionSource,
  Track,
  TrackId,
  TrackObservation,
  UtcMillis,
  Vec2,
} from '@sentinel/shared-types';
import { asId } from '@sentinel/shared-types';
import { bearingDegrees, haversineDistance, projectDetection } from '@sentinel/geometry';

/**
 * Single-camera multi-object tracker.
 *
 * Associates detections to persistent tracks by IoU with motion compensation.
 * Deliberately not a full Kalman filter: at the 5-15 fps this platform runs
 * inference, a constant-velocity predictor plus IoU gating matches nearly as
 * well, has no tuning matrices to get wrong, and stays fully deterministic -
 * which is what lets the same recorded scenario produce byte-identical tracks in
 * a test and in production.
 *
 * The property that matters operationally is gap tolerance: a person walking
 * behind a pillar must come out the other side as the *same* track. A tracker
 * that splits identity there turns one incident into three and destroys
 * cross-camera correlation before it starts.
 */

export type TrackerOptions = {
  /** Minimum IoU for a detection to be considered the same object. */
  readonly iouThreshold?: number;
  /**
   * Fallback gate for small, fast-moving boxes, as a multiple of object size.
   * A detection whose centre lands within this radius of the predicted position
   * associates even when the boxes do not overlap.
   */
  readonly gateFactor?: number;
  /** How long a track coasts without detections before it is closed. */
  readonly maxGapMillis?: number;
  /** Consecutive detections required before a track is reported at all. */
  readonly minHitsToConfirm?: number;
  /** Maximum trajectory points retained per track. */
  readonly maxTrajectoryPoints?: number;
  /**
   * Window over which ground speed and heading are averaged, in milliseconds.
   * Too short and projection noise dominates; too long and a turn is missed.
   */
  readonly motionWindowMillis?: number;
  /** Camera pose used to project detections onto the map. */
  readonly pose?: CameraPose | null;
};

const DEFAULTS = {
  iouThreshold: 0.2,
  gateFactor: 2.5,
  maxGapMillis: 2000,
  minHitsToConfirm: 2,
  maxTrajectoryPoints: 120,
  motionWindowMillis: 3000,
} as const;

/** Intersection-over-union of two normalised boxes. */
export const iou = (a: BoundingBox, b: BoundingBox): number => {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w);
  const y2 = Math.min(a.y + a.h, b.y + b.h);

  const w = x2 - x1;
  const h = y2 - y1;
  if (w <= 0 || h <= 0) return 0;

  const intersection = w * h;
  const union = a.w * a.h + b.w * b.h - intersection;
  return union <= 0 ? 0 : intersection / union;
};

export const boxCenter = (box: BoundingBox): Vec2 => ({
  x: box.x + box.w / 2,
  y: box.y + box.h / 2,
});

type TrackState = {
  id: TrackId;
  objectClass: DetectedClass;
  firstSeen: UtcMillis;
  lastSeen: UtcMillis;
  /** Last frame in which a detection actually matched, as opposed to coasting. */
  lastDetectedAt: UtcMillis;
  box: BoundingBox;
  /** Normalised image units per millisecond. */
  velocity: Vec2;
  confidence: number;
  hits: number;
  trajectory: Vec2[];
  position: PositionEstimate | null;
  /** Recent ground positions, used to derive speed and heading over the map. */
  groundHistory: Array<{ at: UtcMillis; point: LatLon }>;
  confirmed: boolean;
};

/** What the tracker emitted for one frame. */
export type TrackerUpdate = {
  readonly tracks: readonly Track[];
  readonly observations: readonly TrackObservation[];
  /** Tracks closed on this frame, for downstream state cleanup. */
  readonly ended: readonly TrackId[];
};

export class Tracker {
  readonly #cameraId: CameraId;
  readonly #options: Required<Omit<TrackerOptions, 'pose'>>;
  readonly #states = new Map<TrackId, TrackState>();
  #pose: CameraPose | null;
  #sequence = 0;

  constructor(cameraId: CameraId, options: TrackerOptions = {}) {
    this.#cameraId = cameraId;
    this.#pose = options.pose ?? null;
    this.#options = {
      iouThreshold: options.iouThreshold ?? DEFAULTS.iouThreshold,
      gateFactor: options.gateFactor ?? DEFAULTS.gateFactor,
      maxGapMillis: options.maxGapMillis ?? DEFAULTS.maxGapMillis,
      minHitsToConfirm: options.minHitsToConfirm ?? DEFAULTS.minHitsToConfirm,
      maxTrajectoryPoints: options.maxTrajectoryPoints ?? DEFAULTS.maxTrajectoryPoints,
      motionWindowMillis: options.motionWindowMillis ?? DEFAULTS.motionWindowMillis,
    };
  }

  /** Update the pose after an operator moves the camera on the map. */
  setPose(pose: CameraPose | null): void {
    this.#pose = pose;
  }

  get activeTrackCount(): number {
    return this.#states.size;
  }

  /**
   * Predict where a track's box will be at `at`, from its last known box and
   * velocity. This is what lets a track survive a detection gap: it keeps moving
   * through the occlusion so the reacquired detection still overlaps it.
   */
  #predict(state: TrackState, at: UtcMillis): BoundingBox {
    const dt = at - state.lastSeen;
    if (dt <= 0) return state.box;
    return {
      x: state.box.x + state.velocity.x * dt,
      y: state.box.y + state.velocity.y * dt,
      w: state.box.w,
      h: state.box.h,
    };
  }

  /**
   * How well a detection matches a predicted track box, or null if it cannot.
   *
   * Two tiers, because IoU alone is not enough. A person 25 m from a wide-angle
   * camera occupies a box under two percent of the frame width, and at walking
   * pace crosses half a box width between inference frames. Consecutive
   * detections of the same person then overlap by barely a third, and detector
   * jitter regularly pushes that under any sane IoU threshold - so a pure-IoU
   * tracker shatters one person into a dozen tracks, which destroys both dwell
   * timing and cross-camera correlation.
   *
   * So: a real overlap always wins (scored above 1), and when overlap fails, a
   * detection whose centre lands within a size-scaled gate of the *predicted*
   * position is still accepted, ranked by proximity (scored below 1). The gate
   * scales with object size, so it stays tight for large nearby objects and
   * generous only where the geometry demands it.
   */
  #associationScore(predicted: BoundingBox, detected: BoundingBox): number | null {
    const overlap = iou(predicted, detected);
    if (overlap >= this.#options.iouThreshold) return 1 + overlap;

    const gate =
      this.#options.gateFactor *
      Math.max(predicted.w, predicted.h, detected.w, detected.h);
    if (gate <= 0) return null;

    const a = boxCenter(predicted);
    const b = boxCenter(detected);
    const separation = Math.hypot(b.x - a.x, b.y - a.y);
    if (separation > gate) return null;

    return 1 - separation / gate;
  }

  /**
   * Feed one frame's detections.
   *
   * Detections must belong to this tracker's camera and carry the frame time.
   * Association is greedy by descending score, which for the handful of objects a
   * single camera sees at once is both optimal in practice and stable - the same
   * input always yields the same assignment, with no dependence on iteration
   * order.
   */
  update(detections: readonly Detection[], at: UtcMillis): TrackerUpdate {
    const candidates: Array<{ trackId: TrackId; index: number; score: number }> = [];

    const predicted = new Map<TrackId, BoundingBox>();
    for (const [trackId, state] of this.#states) {
      predicted.set(trackId, this.#predict(state, at));
    }

    for (const [trackId, state] of this.#states) {
      const box = predicted.get(trackId);
      if (box === undefined) continue;

      for (let i = 0; i < detections.length; i += 1) {
        const detection = detections[i];
        if (detection === undefined) continue;
        // Class changes are treated as different objects: a "person" track must
        // never silently absorb a "vehicle" detection.
        if (detection.objectClass !== state.objectClass) continue;

        const score = this.#associationScore(box, detection.box);
        if (score !== null) candidates.push({ trackId, index: i, score });
      }
    }

    // Greedy assignment, highest overlap first. Ties break on track id so the
    // result never depends on Map iteration order.
    candidates.sort((a, b) => b.score - a.score || (a.trackId < b.trackId ? -1 : 1));

    const claimedTracks = new Set<TrackId>();
    const claimedDetections = new Set<number>();
    const observations: TrackObservation[] = [];

    for (const candidate of candidates) {
      if (claimedTracks.has(candidate.trackId) || claimedDetections.has(candidate.index)) continue;

      const detection = detections[candidate.index];
      const state = this.#states.get(candidate.trackId);
      if (detection === undefined || state === undefined) continue;

      claimedTracks.add(candidate.trackId);
      claimedDetections.add(candidate.index);

      this.#applyDetection(state, detection, at);
      observations.push(this.#observe(state, at, false));
    }

    // Unmatched detections start new tracks.
    for (let i = 0; i < detections.length; i += 1) {
      if (claimedDetections.has(i)) continue;
      const detection = detections[i];
      if (detection === undefined) continue;

      const state = this.#createTrack(detection, at);
      this.#states.set(state.id, state);
      if (state.confirmed) observations.push(this.#observe(state, at, false));
    }

    // Unmatched tracks coast, then expire.
    const ended: TrackId[] = [];
    for (const [trackId, state] of this.#states) {
      if (claimedTracks.has(trackId)) continue;

      if (at - state.lastDetectedAt > this.#options.maxGapMillis) {
        this.#states.delete(trackId);
        ended.push(trackId);
        continue;
      }

      if (state.confirmed) {
        // Coast: advance the box along its velocity so the track keeps a plausible
        // position through the occlusion, flagged as interpolated so nothing
        // downstream mistakes it for a real observation.
        const box = this.#predict(state, at);
        state.box = clampBox(box);
        state.lastSeen = at;
        state.position = projectDetection(this.#pose, state.box);
        this.#recordGround(state, at);
        observations.push(this.#observe(state, at, true));
      }
    }

    return { tracks: this.tracks(), observations, ended };
  }

  #applyDetection(state: TrackState, detection: Detection, at: UtcMillis): void {
    const dt = at - state.lastSeen;
    const previous = boxCenter(state.box);
    const next = boxCenter(detection.box);

    if (dt > 0) {
      // Exponential smoothing on velocity: responsive enough to follow a turn,
      // damped enough that one noisy box does not fling the prediction away.
      const instant = { x: (next.x - previous.x) / dt, y: (next.y - previous.y) / dt };
      state.velocity = {
        x: state.velocity.x * 0.6 + instant.x * 0.4,
        y: state.velocity.y * 0.6 + instant.y * 0.4,
      };
    }

    state.box = detection.box;
    state.lastSeen = at;
    state.lastDetectedAt = at;
    state.confidence = state.confidence * 0.7 + detection.confidence * 0.3;
    state.hits += 1;
    if (state.hits >= this.#options.minHitsToConfirm) state.confirmed = true;

    state.trajectory.push(next);
    if (state.trajectory.length > this.#options.maxTrajectoryPoints) state.trajectory.shift();

    state.position = projectDetection(this.#pose, detection.box);
    this.#recordGround(state, at);
  }

  #createTrack(detection: Detection, at: UtcMillis): TrackState {
    this.#sequence += 1;
    // Deterministic, readable, and unique within a camera's lifetime.
    const id = asId<TrackId>(`${this.#cameraId}:${at}:${this.#sequence}`);

    return {
      id,
      objectClass: detection.objectClass,
      firstSeen: at,
      lastSeen: at,
      lastDetectedAt: at,
      box: detection.box,
      velocity: { x: 0, y: 0 },
      confidence: detection.confidence,
      hits: 1,
      trajectory: [boxCenter(detection.box)],
      position: projectDetection(this.#pose, detection.box),
      groundHistory: [],
      confirmed: this.#options.minHitsToConfirm <= 1,
    };
  }

  /**
   * Append a ground position, keeping only the motion window.
   *
   * Camera-fallback positions are excluded: they are the camera's own location,
   * not the object's, so feeding them in would compute the speed of a stationary
   * mast and report it as the target's.
   */
  #recordGround(state: TrackState, at: UtcMillis): void {
    const position = state.position;
    if (position === null || position.source !== ('GROUND_PROJECTION' satisfies PositionSource)) {
      return;
    }

    state.groundHistory.push({ at, point: position.point });

    const cutoff = at - this.#options.motionWindowMillis;
    while (state.groundHistory.length > 2 && (state.groundHistory[0]?.at ?? at) < cutoff) {
      state.groundHistory.shift();
    }
  }

  /**
   * Ground speed and heading over the motion window.
   *
   * Null when the camera has no pose, or when the object has not moved far
   * enough to distinguish real movement from projection jitter - a heading
   * derived from noise is worse than no heading, because correlation would
   * weigh it as evidence.
   */
  #motion(state: TrackState): { speedMps: number | null; headingDegrees: number | null } {
    const first = state.groundHistory[0];
    const last = state.groundHistory[state.groundHistory.length - 1];
    if (first === undefined || last === undefined || first === last) {
      return { speedMps: null, headingDegrees: null };
    }

    const seconds = (last.at - first.at) / 1000;
    if (seconds <= 0) return { speedMps: null, headingDegrees: null };

    const distance = haversineDistance(first.point, last.point);
    const uncertainty = state.position?.radiusMeters ?? 0;
    if (distance < Math.max(1, uncertainty)) {
      return { speedMps: 0, headingDegrees: null };
    }

    return {
      speedMps: distance / seconds,
      headingDegrees: bearingDegrees(first.point, last.point),
    };
  }

  #observe(state: TrackState, at: UtcMillis, interpolated: boolean): TrackObservation {
    return {
      trackId: state.id,
      cameraId: this.#cameraId,
      at,
      box: state.box,
      confidence: state.confidence,
      position: state.position,
      interpolated,
    };
  }

  /** Every confirmed, currently-live track. */
  tracks(): readonly Track[] {
    const result: Track[] = [];
    for (const state of this.#states.values()) {
      if (!state.confirmed) continue;
      result.push(this.#toTrack(state));
    }
    return result;
  }

  track(trackId: TrackId): Track | undefined {
    const state = this.#states.get(trackId);
    return state === undefined || !state.confirmed ? undefined : this.#toTrack(state);
  }

  #toTrack(state: TrackState): Track {
    const motion = this.#motion(state);
    return {
      id: state.id,
      cameraId: this.#cameraId,
      objectClass: state.objectClass,
      firstSeen: state.firstSeen,
      lastSeen: state.lastSeen,
      confidence: state.confidence,
      observationCount: state.hits,
      currentBox: state.box,
      currentPosition: state.position,
      trajectory: [...state.trajectory],
      speedMps: motion.speedMps,
      headingDegrees: motion.headingDegrees,
      zoneIds: [],
      embedding: null,
      active: true,
    };
  }

  /** Close every live track, e.g. when a camera goes offline. */
  reset(): readonly TrackId[] {
    const ended = [...this.#states.keys()];
    this.#states.clear();
    return ended;
  }
}

/** Keep a coasting box inside the frame so it cannot drift off into nonsense. */
const clampBox = (box: BoundingBox): BoundingBox => ({
  x: Math.min(Math.max(box.x, -box.w), 1),
  y: Math.min(Math.max(box.y, -box.h), 1),
  w: box.w,
  h: box.h,
});
