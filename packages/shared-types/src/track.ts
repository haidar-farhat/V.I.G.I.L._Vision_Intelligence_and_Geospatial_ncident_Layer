import type { CameraId, ModelId, TrackId, UtcMillis, ZoneId } from './ids.ts';
import type { DetectedClass } from './enums.ts';
import type { BoundingBox, PositionEstimate, Vec2 } from './geo.ts';

/** A single model output for one frame. The rawest unit the system keeps. */
export type Detection = {
  readonly cameraId: CameraId;
  /** Time the frame was captured, as reported by the pipeline (UTC). */
  readonly frameAt: UtcMillis;
  readonly objectClass: DetectedClass;
  readonly confidence: number;
  readonly box: BoundingBox;
  /** Which model produced this. Every detection is traceable to an artifact. */
  readonly modelId: ModelId;
};

/** One frame's worth of a track: where it was and how sure we are. */
export type TrackObservation = {
  readonly trackId: TrackId;
  readonly cameraId: CameraId;
  readonly at: UtcMillis;
  readonly box: BoundingBox;
  readonly confidence: number;
  /** Map position, present only once the camera has a pose. */
  readonly position: PositionEstimate | null;
  /** Whether a detection actually landed this frame, or the track coasted. */
  readonly interpolated: boolean;
};

/**
 * A persistent object identity within one camera.
 *
 * Tracks survive short detection gaps: a person walking behind a pillar is the
 * same track, not two. Cross-camera identity is never asserted here - that is a
 * scored association, held separately (see `TrackAssociation`).
 */
export type Track = {
  readonly id: TrackId;
  readonly cameraId: CameraId;
  readonly objectClass: DetectedClass;
  readonly firstSeen: UtcMillis;
  readonly lastSeen: UtcMillis;
  readonly confidence: number;
  readonly observationCount: number;

  readonly currentBox: BoundingBox;
  readonly currentPosition: PositionEstimate | null;

  /** Recent path in normalised image space, oldest first. */
  readonly trajectory: readonly Vec2[];
  /** Metres per second over the ground, null without a camera pose. */
  readonly speedMps: number | null;
  /** Compass bearing of travel in degrees, null when stationary or unposed. */
  readonly headingDegrees: number | null;

  readonly zoneIds: readonly ZoneId[];
  /** Optional appearance vector for cross-camera matching. Not biometric. */
  readonly embedding: readonly number[] | null;
  readonly active: boolean;
};

/**
 * A scored hypothesis that two tracks on different cameras are the same object.
 *
 * The system never claims certainty here. `score` and `reasons` are surfaced to
 * the operator together, so a 62% association reads as a suggestion rather than
 * a fact.
 */
export type TrackAssociation = {
  readonly fromTrackId: TrackId;
  readonly toTrackId: TrackId;
  readonly fromCameraId: CameraId;
  readonly toCameraId: CameraId;
  readonly departedAt: UtcMillis;
  readonly arrivedAt: UtcMillis;
  /** 0..1. */
  readonly score: number;
  readonly reasons: readonly AssociationReason[];
};

/** One weighted contribution to an association score, kept for explainability. */
export type AssociationReason = {
  readonly code: AssociationReasonCode;
  readonly detail: string;
  readonly contribution: number;
};

export const AssociationReasonCode = {
  ClassMatch: 'CLASS_MATCH',
  ClassMismatch: 'CLASS_MISMATCH',
  TravelTimePlausible: 'TRAVEL_TIME_PLAUSIBLE',
  TravelTimeImplausible: 'TRAVEL_TIME_IMPLAUSIBLE',
  TopologyEdgeKnown: 'TOPOLOGY_EDGE_KNOWN',
  TopologyEdgeUnknown: 'TOPOLOGY_EDGE_UNKNOWN',
  DirectionConsistent: 'DIRECTION_CONSISTENT',
  DirectionInconsistent: 'DIRECTION_INCONSISTENT',
  AppearanceSimilar: 'APPEARANCE_SIMILAR',
  AppearanceDissimilar: 'APPEARANCE_DISSIMILAR',
} as const;
export type AssociationReasonCode =
  (typeof AssociationReasonCode)[keyof typeof AssociationReasonCode];
