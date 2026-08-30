import type {
  CameraId,
  EventId,
  EvidenceId,
  IncidentId,
  ModelId,
  RecordingId,
  TrackId,
  UserId,
  UtcMillis,
  ZoneId,
} from './ids.ts';
import type { IncidentStatus, Severity } from './enums.ts';
import type { PositionEstimate } from './geo.ts';

/**
 * The operator's unit of work.
 *
 * One incident aggregates every correlated event from every camera that saw the
 * same situation. This is the layer that exists to prevent alert fatigue: the
 * measure of the system is how few incidents it raises, not how many events it
 * detects.
 */
export type Incident = {
  readonly id: IncidentId;
  readonly title: string;
  readonly severity: Severity;
  readonly status: IncidentStatus;

  readonly openedAt: UtcMillis;
  readonly updatedAt: UtcMillis;
  readonly closedAt: UtcMillis | null;

  readonly position: PositionEstimate | null;
  readonly cameraIds: readonly CameraId[];
  readonly zoneIds: readonly ZoneId[];
  readonly trackIds: readonly TrackId[];
  readonly eventIds: readonly EventId[];
  readonly evidenceIds: readonly EvidenceId[];

  readonly risk: RiskAssessment;
  /** Present only once an analyst run has completed. Always evidence-bound. */
  readonly aiSummary: AiIncidentReport | null;
  readonly assessment: HumanAssessment | null;

  readonly acknowledgedBy: UserId | null;
  readonly acknowledgedAt: UtcMillis | null;
};

/** One entry on the incident's synchronised timeline. */
export type IncidentTimelineEntry = {
  readonly at: UtcMillis;
  readonly kind: 'EVENT' | 'ASSOCIATION' | 'OPERATOR_ACTION' | 'SYSTEM';
  readonly label: string;
  readonly eventId: EventId | null;
  readonly cameraId: CameraId | null;
};

/**
 * Explainable risk.
 *
 * `score` exists for developers and tuning; operators are shown `contributions`.
 * A number nobody can account for is worse than no number, so the two are never
 * separated.
 */
export type RiskAssessment = {
  readonly score: number;
  readonly severity: Severity;
  readonly contributions: readonly RiskContribution[];
  readonly assessedAt: UtcMillis;
};

export type RiskContribution = {
  readonly code: RiskFactor;
  readonly detail: string;
  /** Signed: mitigating factors are negative. */
  readonly points: number;
};

export const RiskFactor = {
  BaseEventType: 'BASE_EVENT_TYPE',
  RestrictedZone: 'RESTRICTED_ZONE',
  AfterHours: 'AFTER_HOURS',
  ExtendedDuration: 'EXTENDED_DURATION',
  MultiCameraConfirmation: 'MULTI_CAMERA_CONFIRMATION',
  RepeatedBehaviour: 'REPEATED_BEHAVIOUR',
  GroupSize: 'GROUP_SIZE',
  CriticalAssetProximity: 'CRITICAL_ASSET_PROXIMITY',
  KnownNormalCondition: 'KNOWN_NORMAL_CONDITION',
  LowDetectionConfidence: 'LOW_DETECTION_CONFIDENCE',
} as const;
export type RiskFactor = (typeof RiskFactor)[keyof typeof RiskFactor];

/** The operator's verdict. Separate from, and authoritative over, the AI's. */
export type HumanAssessment = {
  readonly by: UserId;
  readonly at: UtcMillis;
  readonly status: IncidentStatus;
  readonly reason?: string;
};

/**
 * An operator note. Append-only: notes are never edited or deleted, because an
 * incident record that can be rewritten after the fact is not evidence.
 */
export type IncidentNote = {
  readonly incidentId: IncidentId;
  readonly by: UserId;
  readonly at: UtcMillis;
  readonly text: string;
};

// ------------------------------------------------------------ AI analyst output

/**
 * The analyst's report.
 *
 * The OBSERVED / INFERRED / UNKNOWN partition is mandatory, not stylistic: an
 * operator must be able to tell at a glance which statements are recorded fact
 * and which are the model's reasoning. Every statement carries the evidence it
 * rests on.
 */
export type AiIncidentReport = {
  readonly incidentId: IncidentId;
  readonly generatedAt: UtcMillis;
  readonly modelId: ModelId;
  readonly promptVersion: string;
  /** Exactly the evidence the model was given. Nothing else was available to it. */
  readonly inputEvidenceIds: readonly EvidenceId[];
  readonly inputEventIds: readonly EventId[];

  readonly summary: string;
  readonly observed: readonly GroundedStatement[];
  readonly inferred: readonly GroundedStatement[];
  readonly unknown: readonly string[];
  readonly operatorQuestions: readonly string[];
  /** True when the model declined for lack of evidence. */
  readonly insufficientEvidence: boolean;
};

/** A statement that cannot exist without the evidence that supports it. */
export type GroundedStatement = {
  readonly text: string;
  readonly eventIds: readonly EventId[];
  readonly evidenceIds: readonly EvidenceId[];
  readonly cameraIds: readonly CameraId[];
  /** 0..1. Absent confidence is not permitted. */
  readonly confidence: number;
};

// -------------------------------------------------------------------- evidence

export const EvidenceKind = {
  VideoSegment: 'VIDEO_SEGMENT',
  Frame: 'FRAME',
  Thumbnail: 'THUMBNAIL',
  EventJson: 'EVENT_JSON',
  TrackJson: 'TRACK_JSON',
  MapSnapshot: 'MAP_SNAPSHOT',
} as const;
export type EvidenceKind = (typeof EvidenceKind)[keyof typeof EvidenceKind];

/**
 * A reference to stored evidence.
 *
 * Content-addressed by SHA-256: the same video segment referenced by five
 * incidents is stored once, and any copy can be verified against the hash long
 * after export.
 */
export type Evidence = {
  readonly id: EvidenceId;
  readonly kind: EvidenceKind;
  readonly cameraId: CameraId | null;
  readonly recordingId: RecordingId | null;
  readonly capturedAt: UtcMillis;
  readonly durationMillis: number | null;
  /** Path relative to the evidence root. Never an absolute or operator-supplied path. */
  readonly relativePath: string;
  readonly sha256: string;
  readonly sizeBytes: number;
  readonly mimeType: string;
  /** Evidence is exempt from routine retention cleanup while this is true. */
  readonly retainedForIncident: boolean;
};
