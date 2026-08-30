import type {
  CameraId,
  EventId,
  EvidenceId,
  IncidentId,
  ModelId,
  NodeId,
  RuleId,
  TrackId,
  UtcMillis,
  ZoneId,
} from './ids.ts';
import type {
  CrossingDirection,
  DetectedClass,
  EventStatus,
  EventType,
  Severity,
} from './enums.ts';
import type { PositionEstimate } from './geo.ts';

/**
 * A meaningful occurrence, produced by a rule from zone observations.
 *
 * An event is not an alert. Most events are never shown to anyone directly; they
 * are raw material for correlation. Alerting happens at the incident layer, which
 * is what keeps three cameras seeing one person from becoming three alerts.
 */
export type SecurityEvent = {
  readonly id: EventId;
  readonly type: EventType;
  readonly severity: Severity;
  readonly status: EventStatus;

  /** When the event occurred (UTC), as opposed to when it was recorded. */
  readonly occurredAt: UtcMillis;
  /** When the control node durably accepted it. Clock skew is preserved, not hidden. */
  readonly recordedAt: UtcMillis;

  readonly cameraId: CameraId;
  readonly nodeId: NodeId;
  readonly zoneIds: readonly ZoneId[];
  readonly trackIds: readonly TrackId[];
  readonly objectClass: DetectedClass | null;

  /** 0..1, propagated from detection and rule confidence. */
  readonly confidence: number;
  readonly position: PositionEstimate | null;

  /** Rule that fired. Null for system events such as CameraOffline. */
  readonly ruleId: RuleId | null;
  /** Model that produced the underlying detections, when there were any. */
  readonly modelId: ModelId | null;

  readonly evidenceIds: readonly EvidenceId[];
  readonly incidentId: IncidentId | null;

  /** Human-readable statement of what was observed. Never a judgement. */
  readonly summary: string;
  /** Structured detail; must never contain credentials or raw frames. */
  readonly detail: Readonly<Record<string, string | number | boolean | null>>;
};

/**
 * Deterministic identity input for an event.
 *
 * A worker that reconnects after an outage replays its buffer. Deriving the event
 * id from these fields makes the replay an idempotent upsert instead of a
 * duplicate, and makes reconciliation independent of arrival order.
 */
export type EventIdentity = {
  readonly nodeId: NodeId;
  readonly cameraId: CameraId;
  readonly ruleId: RuleId | null;
  readonly type: EventType;
  readonly trackId: TrackId | null;
  /** Occurrence time quantised to the rule's dedup bucket. */
  readonly timeBucket: number;
};

// ------------------------------------------------------------------ rules

/**
 * Rule condition. Structured data, never executable code: a rule arriving from
 * the database or an import can be inspected, diffed and audited, and can never
 * be a code-execution vector.
 */
export type RuleCondition = {
  readonly objectClasses: readonly DetectedClass[];
  readonly zoneIds: readonly ZoneId[];
  readonly cameraIds: readonly CameraId[];
  /** Local-time window, inclusive start, exclusive end. Crosses midnight when start > end. */
  readonly timeWindow: TimeWindow | null;
  readonly direction: CrossingDirection;
  /** Minimum continuous dwell before the rule fires, in milliseconds. */
  readonly minDurationMillis: number;
  readonly confidenceThreshold: number;
  /** Suppression window after firing, per track, in milliseconds. */
  readonly cooldownMillis: number;
  /** Minimum simultaneous tracks required, for group and crowd rules. */
  readonly minTrackCount: number;
};

export type TimeWindow = {
  /** Minutes from local midnight, 0..1439. */
  readonly startMinute: number;
  readonly endMinute: number;
  /** 0 = Sunday. Empty means every day. */
  readonly daysOfWeek: readonly number[];
};

export type RuleAction = {
  readonly eventType: EventType;
  readonly severity: Severity;
  readonly notify: boolean;
  readonly createIncident: boolean;
};

export type Rule = {
  readonly id: RuleId;
  readonly name: string;
  readonly description?: string;
  readonly enabled: boolean;
  readonly when: RuleCondition;
  readonly then: RuleAction;
  readonly createdAt: UtcMillis;
  readonly updatedAt: UtcMillis;
};
