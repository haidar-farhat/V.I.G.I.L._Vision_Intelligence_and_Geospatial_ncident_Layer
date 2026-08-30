import type {
  EventType,
  RiskAssessment,
  RiskContribution,
  SecurityEvent,
  Severity,
  UtcMillis,
  Zone,
  ZoneId,
} from '@sentinel/shared-types';
import { RiskFactor } from '@sentinel/shared-types';
import { withinTimeWindow } from './rules.ts';

/**
 * Explainable risk scoring.
 *
 * The score is additive and every term is retained with the reason that produced
 * it. Developers tune against the number; operators are shown the reasons. A
 * number nobody can account for is worse than no number at all - it invites
 * either blind trust or blanket dismissal, and both are failures.
 *
 * Scoring is pure: same events in, same assessment out, with no dependence on
 * wall-clock time or iteration order.
 */

/** Starting points per event type, before context. */
export const BASE_SCORES: Readonly<Record<string, number>> = Object.freeze({
  PersonEnteredZone: 20,
  PersonExitedZone: 5,
  VehicleEnteredZone: 15,
  VehicleExitedZone: 5,
  VehicleStopped: 15,
  ObjectLoitering: 25,
  LineCrossed: 20,
  AbnormalDirection: 30,
  RepeatedApproach: 35,
  CrowdDetected: 30,
  SmokeDetected: 60,
  FireSuspected: 80,
  CameraTamper: 55,
  CameraOffline: 25,
  MultipleCameraCorrelation: 10,
});

export const DEFAULT_BASE_SCORE = 10;

/** Score thresholds. Deliberately coarse: four buckets an operator can act on. */
export const SEVERITY_THRESHOLDS: Readonly<Record<Severity, number>> = Object.freeze({
  LOW: 0,
  MEDIUM: 30,
  HIGH: 55,
  CRITICAL: 80,
});

export const severityForScore = (score: number): Severity => {
  if (score >= SEVERITY_THRESHOLDS.CRITICAL) return 'CRITICAL';
  if (score >= SEVERITY_THRESHOLDS.HIGH) return 'HIGH';
  if (score >= SEVERITY_THRESHOLDS.MEDIUM) return 'MEDIUM';
  return 'LOW';
};

export type RiskContext = {
  readonly zones: ReadonlyMap<ZoneId, Zone>;
  /** Local-time window treated as "after hours" for this site. */
  readonly afterHours: { readonly startMinute: number; readonly endMinute: number } | null;
  readonly utcOffsetMinutes: number;
  /**
   * Conditions the operator has marked normal, e.g. a delivery window. Each
   * subtracts from the score, with the reason recorded so the mitigation is as
   * visible as the aggravation.
   */
  readonly knownNormalConditions?: readonly {
    readonly zoneId: ZoneId;
    readonly label: string;
    readonly points: number;
  }[];
  readonly assessedAt: UtcMillis;
};

/**
 * Score a set of correlated events as one situation.
 *
 * Scoring the group rather than each event is the point: a person seen by three
 * cameras is one situation with corroboration, not three independent risks.
 */
export const assessRisk = (
  events: readonly SecurityEvent[],
  context: RiskContext,
): RiskAssessment => {
  const contributions: RiskContribution[] = [];

  if (events.length === 0) {
    return {
      score: 0,
      severity: 'LOW',
      contributions: [],
      assessedAt: context.assessedAt,
    };
  }

  // --- base: the most serious single event anchors the score -----------------
  let base = DEFAULT_BASE_SCORE;
  let baseType: EventType = events[0]!.type;
  for (const event of events) {
    const score = BASE_SCORES[event.type] ?? DEFAULT_BASE_SCORE;
    if (score > base) {
      base = score;
      baseType = event.type;
    }
  }
  contributions.push({
    code: RiskFactor.BaseEventType,
    detail: `most serious observation: ${baseType}`,
    points: base,
  });

  // --- restricted or critical zones ------------------------------------------
  const touchedZones = [...new Set(events.flatMap((e) => e.zoneIds))]
    .map((id) => context.zones.get(id))
    .filter((zone): zone is Zone => zone !== undefined);

  const restricted = touchedZones.filter(
    (z) => z.purpose === 'RESTRICTED' || z.purpose === 'NO_ENTRY',
  );
  if (restricted.length > 0) {
    contributions.push({
      code: RiskFactor.RestrictedZone,
      detail: `restricted area: ${restricted.map((z) => z.name).join(', ')}`,
      points: 25,
    });
  }

  const criticalAssets = touchedZones.filter((z) => z.purpose === 'CRITICAL_ASSET');
  if (criticalAssets.length > 0) {
    contributions.push({
      code: RiskFactor.CriticalAssetProximity,
      detail: `protected asset: ${criticalAssets.map((z) => z.name).join(', ')}`,
      points: 20,
    });
  }

  // --- after hours -----------------------------------------------------------
  const first = events.reduce((earliest, e) => (e.occurredAt < earliest.occurredAt ? e : earliest));
  if (context.afterHours !== null) {
    const afterHours = withinTimeWindow(
      { ...context.afterHours, daysOfWeek: [] },
      first.occurredAt,
      context.utcOffsetMinutes,
    );
    if (afterHours) {
      contributions.push({
        code: RiskFactor.AfterHours,
        detail: 'occurred outside scheduled hours',
        points: 20,
      });
    }
  }

  // --- duration --------------------------------------------------------------
  const last = events.reduce((latest, e) => (e.occurredAt > latest.occurredAt ? e : latest));
  const spanSeconds = Math.round((last.occurredAt - first.occurredAt) / 1000);
  if (spanSeconds >= 60) {
    // Capped: a situation lasting an hour is not sixty times worse than one
    // lasting a minute, and an uncapped term would swamp every other factor.
    const points = Math.min(20, Math.floor(spanSeconds / 60) * 5);
    contributions.push({
      code: RiskFactor.ExtendedDuration,
      detail: `sustained for ${spanSeconds} seconds`,
      points,
    });
  }

  // --- corroboration ---------------------------------------------------------
  const cameras = new Set(events.map((e) => e.cameraId));
  if (cameras.size > 1) {
    contributions.push({
      code: RiskFactor.MultiCameraConfirmation,
      detail: `observed by ${cameras.size} cameras`,
      points: Math.min(20, (cameras.size - 1) * 10),
    });
  }

  // --- group size ------------------------------------------------------------
  const tracks = new Set(events.flatMap((e) => e.trackIds));
  if (tracks.size > 1) {
    contributions.push({
      code: RiskFactor.GroupSize,
      detail: `${tracks.size} distinct objects involved`,
      points: Math.min(15, (tracks.size - 1) * 5),
    });
  }

  // --- repetition ------------------------------------------------------------
  const byType = new Map<EventType, number>();
  for (const event of events) byType.set(event.type, (byType.get(event.type) ?? 0) + 1);
  const repeated = [...byType.entries()].filter(([, count]) => count >= 3);
  if (repeated.length > 0) {
    contributions.push({
      code: RiskFactor.RepeatedBehaviour,
      detail: repeated.map(([type, count]) => `${type} x${count}`).join(', '),
      points: 10,
    });
  }

  // --- mitigating: weak detections -------------------------------------------
  const meanConfidence = events.reduce((sum, e) => sum + e.confidence, 0) / events.length;
  if (meanConfidence < 0.6) {
    contributions.push({
      code: RiskFactor.LowDetectionConfidence,
      detail: `mean detection confidence ${(meanConfidence * 100).toFixed(0)}%`,
      points: -15,
    });
  }

  // --- mitigating: operator-declared normal conditions ------------------------
  for (const normal of context.knownNormalConditions ?? []) {
    if (touchedZones.some((z) => z.id === normal.zoneId)) {
      contributions.push({
        code: RiskFactor.KnownNormalCondition,
        detail: normal.label,
        points: -Math.abs(normal.points),
      });
    }
  }

  const raw = contributions.reduce((sum, c) => sum + c.points, 0);
  const score = Math.max(0, Math.min(100, raw));

  return {
    score,
    severity: severityForScore(score),
    contributions,
    assessedAt: context.assessedAt,
  };
};

/**
 * A short operator-facing explanation.
 *
 * Reads as prose, not as a scorecard: the operator needs to know why this is on
 * their screen, not how the arithmetic worked.
 */
export const explainRisk = (assessment: RiskAssessment): string => {
  const aggravating = assessment.contributions
    .filter((c) => c.points > 0 && c.code !== RiskFactor.BaseEventType)
    .map((c) => c.detail);
  const mitigating = assessment.contributions.filter((c) => c.points < 0).map((c) => c.detail);

  const parts: string[] = [`Assessed ${assessment.severity}`];
  if (aggravating.length > 0) parts.push(`raised by: ${aggravating.join('; ')}`);
  if (mitigating.length > 0) parts.push(`reduced by: ${mitigating.join('; ')}`);

  return `${parts.join('. ')}.`;
};
