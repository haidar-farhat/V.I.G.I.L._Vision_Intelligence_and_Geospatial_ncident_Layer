import type {
  AiIncidentReport,
  CameraId,
  EventId,
  EvidenceId,
  GroundedStatement,
  IncidentId,
  ModelId,
  SecurityEvent,
  TrackAssociation,
  UtcMillis,
  Zone,
} from '@sentinel/shared-types';

/**
 * The AI analyst.
 *
 * Its role is to summarise, correlate, explain and answer questions about
 * recorded events. It is explicitly *not* permitted to identify people, infer
 * personal traits, assert criminality, or command anything physical. It is an
 * analyst writing a briefing, not an authority issuing a verdict.
 *
 * The design constraint that makes this trustworthy: the model only ever receives
 * a structured evidence bundle, and every statement it returns must cite the
 * evidence it rests on. A claim with no citation is not a weaker claim - it is
 * rejected. That is checked here, in code, rather than requested in a prompt,
 * because a prompt is a preference and a validator is a guarantee.
 */

/** Exactly what the model is given. It has access to nothing else. */
export type EvidenceBundle = {
  readonly incidentId: IncidentId;
  readonly events: readonly SecurityEvent[];
  readonly associations: readonly TrackAssociation[];
  readonly zones: readonly Zone[];
  readonly cameras: readonly { id: CameraId; name: string }[];
  readonly windowStart: UtcMillis;
  readonly windowEnd: UtcMillis;
  readonly evidenceIds: readonly EvidenceId[];
};

/** The exact response when the evidence does not support a conclusion. */
export const INSUFFICIENT_EVIDENCE = 'Insufficient evidence.';

export type AnalystEngine = {
  readonly modelId: ModelId;
  readonly promptVersion: string;
  /** Deterministic settings are required: an incident report must be reproducible. */
  analyse(bundle: EvidenceBundle): Promise<AiIncidentReport>;
};

// ------------------------------------------------------------------ guardrails

export type GuardrailViolation = {
  readonly code: GuardrailCode;
  readonly detail: string;
  readonly statement?: string;
};

export const GuardrailCode = {
  UncitedStatement: 'UNCITED_STATEMENT',
  UnknownEvidenceReference: 'UNKNOWN_EVIDENCE_REFERENCE',
  MissingConfidence: 'MISSING_CONFIDENCE',
  ProhibitedClaim: 'PROHIBITED_CLAIM',
  ObservationNotGrounded: 'OBSERVATION_NOT_GROUNDED',
} as const;
export type GuardrailCode = (typeof GuardrailCode)[keyof typeof GuardrailCode];

/**
 * Language that asserts identity, criminality or protected traits.
 *
 * The system describes what a sensor recorded: "three people entered Restricted
 * Zone B after hours". It must never produce "the intruder is a known thief" or
 * anything resembling a judgement about who someone is. This list is a safety
 * net over the prompt, not a substitute for it.
 */
const PROHIBITED_PATTERNS: readonly { pattern: RegExp; reason: string }[] = Object.freeze([
  { pattern: /\b(criminal|thief|burglar|perpetrator|suspect is|culprit)\b/i, reason: 'asserts criminality' },
  { pattern: /\b(identified as|recognised as|recognized as|known to be)\s+[A-Z]/, reason: 'asserts identity' },
  { pattern: /\b(male|female|man|woman|boy|girl)\s+aged\b/i, reason: 'infers demographic traits' },
  { pattern: /\b(ethnicity|race|religion|nationality|gender identity)\b/i, reason: 'infers protected traits' },
  { pattern: /\b(arrest|detain|apprehend|prosecute)\b/i, reason: 'recommends enforcement action' },
  { pattern: /\b(I have|system has)\s+(locked|unlocked|activated|dispatched)\b/i, reason: 'claims a physical action' },
]);

/**
 * Validate a report before it is ever shown to an operator or stored.
 *
 * Returns every violation rather than the first, so a failing model can be
 * diagnosed in one pass.
 */
export const validateReport = (
  report: AiIncidentReport,
  bundle: EvidenceBundle,
): readonly GuardrailViolation[] => {
  const violations: GuardrailViolation[] = [];

  const knownEvents = new Set<EventId>(bundle.events.map((e) => e.id));
  const knownEvidence = new Set<EvidenceId>(bundle.evidenceIds);
  const knownCameras = new Set<CameraId>(bundle.cameras.map((c) => c.id));

  const checkProhibited = (text: string): void => {
    for (const { pattern, reason } of PROHIBITED_PATTERNS) {
      if (pattern.test(text)) {
        violations.push({
          code: GuardrailCode.ProhibitedClaim,
          detail: reason,
          statement: text,
        });
      }
    }
  };

  const checkStatement = (statement: GroundedStatement, requireCitation: boolean): void => {
    checkProhibited(statement.text);

    const cites =
      statement.eventIds.length > 0 ||
      statement.evidenceIds.length > 0 ||
      statement.cameraIds.length > 0;

    if (requireCitation && !cites) {
      violations.push({
        code: GuardrailCode.UncitedStatement,
        detail: 'a factual statement carries no evidence reference',
        statement: statement.text,
      });
    }

    if (!Number.isFinite(statement.confidence) || statement.confidence < 0 || statement.confidence > 1) {
      violations.push({
        code: GuardrailCode.MissingConfidence,
        detail: `confidence must be within 0..1, got ${String(statement.confidence)}`,
        statement: statement.text,
      });
    }

    // A citation must point at evidence that was actually supplied. A model that
    // invents a plausible-looking event id is the single most dangerous failure
    // mode here, because the citation is exactly what makes the claim credible.
    for (const eventId of statement.eventIds) {
      if (!knownEvents.has(eventId)) {
        violations.push({
          code: GuardrailCode.UnknownEvidenceReference,
          detail: `cites event ${String(eventId)}, which was not in the evidence bundle`,
          statement: statement.text,
        });
      }
    }
    for (const evidenceId of statement.evidenceIds) {
      if (!knownEvidence.has(evidenceId)) {
        violations.push({
          code: GuardrailCode.UnknownEvidenceReference,
          detail: `cites evidence ${String(evidenceId)}, which was not in the evidence bundle`,
          statement: statement.text,
        });
      }
    }
    for (const cameraId of statement.cameraIds) {
      if (!knownCameras.has(cameraId)) {
        violations.push({
          code: GuardrailCode.UnknownEvidenceReference,
          detail: `cites camera ${String(cameraId)}, which was not in the evidence bundle`,
          statement: statement.text,
        });
      }
    }
  };

  checkProhibited(report.summary);

  // Observations are recorded fact and must always cite. Inferences are the
  // model's reasoning and must cite whatever they were reasoned from.
  for (const statement of report.observed) checkStatement(statement, true);
  for (const statement of report.inferred) checkStatement(statement, true);
  for (const text of report.unknown) checkProhibited(text);

  if (report.observed.length === 0 && report.inferred.length === 0 && !report.insufficientEvidence) {
    violations.push({
      code: GuardrailCode.ObservationNotGrounded,
      detail: 'report contains no grounded statements and did not declare insufficient evidence',
    });
  }

  return violations;
};

/** The report returned when evidence does not support any conclusion. */
export const insufficientEvidenceReport = (
  bundle: EvidenceBundle,
  modelId: ModelId,
  promptVersion: string,
  generatedAt: UtcMillis,
): AiIncidentReport => ({
  incidentId: bundle.incidentId,
  generatedAt,
  modelId,
  promptVersion,
  inputEvidenceIds: bundle.evidenceIds,
  inputEventIds: bundle.events.map((e) => e.id),
  summary: INSUFFICIENT_EVIDENCE,
  observed: [],
  inferred: [],
  unknown: ['No events were available for the requested window.'],
  operatorQuestions: [],
  insufficientEvidence: true,
});

export class GuardrailError extends Error {
  readonly code = 'AI_GUARDRAIL_VIOLATION';
  readonly violations: readonly GuardrailViolation[];
  readonly recoverable = true;

  constructor(violations: readonly GuardrailViolation[]) {
    super(
      `AI report rejected by ${violations.length} guardrail violation(s): ` +
        violations.map((v) => `${v.code} (${v.detail})`).join('; '),
    );
    this.name = 'GuardrailError';
    this.violations = violations;
  }
}

/**
 * Run an analyst and refuse to return anything that violates the guardrails.
 *
 * The failure is deliberately visible: an operator seeing "the analyst produced a
 * report that failed validation" is safe, whereas an operator quietly shown an
 * unvalidated report is not.
 */
export const analyseWithGuardrails = async (
  engine: AnalystEngine,
  bundle: EvidenceBundle,
): Promise<AiIncidentReport> => {
  const report = await engine.analyse(bundle);
  const violations = validateReport(report, bundle);
  if (violations.length > 0) throw new GuardrailError(violations);
  return report;
};
