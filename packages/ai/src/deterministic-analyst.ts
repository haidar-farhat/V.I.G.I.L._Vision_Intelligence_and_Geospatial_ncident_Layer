import type { AiIncidentReport, GroundedStatement, ModelId, UtcMillis } from '@sentinel/shared-types';
import { asId } from '@sentinel/shared-types';
import type { AnalystEngine, EvidenceBundle } from './analyst.ts';
import { INSUFFICIENT_EVIDENCE, insufficientEvidenceReport } from './analyst.ts';

/**
 * A grounded analyst that needs no language model at all.
 *
 * This exists for two reasons, and both matter more than they might appear.
 *
 * First, graceful degradation. A local LLM is optional - many deployments will
 * never install one, and a machine with no spare VRAM should still hand the
 * operator a readable incident narrative. This engine always works.
 *
 * Second, it is grounded *by construction*. Every sentence is assembled from
 * fields of the evidence bundle, so it is structurally incapable of asserting
 * something the evidence does not contain. That makes it the reference
 * implementation the guardrail suite tests against, and the baseline any
 * LLM-backed engine has to beat while satisfying the same validator.
 *
 * It is not a language model and does not pretend to be one: the writing is
 * templated, and the UI labels it as an automatically assembled summary.
 */

export const DETERMINISTIC_ANALYST_MODEL_ID = asId<ModelId>('builtin:deterministic-analyst');
export const DETERMINISTIC_ANALYST_PROMPT_VERSION = 'deterministic-v1';

const pad = (value: number): string => String(value).padStart(2, '0');

/** UTC clock time. Display-layer localisation happens in the UI, not here. */
const clock = (at: UtcMillis): string => {
  const d = new Date(at);
  return `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}:${pad(d.getUTCSeconds())}`;
};

const plural = (count: number, singular: string, pluralForm: string): string =>
  `${count} ${count === 1 ? singular : pluralForm}`;

export const createDeterministicAnalyst = (
  now: () => UtcMillis = () => Date.now() as UtcMillis,
): AnalystEngine => ({
  modelId: DETERMINISTIC_ANALYST_MODEL_ID,
  promptVersion: DETERMINISTIC_ANALYST_PROMPT_VERSION,

  analyse: async (bundle: EvidenceBundle): Promise<AiIncidentReport> => {
    const generatedAt = now();

    if (bundle.events.length === 0) {
      return insufficientEvidenceReport(
        bundle,
        DETERMINISTIC_ANALYST_MODEL_ID,
        DETERMINISTIC_ANALYST_PROMPT_VERSION,
        generatedAt,
      );
    }

    const events = [...bundle.events].sort((a, b) => a.occurredAt - b.occurredAt);
    const cameraName = new Map(bundle.cameras.map((c) => [c.id, c.name]));
    const zoneName = new Map(bundle.zones.map((z) => [z.id, z.name]));

    // --- OBSERVED: one statement per event, each citing exactly its own source.
    const observed: GroundedStatement[] = events.map((event) => {
      const camera = cameraName.get(event.cameraId) ?? String(event.cameraId);
      const zones = event.zoneIds
        .map((id) => zoneName.get(id) ?? String(id))
        .filter((name) => name.length > 0);

      const where = zones.length > 0 ? ` in ${zones.join(' and ')}` : '';
      return {
        text: `At ${clock(event.occurredAt)} UTC, ${camera} recorded ${event.summary}${where}.`,
        eventIds: [event.id],
        evidenceIds: event.evidenceIds.filter((id) => bundle.evidenceIds.includes(id)),
        cameraIds: [event.cameraId],
        confidence: event.confidence,
      };
    });

    // --- INFERRED: only what follows from more than one piece of evidence.
    const inferred: GroundedStatement[] = [];
    const unknown: string[] = [];

    const cameras = [...new Set(events.map((e) => e.cameraId))];
    if (cameras.length > 1) {
      inferred.push({
        text:
          `Activity was recorded by ${plural(cameras.length, 'camera', 'cameras')} ` +
          `(${cameras.map((id) => cameraName.get(id) ?? String(id)).join(', ')}) ` +
          `between ${clock(events[0]!.occurredAt)} and ` +
          `${clock(events[events.length - 1]!.occurredAt)} UTC, ` +
          'which is consistent with movement across the site rather than a single fixed location.',
        eventIds: events.map((e) => e.id),
        evidenceIds: [],
        cameraIds: cameras,
        // Multi-camera agreement is corroboration, but the objects have not been
        // proven to be the same ones - that is what the associations below say.
        confidence: 0.7,
      });
    }

    for (const association of bundle.associations) {
      const from = cameraName.get(association.fromCameraId) ?? String(association.fromCameraId);
      const to = cameraName.get(association.toCameraId) ?? String(association.toCameraId);
      const reasons = association.reasons
        .filter((r) => r.contribution > 0)
        .map((r) => r.detail)
        .join('; ');

      inferred.push({
        text:
          `A track leaving ${from} at ${clock(association.departedAt)} UTC may be the same object ` +
          `as one appearing on ${to} at ${clock(association.arrivedAt)} UTC ` +
          `(association ${Math.round(association.score * 100)}%: ${reasons}).`,
        eventIds: [],
        evidenceIds: [],
        cameraIds: [association.fromCameraId, association.toCameraId],
        confidence: association.score,
      });
    }

    // --- UNKNOWN: stated explicitly, because silence reads as certainty.
    if (bundle.associations.length === 0 && cameras.length > 1) {
      unknown.push(
        'Whether the objects seen by different cameras are the same objects. ' +
          'No cross-camera association met the confidence threshold.',
      );
    }
    if (events.some((e) => e.position === null)) {
      unknown.push(
        'Precise ground positions for some observations. The cameras involved are ' +
          'not calibrated, so those detections are located only to the camera itself.',
      );
    }
    unknown.push('The identity, intent or purpose of any person or vehicle observed.');

    const operatorQuestions = [
      'Was any activity in this window expected, such as scheduled maintenance or a patrol?',
      cameras.length > 1
        ? 'Do the camera views support these observations being the same group?'
        : 'Does the recorded footage support this observation?',
    ];

    const first = events[0]!;
    const last = events[events.length - 1]!;
    const durationSeconds = Math.round((last.occurredAt - first.occurredAt) / 1000);

    const summary =
      `${plural(events.length, 'event was', 'events were')} recorded across ` +
      `${plural(cameras.length, 'camera', 'cameras')} over ${durationSeconds} seconds, ` +
      `beginning at ${clock(first.occurredAt)} UTC. ` +
      `The highest severity recorded was ${
        events.reduce((worst, e) => (e.severity === 'CRITICAL' ? e.severity : worst), first.severity)
      }.`;

    return {
      incidentId: bundle.incidentId,
      generatedAt,
      modelId: DETERMINISTIC_ANALYST_MODEL_ID,
      promptVersion: DETERMINISTIC_ANALYST_PROMPT_VERSION,
      inputEvidenceIds: bundle.evidenceIds,
      inputEventIds: events.map((e) => e.id),
      summary,
      observed,
      inferred,
      unknown,
      operatorQuestions,
      insufficientEvidence: false,
    };
  },
});

export { INSUFFICIENT_EVIDENCE };
