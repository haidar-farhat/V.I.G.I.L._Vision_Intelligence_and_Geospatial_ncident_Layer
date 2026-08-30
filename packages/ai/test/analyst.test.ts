import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import type {
  AiIncidentReport,
  CameraId,
  EventId,
  EvidenceId,
  IncidentId,
  SecurityEvent,
  UtcMillis,
} from '@sentinel/shared-types';
import { asId, utcMillis } from '@sentinel/shared-types';
import type { EvidenceBundle } from '../src/analyst.ts';
import {
  GuardrailError,
  INSUFFICIENT_EVIDENCE,
  analyseWithGuardrails,
  insufficientEvidenceReport,
  validateReport,
} from '../src/analyst.ts';
import {
  DETERMINISTIC_ANALYST_MODEL_ID,
  createDeterministicAnalyst,
} from '../src/deterministic-analyst.ts';

const INCIDENT = asId<IncidentId>('incident-a381');
const CAM_07 = asId<CameraId>('cam-07');
const CAM_08 = asId<CameraId>('cam-08');
const EVIDENCE_1 = asId<EvidenceId>('ev-1');

const event = (id: string, cameraId: CameraId, at: number, summary: string): SecurityEvent => ({
  id: asId<EventId>(id),
  type: 'PersonEnteredZone',
  severity: 'HIGH',
  status: 'NEW',
  occurredAt: utcMillis(at),
  recordedAt: utcMillis(at + 50),
  cameraId,
  nodeId: asId('node-1'),
  zoneIds: [asId('zone-restricted-a')],
  trackIds: [asId('track-1')],
  objectClass: 'person',
  confidence: 0.91,
  position: null,
  ruleId: asId('rule-restricted'),
  modelId: asId('model-1'),
  evidenceIds: [EVIDENCE_1],
  incidentId: INCIDENT,
  summary,
  detail: {},
});

const bundle = (events: readonly SecurityEvent[]): EvidenceBundle => ({
  incidentId: INCIDENT,
  events,
  associations: [],
  zones: [],
  cameras: [
    { id: CAM_07, name: 'Camera 07' },
    { id: CAM_08, name: 'Camera 08' },
  ],
  windowStart: utcMillis(0),
  windowEnd: utcMillis(600_000),
  evidenceIds: [EVIDENCE_1],
});

const baseReport = (overrides: Partial<AiIncidentReport> = {}): AiIncidentReport => ({
  incidentId: INCIDENT,
  generatedAt: utcMillis(1000),
  modelId: asId('model-analyst'),
  promptVersion: 'v1',
  inputEvidenceIds: [EVIDENCE_1],
  inputEventIds: [asId<EventId>('e1')],
  summary: 'Two events were recorded.',
  observed: [
    {
      text: 'Camera 07 recorded a person entering Restricted Zone A.',
      eventIds: [asId<EventId>('e1')],
      evidenceIds: [EVIDENCE_1],
      cameraIds: [CAM_07],
      confidence: 0.9,
    },
  ],
  inferred: [],
  unknown: [],
  operatorQuestions: [],
  insufficientEvidence: false,
  ...overrides,
});

describe('guardrails: evidence grounding', () => {
  const evidence = bundle([event('e1', CAM_07, 100_000, 'a person entering the zone')]);

  test('accepts a properly cited report', () => {
    assert.deepEqual(validateReport(baseReport(), evidence), []);
  });

  test('rejects a factual statement with no citation at all', () => {
    const violations = validateReport(
      baseReport({
        observed: [
          {
            text: 'Three people were present.',
            eventIds: [],
            evidenceIds: [],
            cameraIds: [],
            confidence: 0.8,
          },
        ],
      }),
      evidence,
    );

    assert.equal(violations.length, 1);
    assert.equal(violations[0]?.code, 'UNCITED_STATEMENT');
  });

  test('rejects a citation to evidence that was never supplied', () => {
    // The most dangerous failure mode: an invented citation is exactly what makes
    // a fabricated claim look credible.
    const violations = validateReport(
      baseReport({
        observed: [
          {
            text: 'Camera 07 recorded an entry.',
            eventIds: [asId<EventId>('event-that-does-not-exist')],
            evidenceIds: [],
            cameraIds: [CAM_07],
            confidence: 0.9,
          },
        ],
      }),
      evidence,
    );

    assert.equal(violations[0]?.code, 'UNKNOWN_EVIDENCE_REFERENCE');
    assert.match(violations[0]?.detail ?? '', /not in the evidence bundle/);
  });

  test('rejects a citation to a camera outside the bundle', () => {
    const violations = validateReport(
      baseReport({
        observed: [
          {
            text: 'Camera 42 recorded an entry.',
            eventIds: [asId<EventId>('e1')],
            evidenceIds: [],
            cameraIds: [asId<CameraId>('cam-42')],
            confidence: 0.9,
          },
        ],
      }),
      evidence,
    );

    assert.ok(violations.some((v) => v.code === 'UNKNOWN_EVIDENCE_REFERENCE'));
  });

  test('rejects a statement with no usable confidence', () => {
    for (const confidence of [Number.NaN, -0.1, 1.5]) {
      const violations = validateReport(
        baseReport({
          observed: [
            {
              text: 'Camera 07 recorded an entry.',
              eventIds: [asId<EventId>('e1')],
              evidenceIds: [],
              cameraIds: [CAM_07],
              confidence,
            },
          ],
        }),
        evidence,
      );
      assert.ok(
        violations.some((v) => v.code === 'MISSING_CONFIDENCE'),
        `confidence ${confidence} should be rejected`,
      );
    }
  });

  test('rejects an empty report that did not declare insufficient evidence', () => {
    const violations = validateReport(
      baseReport({ observed: [], inferred: [], insufficientEvidence: false }),
      evidence,
    );
    assert.ok(violations.some((v) => v.code === 'OBSERVATION_NOT_GROUNDED'));
  });

  test('accepts an explicit declaration of insufficient evidence', () => {
    const report = insufficientEvidenceReport(bundle([]), asId('m'), 'v1', utcMillis(0));
    assert.equal(report.summary, INSUFFICIENT_EVIDENCE);
    assert.deepEqual(validateReport(report, bundle([])), []);
  });
});

describe('guardrails: prohibited claims', () => {
  const evidence = bundle([event('e1', CAM_07, 100_000, 'a person entering the zone')]);

  const rejects = (text: string, label: string): void => {
    test(`refuses to ${label}`, () => {
      const violations = validateReport(
        baseReport({
          observed: [
            {
              text,
              eventIds: [asId<EventId>('e1')],
              evidenceIds: [],
              cameraIds: [CAM_07],
              confidence: 0.9,
            },
          ],
        }),
        evidence,
      );

      assert.ok(
        violations.some((v) => v.code === 'PROHIBITED_CLAIM'),
        `"${text}" should have been refused`,
      );
    });
  };

  rejects('The person is a criminal attempting to break in.', 'assert criminality');
  rejects('The individual was identified as Ahmad Khalil from the staff list.', 'assert identity');
  rejects('A male aged 30-40 entered the zone.', 'infer demographic traits');
  rejects('The subject ethnicity appears to be relevant here.', 'infer protected traits');
  rejects('Security should arrest the individual immediately.', 'recommend enforcement');
  rejects('I have locked the north gate in response.', 'claim a physical action');

  test('permits a plain factual observation', () => {
    const violations = validateReport(
      baseReport({
        observed: [
          {
            text: 'Three people entered Restricted Zone B after scheduled hours.',
            eventIds: [asId<EventId>('e1')],
            evidenceIds: [],
            cameraIds: [CAM_07],
            confidence: 0.9,
          },
        ],
      }),
      evidence,
    );
    assert.deepEqual(violations, [], 'describing what a sensor recorded is exactly the job');
  });

  test('scans the summary as well as the statements', () => {
    const violations = validateReport(
      baseReport({ summary: 'A burglar was detected on site.' }),
      evidence,
    );
    assert.ok(violations.some((v) => v.code === 'PROHIBITED_CLAIM'));
  });
});

describe('analyseWithGuardrails', () => {
  const evidence = bundle([event('e1', CAM_07, 100_000, 'a person entering the zone')]);

  test('returns a valid report unchanged', async () => {
    const engine = {
      modelId: asId('m'),
      promptVersion: 'v1',
      analyse: async (): Promise<AiIncidentReport> => baseReport(),
    };

    const report = await analyseWithGuardrails(engine, evidence);
    assert.equal(report.summary, 'Two events were recorded.');
  });

  test('throws rather than returning an invalid report', async () => {
    const engine = {
      modelId: asId('m'),
      promptVersion: 'v1',
      analyse: async (): Promise<AiIncidentReport> =>
        baseReport({
          observed: [
            {
              text: 'The intruder is a known criminal.',
              eventIds: [],
              evidenceIds: [],
              cameraIds: [],
              confidence: 0.99,
            },
          ],
        }),
    };

    await assert.rejects(
      () => analyseWithGuardrails(engine, evidence),
      (error: unknown) => {
        assert.ok(error instanceof GuardrailError);
        assert.equal(error.code, 'AI_GUARDRAIL_VIOLATION');
        // Both the uncited claim and the prohibited claim are reported, so a
        // failing model can be diagnosed in one pass.
        assert.ok(error.violations.length >= 2);
        return true;
      },
    );
  });
});

describe('deterministic analyst', () => {
  const now = (): UtcMillis => utcMillis(999_000);

  test('produces a report that passes its own guardrails', async () => {
    const evidence = bundle([
      event('e1', CAM_07, 100_000, 'a person entering the zone'),
      event('e2', CAM_08, 260_000, 'a person entering the zone'),
    ]);

    const report = await createDeterministicAnalyst(now).analyse(evidence);
    assert.deepEqual(
      validateReport(report, evidence),
      [],
      'the reference implementation must satisfy the validator it is tested against',
    );
  });

  test('cites every observation and records its provenance', async () => {
    const evidence = bundle([event('e1', CAM_07, 100_000, 'a person entering the zone')]);
    const report = await createDeterministicAnalyst(now).analyse(evidence);

    assert.equal(report.modelId, DETERMINISTIC_ANALYST_MODEL_ID);
    assert.equal(report.promptVersion, 'deterministic-v1');
    assert.deepEqual(report.inputEventIds, [asId<EventId>('e1')]);

    for (const statement of report.observed) {
      assert.ok(
        statement.eventIds.length > 0 || statement.cameraIds.length > 0,
        'every observation cites its source',
      );
    }
  });

  test('separates what was observed from what is inferred', async () => {
    const evidence = bundle([
      event('e1', CAM_07, 100_000, 'a person entering the zone'),
      event('e2', CAM_08, 260_000, 'a person entering the zone'),
    ]);

    const report = await createDeterministicAnalyst(now).analyse(evidence);

    assert.equal(report.observed.length, 2, 'one observation per recorded event');
    assert.ok(report.inferred.length > 0, 'multi-camera activity is an inference, not a fact');
    assert.ok(
      report.inferred[0]!.confidence < 1,
      'an inference must never be stated with full confidence',
    );
  });

  test('states what it does not know rather than staying silent', async () => {
    const evidence = bundle([
      event('e1', CAM_07, 100_000, 'a person entering the zone'),
      event('e2', CAM_08, 260_000, 'a person entering the zone'),
    ]);

    const report = await createDeterministicAnalyst(now).analyse(evidence);

    assert.ok(report.unknown.length > 0);
    assert.ok(
      report.unknown.some((u) => /identity|intent|purpose/i.test(u)),
      'identity and intent are always explicitly disclaimed',
    );
    assert.ok(
      report.unknown.some((u) => /same objects/i.test(u)),
      'with no association, cross-camera sameness is explicitly unknown',
    );
  });

  test('declares insufficient evidence for an empty window', async () => {
    const report = await createDeterministicAnalyst(now).analyse(bundle([]));

    assert.equal(report.insufficientEvidence, true);
    assert.equal(report.summary, INSUFFICIENT_EVIDENCE);
    assert.deepEqual(report.observed, []);
  });

  test('is deterministic, so an incident report is reproducible', async () => {
    const evidence = bundle([
      event('e1', CAM_07, 100_000, 'a person entering the zone'),
      event('e2', CAM_08, 260_000, 'a person entering the zone'),
    ]);

    const a = await createDeterministicAnalyst(now).analyse(evidence);
    const b = await createDeterministicAnalyst(now).analyse(evidence);
    assert.deepEqual(a, b);
  });

  test('orders observations chronologically regardless of input order', async () => {
    const later = event('e2', CAM_08, 260_000, 'a person entering the zone');
    const earlier = event('e1', CAM_07, 100_000, 'a person entering the zone');

    const report = await createDeterministicAnalyst(now).analyse(bundle([later, earlier]));

    assert.match(report.observed[0]!.text, /Camera 07/);
    assert.match(report.observed[1]!.text, /Camera 08/);
  });
});
