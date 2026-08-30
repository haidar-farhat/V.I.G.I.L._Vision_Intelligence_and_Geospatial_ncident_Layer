#!/usr/bin/env node
/**
 * Run the vertical slice and print what the operator would see.
 *
 *   node scripts/dev.mjs slice
 *
 * Every stage is the production component; only the camera and detector are
 * simulated. Runs with the network cable unplugged.
 */

import process from 'node:process';
import { perimeterIntrusionScenario, quietScenario } from '../simulator/src/scenario.ts';
import { runScenario } from '../simulator/src/run.ts';

const pad = (n) => String(n).padStart(2, '0');
const clock = (ms) => {
  const d = new Date(ms);
  return `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}:${pad(d.getUTCSeconds())}`;
};

const rule = (label = '') =>
  console.log(label === '' ? '─'.repeat(78) : `── ${label} ${'─'.repeat(Math.max(0, 74 - label.length))}`);

const which = process.argv[2] === 'quiet' ? quietScenario() : perimeterIntrusionScenario();

console.log('');
rule('SENTINEL VISION - VERTICAL SLICE');
console.log(`scenario   ${which.name}   seed ${which.seed}`);
console.log(`            ${which.description}`);
console.log('');

const started = Date.now();
const result = await runScenario(which, { analyse: true });
const wall = Date.now() - started;

rule('PIPELINE');
console.log(`frames processed     ${result.frames}`);
console.log(`detections           ${result.detections}`);
console.log(`wall clock           ${wall} ms`);
console.log('');
console.log('tracks per camera');
for (const [cameraId, count] of result.tracksByCamera) {
  console.log(`  ${String(cameraId).padEnd(10)} ${count}`);
}
console.log('');

rule('SPATIAL ACCURACY vs GROUND TRUTH');
const e = result.positionError;
console.log(`samples              ${e.samples}`);
console.log(`mean error           ${e.meanMeters.toFixed(2)} m`);
console.log(`p95 error            ${e.p95Meters.toFixed(2)} m`);
console.log(`max error            ${e.maxMeters.toFixed(2)} m`);
console.log(`within stated 2-sigma ${(e.withinStatedUncertainty * 100).toFixed(1)} %`);
console.log('');

rule('CROSS-CAMERA ASSOCIATIONS');
if (result.associations.length === 0) {
  console.log('  none scored above threshold');
}
for (const a of result.associations) {
  console.log(
    `  ${String(a.fromCameraId)} -> ${String(a.toCameraId)}  ` +
      `${clock(a.departedAt)} -> ${clock(a.arrivedAt)}  ${(a.score * 100).toFixed(0)}%`,
  );
  for (const reason of a.reasons) {
    console.log(`      ${reason.code.padEnd(24)} ${reason.detail}`);
  }
}
console.log('');

rule('EVENTS');
console.log(`  ${result.events.length} events generated`);
const byType = new Map();
for (const ev of result.events) byType.set(ev.type, (byType.get(ev.type) ?? 0) + 1);
for (const [type, count] of byType) console.log(`    ${type.padEnd(28)} ${count}`);
console.log('');

rule('INCIDENTS');
console.log(`  ${result.incidents.length} incident(s) from ${result.events.length} events`);
console.log('');

for (const incident of result.incidents) {
  console.log(`  ${incident.id}  [${incident.severity}]  ${incident.title}`);
  console.log(`    opened   ${clock(incident.openedAt)} UTC`);
  console.log(`    cameras  ${incident.cameraIds.join(', ')}`);
  console.log(`    events   ${incident.eventIds.length}`);
  console.log('');
  console.log(`    RISK ${incident.risk.score}/100`);
  for (const c of incident.risk.contributions) {
    const sign = c.points >= 0 ? '+' : '';
    console.log(`      ${sign}${String(c.points).padStart(3)}  ${c.code.padEnd(28)} ${c.detail}`);
  }
  console.log('');

  const timeline = result.timelines.get(String(incident.id)) ?? [];
  console.log('    TIMELINE');
  for (const entry of timeline.slice(0, 14)) {
    console.log(`      ${clock(entry.at)}  ${entry.kind.padEnd(12)} ${entry.label}`);
  }
  if (timeline.length > 14) console.log(`      ... ${timeline.length - 14} more`);
  console.log('');

  const report = result.reports.get(String(incident.id));
  if (report !== undefined) {
    console.log('    AI ANALYST  (evidence-bound, guardrails enforced)');
    console.log(`      model ${report.modelId}  prompt ${report.promptVersion}`);
    console.log('');
    console.log(`      SUMMARY: ${report.summary}`);
    console.log('');
    console.log('      OBSERVED');
    for (const s of report.observed.slice(0, 6)) {
      console.log(`        - ${s.text}`);
      console.log(`          evidence: ${[...s.eventIds, ...s.cameraIds].join(', ')}`);
    }
    if (report.observed.length > 6) {
      console.log(`        ... ${report.observed.length - 6} more`);
    }
    console.log('');
    console.log('      INFERRED');
    for (const s of report.inferred) {
      console.log(`        - ${s.text}  [confidence ${(s.confidence * 100).toFixed(0)}%]`);
    }
    console.log('');
    console.log('      UNKNOWN');
    for (const u of report.unknown) console.log(`        - ${u}`);
    console.log('');
    console.log('      OPERATOR QUESTIONS');
    for (const q of report.operatorQuestions) console.log(`        - ${q}`);
    console.log('');
  }
  rule();
}

console.log('');
console.log('This entire run required no network access of any kind.');
console.log('');
