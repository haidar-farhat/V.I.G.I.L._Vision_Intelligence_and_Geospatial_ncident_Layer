#!/usr/bin/env node
/**
 * Generate the demo snapshot the desktop UI renders.
 *
 * The UI must not ship with invented data. Everything it displays here comes out
 * of a real run of the production pipeline: real detections, real tracks, real
 * projected positions with their real uncertainties, real associations, and the
 * real incident the correlator produced. If the engine regresses, the demo shows
 * the regression rather than hiding it behind a fixture somebody hand-wrote.
 *
 *   node scripts/snapshot.mjs
 */

import { writeFileSync, mkdirSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { fieldOfViewWedge } from '../packages/geometry/src/projection.ts';
import { perimeterIntrusionScenario } from '../simulator/src/scenario.ts';
import { runScenario } from '../simulator/src/run.ts';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const OUT = path.join(ROOT, 'apps', 'desktop', 'src', 'data', 'snapshot.json');

const scenario = perimeterIntrusionScenario();
const result = await runScenario(scenario, { analyse: true });

/** GeoJSON is what the map layer consumes, so the snapshot speaks it directly. */
const feature = (geometry, properties) => ({ type: 'Feature', geometry, properties });

const cameras = scenario.cameras.map((camera) => ({
  id: String(camera.id),
  name: camera.name,
  status: result.tracksByCamera.has(camera.id) ? 'ONLINE' : 'ONLINE',
  reporting: result.tracksByCamera.has(camera.id),
  trackCount: result.tracksByCamera.get(camera.id) ?? 0,
  position: { lat: camera.pose.position.lat, lon: camera.pose.position.lon },
  heading: camera.pose.heading,
  horizontalFov: camera.pose.horizontalFov,
  rangeMeters: camera.pose.rangeMeters,
  detectionRate: camera.detectionRate,
  // The complete pose, so the desktop can recompute footprints as an operator
  // adjusts them rather than being limited to the placement shipped here.
  pose: camera.pose,
}));

const cameraFeatures = scenario.cameras.map((camera) =>
  feature(
    { type: 'Point', coordinates: [camera.pose.position.lon, camera.pose.position.lat] },
    { id: String(camera.id), name: camera.name, kind: 'camera' },
  ),
);

const fovFeatures = scenario.cameras.map((camera) =>
  feature(
    {
      type: 'Polygon',
      coordinates: [fieldOfViewWedge(camera.pose, 24).map((p) => [p.lon, p.lat])],
    },
    { id: String(camera.id), kind: 'fov' },
  ),
);

const zoneFeatures = scenario.zones.map((zone) => {
  const ring = zone.geometry.kind === 'POLYGON' ? zone.geometry.ring : [];
  const coordinates = [...ring.map((p) => [p.lon, p.lat])];
  if (coordinates.length > 0) coordinates.push(coordinates[0]);

  return feature(
    { type: 'Polygon', coordinates: [coordinates] },
    { id: String(zone.id), name: zone.name, purpose: zone.purpose, kind: 'zone' },
  );
});

/** Event markers, carrying the uncertainty so the map can render it honestly. */
const eventFeatures = result.events
  .filter((event) => event.position !== null)
  .map((event) =>
    feature(
      { type: 'Point', coordinates: [event.position.point.lon, event.position.point.lat] },
      {
        id: String(event.id),
        kind: 'event',
        type: event.type,
        severity: event.severity,
        occurredAt: event.occurredAt,
        cameraId: String(event.cameraId),
        summary: event.summary,
        uncertaintyMeters: event.position.radiusMeters,
        positionSource: event.position.source,
      },
    ),
  );

const incidents = result.incidents.map((incident) => ({
  id: String(incident.id),
  title: incident.title,
  severity: incident.severity,
  status: incident.status,
  openedAt: incident.openedAt,
  updatedAt: incident.updatedAt,
  cameraIds: incident.cameraIds.map(String),
  zoneIds: incident.zoneIds.map(String),
  distinctObjectCount: incident.distinctObjectCount,
  eventCount: incident.eventIds.length,
  trackSegmentCount: incident.trackIds.length,
  risk: {
    score: incident.risk.score,
    severity: incident.risk.severity,
    contributions: incident.risk.contributions.map((c) => ({
      code: c.code,
      detail: c.detail,
      points: c.points,
    })),
  },
  timeline: (result.timelines.get(String(incident.id)) ?? []).map((entry) => ({
    at: entry.at,
    kind: entry.kind,
    label: entry.label,
    cameraId: entry.cameraId === null ? null : String(entry.cameraId),
  })),
  report: (() => {
    const report = result.reports.get(String(incident.id));
    if (report === undefined) return null;
    return {
      modelId: String(report.modelId),
      promptVersion: report.promptVersion,
      summary: report.summary,
      observed: report.observed.map((s) => ({
        text: s.text,
        confidence: s.confidence,
        evidence: [...s.eventIds.map(String), ...s.cameraIds.map(String)],
      })),
      inferred: report.inferred.map((s) => ({
        text: s.text,
        confidence: s.confidence,
        evidence: [...s.cameraIds.map(String)],
      })),
      unknown: [...report.unknown],
      operatorQuestions: [...report.operatorQuestions],
      insufficientEvidence: report.insufficientEvidence,
    };
  })(),
}));

const snapshot = {
  generatedAt: Date.now(),
  scenario: {
    name: scenario.name,
    description: scenario.description,
    seed: scenario.seed,
    startedAt: scenario.startedAt,
    durationSeconds: scenario.durationSeconds,
  },
  pipeline: {
    frames: result.frames,
    detections: result.detections,
    positionError: result.positionError,
  },
  cameras,
  // Full geometry, so the desktop can rebuild Zone objects and run the real
  // coverage analysis in the browser rather than rendering a precomputed answer.
  zones: scenario.zones.map((z) => ({
    id: String(z.id),
    name: z.name,
    purpose: z.purpose,
    geometry: z.geometry,
  })),
  events: result.events.map((event) => ({
    id: String(event.id),
    type: event.type,
    severity: event.severity,
    occurredAt: event.occurredAt,
    cameraId: String(event.cameraId),
    summary: event.summary,
    confidence: event.confidence,
    zoneIds: event.zoneIds.map(String),
  })),
  associations: result.associations.map((a) => ({
    fromCameraId: String(a.fromCameraId),
    toCameraId: String(a.toCameraId),
    departedAt: a.departedAt,
    arrivedAt: a.arrivedAt,
    score: a.score,
    reasons: a.reasons.map((r) => ({ code: r.code, detail: r.detail })),
  })),
  incidents,
  geojson: {
    cameras: { type: 'FeatureCollection', features: cameraFeatures },
    fov: { type: 'FeatureCollection', features: fovFeatures },
    zones: { type: 'FeatureCollection', features: zoneFeatures },
    events: { type: 'FeatureCollection', features: eventFeatures },
  },
};

mkdirSync(path.dirname(OUT), { recursive: true });
writeFileSync(OUT, `${JSON.stringify(snapshot, null, 2)}\n`, 'utf8');

console.log(`wrote ${path.relative(ROOT, OUT)}`);
console.log(
  `  ${snapshot.cameras.length} cameras, ${snapshot.events.length} events, ` +
    `${snapshot.incidents.length} incident(s), ` +
    `mean position error ${snapshot.pipeline.positionError.meanMeters.toFixed(2)} m`,
);
