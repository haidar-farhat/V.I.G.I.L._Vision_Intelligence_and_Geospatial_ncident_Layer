import type {
  AiIncidentReport,
  CameraId,
  Incident,
  IncidentTimelineEntry,
  NodeId,
  SecurityEvent,
  Track,
  TrackAssociation,
  UtcMillis,
  Zone,
  ZoneId,
} from '@sentinel/shared-types';
import { asId, utcMillis } from '@sentinel/shared-types';
import { haversineDistance } from '@sentinel/geometry';
import { bestAssociation, topologyIndex } from '@sentinel/tracking';
import { CameraPipeline } from '@sentinel/worker/pipeline';
import { Correlator, RuleEngine } from '@sentinel/event-engine';
import type { RiskContext } from '@sentinel/event-engine';
import { analyseWithGuardrails, createDeterministicAnalyst } from '@sentinel/ai';
import type { EvidenceBundle } from '@sentinel/ai';
import { SimulatedSensor } from './world.ts';
import type { Scenario } from './scenario.ts';

/**
 * The vertical slice.
 *
 * Runs a scenario end to end through the **production** components:
 *
 *   simulated camera -> detections -> tracker -> ground projection
 *   -> zone engine -> rule engine -> cross-camera association
 *   -> correlator -> incident -> AI analyst
 *
 * Nothing here is a stand-in for a real component; the simulator only replaces
 * the camera and the detector. That is the point of the exercise: the slice
 * proves the seams between subsystems, and a change that breaks any one of them
 * fails here rather than in the field.
 */

export const SIMULATOR_NODE_ID = asId<NodeId>('node-simulator');

export type SliceResult = {
  readonly scenario: string;
  readonly seed: number;
  readonly frames: number;
  readonly detections: number;
  readonly tracksByCamera: ReadonlyMap<CameraId, number>;
  readonly events: readonly SecurityEvent[];
  readonly associations: readonly TrackAssociation[];
  readonly incidents: readonly Incident[];
  readonly timelines: ReadonlyMap<string, readonly IncidentTimelineEntry[]>;
  readonly reports: ReadonlyMap<string, AiIncidentReport>;
  /** How far the pipeline's map positions sat from simulated ground truth. */
  readonly positionError: PositionErrorStats;
};

export type PositionErrorStats = {
  readonly samples: number;
  readonly meanMeters: number;
  readonly p95Meters: number;
  readonly maxMeters: number;
  /** Fraction of samples that fell inside the uncertainty the system reported. */
  readonly withinStatedUncertainty: number;
};

export type SliceOptions = {
  /** Run the AI analyst over each incident. */
  readonly analyse?: boolean;
  /** Site timezone offset used for after-hours rules. */
  readonly utcOffsetMinutes?: number;
};

/** A track that has left one camera and may reappear on another. */
type DepartedTrack = { readonly track: Track; readonly at: UtcMillis };

export const runScenario = async (
  scenario: Scenario,
  options: SliceOptions = {},
): Promise<SliceResult> => {
  const utcOffsetMinutes = options.utcOffsetMinutes ?? 0;

  const zoneIndex = new Map<ZoneId, Zone>(scenario.zones.map((z) => [z.id, z]));
  const cameraNames = new Map<CameraId, string>(scenario.cameras.map((c) => [c.id, c.name]));

  const riskContext: RiskContext = {
    zones: zoneIndex,
    afterHours: { startMinute: 22 * 60, endMinute: 6 * 60 },
    utcOffsetMinutes,
    assessedAt: scenario.startedAt,
  };

  const rules = new RuleEngine(scenario.rules);
  const correlator = new Correlator(riskContext);
  const topology = topologyIndex(scenario.topology);

  const sensors = scenario.cameras.map((camera) => new SimulatedSensor(camera, scenario.seed));
  const pipelines = new Map<CameraId, CameraPipeline>(
    scenario.cameras.map((camera) => [
      camera.id,
      new CameraPipeline({
        nodeId: SIMULATOR_NODE_ID,
        cameraId: camera.id,
        pose: camera.pose,
        zones: scenario.zones,
        tracker: { minHitsToConfirm: 3, maxGapMillis: 3000 },
        dwellReportIntervalMillis: 1000,
      }),
    ]),
  );

  const events: SecurityEvent[] = [];
  const associations: TrackAssociation[] = [];
  const seenTracks = new Map<CameraId, Set<string>>();
  const departed: DepartedTrack[] = [];
  const liveTracks = new Map<string, Track>();
  /** Departed tracks already matched to an arrival, keeping association one-to-one. */
  const claimed = new Set<string>();

  let frames = 0;
  let detectionCount = 0;

  // --- ground-truth accounting ------------------------------------------------
  const errors: number[] = [];
  let withinUncertainty = 0;

  const steps = Math.floor(scenario.durationSeconds / scenario.frameIntervalSeconds);

  for (let step = 0; step <= steps; step += 1) {
    const elapsed = step * scenario.frameIntervalSeconds;

    for (const sensor of sensors) {
      const observation = sensor.observe(
        scenario.actors,
        elapsed,
        scenario.startedAt,
        scenario.frameIntervalSeconds,
      );

      const pipeline = pipelines.get(observation.cameraId);
      if (pipeline === undefined) continue;

      frames += 1;
      detectionCount += observation.detections.length;

      const output = pipeline.process(observation.detections, observation.at);

      // Measure the pipeline's spatial error against ground truth. Only actors
      // the camera could actually see are counted, and only where the system
      // claimed a real ground projection.
      for (const truth of observation.truth) {
        const nearest = nearestTrackPosition(output.tracks, truth.position);
        if (nearest === null) continue;

        errors.push(nearest.distance);
        if (nearest.distance <= nearest.stated * 2) withinUncertainty += 1;
      }

      for (const track of output.tracks) {
        liveTracks.set(String(track.id), track);

        // First sighting on this camera: try to link it to a recent departure
        // elsewhere. This is the cross-camera hand-off.
        let seen = seenTracks.get(observation.cameraId);
        if (seen === undefined) {
          seen = new Set<string>();
          seenTracks.set(observation.cameraId, seen);
        }

        if (!seen.has(String(track.id))) {
          seen.add(String(track.id));

          const candidates = departed
            .filter((d) => d.track.cameraId !== track.cameraId && !claimed.has(String(d.track.id)))
            .map((d) => d.track);

          const association = bestAssociation(track, candidates, topology);
          if (association !== null) {
            // One-to-one: a departed track is consumed by its best match. Without
            // this, one person leaving camera 07 gets associated with every track
            // that later appears on camera 08, inflating both the association
            // count and the apparent size of the group.
            claimed.add(String(association.fromTrackId));
            associations.push(association);
            correlator.addAssociation(association);
          }
        }
      }

      for (const trackId of output.endedTrackIds) {
        const track = liveTracks.get(String(trackId));
        if (track !== undefined) {
          departed.push({ track, at: observation.at });
          liveTracks.delete(String(trackId));
        }
      }

      // --- rules ------------------------------------------------------------
      for (const { track, observation: zoneObservation } of output.zoneEvents) {
        const zone = zoneIndex.get(zoneObservation.zoneId);
        if (zone === undefined) continue;

        const fired = rules.evaluateAll({
          nodeId: SIMULATOR_NODE_ID,
          cameraId: observation.cameraId,
          track,
          observation: zoneObservation,
          zone,
          concurrentTracksInZone: pipeline.tracksInZone(String(zone.id)),
          utcOffsetMinutes,
        });

        for (const event of fired) {
          events.push(event);
          correlator.ingest(event);
        }
      }
    }
  }

  // Any track still live at the end counts as departed for association purposes.
  for (const track of liveTracks.values()) {
    departed.push({ track, at: utcMillis(scenario.startedAt + scenario.durationSeconds * 1000) });
  }

  // --- assemble results -------------------------------------------------------
  const incidents = correlator.incidents();
  const timelines = new Map<string, readonly IncidentTimelineEntry[]>();
  const reports = new Map<string, AiIncidentReport>();

  for (const incident of incidents) {
    timelines.set(String(incident.id), correlator.timeline(incident.id));
  }

  if (options.analyse === true) {
    const analyst = createDeterministicAnalyst(() =>
      utcMillis(scenario.startedAt + scenario.durationSeconds * 1000),
    );

    for (const incident of incidents) {
      const incidentAssociations = correlator.associationsFor(incident.id);

      // The bundle must name every camera the evidence refers to, not just those
      // that produced events. A camera can observe the group without generating
      // an event - it sees no zone - yet still appear in a cross-camera
      // association. Omitting it left the analyst citing a camera the operator
      // had no context for, which the guardrails correctly rejected.
      const referencedCameras = new Set<CameraId>(incident.cameraIds);
      for (const association of incidentAssociations) {
        referencedCameras.add(association.fromCameraId);
        referencedCameras.add(association.toCameraId);
      }

      const bundle: EvidenceBundle = {
        incidentId: incident.id,
        events: correlator.eventsFor(incident.id),
        associations: incidentAssociations,
        zones: incident.zoneIds
          .map((id) => zoneIndex.get(id))
          .filter((z): z is Zone => z !== undefined),
        cameras: [...referencedCameras].map((id) => ({
          id,
          name: cameraNames.get(id) ?? String(id),
        })),
        windowStart: incident.openedAt,
        windowEnd: incident.updatedAt,
        evidenceIds: incident.evidenceIds,
      };

      // Guardrails apply to the built-in analyst exactly as they would to a
      // local LLM. If the reference implementation cannot satisfy them, nothing
      // can, and the slice should fail loudly here.
      reports.set(String(incident.id), await analyseWithGuardrails(analyst, bundle));
    }
  }

  const tracksByCamera = new Map<CameraId, number>();
  for (const [cameraId, seen] of seenTracks) tracksByCamera.set(cameraId, seen.size);

  return {
    scenario: scenario.name,
    seed: scenario.seed,
    frames,
    detections: detectionCount,
    tracksByCamera,
    events,
    associations,
    incidents,
    timelines,
    reports,
    positionError: summariseErrors(errors, withinUncertainty),
  };
};

/**
 * Closest tracked position to a known truth point, with the uncertainty the
 * system stated for it.
 *
 * Nearest-neighbour matching is adequate here because the simulated actors are
 * metres apart while the pipeline's error is well under a metre; a mismatch would
 * show up as a large error rather than being silently absorbed.
 */
const nearestTrackPosition = (
  tracks: readonly Track[],
  truth: { lat: number; lon: number },
): { distance: number; stated: number } | null => {
  let best: { distance: number; stated: number } | null = null;

  for (const track of tracks) {
    const position = track.currentPosition;
    if (position === null || position.source !== 'GROUND_PROJECTION') continue;

    const distance = haversineDistance(position.point, truth);
    if (best === null || distance < best.distance) {
      best = { distance, stated: position.radiusMeters };
    }
  }

  return best;
};

const summariseErrors = (errors: readonly number[], within: number): PositionErrorStats => {
  if (errors.length === 0) {
    return { samples: 0, meanMeters: 0, p95Meters: 0, maxMeters: 0, withinStatedUncertainty: 1 };
  }

  const sorted = [...errors].sort((a, b) => a - b);
  const sum = sorted.reduce((total, value) => total + value, 0);
  const p95Index = Math.min(sorted.length - 1, Math.floor(sorted.length * 0.95));

  return {
    samples: sorted.length,
    meanMeters: sum / sorted.length,
    p95Meters: sorted[p95Index] ?? 0,
    maxMeters: sorted[sorted.length - 1] ?? 0,
    withinStatedUncertainty: within / sorted.length,
  };
};
