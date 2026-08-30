import type {
  CameraId,
  CameraPose,
  Detection,
  NodeId,
  Track,
  TrackId,
  TrackObservation,
  UtcMillis,
  Zone,
  ZoneObservation,
} from '@sentinel/shared-types';
import { PositionSource } from '@sentinel/shared-types';
import { ZoneEngine } from '@sentinel/geometry';
import { Tracker } from '@sentinel/tracking';
import type { TrackerOptions } from '@sentinel/tracking';

/**
 * The edge pipeline for a single camera.
 *
 * Detections in, tracks and zone observations out. This is the boundary at which
 * a worker node stops producing raw model output and starts producing statements
 * about the world - and it is deliberately the last stage that runs at the edge.
 * Everything downstream (rules, correlation, incidents) is a control-node concern,
 * so a worker that loses its control node can keep running this indefinitely and
 * buffer the results.
 *
 * The pipeline holds no opinions about what anything *means*. It does not know
 * what a restricted zone is, only that a track entered a polygon.
 */

export type CameraPipelineOptions = {
  readonly nodeId: NodeId;
  readonly cameraId: CameraId;
  readonly pose: CameraPose | null;
  readonly zones: readonly Zone[];
  readonly tracker?: TrackerOptions;
  readonly dwellReportIntervalMillis?: number;
};

/** One frame's worth of output. */
export type PipelineOutput = {
  readonly cameraId: CameraId;
  readonly at: UtcMillis;
  readonly tracks: readonly Track[];
  readonly observations: readonly TrackObservation[];
  /** Zone transitions, paired with the track that caused each one. */
  readonly zoneEvents: readonly { readonly track: Track; readonly observation: ZoneObservation }[];
  readonly endedTrackIds: readonly TrackId[];
};

export class CameraPipeline {
  readonly cameraId: CameraId;
  readonly nodeId: NodeId;
  readonly #tracker: Tracker;
  readonly #zones: ZoneEngine;
  #pose: CameraPose | null;
  #framesProcessed = 0;
  #detectionsProcessed = 0;

  constructor(options: CameraPipelineOptions) {
    this.cameraId = options.cameraId;
    this.nodeId = options.nodeId;
    this.#pose = options.pose;

    this.#tracker = new Tracker(options.cameraId, {
      ...options.tracker,
      pose: options.pose,
    });

    this.#zones = new ZoneEngine(options.zones, {
      ...(options.dwellReportIntervalMillis === undefined
        ? {}
        : { dwellReportIntervalMillis: options.dwellReportIntervalMillis }),
    });
  }

  /** Update the pose when an operator repositions the camera on the map. */
  setPose(pose: CameraPose | null): void {
    this.#pose = pose;
    this.#tracker.setPose(pose);
  }

  setZone(zone: Zone): void {
    this.#zones.setZone(zone);
  }

  get stats(): { frames: number; detections: number; activeTracks: number; zones: number } {
    return {
      frames: this.#framesProcessed,
      detections: this.#detectionsProcessed,
      activeTracks: this.#tracker.activeTrackCount,
      zones: this.#zones.zoneCount,
    };
  }

  /**
   * Process one frame of detections.
   *
   * Zone membership is only evaluated for tracks with a real ground projection.
   * A camera-fallback position is the *camera's* location, not the object's;
   * testing it against a zone would report every detection on an uncalibrated
   * camera as being wherever the mast happens to stand, which is how a system
   * ends up confidently wrong.
   */
  process(detections: readonly Detection[], at: UtcMillis): PipelineOutput {
    this.#framesProcessed += 1;
    this.#detectionsProcessed += detections.length;

    const update = this.#tracker.update(detections, at);

    const zoneEvents: { track: Track; observation: ZoneObservation }[] = [];

    for (const track of update.tracks) {
      const position = track.currentPosition;
      if (position === null) continue;
      if (position.source !== PositionSource.GroundProjection) continue;

      const observations = this.#zones.update(track.id, position.point, at);
      for (const observation of observations) {
        zoneEvents.push({ track, observation });
      }
    }

    // Release per-track state for anything that ended, so memory stays bounded
    // by the number of live tracks rather than by uptime.
    for (const trackId of update.ended) this.#zones.forget(trackId);

    return {
      cameraId: this.cameraId,
      at,
      tracks: update.tracks,
      observations: update.observations,
      zoneEvents,
      endedTrackIds: update.ended,
    };
  }

  /** Tracks currently inside a zone, for group and crowd rule conditions. */
  tracksInZone(zoneId: string): number {
    let count = 0;
    for (const track of this.#tracker.tracks()) {
      if (this.#zones.zonesFor(track.id).some((id) => String(id) === zoneId)) count += 1;
    }
    return count;
  }

  /** Close everything, e.g. when the camera goes offline. */
  reset(): readonly TrackId[] {
    const ended = this.#tracker.reset();
    for (const trackId of ended) this.#zones.forget(trackId);
    return ended;
  }

  get pose(): CameraPose | null {
    return this.#pose;
  }
}
