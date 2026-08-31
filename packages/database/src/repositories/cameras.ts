import type {
  Camera,
  CameraHealth,
  CameraId,
  CameraProfile,
  CameraProfileId,
  CameraStatus,
  CameraTopologyEdge,
  CredentialsRef,
  LocationId,
  NodeId,
  UtcMillis,
  ZoneId,
} from '@sentinel/shared-types';
import { asId, utcMillis } from '@sentinel/shared-types';
import type { SqlDriver, SqlValue } from '../driver.ts';

/**
 * Camera persistence.
 *
 * The one rule that shapes everything here: **no credential is ever written**.
 * The `Camera` domain object carries a `credentialsRef`, which is an opaque
 * handle into the OS keychain, and this repository stores that handle and nothing
 * else. There is no column a password could go into, and a test walks the schema
 * to keep it that way.
 *
 * Structured sub-objects (pose, intrinsics, AI policy, recording policy) are
 * stored as JSON rather than exploded into columns. They are read and written
 * whole, never queried into, and flattening them would produce forty columns that
 * migrate every time a policy field is added.
 */

const boolToInt = (value: boolean): number => (value ? 1 : 0);

/**
 * SQLite has no boolean type; these columns are integers.
 *
 * PostgreSQL, behind the same driver interface, may hand back a bigint for the
 * same column, so both are accepted. Anything else is false rather than truthy -
 * a corrupted value defaulting to "PTZ enabled" would silently permit a
 * privileged action on a camera that cannot perform it.
 */
const intToBool = (value: SqlValue): boolean => value === 1 || value === 1n;

const toJson = (value: unknown): string | null =>
  value === null || value === undefined ? null : JSON.stringify(value);

/**
 * Parse a JSON column.
 *
 * A row written by a newer version, or corrupted on disk, must not take the whole
 * camera list down. Returning the fallback keeps the rest of the record usable
 * and surfaces the damage as a missing pose rather than an exception during
 * start-up.
 */
const fromJson = <T>(value: SqlValue, fallback: T): T => {
  if (typeof value !== 'string' || value === '') return fallback;
  try {
    return JSON.parse(value) as T;
  } catch {
    return fallback;
  }
};

type CameraRow = {
  id: string;
  name: string;
  description: string | null;
  manufacturer: string | null;
  model: string | null;
  serial_number: string | null;
  protocol: string;
  host: string;
  port: number;
  onvif_capabilities: string | null;
  credentials_ref: string | null;
  worker_node_id: string | null;
  location_id: string | null;
  pose: string | null;
  intrinsics: string | null;
  ai_policy: string;
  recording_policy: string;
  status: string;
  last_seen: number | null;
  ptz_supported: number;
  created_at: number;
  updated_at: number;
};

export class CameraRepository {
  readonly #db: SqlDriver;

  constructor(db: SqlDriver) {
    this.#db = db;
  }

  #toDomain(row: CameraRow, zoneIds: readonly ZoneId[]): Camera {
    return {
      id: asId<CameraId>(row.id),
      name: row.name,
      protocol: row.protocol as Camera['protocol'],
      host: row.host,
      port: row.port,
      workerNodeId: row.worker_node_id === null ? null : asId<NodeId>(row.worker_node_id),
      locationId: row.location_id === null ? null : asId<LocationId>(row.location_id),
      pose: fromJson<Camera['pose']>(row.pose, null),
      intrinsics: fromJson<Camera['intrinsics']>(row.intrinsics, null),
      zoneIds,
      ai: fromJson<Camera['ai']>(row.ai_policy, DEFAULT_AI_POLICY),
      recording: fromJson<Camera['recording']>(row.recording_policy, DEFAULT_RECORDING_POLICY),
      status: row.status as CameraStatus,
      lastSeen: row.last_seen === null ? null : utcMillis(row.last_seen),
      ptzSupported: intToBool(row.ptz_supported),
      createdAt: utcMillis(row.created_at),
      updatedAt: utcMillis(row.updated_at),
      ...(row.description === null ? {} : { description: row.description }),
      ...(row.manufacturer === null ? {} : { manufacturer: row.manufacturer }),
      ...(row.model === null ? {} : { model: row.model }),
      ...(row.serial_number === null ? {} : { serialNumber: row.serial_number }),
      ...(row.credentials_ref === null
        ? {}
        : { credentialsRef: asId<CredentialsRef>(row.credentials_ref) }),
      ...(row.onvif_capabilities === null
        ? {}
        : { onvifCapabilities: fromJson<string[]>(row.onvif_capabilities, []) }),
    };
  }

  #zonesFor(cameraId: CameraId): readonly ZoneId[] {
    return this.#db
      .query<{ zone_id: string }>('SELECT zone_id FROM camera_zone_links WHERE camera_id = ?', [
        cameraId,
      ])
      .map((row) => asId<ZoneId>(row.zone_id));
  }

  get(cameraId: CameraId): Camera | undefined {
    const row = this.#db.queryOne<CameraRow>('SELECT * FROM cameras WHERE id = ?', [cameraId]);
    return row === undefined ? undefined : this.#toDomain(row, this.#zonesFor(cameraId));
  }

  list(): readonly Camera[] {
    const rows = this.#db.query<CameraRow>('SELECT * FROM cameras ORDER BY name');
    return rows.map((row) => this.#toDomain(row, this.#zonesFor(asId<CameraId>(row.id))));
  }

  /** Cameras assigned to a worker, which is what that worker asks for on start-up. */
  listForNode(nodeId: NodeId): readonly Camera[] {
    const rows = this.#db.query<CameraRow>(
      'SELECT * FROM cameras WHERE worker_node_id = ? ORDER BY name',
      [nodeId],
    );
    return rows.map((row) => this.#toDomain(row, this.#zonesFor(asId<CameraId>(row.id))));
  }

  /** Cameras nobody is processing. A camera in this list is recording nothing. */
  listUnassigned(): readonly Camera[] {
    const rows = this.#db.query<CameraRow>(
      'SELECT * FROM cameras WHERE worker_node_id IS NULL ORDER BY name',
    );
    return rows.map((row) => this.#toDomain(row, this.#zonesFor(asId<CameraId>(row.id))));
  }

  /**
   * Insert or update a camera and its zone links atomically.
   *
   * One transaction, because a camera saved without its zones is a camera that
   * silently monitors nothing - a failure that looks exactly like a working
   * configuration until an intrusion goes unreported.
   */
  save(camera: Camera): void {
    this.#db.transaction(() => {
      this.#db.exec(
        `INSERT INTO cameras (
           id, name, description, manufacturer, model, serial_number, protocol, host, port,
           onvif_capabilities, credentials_ref, worker_node_id, location_id, pose, intrinsics,
           ai_policy, recording_policy, status, last_seen, ptz_supported, created_at, updated_at
         ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
         ON CONFLICT(id) DO UPDATE SET
           name = excluded.name,
           description = excluded.description,
           manufacturer = excluded.manufacturer,
           model = excluded.model,
           serial_number = excluded.serial_number,
           protocol = excluded.protocol,
           host = excluded.host,
           port = excluded.port,
           onvif_capabilities = excluded.onvif_capabilities,
           credentials_ref = excluded.credentials_ref,
           worker_node_id = excluded.worker_node_id,
           location_id = excluded.location_id,
           pose = excluded.pose,
           intrinsics = excluded.intrinsics,
           ai_policy = excluded.ai_policy,
           recording_policy = excluded.recording_policy,
           status = excluded.status,
           last_seen = excluded.last_seen,
           ptz_supported = excluded.ptz_supported,
           updated_at = excluded.updated_at`,
        [
          camera.id,
          camera.name,
          camera.description ?? null,
          camera.manufacturer ?? null,
          camera.model ?? null,
          camera.serialNumber ?? null,
          camera.protocol,
          camera.host,
          camera.port,
          toJson(camera.onvifCapabilities ?? null),
          camera.credentialsRef ?? null,
          camera.workerNodeId,
          camera.locationId,
          toJson(camera.pose),
          toJson(camera.intrinsics),
          JSON.stringify(camera.ai),
          JSON.stringify(camera.recording),
          camera.status,
          camera.lastSeen,
          boolToInt(camera.ptzSupported),
          camera.createdAt,
          camera.updatedAt,
        ],
      );

      this.#db.exec('DELETE FROM camera_zone_links WHERE camera_id = ?', [camera.id]);
      for (const zoneId of camera.zoneIds) {
        this.#db.exec(
          'INSERT OR IGNORE INTO camera_zone_links (camera_id, zone_id) VALUES (?, ?)',
          [camera.id, zoneId],
        );
      }
    });
  }

  /**
   * Update status and last-seen.
   *
   * Separate from `save` because health changes every few seconds while
   * configuration changes rarely. Rewriting an entire camera row to record that
   * it is still online would make the busiest write in the system also the widest.
   */
  updateStatus(cameraId: CameraId, status: CameraStatus, lastSeen: UtcMillis | null): void {
    this.#db.exec('UPDATE cameras SET status = ?, last_seen = ?, updated_at = ? WHERE id = ?', [
      status,
      lastSeen,
      Date.now(),
      cameraId,
    ]);
  }

  assignToNode(cameraId: CameraId, nodeId: NodeId | null): void {
    this.#db.exec('UPDATE cameras SET worker_node_id = ?, updated_at = ? WHERE id = ?', [
      nodeId,
      Date.now(),
      cameraId,
    ]);
  }

  /**
   * Delete a camera.
   *
   * Returns the credential reference so the caller can purge the keychain entry.
   * Deleting the row without that would leave an orphaned secret behind for a
   * camera that no longer exists - and nothing would ever look at it again to
   * notice.
   */
  remove(cameraId: CameraId): { readonly removed: boolean; readonly credentialsRef: CredentialsRef | null } {
    return this.#db.transaction(() => {
      const existing = this.#db.queryOne<{ credentials_ref: string | null }>(
        'SELECT credentials_ref FROM cameras WHERE id = ?',
        [cameraId],
      );
      if (existing === undefined) return { removed: false, credentialsRef: null };

      this.#db.exec('DELETE FROM cameras WHERE id = ?', [cameraId]);

      return {
        removed: true,
        credentialsRef:
          existing.credentials_ref === null ? null : asId<CredentialsRef>(existing.credentials_ref),
      };
    });
  }

  // ------------------------------------------------------------------ profiles

  saveProfiles(cameraId: CameraId, profiles: readonly CameraProfile[]): void {
    this.#db.transaction(() => {
      this.#db.exec('DELETE FROM camera_profiles WHERE camera_id = ?', [cameraId]);
      for (const profile of profiles) {
        this.#db.exec(
          `INSERT INTO camera_profiles
             (id, camera_id, kind, name, path, codec, width, height, fps, bitrate_kbps, keyframe_interval_seconds)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
          [
            profile.id,
            cameraId,
            profile.kind,
            profile.name,
            profile.path,
            profile.codec,
            profile.width,
            profile.height,
            profile.fps,
            profile.bitrateKbps,
            profile.keyframeIntervalSeconds ?? null,
          ],
        );
      }
    });
  }

  profiles(cameraId: CameraId): readonly CameraProfile[] {
    return this.#db
      .query<{
        id: string;
        camera_id: string;
        kind: string;
        name: string;
        path: string;
        codec: string;
        width: number;
        height: number;
        fps: number;
        bitrate_kbps: number;
        keyframe_interval_seconds: number | null;
      }>('SELECT * FROM camera_profiles WHERE camera_id = ? ORDER BY kind', [cameraId])
      .map((row) => ({
        id: asId<CameraProfileId>(row.id),
        cameraId: asId<CameraId>(row.camera_id),
        kind: row.kind as CameraProfile['kind'],
        name: row.name,
        path: row.path,
        codec: row.codec,
        width: row.width,
        height: row.height,
        fps: row.fps,
        bitrateKbps: row.bitrate_kbps,
        ...(row.keyframe_interval_seconds === null
          ? {}
          : { keyframeIntervalSeconds: row.keyframe_interval_seconds }),
      }));
  }

  // ------------------------------------------------------------------ topology

  saveTopologyEdge(edge: CameraTopologyEdge): void {
    this.#db.exec(
      `INSERT INTO camera_topology
         (from_camera_id, to_camera_id, distance_meters, min_travel_seconds,
          expected_travel_seconds, max_travel_seconds, confidence, bidirectional)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?)
       ON CONFLICT(from_camera_id, to_camera_id) DO UPDATE SET
         distance_meters = excluded.distance_meters,
         min_travel_seconds = excluded.min_travel_seconds,
         expected_travel_seconds = excluded.expected_travel_seconds,
         max_travel_seconds = excluded.max_travel_seconds,
         confidence = excluded.confidence,
         bidirectional = excluded.bidirectional`,
      [
        edge.fromCameraId,
        edge.toCameraId,
        edge.distanceMeters,
        edge.minTravelSeconds,
        edge.expectedTravelSeconds,
        edge.maxTravelSeconds,
        edge.confidence,
        boolToInt(edge.bidirectional),
      ],
    );
  }

  topology(): readonly CameraTopologyEdge[] {
    return this.#db
      .query<{
        from_camera_id: string;
        to_camera_id: string;
        distance_meters: number;
        min_travel_seconds: number;
        expected_travel_seconds: number;
        max_travel_seconds: number;
        confidence: number;
        bidirectional: number;
      }>('SELECT * FROM camera_topology')
      .map((row) => ({
        fromCameraId: asId<CameraId>(row.from_camera_id),
        toCameraId: asId<CameraId>(row.to_camera_id),
        distanceMeters: row.distance_meters,
        minTravelSeconds: row.min_travel_seconds,
        expectedTravelSeconds: row.expected_travel_seconds,
        maxTravelSeconds: row.max_travel_seconds,
        confidence: row.confidence,
        bidirectional: intToBool(row.bidirectional),
      }));
  }

  /** Health snapshot, derived from the stored status. Live metrics come from the worker. */
  health(cameraId: CameraId, observedAt: UtcMillis): CameraHealth | undefined {
    const camera = this.get(cameraId);
    if (camera === undefined) return undefined;

    return {
      cameraId,
      status: camera.status,
      observedAt,
      fps: 0,
      targetFps: camera.ai.enabled ? camera.ai.idleFps : 0,
      inferenceFps: 0,
      droppedFrames: 0,
      decodeErrors: 0,
      reconnectCount: 0,
      bitrateKbps: 0,
      latencyMs: 0,
      pingMs: null,
    };
  }
}

export const DEFAULT_AI_POLICY: Camera['ai'] = Object.freeze({
  enabled: true,
  idleFps: 2,
  activeFps: 12,
  classes: Object.freeze(['person', 'vehicle']),
  confidenceThreshold: 0.5,
  trackingEnabled: true,
  eventGenerationEnabled: true,
});

export const DEFAULT_RECORDING_POLICY: Camera['recording'] = Object.freeze({
  mode: 'EVENT',
  segmentSeconds: 60,
  retentionDays: 7,
  preEventSeconds: 10,
  postEventSeconds: 30,
});
