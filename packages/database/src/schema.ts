import type { Migration } from './migrations.ts';

/**
 * The schema.
 *
 * Conventions that hold throughout:
 *
 *  - Every timestamp is INTEGER milliseconds since the Unix epoch, **UTC**. Local
 *    time exists only in the presentation layer.
 *  - Camera-reported time and node-received time are separate columns and are
 *    never reconciled. Clock skew between a camera and a node is evidence about
 *    the deployment, not noise to be smoothed away.
 *  - No table holds a credential. `cameras.credentials_ref` is an opaque handle
 *    into the OS keychain.
 *  - `audit_logs`, `incident_notes` and `ai_inferences` are append-only. There is
 *    no code path that updates or deletes them; a record that can be rewritten
 *    after the fact is not an audit trail.
 *  - SQL is kept portable so the same migrations can run against PostgreSQL.
 */

const initial: Migration = {
  version: 1,
  name: 'initial_schema',
  up: `
--------------------------------------------------------------------- identity
CREATE TABLE users (
  id            TEXT PRIMARY KEY,
  username      TEXT NOT NULL UNIQUE,
  display_name  TEXT NOT NULL,
  -- Argon2id or scrypt encoded string. Never a bare hash, never reversible.
  password_hash TEXT NOT NULL,
  roles         TEXT NOT NULL,
  active        INTEGER NOT NULL DEFAULT 1,
  created_at    INTEGER NOT NULL,
  last_login_at INTEGER
);

CREATE TABLE nodes (
  id                   TEXT PRIMARY KEY,
  name                 TEXT NOT NULL,
  roles                TEXT NOT NULL,
  status               TEXT NOT NULL,
  addresses            TEXT NOT NULL,
  app_version          TEXT NOT NULL,
  protocol_version     INTEGER NOT NULL,
  hardware             TEXT NOT NULL,
  capabilities         TEXT NOT NULL,
  identity_fingerprint TEXT NOT NULL,
  last_heartbeat       INTEGER,
  paired_at            INTEGER
);

-------------------------------------------------------------------- locations
CREATE TABLE locations (
  id         TEXT PRIMARY KEY,
  name       TEXT NOT NULL,
  kind       TEXT NOT NULL,
  parent_id  TEXT REFERENCES locations(id) ON DELETE SET NULL,
  latitude   REAL,
  longitude  REAL,
  altitude   REAL,
  indoor_frame TEXT,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);
CREATE INDEX idx_locations_parent ON locations(parent_id);

---------------------------------------------------------------------- cameras
CREATE TABLE cameras (
  id               TEXT PRIMARY KEY,
  name             TEXT NOT NULL,
  description      TEXT,
  manufacturer     TEXT,
  model            TEXT,
  serial_number    TEXT,
  protocol         TEXT NOT NULL,
  host             TEXT NOT NULL,
  port             INTEGER NOT NULL,
  onvif_capabilities TEXT,
  -- Opaque keychain handle. The secret itself never reaches this database.
  credentials_ref  TEXT,
  worker_node_id   TEXT REFERENCES nodes(id) ON DELETE SET NULL,
  location_id      TEXT REFERENCES locations(id) ON DELETE SET NULL,
  pose             TEXT,
  intrinsics       TEXT,
  ai_policy        TEXT NOT NULL,
  recording_policy TEXT NOT NULL,
  status           TEXT NOT NULL DEFAULT 'UNKNOWN',
  last_seen        INTEGER,
  ptz_supported    INTEGER NOT NULL DEFAULT 0,
  created_at       INTEGER NOT NULL,
  updated_at       INTEGER NOT NULL
);
CREATE INDEX idx_cameras_node ON cameras(worker_node_id);
CREATE INDEX idx_cameras_status ON cameras(status);

CREATE TABLE camera_profiles (
  id          TEXT PRIMARY KEY,
  camera_id   TEXT NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
  kind        TEXT NOT NULL,
  name        TEXT NOT NULL,
  -- Stream path only. A full rtsp://user:pass@host URL must never be stored.
  path        TEXT NOT NULL,
  codec       TEXT NOT NULL,
  width       INTEGER NOT NULL,
  height      INTEGER NOT NULL,
  fps         REAL NOT NULL,
  bitrate_kbps INTEGER NOT NULL,
  keyframe_interval_seconds REAL
);
CREATE INDEX idx_camera_profiles_camera ON camera_profiles(camera_id);

CREATE TABLE camera_topology (
  from_camera_id          TEXT NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
  to_camera_id            TEXT NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
  distance_meters         REAL NOT NULL,
  min_travel_seconds      REAL NOT NULL,
  expected_travel_seconds REAL NOT NULL,
  max_travel_seconds      REAL NOT NULL,
  confidence              REAL NOT NULL,
  bidirectional           INTEGER NOT NULL DEFAULT 1,
  PRIMARY KEY (from_camera_id, to_camera_id)
);

------------------------------------------------------------------------ zones
CREATE TABLE zones (
  id             TEXT PRIMARY KEY,
  name           TEXT NOT NULL,
  purpose        TEXT NOT NULL,
  geometry       TEXT NOT NULL,
  location_id    TEXT REFERENCES locations(id) ON DELETE SET NULL,
  parent_zone_id TEXT REFERENCES zones(id) ON DELETE SET NULL,
  active         INTEGER NOT NULL DEFAULT 1,
  created_at     INTEGER NOT NULL,
  updated_at     INTEGER NOT NULL
);

CREATE TABLE camera_zone_links (
  camera_id TEXT NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
  zone_id   TEXT NOT NULL REFERENCES zones(id) ON DELETE CASCADE,
  PRIMARY KEY (camera_id, zone_id)
);

----------------------------------------------------------------------- tracks
CREATE TABLE tracks (
  id            TEXT PRIMARY KEY,
  camera_id     TEXT NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
  object_class  TEXT NOT NULL,
  first_seen    INTEGER NOT NULL,
  last_seen     INTEGER NOT NULL,
  confidence    REAL NOT NULL,
  observation_count INTEGER NOT NULL,
  speed_mps     REAL,
  heading_degrees REAL,
  embedding     BLOB,
  active        INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX idx_tracks_camera_time ON tracks(camera_id, first_seen);

CREATE TABLE track_observations (
  track_id     TEXT NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
  at           INTEGER NOT NULL,
  box_x        REAL NOT NULL,
  box_y        REAL NOT NULL,
  box_w        REAL NOT NULL,
  box_h        REAL NOT NULL,
  confidence   REAL NOT NULL,
  latitude     REAL,
  longitude    REAL,
  -- The uncertainty is stored with the position, never separated from it.
  uncertainty_meters REAL,
  position_source TEXT,
  interpolated INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (track_id, at)
);

CREATE TABLE track_associations (
  from_track_id TEXT NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
  to_track_id   TEXT NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
  score         REAL NOT NULL,
  reasons       TEXT NOT NULL,
  departed_at   INTEGER NOT NULL,
  arrived_at    INTEGER NOT NULL,
  PRIMARY KEY (from_track_id, to_track_id)
);

------------------------------------------------------------------------ rules
CREATE TABLE rules (
  id          TEXT PRIMARY KEY,
  name        TEXT NOT NULL,
  description TEXT,
  enabled     INTEGER NOT NULL DEFAULT 1,
  -- Structured JSON, never executable code.
  condition   TEXT NOT NULL,
  action      TEXT NOT NULL,
  created_at  INTEGER NOT NULL,
  updated_at  INTEGER NOT NULL
);

----------------------------------------------------------------------- events
CREATE TABLE events (
  id           TEXT PRIMARY KEY,
  type         TEXT NOT NULL,
  severity     TEXT NOT NULL,
  status       TEXT NOT NULL DEFAULT 'NEW',
  -- When it happened, per the observing node.
  occurred_at  INTEGER NOT NULL,
  -- When the control node durably accepted it. Skew is preserved, not hidden.
  recorded_at  INTEGER NOT NULL,
  camera_id    TEXT REFERENCES cameras(id) ON DELETE SET NULL,
  node_id      TEXT NOT NULL,
  object_class TEXT,
  confidence   REAL NOT NULL,
  latitude     REAL,
  longitude    REAL,
  uncertainty_meters REAL,
  position_source TEXT,
  rule_id      TEXT REFERENCES rules(id) ON DELETE SET NULL,
  model_id     TEXT,
  incident_id  TEXT,
  summary      TEXT NOT NULL,
  detail       TEXT NOT NULL
);
CREATE INDEX idx_events_occurred ON events(occurred_at);
CREATE INDEX idx_events_camera_time ON events(camera_id, occurred_at);
CREATE INDEX idx_events_incident ON events(incident_id);
CREATE INDEX idx_events_type_time ON events(type, occurred_at);

CREATE TABLE event_zones (
  event_id TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
  zone_id  TEXT NOT NULL REFERENCES zones(id) ON DELETE CASCADE,
  PRIMARY KEY (event_id, zone_id)
);

CREATE TABLE event_tracks (
  event_id TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
  track_id TEXT NOT NULL,
  PRIMARY KEY (event_id, track_id)
);

-------------------------------------------------------------------- incidents
CREATE TABLE incidents (
  id            TEXT PRIMARY KEY,
  title         TEXT NOT NULL,
  severity      TEXT NOT NULL,
  status        TEXT NOT NULL DEFAULT 'NEW',
  opened_at     INTEGER NOT NULL,
  updated_at    INTEGER NOT NULL,
  closed_at     INTEGER,
  latitude      REAL,
  longitude     REAL,
  uncertainty_meters REAL,
  distinct_object_count INTEGER NOT NULL DEFAULT 0,
  risk_score    INTEGER NOT NULL,
  -- Every contribution with its reason. The score alone is not auditable.
  risk_contributions TEXT NOT NULL,
  acknowledged_by TEXT REFERENCES users(id) ON DELETE SET NULL,
  acknowledged_at INTEGER
);
CREATE INDEX idx_incidents_status_time ON incidents(status, opened_at);
CREATE INDEX idx_incidents_severity ON incidents(severity);

CREATE TABLE incident_events (
  incident_id TEXT NOT NULL REFERENCES incidents(id) ON DELETE CASCADE,
  event_id    TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
  PRIMARY KEY (incident_id, event_id)
);

-- Append-only. An incident record that can be rewritten is not evidence.
CREATE TABLE incident_notes (
  id          TEXT PRIMARY KEY,
  incident_id TEXT NOT NULL REFERENCES incidents(id) ON DELETE CASCADE,
  user_id     TEXT NOT NULL REFERENCES users(id),
  at          INTEGER NOT NULL,
  text        TEXT NOT NULL
);
CREATE INDEX idx_incident_notes_incident ON incident_notes(incident_id, at);

--------------------------------------------------------------------- evidence
CREATE TABLE recordings (
  id          TEXT PRIMARY KEY,
  camera_id   TEXT NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
  started_at  INTEGER NOT NULL,
  ended_at    INTEGER,
  relative_path TEXT NOT NULL,
  size_bytes  INTEGER NOT NULL DEFAULT 0,
  codec       TEXT NOT NULL,
  mode        TEXT NOT NULL
);
CREATE INDEX idx_recordings_camera_time ON recordings(camera_id, started_at);

CREATE TABLE evidence (
  id           TEXT PRIMARY KEY,
  kind         TEXT NOT NULL,
  camera_id    TEXT REFERENCES cameras(id) ON DELETE SET NULL,
  recording_id TEXT REFERENCES recordings(id) ON DELETE SET NULL,
  captured_at  INTEGER NOT NULL,
  duration_millis INTEGER,
  relative_path TEXT NOT NULL,
  sha256       TEXT NOT NULL,
  size_bytes   INTEGER NOT NULL,
  mime_type    TEXT NOT NULL,
  -- Exempts this item from routine retention cleanup.
  retained_for_incident INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX idx_evidence_sha ON evidence(sha256);

CREATE TABLE incident_evidence (
  incident_id TEXT NOT NULL REFERENCES incidents(id) ON DELETE CASCADE,
  evidence_id TEXT NOT NULL REFERENCES evidence(id) ON DELETE CASCADE,
  PRIMARY KEY (incident_id, evidence_id)
);

------------------------------------------------------------------------- ai
-- Append-only. Records exactly what the analyst was given and what it produced.
CREATE TABLE ai_inferences (
  id             TEXT PRIMARY KEY,
  incident_id    TEXT REFERENCES incidents(id) ON DELETE CASCADE,
  model_id       TEXT NOT NULL,
  prompt_version TEXT NOT NULL,
  generated_at   INTEGER NOT NULL,
  input_event_ids TEXT NOT NULL,
  input_evidence_ids TEXT NOT NULL,
  output         TEXT NOT NULL,
  insufficient_evidence INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX idx_ai_inferences_incident ON ai_inferences(incident_id);

CREATE TABLE model_registry (
  id            TEXT PRIMARY KEY,
  name          TEXT NOT NULL,
  version       TEXT NOT NULL,
  kind          TEXT NOT NULL,
  format        TEXT NOT NULL,
  classes       TEXT NOT NULL,
  input_width   INTEGER NOT NULL,
  input_height  INTEGER NOT NULL,
  runtime       TEXT NOT NULL,
  backend       TEXT NOT NULL,
  precision     TEXT NOT NULL,
  sha256        TEXT NOT NULL,
  size_bytes    INTEGER NOT NULL,
  license       TEXT NOT NULL,
  relative_path TEXT NOT NULL,
  installed_at  INTEGER NOT NULL,
  enabled       INTEGER NOT NULL DEFAULT 1
);

--------------------------------------------------------------------- platform
CREATE TABLE alerts (
  id          TEXT PRIMARY KEY,
  incident_id TEXT REFERENCES incidents(id) ON DELETE CASCADE,
  channel     TEXT NOT NULL,
  created_at  INTEGER NOT NULL,
  delivered_at INTEGER,
  acknowledged_at INTEGER,
  detail      TEXT
);

CREATE TABLE map_packages (
  id           TEXT PRIMARY KEY,
  name         TEXT NOT NULL,
  region       TEXT NOT NULL,
  tile_type    TEXT NOT NULL,
  min_zoom     INTEGER NOT NULL,
  max_zoom     INTEGER NOT NULL,
  bounds       TEXT NOT NULL,
  size_bytes   INTEGER NOT NULL,
  sha256       TEXT NOT NULL,
  relative_path TEXT NOT NULL,
  is_default   INTEGER NOT NULL DEFAULT 0,
  imported_at  INTEGER NOT NULL
);

CREATE TABLE system_settings (
  key        TEXT PRIMARY KEY,
  value      TEXT NOT NULL,
  updated_at INTEGER NOT NULL,
  updated_by TEXT REFERENCES users(id) ON DELETE SET NULL
);

-- Append-only.
CREATE TABLE audit_logs (
  id          TEXT PRIMARY KEY,
  at          INTEGER NOT NULL,
  user_id     TEXT REFERENCES users(id) ON DELETE SET NULL,
  node_id     TEXT NOT NULL,
  request_id  TEXT NOT NULL,
  action      TEXT NOT NULL,
  target_type TEXT NOT NULL,
  target_id   TEXT NOT NULL,
  -- Redacted snapshots. These describe a change; they never carry a credential.
  before_state TEXT,
  after_state  TEXT,
  outcome     TEXT NOT NULL,
  detail      TEXT
);
CREATE INDEX idx_audit_at ON audit_logs(at);
CREATE INDEX idx_audit_action ON audit_logs(action, at);
CREATE INDEX idx_audit_target ON audit_logs(target_type, target_id);
`,
  down: `
DROP TABLE IF EXISTS audit_logs;
DROP TABLE IF EXISTS system_settings;
DROP TABLE IF EXISTS map_packages;
DROP TABLE IF EXISTS alerts;
DROP TABLE IF EXISTS model_registry;
DROP TABLE IF EXISTS ai_inferences;
DROP TABLE IF EXISTS incident_evidence;
DROP TABLE IF EXISTS evidence;
DROP TABLE IF EXISTS recordings;
DROP TABLE IF EXISTS incident_notes;
DROP TABLE IF EXISTS incident_events;
DROP TABLE IF EXISTS incidents;
DROP TABLE IF EXISTS event_tracks;
DROP TABLE IF EXISTS event_zones;
DROP TABLE IF EXISTS events;
DROP TABLE IF EXISTS rules;
DROP TABLE IF EXISTS track_associations;
DROP TABLE IF EXISTS track_observations;
DROP TABLE IF EXISTS tracks;
DROP TABLE IF EXISTS camera_zone_links;
DROP TABLE IF EXISTS zones;
DROP TABLE IF EXISTS camera_topology;
DROP TABLE IF EXISTS camera_profiles;
DROP TABLE IF EXISTS cameras;
DROP TABLE IF EXISTS locations;
DROP TABLE IF EXISTS nodes;
DROP TABLE IF EXISTS users;
`,
};

/**
 * Retention is tiered and applied independently per class of data.
 *
 * Ordinary recordings expire on a short cycle; incident evidence does not expire
 * with them, because the moment a recording becomes evidence it stops being
 * routine footage. Audit logs outlive both.
 */
const retentionPolicies: Migration = {
  version: 2,
  name: 'retention_policies',
  up: `
CREATE TABLE retention_policies (
  id            TEXT PRIMARY KEY,
  data_class    TEXT NOT NULL UNIQUE,
  retain_days   INTEGER NOT NULL,
  -- When true, routine cleanup may never remove this class.
  protected     INTEGER NOT NULL DEFAULT 0,
  updated_at    INTEGER NOT NULL
);

INSERT INTO retention_policies (id, data_class, retain_days, protected, updated_at) VALUES
  ('rp-recordings',        'NORMAL_RECORDING',  7,    0, 0),
  ('rp-event-recordings',  'EVENT_RECORDING',   30,   0, 0),
  ('rp-incident-evidence', 'INCIDENT_EVIDENCE', 3650, 1, 0),
  ('rp-audit',             'AUDIT_LOG',         3650, 1, 0);
`,
  down: `DROP TABLE IF EXISTS retention_policies;`,
};

export const MIGRATIONS: readonly Migration[] = Object.freeze([initial, retentionPolicies]);
