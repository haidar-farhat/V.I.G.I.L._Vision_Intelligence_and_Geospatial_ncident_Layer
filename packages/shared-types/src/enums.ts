/**
 * Closed vocabularies of the domain.
 *
 * These are const objects rather than TypeScript `enum`s: the whole codebase is
 * compiled with `erasableSyntaxOnly` so that services run directly on Node's
 * type-stripping loader with no build step. A const object plus a derived union
 * gives the same ergonomics, erases completely, and serialises as a plain string
 * over the wire and into the database.
 */

// ---------------------------------------------------------------- node / system

export const NodeRole = {
  Control: 'CONTROL',
  Worker: 'WORKER',
  Storage: 'STORAGE',
  Hybrid: 'HYBRID',
} as const;
export type NodeRole = (typeof NodeRole)[keyof typeof NodeRole];

export const NodeStatus = {
  Online: 'ONLINE',
  Degraded: 'DEGRADED',
  Offline: 'OFFLINE',
  Draining: 'DRAINING',
  Pending: 'PENDING_PAIRING',
  Unknown: 'UNKNOWN',
} as const;
export type NodeStatus = (typeof NodeStatus)[keyof typeof NodeStatus];

export const OperatingMode = {
  Standalone: 'STANDALONE',
  LanDistributed: 'LAN_DISTRIBUTED',
} as const;
export type OperatingMode = (typeof OperatingMode)[keyof typeof OperatingMode];

// ------------------------------------------------------------------- cameras

export const CameraStatus = {
  Online: 'ONLINE',
  Degraded: 'DEGRADED',
  Offline: 'OFFLINE',
  Unknown: 'UNKNOWN',
} as const;
export type CameraStatus = (typeof CameraStatus)[keyof typeof CameraStatus];

export const CameraProtocol = {
  Rtsp: 'RTSP',
  Onvif: 'ONVIF',
  Usb: 'USB',
  File: 'FILE',
  Simulated: 'SIMULATED',
} as const;
export type CameraProtocol = (typeof CameraProtocol)[keyof typeof CameraProtocol];

export const StreamKind = {
  Main: 'MAIN',
  Sub: 'SUB',
} as const;
export type StreamKind = (typeof StreamKind)[keyof typeof StreamKind];

export const RecordingMode = {
  Continuous: 'CONTINUOUS',
  Motion: 'MOTION',
  Event: 'EVENT',
  Manual: 'MANUAL',
  Off: 'OFF',
} as const;
export type RecordingMode = (typeof RecordingMode)[keyof typeof RecordingMode];

// ------------------------------------------------------------------- detection

/**
 * Physical, non-biometric object classes.
 *
 * Deliberately excludes identity, demographic and biometric categories: the
 * platform tracks objects, never people-as-identities (see docs/SECURITY.md,
 * "Privacy by design"). Models may report other classes; the event engine treats
 * unknown class strings as opaque and never assigns them meaning.
 */
export const ObjectClass = {
  Person: 'person',
  Vehicle: 'vehicle',
  Car: 'car',
  Truck: 'truck',
  Bus: 'bus',
  Motorcycle: 'motorcycle',
  Bicycle: 'bicycle',
  Animal: 'animal',
  Bag: 'bag',
  Package: 'package',
  Smoke: 'smoke',
  Fire: 'fire',
} as const;
export type ObjectClass = (typeof ObjectClass)[keyof typeof ObjectClass];

/** Any class string a model may emit. Unknown values remain valid but unmapped. */
export type DetectedClass = ObjectClass | (string & {});

// ---------------------------------------------------------------------- zones

export const ZoneKind = {
  Polygon: 'POLYGON',
  Rectangle: 'RECTANGLE',
  Line: 'LINE',
  Circle: 'CIRCLE',
  Corridor: 'CORRIDOR',
} as const;
export type ZoneKind = (typeof ZoneKind)[keyof typeof ZoneKind];

export const ZonePurpose = {
  Restricted: 'RESTRICTED',
  Monitoring: 'MONITORING',
  Perimeter: 'PERIMETER',
  NoEntry: 'NO_ENTRY',
  Parking: 'PARKING',
  Loading: 'LOADING',
  CriticalAsset: 'CRITICAL_ASSET',
} as const;
export type ZonePurpose = (typeof ZonePurpose)[keyof typeof ZonePurpose];

export const CrossingDirection = {
  Any: 'ANY',
  In: 'IN',
  Out: 'OUT',
  Forward: 'FORWARD',
  Backward: 'BACKWARD',
} as const;
export type CrossingDirection = (typeof CrossingDirection)[keyof typeof CrossingDirection];

// --------------------------------------------------------------------- events

export const EventType = {
  PersonEnteredZone: 'PersonEnteredZone',
  PersonExitedZone: 'PersonExitedZone',
  VehicleEnteredZone: 'VehicleEnteredZone',
  VehicleExitedZone: 'VehicleExitedZone',
  VehicleStopped: 'VehicleStopped',
  ObjectLoitering: 'ObjectLoitering',
  LineCrossed: 'LineCrossed',
  AbnormalDirection: 'AbnormalDirection',
  RepeatedApproach: 'RepeatedApproach',
  CrowdDetected: 'CrowdDetected',
  SmokeDetected: 'SmokeDetected',
  FireSuspected: 'FireSuspected',
  CameraTamper: 'CameraTamper',
  CameraOffline: 'CameraOffline',
  MultipleCameraCorrelation: 'MultipleCameraCorrelation',
} as const;
export type EventType = (typeof EventType)[keyof typeof EventType];

export const Severity = {
  Low: 'LOW',
  Medium: 'MEDIUM',
  High: 'HIGH',
  Critical: 'CRITICAL',
} as const;
export type Severity = (typeof Severity)[keyof typeof Severity];

/** Ordinal ranking, for comparisons such as `severity >= HIGH`. */
export const SEVERITY_ORDER: Readonly<Record<Severity, number>> = Object.freeze({
  LOW: 0,
  MEDIUM: 1,
  HIGH: 2,
  CRITICAL: 3,
});

export const compareSeverity = (a: Severity, b: Severity): number =>
  SEVERITY_ORDER[a] - SEVERITY_ORDER[b];

export const EventStatus = {
  New: 'NEW',
  Correlated: 'CORRELATED',
  Suppressed: 'SUPPRESSED',
  Dismissed: 'DISMISSED',
} as const;
export type EventStatus = (typeof EventStatus)[keyof typeof EventStatus];

// ------------------------------------------------------------------ incidents

export const IncidentStatus = {
  New: 'NEW',
  Acknowledged: 'ACKNOWLEDGED',
  Investigating: 'INVESTIGATING',
  Resolved: 'RESOLVED',
  FalsePositive: 'FALSE_POSITIVE',
  Archived: 'ARCHIVED',
} as const;
export type IncidentStatus = (typeof IncidentStatus)[keyof typeof IncidentStatus];

// ------------------------------------------------------------- access control

export const Role = {
  Admin: 'ADMIN',
  Operator: 'OPERATOR',
  Analyst: 'ANALYST',
  Viewer: 'VIEWER',
} as const;
export type Role = (typeof Role)[keyof typeof Role];

// ------------------------------------------------------------------ locations

export const LocationKind = {
  Site: 'SITE',
  Building: 'BUILDING',
  Floor: 'FLOOR',
  Room: 'ROOM',
  Zone: 'ZONE',
  Asset: 'ASSET',
  Checkpoint: 'CHECKPOINT',
} as const;
export type LocationKind = (typeof LocationKind)[keyof typeof LocationKind];

// --------------------------------------------------------------------- models

export const ModelKind = {
  Detector: 'DETECTOR',
  Tracker: 'TRACKER',
  Classifier: 'CLASSIFIER',
  Segmenter: 'SEGMENTER',
  PoseEstimator: 'POSE_ESTIMATOR',
  EmbeddingModel: 'EMBEDDING',
  Vlm: 'VLM',
  Llm: 'LLM',
} as const;
export type ModelKind = (typeof ModelKind)[keyof typeof ModelKind];

export const ComputeBackend = {
  Cpu: 'CPU',
  Cuda: 'CUDA',
  DirectMl: 'DIRECTML',
  Metal: 'METAL',
  Rocm: 'ROCM',
  Vulkan: 'VULKAN',
  Simulated: 'SIMULATED',
} as const;
export type ComputeBackend = (typeof ComputeBackend)[keyof typeof ComputeBackend];

export const Precision = {
  Fp32: 'FP32',
  Fp16: 'FP16',
  Int8: 'INT8',
} as const;
export type Precision = (typeof Precision)[keyof typeof Precision];

// ------------------------------------------------------------- workload modes

export const WorkloadProfile = {
  HighQuality: 'HIGH_QUALITY',
  Balanced: 'BALANCED',
  LowLatency: 'LOW_LATENCY',
  PowerSaver: 'POWER_SAVER',
  Custom: 'CUSTOM',
} as const;
export type WorkloadProfile = (typeof WorkloadProfile)[keyof typeof WorkloadProfile];
