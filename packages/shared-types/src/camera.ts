import type {
  CameraId,
  CameraProfileId,
  CredentialsRef,
  LocationId,
  NodeId,
  UtcMillis,
  ZoneId,
} from './ids.ts';
import type {
  CameraProtocol,
  CameraStatus,
  DetectedClass,
  RecordingMode,
  StreamKind,
} from './enums.ts';
import type { CameraIntrinsics, CameraPose } from './geo.ts';

/**
 * A media stream a camera exposes. Discovered via ONVIF media profiles, or
 * entered by hand for devices that only speak RTSP.
 */
export type CameraProfile = {
  readonly id: CameraProfileId;
  readonly cameraId: CameraId;
  readonly kind: StreamKind;
  readonly name: string;
  /**
   * Stream path **without credentials**. Never a full `rtsp://user:pass@host` URL:
   * the credentials are resolved from the keychain at connect time and never
   * placed in a string that could reach a log, an error, or the UI.
   */
  readonly path: string;
  readonly codec: string;
  readonly width: number;
  readonly height: number;
  readonly fps: number;
  readonly bitrateKbps: number;
  readonly keyframeIntervalSeconds?: number;
};

/** Per-camera AI policy. Each camera decides how much compute it deserves. */
export type CameraAiPolicy = {
  readonly enabled: boolean;
  /** Inference rate while nothing is happening. */
  readonly idleFps: number;
  /** Inference rate once an event is in progress on this camera. */
  readonly activeFps: number;
  readonly classes: readonly DetectedClass[];
  readonly confidenceThreshold: number;
  readonly trackingEnabled: boolean;
  readonly eventGenerationEnabled: boolean;
};

export type CameraRecordingPolicy = {
  readonly mode: RecordingMode;
  readonly segmentSeconds: number;
  readonly retentionDays: number;
  /** Seconds of buffered video kept before an event's first frame. */
  readonly preEventSeconds: number;
  readonly postEventSeconds: number;
};

export type Camera = {
  readonly id: CameraId;
  readonly name: string;
  readonly description?: string;

  readonly manufacturer?: string;
  readonly model?: string;
  readonly serialNumber?: string;

  readonly protocol: CameraProtocol;
  readonly host: string;
  readonly port: number;
  readonly onvifCapabilities?: readonly string[];

  /**
   * Opaque handle into the OS keychain. Resolving it requires the security
   * package and an authorised caller; the handle itself carries no secret.
   */
  readonly credentialsRef?: CredentialsRef;

  /** Worker node responsible for this camera. Null while unassigned. */
  readonly workerNodeId: NodeId | null;
  readonly locationId: LocationId | null;

  /** Absent until the operator places the camera on the map. */
  readonly pose: CameraPose | null;
  readonly intrinsics: CameraIntrinsics | null;

  readonly zoneIds: readonly ZoneId[];
  readonly ai: CameraAiPolicy;
  readonly recording: CameraRecordingPolicy;

  readonly status: CameraStatus;
  readonly lastSeen: UtcMillis | null;
  readonly ptzSupported: boolean;

  readonly createdAt: UtcMillis;
  readonly updatedAt: UtcMillis;
};

/** Live health, sampled by the owning worker. Never persisted at full rate. */
export type CameraHealth = {
  readonly cameraId: CameraId;
  readonly status: CameraStatus;
  readonly observedAt: UtcMillis;
  readonly fps: number;
  readonly targetFps: number;
  readonly inferenceFps: number;
  readonly droppedFrames: number;
  readonly decodeErrors: number;
  readonly reconnectCount: number;
  readonly bitrateKbps: number;
  readonly latencyMs: number;
  /** Reachability in milliseconds, or null when the probe itself failed. */
  readonly pingMs: number | null;
  readonly lastErrorCode?: string;
};

/** A camera found on the LAN but not yet added. Carries no credentials. */
export type DiscoveredCamera = {
  readonly host: string;
  readonly port: number;
  readonly name?: string;
  readonly manufacturer?: string;
  readonly model?: string;
  readonly onvif: boolean;
  readonly rtsp: boolean;
  readonly discoveredAt: UtcMillis;
  readonly requiresAuthentication: boolean;
};

/**
 * A directed edge of the camera topology graph: "an object leaving A plausibly
 * reappears at B after roughly this long". Operator-editable, and the single
 * biggest lever on cross-camera correlation quality.
 */
export type CameraTopologyEdge = {
  readonly fromCameraId: CameraId;
  readonly toCameraId: CameraId;
  readonly distanceMeters: number;
  readonly minTravelSeconds: number;
  readonly expectedTravelSeconds: number;
  readonly maxTravelSeconds: number;
  /** Operator confidence that this transition is real, 0..1. */
  readonly confidence: number;
  readonly bidirectional: boolean;
};
