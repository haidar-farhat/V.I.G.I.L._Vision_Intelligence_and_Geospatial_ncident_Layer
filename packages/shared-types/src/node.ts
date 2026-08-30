import type { CameraId, NodeId, UtcMillis } from './ids.ts';
import type { ComputeBackend, NodeRole, NodeStatus, Precision } from './enums.ts';

/** A GPU or other accelerator discovered on a node. */
export type ComputeDevice = {
  readonly name: string;
  readonly backend: ComputeBackend;
  readonly vramMb: number | null;
  readonly driverVersion: string | null;
  readonly supportedPrecisions: readonly Precision[];
  /** Rough guidance for the scheduler, not a hard limit. */
  readonly recommendedConcurrentStreams: number;
};

export type NodeCapabilities = {
  readonly canDecode: boolean;
  readonly canInfer: boolean;
  readonly canRecord: boolean;
  readonly canRunVlm: boolean;
  readonly canRunLlm: boolean;
  readonly maxCameras: number | null;
};

export type NodeHardware = {
  readonly os: string;
  readonly osVersion: string;
  readonly arch: string;
  readonly cpuModel: string;
  readonly cpuCores: number;
  readonly memoryMb: number;
  readonly devices: readonly ComputeDevice[];
};

/**
 * A participating machine.
 *
 * `id` is derived from the node's public key, so a node cannot rename itself into
 * another node's trust slot. A node may hold several roles at once.
 */
export type SentinelNode = {
  readonly id: NodeId;
  readonly name: string;
  readonly roles: readonly NodeRole[];
  readonly status: NodeStatus;
  readonly addresses: readonly string[];
  readonly appVersion: string;
  readonly protocolVersion: number;
  readonly hardware: NodeHardware;
  readonly capabilities: NodeCapabilities;
  readonly assignedCameraIds: readonly CameraId[];
  readonly lastHeartbeat: UtcMillis | null;
  /** Fingerprint of the node's public key, rendered as words for humans to compare. */
  readonly identityFingerprint: string;
  readonly pairedAt: UtcMillis | null;
};

/** Periodic health from a worker. Cheap enough to send every second. */
export type NodeHeartbeat = {
  readonly nodeId: NodeId;
  readonly at: UtcMillis;
  readonly cpuPercent: number;
  readonly memoryUsedMb: number;
  readonly gpuPercent: number | null;
  readonly vramUsedMb: number | null;
  readonly diskFreeMb: number;
  readonly camerasOnline: number;
  readonly camerasTotal: number;
  readonly inferenceFps: number;
  readonly queueDepth: number;
  readonly droppedFrames: number;
  /** Events held locally because the control node was unreachable. */
  readonly bufferedEvents: number;
  /** Node clock minus control clock, in milliseconds. Surfaced, never silently corrected. */
  readonly clockOffsetMillis: number | null;
};

/** A node that has announced itself but has not yet been approved by a human. */
export type PendingNode = {
  readonly nodeId: NodeId;
  readonly name: string;
  readonly addresses: readonly string[];
  readonly identityFingerprint: string;
  readonly firstSeen: UtcMillis;
  readonly roles: readonly NodeRole[];
};
