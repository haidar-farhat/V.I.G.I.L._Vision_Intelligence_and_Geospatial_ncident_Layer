import type { AuditId, NodeId, RequestId, UserId, UtcMillis } from './ids.ts';
import type { Role } from './enums.ts';

export type User = {
  readonly id: UserId;
  readonly username: string;
  readonly displayName: string;
  readonly roles: readonly Role[];
  readonly active: boolean;
  readonly createdAt: UtcMillis;
  readonly lastLoginAt: UtcMillis | null;
};

/**
 * Granular permissions.
 *
 * Roles are shorthand for sets of these; authorization is always checked against
 * the permission, never against the role name, so adding a role never
 * accidentally widens an existing check.
 */
export const Permission = {
  CameraView: 'camera:view',
  CameraCreate: 'camera:create',
  CameraUpdate: 'camera:update',
  CameraDelete: 'camera:delete',
  CameraPtz: 'camera:ptz',

  ZoneView: 'zone:view',
  ZoneEdit: 'zone:edit',

  RuleView: 'rule:view',
  RuleEdit: 'rule:edit',

  EventView: 'event:view',
  IncidentView: 'incident:view',
  IncidentAcknowledge: 'incident:acknowledge',
  IncidentResolve: 'incident:resolve',
  IncidentExport: 'incident:export',

  EvidenceView: 'evidence:view',
  EvidenceDelete: 'evidence:delete',

  RecordingView: 'recording:view',
  RetentionEdit: 'retention:edit',

  NodeView: 'node:view',
  NodePair: 'node:pair',
  NodeEdit: 'node:edit',

  ModelView: 'model:view',
  ModelInstall: 'model:install',

  MapImport: 'map:import',

  UserView: 'user:view',
  UserEdit: 'user:edit',

  AuditView: 'audit:view',
  SettingsEdit: 'settings:edit',
  DiagnosticsRun: 'diagnostics:run',
} as const;
export type Permission = (typeof Permission)[keyof typeof Permission];

/**
 * Actions that are destructive, irreversible, or physically consequential.
 *
 * Holding the permission is not enough: the UI requires explicit confirmation and
 * the API records an audit entry regardless of outcome.
 */
export const HIGH_RISK_PERMISSIONS: readonly Permission[] = Object.freeze([
  Permission.CameraDelete,
  Permission.EvidenceDelete,
  Permission.IncidentExport,
  Permission.RetentionEdit,
  Permission.CameraPtz,
  Permission.NodeEdit,
  Permission.RuleEdit,
  Permission.UserEdit,
]);

export const ROLE_PERMISSIONS: Readonly<Record<Role, readonly Permission[]>> = Object.freeze({
  ADMIN: Object.freeze(Object.values(Permission)),
  OPERATOR: Object.freeze([
    Permission.CameraView,
    Permission.CameraUpdate,
    Permission.CameraPtz,
    Permission.ZoneView,
    Permission.ZoneEdit,
    Permission.RuleView,
    Permission.EventView,
    Permission.IncidentView,
    Permission.IncidentAcknowledge,
    Permission.IncidentResolve,
    Permission.IncidentExport,
    Permission.EvidenceView,
    Permission.RecordingView,
    Permission.NodeView,
    Permission.ModelView,
    Permission.DiagnosticsRun,
  ]),
  ANALYST: Object.freeze([
    Permission.CameraView,
    Permission.ZoneView,
    Permission.RuleView,
    Permission.EventView,
    Permission.IncidentView,
    Permission.IncidentAcknowledge,
    Permission.IncidentExport,
    Permission.EvidenceView,
    Permission.RecordingView,
    Permission.NodeView,
    Permission.ModelView,
    Permission.AuditView,
  ]),
  VIEWER: Object.freeze([
    Permission.CameraView,
    Permission.ZoneView,
    Permission.EventView,
    Permission.IncidentView,
    Permission.RecordingView,
  ]),
});

// ----------------------------------------------------------------- audit trail

export const AuditAction = {
  UserCreated: 'USER_CREATED',
  UserUpdated: 'USER_UPDATED',
  UserLogin: 'USER_LOGIN',
  UserLoginFailed: 'USER_LOGIN_FAILED',
  CameraAdded: 'CAMERA_ADDED',
  CameraUpdated: 'CAMERA_UPDATED',
  CameraRemoved: 'CAMERA_REMOVED',
  CameraCredentialChanged: 'CAMERA_CREDENTIAL_CHANGED',
  CameraPtzCommand: 'CAMERA_PTZ_COMMAND',
  ZoneCreated: 'ZONE_CREATED',
  ZoneUpdated: 'ZONE_UPDATED',
  ZoneDeleted: 'ZONE_DELETED',
  RuleChanged: 'RULE_CHANGED',
  IncidentAcknowledged: 'INCIDENT_ACKNOWLEDGED',
  IncidentResolved: 'INCIDENT_RESOLVED',
  IncidentMarkedFalsePositive: 'INCIDENT_MARKED_FALSE_POSITIVE',
  EvidenceExported: 'EVIDENCE_EXPORTED',
  EvidenceDeleted: 'EVIDENCE_DELETED',
  NodePaired: 'NODE_PAIRED',
  NodeRemoved: 'NODE_REMOVED',
  ModelChanged: 'MODEL_CHANGED',
  RetentionChanged: 'RETENTION_CHANGED',
  SettingsChanged: 'SETTINGS_CHANGED',
  MapPackageImported: 'MAP_PACKAGE_IMPORTED',
} as const;
export type AuditAction = (typeof AuditAction)[keyof typeof AuditAction];

/**
 * An immutable record of a privileged action.
 *
 * `before` and `after` are redacted snapshots: they describe what changed without
 * ever carrying a credential. The table is append-only.
 */
export type AuditRecord = {
  readonly id: AuditId;
  readonly at: UtcMillis;
  readonly userId: UserId | null;
  readonly nodeId: NodeId;
  readonly requestId: RequestId;
  readonly action: AuditAction;
  readonly targetType: string;
  readonly targetId: string;
  readonly before: Readonly<Record<string, unknown>> | null;
  readonly after: Readonly<Record<string, unknown>> | null;
  readonly outcome: 'SUCCESS' | 'DENIED' | 'ERROR';
  readonly detail?: string;
};
