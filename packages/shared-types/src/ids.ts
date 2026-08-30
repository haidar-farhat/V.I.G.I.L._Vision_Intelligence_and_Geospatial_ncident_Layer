/**
 * Branded identifier types.
 *
 * A `CameraId` and an `IncidentId` are both strings at runtime, but the compiler
 * refuses to substitute one for the other. In a system that threads dozens of
 * identifiers through correlation, evidence and audit paths, this eliminates an
 * entire category of silent mix-ups at zero runtime cost.
 */

declare const brand: unique symbol;

export type Brand<T, B extends string> = T & { readonly [brand]: B };

export type NodeId = Brand<string, 'NodeId'>;
export type CameraId = Brand<string, 'CameraId'>;
export type CameraProfileId = Brand<string, 'CameraProfileId'>;
export type LocationId = Brand<string, 'LocationId'>;
export type ZoneId = Brand<string, 'ZoneId'>;
export type TrackId = Brand<string, 'TrackId'>;
export type EventId = Brand<string, 'EventId'>;
export type IncidentId = Brand<string, 'IncidentId'>;
export type RuleId = Brand<string, 'RuleId'>;
export type UserId = Brand<string, 'UserId'>;
export type ModelId = Brand<string, 'ModelId'>;
export type RecordingId = Brand<string, 'RecordingId'>;
export type EvidenceId = Brand<string, 'EvidenceId'>;
export type AlertId = Brand<string, 'AlertId'>;
export type MapPackageId = Brand<string, 'MapPackageId'>;
export type AuditId = Brand<string, 'AuditId'>;
export type RequestId = Brand<string, 'RequestId'>;
export type CredentialsRef = Brand<string, 'CredentialsRef'>;

/** Milliseconds since the Unix epoch, always UTC. Never a local-time value. */
export type UtcMillis = Brand<number, 'UtcMillis'>;

/** Construct a `UtcMillis` from a number that is known to be epoch-UTC. */
export const utcMillis = (n: number): UtcMillis => n as UtcMillis;

/**
 * Cast a plain string into a branded id.
 *
 * Deliberately explicit and greppable: every place raw input crosses into the
 * typed domain is visible in a search for `asId(`.
 */
export const asId = <T extends Brand<string, string>>(raw: string): T => raw as T;

export const idToString = (id: Brand<string, string>): string => id as unknown as string;
