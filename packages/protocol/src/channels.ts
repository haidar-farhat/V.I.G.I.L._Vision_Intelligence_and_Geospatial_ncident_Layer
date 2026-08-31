/**
 * The realtime channels a client may subscribe to.
 *
 * Subscription is explicit and per-channel rather than a firehose, because the
 * cost that matters is the operator's attention as much as the bandwidth: a
 * camera-wall window has no use for incident updates, and an investigation window
 * has no use for per-frame track motion.
 */

export const Channel = {
  /** Security events as they are generated. */
  Events: 'events',
  /** Camera status transitions: ONLINE, DEGRADED, OFFLINE. */
  CameraStatus: 'camera_status',
  /** Track positions. The highest-volume channel by a wide margin. */
  Tracks: 'tracks',
  /** Incidents opening, updating and being acknowledged. */
  Incidents: 'incidents',
  /** Node heartbeats and pairing state. */
  Nodes: 'nodes',
  /** Service health, storage, diagnostics. */
  System: 'system',
} as const;
export type Channel = (typeof Channel)[keyof typeof Channel];

export const ALL_CHANNELS: readonly Channel[] = Object.freeze(Object.values(Channel));

export const isChannel = (value: string): value is Channel =>
  (ALL_CHANNELS as readonly string[]).includes(value);

/** Message kinds carried over the realtime connection. */
export const MessageKind = {
  // client -> server
  Subscribe: 'subscribe',
  Unsubscribe: 'unsubscribe',
  Ping: 'ping',

  // server -> client
  Welcome: 'welcome',
  Subscribed: 'subscribed',
  Event: 'event',
  CameraStatus: 'camera.status',
  Track: 'track',
  IncidentOpened: 'incident.opened',
  IncidentUpdated: 'incident.updated',
  NodeHeartbeat: 'node.heartbeat',
  SystemHealth: 'system.health',
  Pong: 'pong',
  Error: 'error',
} as const;
export type MessageKind = (typeof MessageKind)[keyof typeof MessageKind];

/** Which channel a server message belongs to, for routing to subscribers. */
export const channelForKind = (kind: string): Channel | null => {
  switch (kind) {
    case MessageKind.Event:
      return Channel.Events;
    case MessageKind.CameraStatus:
      return Channel.CameraStatus;
    case MessageKind.Track:
      return Channel.Tracks;
    case MessageKind.IncidentOpened:
    case MessageKind.IncidentUpdated:
      return Channel.Incidents;
    case MessageKind.NodeHeartbeat:
      return Channel.Nodes;
    case MessageKind.SystemHealth:
      return Channel.System;
    default:
      return null;
  }
};
