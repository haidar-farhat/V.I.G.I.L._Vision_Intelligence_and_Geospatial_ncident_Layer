# Protocol

> **Status:** specified, not implemented. See [STATUS.md](../STATUS.md). The parts
> that make reconciliation correct — deterministic event identity and idempotent
> correlation — *are* implemented and tested; the transport is not.

## Principles

1. **Every message is versioned.** An unversioned internal API is a future outage.
2. **A version mismatch is a hard, explicit failure** with a human-readable
   message. Never a best-effort parse.
3. **Every message is identified and timestamped**, so replay can be refused and
   traces can be correlated across nodes.
4. **The worker is autonomous.** The protocol exists to synchronise a worker that
   would keep working perfectly well without it.

## Envelope

Every message on every channel:

```jsonc
{
  "v": 1,                                  // protocol version
  "request_id": "01J8XG...",               // unique per message, replay-checked
  "timestamp": 1718158440123,              // UTC milliseconds, window-checked
  "node_id": "nd_7f3a...",                 // derived from the sender's public key
  "kind": "observation.batch",
  "payload": { }
}
```

`request_id` and `timestamp` together are what `ReplayGuard` checks: a captured
message replayed later fails on both the id and the window.

## Channels

| Channel | Transport | Direction | Purpose |
|---|---|---|---|
| `control` | REST over HTTPS (mTLS) | control → worker | configuration, camera assignment, drain, restart |
| `telemetry` | WebSocket | worker → control | heartbeats, camera health, load |
| `observations` | WebSocket | worker → control | tracks, zone observations, events |
| `discovery` | mDNS/DNS-SD + UDP | broadcast | node presence on the local link |
| `ui` | WebSocket | api → desktop | events, camera_status, tracks, incidents, nodes, system |

## Discovery

Service type `_sentinel._tcp.local`, advertising:

```
node_id, name, roles, protocol version, app version, port, fingerprint
```

No external discovery service, no rendezvous server, no STUN/TURN. Manual IP
entry is always available as a fallback, because mDNS is blocked on plenty of
managed switches.

## Pairing

See [SECURITY.md](SECURITY.md). Summary: a discovered node appears as *pending*
and does nothing until a human compares a six-word fingerprint off-channel and
approves it. Approval issues a certificate; from then on all control traffic is
mTLS 1.3 with certificates pinned by `node_id`.

## Observations

The worker sends batches. Ordering within a batch is by `occurred_at`; batches
carry a monotonic `sequence` per node.

```jsonc
{
  "kind": "observation.batch",
  "payload": {
    "sequence": 4821,
    "camera_id": "cam-07",
    "events": [ /* SecurityEvent */ ],
    "tracks": [ /* Track */ ],
    "zone_observations": [ /* ZoneObservation */ ]
  }
}
```

The control node acknowledges a sequence. The worker retains everything after the
last acknowledged sequence.

## Disconnection and reconciliation

This is the part that has to be right, because it is the part nobody tests in the
field until it matters.

**On losing the control node**, a worker:

1. Keeps decoding, detecting, tracking and recording. The sensor does not degrade.
2. Appends to a durable local buffer with a monotonic sequence.
3. Reconnects with bounded exponential backoff plus jitter — never a tight loop
   that turns one outage into a broadcast storm.

**On reconnecting**, it replays from the last acknowledged sequence. Delivery is
at-least-once, so the control node will see duplicates. That is fine, because:

- **Event ids are deterministic**, derived from
  `(node_id, camera_id, rule_id, event_type, track_id, time_bucket)`. A replayed
  event has the same id it had the first time, so persisting it is an idempotent
  upsert rather than a duplicate row.
- **Correlation is idempotent**: attaching an event already present in an incident
  is a no-op.
- **Incident ids are deterministic** too, derived from the event that opened them.

The result converges regardless of the order batches arrive in — which matters,
because after a multi-node outage they will not arrive in order.

Both properties are implemented and covered by tests today
(`services/event-engine/test/correlation.test.ts`), even though the transport
that would exercise them is not built.

## Clock handling

Timestamps are UTC milliseconds everywhere. A worker reports its offset from the
control node in each heartbeat, and the UI surfaces it. **Skew is never silently
corrected**: `events.occurred_at` (per the observing node) and
`events.recorded_at` (per the control node) are separate columns, because the
difference between them is evidence about the deployment.

## Rate limits

Every entry point is limited: login, pairing, REST routes, WebSocket messages,
file imports, export generation. Limits are per-key so one noisy node cannot lock
out another.

## Versioning policy

- The protocol version increments on any wire-visible change.
- A control node states the versions it accepts; a worker outside that range is
  refused with an explanatory message and shown in the UI as needing an upgrade.
- Payloads are validated at the boundary. Untrusted input is never fed to a
  consumer that assumes it is well-formed.
