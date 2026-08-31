# Protocol

> **Status:** none of this is built in the current codebase. It was implemented
> and tested in the TypeScript prototype — versioned envelopes, the RFC 6455 codec
> and handshake, and the `ui` channel over a running server — and removed with it.
> This document is the design the rebuild will follow, and the reasoning in it is
> what carried over. See [STATUS.md](../STATUS.md).

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

Both properties were implemented and covered by tests in the prototype, and both
are requirements on the rebuild rather than nice-to-haves: without them,
at-least-once delivery after a multi-node outage produces duplicate events and
duplicate incidents, which is the alert-fatigue failure mode this system exists
to avoid.

## Clock handling

Timestamps are UTC milliseconds everywhere. A worker reports its offset from the
control node in each heartbeat, and the UI surfaces it. **Skew is never silently
corrected**: `events.occurred_at` (per the observing node) and
`events.recorded_at` (per the control node) are separate columns, because the
difference between them is evidence about the deployment.

## The realtime connection

Designed. Was driven by 58 tests over real loopback sockets in the prototype; not yet rebuilt.

```
 client                                        server
   |  GET /ws?token=<session>  Upgrade: websocket
   |------------------------------------------------>
   |                     session checked BEFORE the handshake completes
   |  <---- 101 Switching Protocols ----  or  ---- 401 ----
   |
   |  <---- welcome { channels, heartbeatIntervalMillis } ----
   |  ---- subscribe { channels: ["events","incidents"] } ---->
   |  <---- subscribed { subscribed, accepted, rejected } ----
   |  <---- event / incident.opened / track / ... ----
   |  <---- ping ----      (every 30s)
   |  ---- pong ---->
```

Decisions worth knowing:

- **The session is verified before the upgrade completes.** Upgrading first and
  authenticating afterwards leaves a window in which an unauthenticated socket is
  attached to the hub. Logging out closes that operator's stream too.
- **Subscriptions are explicit and per-channel.** The cost that matters is the
  operator's attention as much as the bandwidth: a camera-wall window has no use
  for incident updates. A refused channel is *named* in the reply — a silently
  dropped subscription produces a client waiting forever for a channel it never
  joined.
- **Outbound queues are bounded.** A console whose machine went to sleep still
  holds a socket, and without a cap its backlog grows until the process dies,
  taking every other console with it. One disconnected operator is recoverable; a
  dead control node is not.
- **Heartbeats every 30 seconds.** TCP will not report a yanked cable for minutes,
  so without them a dead connection and an idle one are indistinguishable.
- **Clients cannot inject server messages.** A client sending `event` is refused:
  authentication happens once, validation happens per message.

### Why the codec is written rather than imported

The frame codec parses bytes that arrived from the network, and a security
appliance's dependency list is part of its attack surface. Writing it also makes
the limits decisions taken here rather than inherited from a library's defaults.
Four rules the specification requires and implementations routinely miss, all of
which the rebuild must enforce and test:

| Rule | Why |
|---|---|
| Unmasked client frames fail the connection | Masking exists so a frame cannot be crafted to look like an HTTP request to a proxy |
| Declared length refused from the header alone | A peer must not be able to claim 2 GB and have it allocated on its say-so |
| A 64-bit length above 2^53 cannot round into acceptability | Silent precision loss would turn a refusal into an allocation |
| Fragmentation cannot exceed the message cap | Otherwise the per-frame limit is trivially bypassed |

## REST surface

Versioned, permission-checked, and audited. Every route declares the permission
it requires; `permission: null` is an explicit, greppable decision rather than an
omission, and only health and login use it.

Order of operations per request: **match, authenticate, rate-limit, authorise,
run**. Rate limiting before authorisation means a flood cannot be used to probe
which routes exist.

Every response carries `X-Request-Id`, and the same id appears in the audit record
— so "the export that failed yesterday" is an answerable question.

Errors are `{ error: { code, message, recoverable } }`. Internal failures are
logged redacted and returned sanitised: an internal message can carry a file path,
a query or a credential.

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
