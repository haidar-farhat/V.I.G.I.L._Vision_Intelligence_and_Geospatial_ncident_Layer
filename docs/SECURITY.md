# Security

## Threat model

The system holds continuous video of a physical site, the credentials to every
camera on it, and a map of where those cameras cannot see. An attacker who
obtains any of those gains more than they would from most databases.

### Assumed adversaries

| Adversary | Capability | Primary controls |
|---|---|---|
| **LAN attacker** | On the same subnet. Can send arbitrary traffic to any node and impersonate an unpaired device. | mTLS, pinned node identity, explicit human-approved pairing, replay protection, rate limits |
| **Hostile camera** | A compromised or counterfeit device on a trusted address. Sends malformed RTSP/ONVIF. | Decoder isolation, bounded queues, restart-on-wedge, no parsing in a privileged process |
| **Curious insider** | A valid low-privilege account. | Permission-based authorization, audit on every privileged action, confirmation on destructive ones |
| **Evidence tamperer** | Wants a recording to disappear or change. | Content-addressed evidence, SHA-256 manifests, append-only audit and notes |
| **Supply chain** | A malicious or compromised dependency. | Zero third-party runtime dependencies in core packages; egress guard; no auto-update |
| **Physical thief** | Takes the machine. | Secrets in the OS keychain rather than in files; disk encryption is the operator's responsibility and is documented as such |

### Explicitly out of scope

- A compromised control node with an authenticated admin session. At that point
  the attacker *is* the operator.
- Host operating system compromise.
- Coercion of an authorised operator.

## The LAN is hostile

The single most common mistake in this product category is treating the local
network as a trust boundary. It is not. Possession of a LAN address grants
nothing.

```
 worker                                     control
   |  discovery announce (node_id, pubkey fingerprint, caps)
   |------------------------------------------>
   |                     operator sees a pending node and compares a six-word
   |                     fingerprint off-channel before approving
   |  <---- pairing challenge (nonce, control cert) ----
   |  ---- signed response + CSR ------------->
   |  <---- signed worker cert, trust anchor ----------
   |
   |  === mTLS 1.3 from here; certificates pinned by node_id ===
```

`node_id` is **derived from the node's public key**, so a node cannot rename
itself into another node's trust slot.

Every message carries a request id and a timestamp. `ReplayGuard` refuses a
message outside a narrow time window or with a request id it has already seen, and
bounds its own memory to that window so an attacker cannot drive it into
exhaustion.

## Credentials

**Camera passwords never enter the database, a log, an API payload, a URL visible
to frontend code, an error, or an export.**

The enforcement is structural rather than procedural:

- `Secret<T>` holds the value. `toString`, `toJSON` and the Node inspection hook
  all render `[redacted]`, so string interpolation, `JSON.stringify`, a thrown
  error and `console.log` all fail safe. The value escapes only through an
  explicit `.expose()`, which is greppable and reviewable.
- `buildRtspUrl` is the only function permitted to join a credential to a URL, and
  it returns the result still wrapped.
- `redact()` is the backstop for structures arriving from ONVIF responses and
  imported configuration where the wrapper was never applied. It handles cycles
  and bounds depth, so redaction can never be the thing that crashes the logger.
- The database stores `credentials_ref`, an opaque handle into the OS keychain
  (Windows Credential Manager, macOS Keychain, Linux Secret Service).

A test suite asserts the password does not appear in any serialisation,
inspection or error path, and a schema test walks every column in the database
looking for anything credential-shaped.

## Zero WAN, enforced

The offline guarantee is only worth something if code enforces it.

`EgressGuard` classifies every outbound host before a socket opens and refuses
anything outside the private ranges. Notably, a public DNS name is reported as
`UNRESOLVED` rather than `PUBLIC` — the system will not resolve it to find out,
because resolution itself would require the Internet. Both are refused; they are
simply not conflated.

`npm run lint` fails the build on a cloud SDK import, an analytics package, or a
hard-coded external URL anywhere in the source.

The error message when a connection is refused explains the design decision
rather than just denying, because whoever hits it needs to know it is deliberate:

> Refused to contact "…". Sentinel Vision operates without Internet access by
> design and never falls back to an online service.

## Authorization

Checks are always against a **permission**, never a role name, so adding or
widening a role cannot accidentally open a door elsewhere.

| Role | Can |
|---|---|
| `VIEWER` | See cameras, zones, events, incidents, recordings |
| `OPERATOR` | The above, plus acknowledge/resolve incidents, edit zones, control PTZ, export |
| `ANALYST` | Review, export, and read the audit log; cannot operate cameras |
| `ADMIN` | Everything |

An inactive account holds no permissions regardless of its roles.

Authorization failures state what was required, never what the caller holds —
enumerating a user's permissions to an unauthorised caller is itself a leak.

### High-risk actions

Delete camera, delete evidence, export incident, change retention, control PTZ,
modify node, change rules, edit users. Holding the permission is not sufficient:
the UI requires explicit confirmation, and the action is audited whether it
succeeds, is denied, or errors.

## Audit

`audit_logs` is append-only. Each record carries timestamp, user, node, request
id, action, target, redacted before/after snapshots, and outcome. There is no code
path that updates or deletes a row: a trail that can be rewritten after the fact
is not a trail.

## Privacy by design

- No facial recognition. No biometric identification. No identity database.
- Tracking is appearance-based and identity-free; the optional embedding used for
  cross-camera association is a similarity vector, not an identifier, and is never
  matched against any enrolled set.
- Detection classes are physical and non-biometric.
- The AI analyst is forbidden from asserting identity, inferring protected traits,
  or claiming criminality, and reports violating that are rejected before display.
- Optional face blurring for exported evidence.

## Reporting a vulnerability

This is a portfolio and reference implementation, not a deployed product. If you
find a flaw, open an issue describing the impact and the reproduction. Do not
include real credentials, real footage, or a real site's camera topology.
