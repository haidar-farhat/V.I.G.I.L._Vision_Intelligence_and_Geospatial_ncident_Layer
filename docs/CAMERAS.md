# Cameras

How Sentinel Vision finds, authenticates to, and ingests from IP cameras.

> **Status:** video decode from a file works and is tested against real encoded
> media. RTSP is a code path with a reachability pre-check, and **no physical
> camera has ever been contacted**. Discovery and ONVIF were implemented and
> tested against mock devices in the TypeScript prototype and were removed with
> it; the reasoning below is what the rebuild will follow.
> See [STATUS.md](../STATUS.md).

---

## The shape of the problem

IP cameras are the least standards-compliant hardware most engineers integrate
with. ONVIF is a large specification that every vendor implements a subset of,
differently; RTSP is a small specification that every vendor implements a
superset of, differently. Assume nothing, validate everything, and report failures
precisely enough that an integrator on a ladder knows which screw to turn.

Three consequences run through this whole layer:

1. **Everything on the wire is untrusted.** A camera may be counterfeit,
   compromised, or simply badly written — and the third is overwhelmingly the most
   common. Every parser is bounded, and no device-supplied value is acted on
   without validation.
2. **A credential must never enter a URL.** This is the single most common way
   camera passwords end up in log files, and the countermeasure is structural
   rather than procedural.
3. **Failures are not interchangeable.** "Connection failed" is not a diagnosis.

---

## Onboarding flow

```
 discover (WS-Discovery)          or         enter host + RTSP path by hand
        |                                              |
        v                                              |
 authenticate (WS-Security)                            |
        |                                              |
        v                                              |
 read device info + capabilities                       |
        |                                              |
        v                                              |
 read media profiles  --->  choose inference + recording streams
        |                                              |
        v                                              v
 resolve stream URI  ------------------->  RTSP handshake + codec check
                                                       |
                                                       v
                                            place on map, draw zones, save
```

Every step reports individually, with a remedy. See `testCameraConnection`.

---

## Discovery

ONVIF WS-Discovery: a SOAP Probe sent to the multicast group
`239.255.255.250:3702`, with `ProbeMatch` responses collected for a few seconds.

- **Link-local by construction.** Multicast TTL is 1. The group is not routed off
  the subnet, there is no registry, and no external name is ever resolved.
  Discovery works with the Internet unplugged because it cannot leave the LAN.
- **Every interface is probed separately.** A security host is usually
  multi-homed — a management NIC and a camera VLAN — and a socket bound to the
  wrong one finds nothing while reporting success. That is indistinguishable from
  "there are no cameras" unless each interface is probed on its own.
- **Several probes, not one.** Cameras drop multicast under load. A single probe
  routinely misses a device that is present and working.
- **Advertised addresses are validated.** A `ProbeMatch` carries device-supplied
  `XAddrs`, and a hostile device can claim any address it likes. Anything outside
  the private ranges is discarded rather than followed.

Before authentication, the only identifying detail available is the ONVIF *scope*
vocabulary (`onvif://www.onvif.org/name/Front%20Door`). That is what an operator
uses to tell two unconfigured cameras apart, so it is parsed and displayed.

Devices with no ONVIF, or with ONVIF disabled, are added by host and RTSP path.
Requiring ONVIF would exclude hardware that is otherwise perfectly serviceable.

---

## Authentication

### ONVIF: WS-Security UsernameToken

`PasswordDigest = Base64(SHA1(nonce + created + password))`.

The digest goes on the wire; the password never leaves the function that computes
it. The nonce and timestamp are what stop a captured token being replayed
indefinitely.

SHA-1 here is not a choice — it is what the UsernameToken profile specifies and
what every device implements. It is a replay-limited token rather than a
signature, so its collision weakness is not the relevant property. It is worth
knowing why it appears in a codebase that otherwise uses SHA-256.

### RTSP: Digest, and Basic under protest

Digest (RFC 7616) is preferred and is what almost every camera offers. Basic is
implemented because some devices offer nothing else, but it transmits the password
reversibly, so it is refused unless explicitly enabled per camera.

Two decisions worth knowing:

- **An unimplemented Digest algorithm is refused, not downgraded.** Silently
  falling back to MD5 when a camera asks for SHA-256 produces a failure
  indistinguishable from a wrong password, sending an integrator to re-check
  credentials that were correct all along.
- **A 401 is answered exactly once.** Retrying a rejected credential in a loop is
  how an account gets locked out and how a device log fills with failed-auth
  entries somebody later has to explain. The supervised source goes further: an
  authentication failure stops the camera entirely rather than retrying on a
  timer, because that failure will not resolve itself.

### Where credentials live

In the OS keychain. The database stores only an opaque reference; there is no
column a password could occupy, and a schema test enforces that.

---

## Streams and profiles

A camera typically exposes a **main stream** and a **sub stream**. The assignment
is the single most consequential sizing decision in a deployment:

| Stream | Used for | Why |
|---|---|---|
| Sub (≈640×360, 10 fps) | Inference | Cost scales with pixels; detection quality plateaus well below 4K |
| Main (up to 4K, 25 fps) | Recording | This is the evidence |

Running a detector on a main stream is the most common way a deployment ends up
needing four times the hardware it actually requires. A camera offering only one
profile raises a warning saying exactly that.

### The stream URI trap

ONVIF `GetStreamUri` returns a complete URL, and real firmware genuinely returns
`rtsp://user:pass@host/path`. An integration that passes that string around is
how camera passwords reach log files.

So the URI is **decomposed on arrival** into host, port and path, and never
reassembled with a credential. The RTSP client authenticates in a header instead.
Because no credentialed URL is ever constructed, every error message, log line and
diagnostic in this layer can quote the endpoint verbatim — which is why they do.

The returned host is also treated with suspicion: a NATed camera reports an
address only it can reach, and a hostile one could report somebody else's. The
address the device was actually reached on wins.

---

## RTSP handshake

```
OPTIONS   -> discover methods, learn whether GET_PARAMETER exists
DESCRIBE  -> SDP: media tracks, codecs, control URLs
SETUP     -> transport negotiation, session id, session timeout
PLAY      -> start streaming
...
GET_PARAMETER (or OPTIONS) every timeout/2   -> keep-alive
TEARDOWN
```

- **Interleaved TCP by default.** It traverses the firewalls and NAT a security
  LAN invariably has, and a dropped UDP stream is indistinguishable from a dead
  camera until the RTCP timeout expires.
- **Keep-alive prefers GET_PARAMETER** because it is session-scoped. A camera that
  has silently dropped the session still answers OPTIONS happily, so a client using
  OPTIONS never notices until frames stop.
- **Keep-alive is sent at half the advertised timeout**, so one dropped message on
  a lossy link does not end the session.
- **Control URL resolution** handles absolute, relative and `*` forms, because
  cameras use all three. Getting it wrong produces a SETUP against a URL the device
  does not recognise, which most report as a bare 454 with no explanation.

### Bounded parsing

Response size, header count and SDP size are all capped. A device cannot force
unbounded allocation by declaring a `Content-Length` it never satisfies. Malformed
SDP lines are reported as warnings rather than fatal — refusing to use a camera
because one attribute line is malformed would take working hardware offline over a
cosmetic defect.

### No XML library

ONVIF responses are parsed by a bounded, targeted extractor rather than a general
XML parser. XXE, billion-laughs entity expansion and DTD fetches are all reachable
from parsing untrusted XML permissively, and what is actually needed here is a few
dozen scalar fields. The extractor has no machinery to resolve an entity, follow a
DTD, or make a request, and any response containing a doctype is refused outright —
no legitimate device sends one.

---

## Supervision and recovery

A camera that has run for six months will still drop. The supervised source:

- **Reconnects with bounded exponential backoff and full jitter, forever.** Jitter
  is the part most implementations omit and the part that matters: without it,
  forty cameras that dropped when a switch failed retry in lockstep indefinitely.
- **Stops on faults that will not self-heal.** Wrong credentials, an audio-only
  stream, an undecodable codec. These need a configuration change, and retrying
  them on a timer does harm.
- **Reports honest state.** A camera reconnecting every ten seconds is connected
  at any given instant. Reporting it ONLINE puts a green dot over a real fault, so
  repeated reconnects within a window report DEGRADED with the count.
- **Names the stage that failed.** `RTSP_CONNECT_FAILED`, `RTSP_TIMEOUT`,
  `RTSP_NO_VIDEO_TRACK`, `RTSP_UNSUPPORTED_CODEC`, `ONVIF_AUTH_FAILED` are
  different problems requiring different actions.

---

## Diagnosing a camera that will not add

| Symptom | Usual cause |
|---|---|
| ONVIF unreachable, RTSP fine | ONVIF disabled in the camera's web interface — very common on shipped defaults |
| Credentials rejected on ONVIF, accepted on the web UI | Many cameras keep a separate ONVIF user list; the account must be created explicitly |
| "not RTSP" on connect | Port 80 or 8000 is the web interface, not the stream |
| Connects, then times out | The camera's concurrent-stream limit is reached; disconnect other clients |
| SETUP refused (461) | Transport rejected, usually the same stream-limit problem |
| No video track | Audio-only profile enabled |
| Unsupported codec | Profile set to something outside H.264/H.265; change it on the camera |
| Discovered but unreachable | Camera on a different VLAN, or advertising a NATed address |

The connection test reports each of these distinctly, with the remedy attached. A
test asserts that no failed check may ship without one.

---

## Testing

Both protocol layers are tested against mock devices that misbehave the way real
cameras do: demanding Digest before answering anything, offering only Basic,
splitting responses across segments that end mid-header and mid-body, lying about
`Content-Length`, answering in HTTP, embedding credentials in a stream URI,
reporting an unreachable host, and staying silent.

That is a real bar, and it is not the same bar as hardware. A mock encodes the
author's reading of a specification; a camera encodes its vendor's. The gap
closes only on real devices.
