# Roadmap

[STATUS.md](STATUS.md) records what is true today. This file records what I think
is still between that and a system I would be willing to put in front of a real
site, in the order I would build it and with the reason each item comes when it
does.

It is an opinion, not a specification. Where it disagrees with
[ARCHITECTURE.md](ARCHITECTURE.md) the architecture document is the design and
this is the sequencing.

[FEATURES.md](FEATURES.md) is the product definition — every capability the
system is meant to have, with its state. Nothing below is displaced by it, and
five items here turn out to gate the large majority of everything still marked
`PLAN` there: **recording** (1.1), **appearance features** (1.3), **the headless
daemon** (1.2), **accounts** (3.1) and **offline GIS** (4.3).

---

## 1. Where the line actually falls

The spine is built and the perimeter is not. Everything that turns pixels into a
justified conclusion works and is tested; everything that makes it a *system* —
running unattended, on more than one machine, with video to show for it — is
absent.

```mermaid
flowchart LR
    subgraph done["BUILT · 547 tests"]
        direction TB
        D1["decode · detect · track · project"]
        D2["zones · rules · events"]
        D3["correlation · identity · risk"]
        D4["persistence · audit · export"]
        D5["operator console"]
        D6["<b>local cameras</b><br/>through each OS's own capture API"]
    end

    subgraph missing["MISSING · the system around it"]
        direction TB
        M1["<b>recording</b><br/>evidence has no video"]
        M2["<b>headless daemon</b><br/>the console IS the app"]
        M3["<b>networking</b><br/>one machine only"]
        M4["<b>identity</b><br/>no user, no secret store"]
    end

    subgraph unproven["UNPROVEN · never met reality"]
        direction TB
        U1["real footage"]
        U2["trained weights"]
        U3["a physical <i>IP</i> camera"]
        U4["CI has never run"]
    end

    done --> missing
    done --> unproven

    style done fill:#1e3f2f,stroke:#4ade80,color:#e2e8f0
    style missing fill:#4c1d24,stroke:#f87171,color:#e2e8f0
    style unproven fill:#4a3f1e,stroke:#fbbf24,color:#e2e8f0
```

**The one-sentence version:** this is a very well-tested analysis library with a
demo application on top of it, and the gap to a product is recording, a headless
runtime, and having pointed it at something real. It has now been pointed at one
real thing — a camera attached to the machine — and that immediately produced
two surprises local reasoning had not, which is the argument for the rest of
Tier 0.

---

## 2. What I consider genuinely done

Listed only where I would defend the claim, not merely where code exists.

| Area | Why I count it done |
|---|---|
| Ground projection with uncertainty | The mathematics is derived, tested against hand arithmetic, round-trips with the inverse to 1e-6, and refuses rather than guessing when the ray misses the ground. Uncertainty is part of the position, never separated from it. |
| Field of view as an annular sector | A downward-tilted camera cannot see its own mast, and the footprint says so. The buffer is sized for what the core produces and a truncated ring is refused. |
| Three-state semantics, everywhere | Motion is unknown / stationary / moving. Zone membership is inside / outside / uncertain. Position is projected / fallback. Nothing collapses an unknown into a zero. |
| Tracking | Elliptical gating on per-axis extents, coasting with a gap budget, motion withheld until it spans 1.2 s. Its failures — 4 objects for 3 people, 7 identity switches — are measured and bounded by tests. |
| Event → incident correlation | Union-find object identity, transitive across cameras, deterministic ids so replay is idempotent. 13 events become 1 incident on the reference scene. |
| Persistence and evidence | WAL SQLite, forward migrations each with a reversal, append-only audit, and an export folder verifiable by somebody who has only the folder. |
| The C ABI boundary | Version-checked and layout-checked before a single call, with round-trip tests that catch a same-width field swap the size guard cannot. |
| Credential handling | Structural, not procedural: one private slot, one read, everything downstream given the redacted form — including every exception. Nine awkward URLs that each caused a real leak are now tests. |
| Zero WAN | Three enforcement points at three different times, all three tested, and the static one tested against deliberately bad source. |
| Local camera capture | Each platform's own device interface and capture API, with no third-party dependency. Enumeration opens nothing; an index nothing has opened is reported as *assumed*, because there is no supported mapping from an OS device to a capture index. Run end to end against a real webcam. |

---

## 3. The list

Ordered by what I would actually do next, not by size.

```mermaid
flowchart TD
    T0["<b>TIER 0 — establish truth</b><br/>real footage · trained model · an IP camera · run CI"]
    T1["<b>TIER 1 — become a security system</b><br/>recording · headless daemon · appearance re-ID"]
    T2["<b>TIER 2 — become a multi-node system</b><br/>control plane · pairing · worker autonomy"]
    T3["<b>TIER 3 — become operable</b><br/>auth · keychain · discovery · retention · health"]
    T4["<b>TIER 4 — become a product</b><br/>zone drawing · replay · maps · analyst · packaging"]
    T5["<b>TIER 5 — become trustworthy</b><br/>occlusion · terrain · soak · fuzzing"]

    T0 --> T1 --> T2 --> T3 --> T4 --> T5
    T0 -.->|"every number in STATUS<br/>is provisional until this"| T5

    style T0 fill:#4c1d24,stroke:#f87171,color:#e2e8f0
    style T1 fill:#4c1d24,stroke:#f87171,color:#e2e8f0
    style T2 fill:#4a3f1e,stroke:#fbbf24,color:#e2e8f0
    style T3 fill:#4a3f1e,stroke:#fbbf24,color:#e2e8f0
    style T4 fill:#334155,stroke:#94a3b8,color:#e2e8f0
    style T5 fill:#334155,stroke:#94a3b8,color:#e2e8f0
```

### Tier 0 — establish truth

Nothing below this tier means what it appears to mean until this tier is done.
Every accuracy number in STATUS.md is currently a statement about generated
geometry.

- [ ] **0.1 · Run the pipeline over real footage.** One hour of genuine CCTV,
      operator-supplied, kept out of version control. Re-measure recall, mean
      IoU, fragments per frame, identity switches and object count against it,
      and record both numbers side by side rather than replacing the synthetic
      ones. *Done when:* STATUS.md's detection and tracking tables have a
      "real footage" column, however bad it reads.
- [ ] **0.2 · Run a trained ONNX detector.** A person/vehicle model the operator
      exports themselves — nothing downloaded. This exercises class handling,
      confidence calibration and NMS against a model that actually classifies,
      all of which are currently tested against a brightness fixture.
      *Done when:* `class_label` in a stored event names something a model
      believed, not `UNCLASSIFIED`.
- [x] **0.3a · Contact a camera attached to the machine.** Done. `device:N`,
      through each platform's own capture API. The first real hardware this
      system has touched, and it immediately produced two surprises local
      reasoning had not: Media Foundation refuses the development machine's
      integrated camera and DirectShow opens it, and a Windows Hello infrared
      sensor lists as a camera and opens on nothing.
- [ ] **0.3b · Contact one physical IP camera.** RTSP is still a code path, not
      a capability. Expect to find: timestamps that jump, streams that reconnect
      with a different resolution, ONVIF that lies about profiles, and at least
      one credential-handling assumption that is wrong.
      *Done when:* a real IP camera runs for an hour without a reconnect storm.
- [ ] **0.4 · Push once and let CI run.** The workflow targets three platforms
      and Python 3.12; development is on Windows and 3.14. It has never executed.
      *Done when:* the offline job is green on Linux.

### Tier 1 — become a security system

- [x] **1.1 · Continuous recording.** **Done at the engine level, 2026-09-02.**
      Segmented `mp4v` per camera on its own writer thread; pre-event footage by
      construction (continuous + a lead window at export, which made a ring
      buffer unnecessary); retention by age, size and free space, audited, with
      preserved evidence untouchable; an exported incident folder contains its
      clips, the manifest hashes them, and `footage.json` times every gap.
      *Still open from this item:* the console toggle (recording is CLI-only),
      scheduled retention, and true decode/analytic independence — the last is
      1.2's job, and the recorder is already shaped for it (fed before analysis,
      behind a bounded queue).
- [~] **1.2 · A headless engine daemon.** **The engine half is done,
      2026-09-03.** `sentinel node` runs cameras unattended, correlates across
      all of them on a cadence, records, persists and audits, with **no Qt
      import anywhere in the process** — asserted by two tests that import the
      node, and then every engine module, in a subprocess and fail if `PySide`
      appears in `sys.modules`.
      **And the console is now a client of it, same day.** `ConsoleWindow` owns
      no store, no zones, no rule set, no correlation loop and no analysis
      thread; it owns widgets and calls `node.poll()` on a 33 ms repaint. As
      predicted it was a deletion: 623 lines removed against 307 added, and
      `worker.py` — the whole 240-line `QThread` adaptor — is gone.

      *Still open from this item:* true decode/analytic independence. Recording
      survives a slow analytic but not a dead decode, because both still share
      one loop. That needs the decode thread to fan out to a recorder queue and
      an analytic queue separately, and it is the last piece.
- [ ] **1.3 · Appearance-based association.** The tracker reports 4 objects for
      3 people and switches identity 7 times in 388 observations, because
      position and box geometry are all it has. A small appearance descriptor —
      even a colour histogram over the box, before reaching for a re-ID model —
      would fix both, and would make cross-camera association something better
      than "same place, near enough in time".
      *Done when:* object count matches ground truth on the reference scene and
      identity switches drop, both measured.
- [ ] **1.4 · A detector that sees a stationary object.** Background subtraction
      loses the loiterer — 0.49 recall against 0.74 and 0.65 — which is the case
      a security system most needs. Tier 0.2 is the real answer; the interim is a
      hybrid that runs a full-frame detector on a slow cadence and uses motion
      only for proposals between passes.
      *Done when:* the loiterer's recall is the best of the three, not the worst.

### Tier 2 — become a multi-node system

The architecture's whole premise is cameras on several machines correlated
centrally. None of it exists; `services/` is an empty directory left by the
TypeScript prototype.

- [ ] **2.1 · Control plane.** REST for state, WebSocket for the live stream of
      events and track updates, over mTLS, bound to the LAN only. The egress
      guard must be hoisted out of `decode.py` into something every outbound path
      routes through before this lands, or the zero-WAN guarantee quietly becomes
      decode-only in fact as well as in the documentation.
- [ ] **2.2 · Node discovery and human-approved pairing.** mDNS announce, a
      six-word fingerprint the operator compares off-channel, and a pinned node
      identity afterwards. *Treat the LAN as hostile* is the rule; possession of
      a LAN address must grant nothing.
- [ ] **2.3 · Worker autonomy and reconciliation.** A worker keeps decoding,
      detecting, recording and raising events when the control node is gone, and
      reconciles on reconnect. Deterministic event ids already make the
      reconciliation an upsert rather than a merge — that was the point of them.
- [ ] **2.4 · Replay protection and rate limits** on every accepted message,
      bounded so an attacker cannot drive the replay window into exhaustion.

### Tier 3 — become operable

- [ ] **3.1 · Authentication and authorization.** Permission-based, never role
      name checks. The audit half is built and currently attributes every action
      to the literal string `"console"` — there is nobody to name.
- [ ] **3.2 · Secret storage in the OS keychain.** No secret is persisted at all
      today, which is only safe because no camera can be saved with credentials.
      The moment 3.3 lands, this is required, not optional. The `credentials_ref`
      column already exists and holds nothing.
- [ ] **3.3 · Camera discovery (ONVIF / mDNS).** An operator should not be typing
      RTSP URLs. Discovery is also where a hostile camera first touches the
      system, so the parser needs to be treated as such. Local devices are
      already discovered this way — through the operating system rather than the
      network — and `sentinel.devices` is the shape the network version should
      follow: enumerate without connecting, confirm by opening, and never claim
      an identity that has not been established.
- [ ] **3.4 · Retention, disk budget and health.** A node that fills its disk
      stops being a security system. Needs a disk watermark, an eviction policy
      that is audited, per-camera health (frames, drops, reconnects, last-seen)
      and an alert when a camera has been dark long enough to matter — the
      absence of events is itself a signal, and nothing currently reports it.
- [ ] **3.5 · Packaging.** There is no way for an operator to install this. A
      signed installer per platform, with the core shipped as a built library so
      no Rust toolchain is required.

### Tier 4 — become a product

- [ ] **4.1 · Draw a zone on the plan view.** *Add zone* currently places a
      square at a computed radius and the README says so. The zone engine already
      takes any polygon; what is missing is the editor — click to place vertices,
      drag to adjust, snap to a footprint edge, with the live count of how much
      of the polygon each camera can actually adjudicate.
- [ ] **4.2 · Incident replay.** Scrub an incident's timeline with its video and
      its events on the same axis, across every camera that saw it. This is the
      screen an operator actually lives in, and it depends on 1.1.
- [ ] **4.3 · Map package import.** Sandboxed to its directory, offline, with the
      georeferencing checked rather than trusted. A site plan under the metric
      grid is worth more than the grid alone.
- [ ] **4.4 · The grounded AI analyst.** The third pillar — *"the AI is an
      analyst"* — is entirely unbuilt. When it lands it must be structurally
      incapable of asserting anything not present in the evidence it was given,
      it must cite the events it reasoned from, and it must never touch a camera
      or a security action. Local inference only. I would build this **last**,
      because an analyst reasoning over a system that cannot record and has never
      seen a real object would be generating confident prose about nothing.

### Tier 5 — become trustworthy

- [ ] **5.1 · Occlusion and visibility.** Coverage today is a geometric upper
      bound: it models what a camera can *reach*, not what it can *see*. Nothing
      occludes anything. A building on the plan should subtract from the
      footprint, and a zone behind it should report `UNCERTAIN`, not covered.
- [ ] **5.2 · Non-flat ground and lens distortion.** Projection assumes a plane
      and a perfect pinhole. Real sites have slopes and real lenses have
      distortion, and both bias position in ways the current uncertainty model
      does not account for — it is honest about its inputs, and its inputs are
      ideal.
- [ ] **5.3 · A 72-hour soak.** Memory, file handles, thread count, database
      size, disk. The statistics are now bounded; nothing else has been watched
      over a long run.
- [ ] **5.4 · Fuzz the hostile paths.** A malformed stream, a malicious map
      package, an ONVIF response designed to break a parser. *Hostile camera* is
      in the threat model and nothing has been fuzzed.

---

## 4. Things I would change about what exists

Not gaps — decisions I think are wrong, or right-for-now and about to stop being.

| What | Why it bothers me | What I would do |
|---|---|---|
| `apps/console/sentinel_console/app.py`, 963 lines | It owns the store, the rule set, the session lifecycle, the correlation loop and the layout. A `QMainWindow` should not be the orchestrator. | Extract a `Node` object that owns cameras, rules, correlation and persistence with no Qt import. This *is* item 1.2 — the refactor and the daemon are the same work. |
| The egress guard lives inside `VideoSource` | It is correct and it is decode-only. The documentation now says so plainly, but the first non-decode outbound path will silently escape it. | Hoist it before Tier 2, not during. |
| `ZONE_EXIT` and `PERIMETER_BREACH` declared but unraised | Labelled and test-pinned, so it is honest — but two of six event types being reserved is a smell. | Either write the line-crossing predicate `PERIMETER_BREACH` needs, or drop it from the design. |
| Risk scoring weights are constants in source | They are explainable and reproducible, which is the important half. But every site is different, and a weight nobody can change is a weight somebody works around. | Make them a stored, audited, per-site configuration once there is a site to configure. |
| `models/` and `map-data/` are empty directories with READMEs | Correct — nothing may be committed or downloaded. But there is no *import* path either, so an operator has nowhere to put a file from inside the application. | Sandboxed import with digest recording, alongside 0.2 and 4.3. |
| The reference scene is the only scene | It is a good one and it is one. Every measurement is a measurement of it. | A second scene with different geometry, before trusting any tuning decision made against the first. |

---

## 5. If I could only do three things

1. **Recording (1.1).** A security system that does not record is not one. It is
   also the dependency under the screen an operator would actually use.
2. **Real footage and a trained model (0.1, 0.2).** Not because the numbers will
   look good — they will look worse — but because every claim in STATUS.md is
   provisional until then, and the project's own rule is that a capability is not
   complete because it demonstrates.
3. **The headless daemon (1.2).** Everything multi-node is blocked behind it, and
   it is a refactor of code that already exists rather than new territory.

Everything else is real work and none of it changes what this is until those are
done.
