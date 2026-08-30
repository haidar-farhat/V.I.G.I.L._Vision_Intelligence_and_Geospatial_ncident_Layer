# AI

## Position

The AI is an analyst, not an authority. It may summarise, correlate, explain and
answer questions about recorded events. It may not identify people, infer
personal traits, assert criminality, or command anything physical.

That is not a policy statement bolted onto a prompt. It is enforced by a validator
that rejects reports before an operator sees them, because **a prompt is a
preference and a validator is a guarantee**.

## Pipeline

```
  every frame          selected frames only        selected events only
      |                        |                           |
      v                        v                           v
 +---------+            +------------+              +--------------+
 |Detector |--tracks--->|    VLM     |--structured->|  LLM analyst |
 | (fast)  |            | (targeted) |  observation |  (summarize) |
 +---------+            +------------+              +--------------+
```

**The VLM never sees the whole stream.** It is invoked on a handful of
representative frames for candidate events only. This is the difference between a
system that runs on one GPU and one that cannot exist. Running a vision-language
model at frame rate across sixteen cameras is not a tuning problem; it is a
category error.

## Model abstraction

`Detector`, `Classifier`, `EmbeddingModel`, `Vlm` and `AnalystEngine` are
interfaces. The event engine consumes **typed observations**, never model output,
so a detector can be swapped — YOLO for RT-DETR, ONNX for TensorRT, GPU for CPU —
without a line changing downstream.

That indirection is what makes the model registry meaningful: the system can
honestly record *which artifact* produced a detection precisely because nothing
else in the pipeline assumes what it was.

## Model registry

Every model records `model_id, name, version, format, classes, input size,
runtime, precision, sha256, license, installed_at`. Every event records the model
that produced its detections.

Six months later, "why did the system flag this?" is answerable down to the exact
file and its hash. Without that, a detection is an assertion nobody can check.

- **The SHA-256 is required.** Without it there is no way to prove the file backing
  an old detection is the file that produced it.
- **The licence is required.** Redistribution and deployment both depend on it.
- **Nothing is ever auto-downloaded.** Models are imported from local files and
  validated: checksum, format, metadata, supported runtime, hardware
  compatibility. The relative path is checked for traversal.

## Device selection

`selectDevice` prefers the accelerator with the most VRAM, because stream count is
bounded by memory long before it is bounded by compute. With no accelerator it
falls back to CPU **with a stated reason** rather than failing — losing the GPU
must degrade the system, not stop it.

## Adaptive inference

A camera watching an empty car park does not deserve the same compute as one with
someone climbing a fence.

```
idle camera        2 fps
active camera     15 fps
live display      30 fps (display is not inference)
```

`AdaptiveSampler` runs each camera at its idle rate and steps up the moment
something happens, holding the higher rate for a cooldown. This is what allows one
machine to cover far more cameras than a fixed-rate design.

## The analyst contract

The analyst receives an `EvidenceBundle` and nothing else:

```ts
{ incidentId, events, associations, zones, cameras, windowStart, windowEnd, evidenceIds }
```

It returns an `AiIncidentReport` partitioned into **OBSERVED / INFERRED /
UNKNOWN**. That partition is mandatory, not stylistic: an operator must be able to
tell at a glance which statements are recorded fact and which are the model's
reasoning.

### Guardrails (tested)

| Rule | Enforcement |
|---|---|
| Every factual statement cites evidence | `UNCITED_STATEMENT` |
| Citations must reference evidence actually supplied | `UNKNOWN_EVIDENCE_REFERENCE` |
| Every statement carries a confidence in 0..1 | `MISSING_CONFIDENCE` |
| No identity, criminality, protected traits, enforcement, or physical action | `PROHIBITED_CLAIM` |
| An empty report must declare insufficient evidence | `OBSERVATION_NOT_GROUNDED` |

A report failing any of these **throws** rather than being displayed. An operator
shown "the analyst produced a report that failed validation" is safe; an operator
quietly shown an unvalidated report is not.

`UNKNOWN_EVIDENCE_REFERENCE` is the most important of these. A model that invents
a plausible-looking event id is the single most dangerous failure mode available
to it, because the citation is exactly what makes the claim credible. During
development this rule caught a real bug — the bundle omitted a camera that
appeared only in an association, and the analyst cited a camera the operator had
no context for.

When evidence does not support a conclusion, the response is exactly:

> Insufficient evidence.

## The deterministic analyst

Ships and runs with no language model at all. Two reasons, both load-bearing:

1. **Graceful degradation.** A local LLM is optional. A machine with no spare VRAM
   should still hand the operator a readable incident narrative.
2. **Grounded by construction.** Every sentence is assembled from fields of the
   evidence bundle, so it is structurally incapable of asserting something the
   evidence does not contain. That makes it the reference implementation the
   guardrail suite tests against, and the baseline any LLM-backed engine must beat
   while satisfying the same validator.

It is not a language model and the UI does not present it as one.

## Local LLM engines

Pluggable, and never vendor-specific. Configuration is endpoint, model, context,
temperature, max tokens. The endpoint must resolve to a loopback or private
address; the egress guard enforces it.

**Temperature defaults to 0.** An incident report that differs between two runs
over the same evidence is not a report, it is an opinion.

## Prompt versioning

Prompts are configuration, not prose buried in code. Each has `prompt_id`,
`version`, `purpose`, `created_at`, and every generated report cites the version
that produced it. `ai_inferences` is append-only and records the model, prompt
version, input evidence ids, output and timestamp.

Raw prompts containing operational secrets are not stored.

## Privacy

No facial recognition. No biometric identification. No identity database. The
optional appearance embedding used for cross-camera association is a similarity
vector compared only against other tracks in the same time window — never against
an enrolled set, because there is no enrolled set and no code path to create one.
