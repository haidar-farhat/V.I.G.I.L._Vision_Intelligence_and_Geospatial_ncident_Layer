# Testing

## Runner

`node:test`, built into Node. No runner to configure, no dependency to audit, and
the suite runs with the network cable unplugged.

```bash
npm test              # everything
npm run test:watch    # re-run on change
node --test "packages/geometry/test/*.test.ts"   # one package
```

**247 tests**, running in well under a second. That number matters: a suite fast
enough to run on every save is a suite that actually gets run.

## What is tested, and why those things

### Unit — the deterministic core

`geometry`, `tracking`, `event-engine`, `security`, `ai`, `database`. These are
pure and seedable, so their tests are exact rather than statistical.

The tests worth knowing about are the ones that encode a *product* decision rather
than a code path:

| Test | Protects |
|---|---|
| a single pass produces exactly one ENTERED and one EXITED | zone boundaries must not chatter, or dwell timing and alert volume both break |
| planar distance agrees with haversine to under a centimetre | the local-frame approximation is safe at site scale |
| uncertainty grows super-linearly toward the horizon | the map must not show false precision |
| the footprint excludes the blind foreground | an operator must not believe a camera covers ground it cannot see |
| a track survives an occlusion with the same id | fragmenting identity destroys correlation before it starts |
| a physically impossible hand-off is rejected | an impossible candidate must never reach a review queue |
| an unknown route never outscores a known one | missing evidence must not be normalised away |
| three cameras seeing one person produce ONE incident | the entire purpose of the product |
| a replayed event is not counted twice | worker reconnect is at-least-once |
| the password never appears in any serialisation | credential containment |
| a public host is refused with an explanatory error | zero WAN |
| no schema column can hold a credential | credential containment, structurally |
| a position is never stored without its uncertainty | false precision, structurally |
| an AI report citing unsupplied evidence is rejected | fabricated citations are the worst AI failure mode |

### Integration — the vertical slice

`simulator/test/slice.test.ts` runs a full scenario through the **production**
pipeline: detections → tracker → ground projection → zone engine → rules →
cross-camera association → correlator → incident → analyst. Only the camera and
the detector are simulated.

This is the difference between testing components and testing a system. Four real
defects surfaced the first time the slice ran end to end — a tracker that
fragmented three people into 31 tracks, a broken association chain, incidents
counting track segments as people, and an evidence bundle omitting a camera — none
of which any unit test would have caught, because each lived in a seam.

### Ground-truth measurement

The simulator knows where its actors actually are, so the slice does not merely
check that the pipeline produces *a* position — it measures the error:

```
samples               1119
mean error            0.52 m
p95 error             1.27 m
within stated 2-sigma 100.0 %
```

That last line is the important one. It asserts the uncertainty the system reports
actually covers the error it makes. A confident-looking dot that is wrong is worse
than an honest wide ellipse, and this is how that stays true as the code changes.

### Determinism

Every scenario is seeded. The slice asserts that two runs of the same seed produce
identical event ids, identical incident ids and identical risk scores, and that a
*different* seed changes the noise but not the conclusion.

An expectation that depends on unseeded randomness is not an expectation.

### The quiet site

An empty site must raise nothing. This is the hardest test to pass in a real
deployment and the one that decides whether an operator keeps the system switched
on, so it is asserted explicitly rather than assumed.

## Architectural tests

`npm run lint` fails the build on layering violations, cloud SDK imports,
hard-coded external URLs, credential-shaped literals, unfinished markers, and
syntax that does not survive type-stripping. These are tests; they simply run
against the source rather than against behaviour.

## Writing tests here

- **Name the property, not the function.** `a track survives an occlusion and
  keeps its identity` says what breaks if it fails; `test tracker update` does not.
- **Assert the reason in the message.** A failure should explain why the property
  matters, not just that a number differed.
- **Use exact values where the maths allows it.** A camera at 45 degrees down lands
  its image centre at exactly one mount height; that is checkable by hand and
  worth checking that way.
- **Use tolerances honestly.** Floating point never lands on exact decimals, and
  writing `assert.equal(iou(a, a), 1)` produces a test that fails at 1.0000000000009
  for no useful reason.
- **Prefer a fixture that carries a real credential** in the security suite. The
  whole point is that it must not appear in the output.

## Not yet covered

Named here rather than left to be discovered:

- **Chaos and partition testing.** The simulator injects detector dropout and false
  positives. It does not yet kill workers, delay networks, corrupt streams, fill
  disks or restart the database.
- **Security testing beyond unit level.** Authentication, malformed packets, path
  traversal on import, and injection are specified in
  [SECURITY.md](SECURITY.md) but have no integration coverage, because the
  transport and API they would target are not built.
- **Performance.** The numbers in the specification are design targets, not
  measurements. Nothing has been benchmarked against real video.
- **Cross-platform.** Developed and run on Windows. Nothing is platform-specific by
  design — no shell scripts, no native modules, no absolute paths — but Linux and
  macOS have not been exercised.
