# Development

## Requirements

**Node 22.6 or later** (Node 24 recommended). That is the entire requirement for
the core, and it is deliberate:

- Node's **type-stripping loader** runs TypeScript directly, so services have no
  build step. `node simulator/src/cli.ts` just works.
- **`node:sqlite`** is the embedded database, so standalone mode needs no database
  service installed.
- **`node:test`** is the test runner, so there is no runner to configure and the
  suite runs offline.

`typescript` is the only dev dependency, used for typechecking. The desktop app
additionally needs a **Rust toolchain** for Tauri, and its own npm dependencies.

```bash
npm install
npm test
```

## The zero-dependency rule

Everything under `packages/` and `services/` has **no third-party runtime
dependencies**. This is a hard rule, not a preference:

- For a security product the supply chain is part of the threat model. A
  transitive dependency that phones home would silently break the platform's
  headline guarantee.
- The domain becomes auditable by reading rather than by trusting.
- The suite runs in well under a second, which changes how often it gets run.

Adding a runtime dependency to a core package requires an architecture decision
recorded in ARCHITECTURE.md. The desktop app is exempt: React, MapLibre and Tauri
are vendored at build time and ship inside the bundle.

## Erasable syntax only

Because the loader strips types rather than compiling them, source must contain
**only erasable type syntax**:

| Not allowed | Use instead |
|---|---|
| `enum Foo {}` | `const Foo = {...} as const` plus a derived union |
| `namespace Foo {}` | a module |
| `constructor(private x: T)` | an explicit field assignment |

Relative imports carry a `.ts` extension. `npm run lint` and `tsc` both catch
violations.

## Layout and layering

Dependencies point **inward**:

```
apps/  ->  services/  ->  packages/
```

A package never imports a service or an app. A service never imports an app.
`npm run lint` fails the build on violations, because a rule that lives only in a
document erodes.

## Commands

| Command | Does |
|---|---|
| `npm test` | Full suite |
| `npm run test:watch` | Re-run on change |
| `npm run typecheck` | Strict TypeScript across the workspace |
| `npm run lint` | Layering, zero-WAN, secrets, placeholders, erasable syntax |
| `npm run slice` | End-to-end vertical slice, printed |
| `npm run simulator` | Camera simulator alone |
| `npm run db <cmd>` | `migrate` / `rollback` / `status` |

All identical on Windows, Linux and macOS: every command routes through Node, so
there is no `.sh` / `.ps1` pair to drift apart.

## Working on the pipeline

The fastest loop for anything touching detection, tracking, zones, rules or
correlation is the slice:

```bash
npm run slice
```

It runs a scripted scenario through the production pipeline with a fixed seed and
prints every stage. Because the simulator carries **ground truth**, it also
reports the pipeline's real spatial error rather than assuming it away. A change
that degrades tracking or projection shows up immediately as a worse number, and
`simulator/test/slice.test.ts` fails on the ones that matter.

Add a scenario in `simulator/src/scenario.ts`. Keep it seeded: an expectation that
depends on unseeded randomness is not an expectation.

## Adding a feature

1. Model it in `packages/shared-types` first. Every other layer agrees there.
2. Put the logic in the innermost layer that can hold it. Pure geometry belongs in
   `geometry`, not in a service.
3. Write the test with the feature, not after it.
4. Update `ARCHITECTURE.md` in the same commit as any architectural change.
5. Update `STATUS.md`. A capability is not `TESTED` because a UI for it exists.

## Commits

Conventional commits, scoped to the package or service:

```
feat(tracking): ...
fix(event-engine): ...
docs: ...
build: ...
security: ...
```

Explain *why* in the body, especially for a non-obvious trade-off. The commit log
is the only place a future reader will find the reasoning behind a threshold, a
weighting or a rejected alternative.
