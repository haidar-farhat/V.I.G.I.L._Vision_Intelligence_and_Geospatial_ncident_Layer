# Database

## Choice of engine

**Standalone mode uses `node:sqlite`.** It ships inside Node, so a single-machine
installation needs no database service, no daemon to supervise, and no
third-party dependency in the supply chain. All three matter for something
expected to run unattended on an isolated network for months.

**Multi-node deployments use PostgreSQL** behind the same `SqlDriver` interface.
Repositories never see the difference; migrations are authored once in portable
SQL.

```
        repositories (typed, domain-shaped)
                    |
            +-------v--------+
            |   SqlDriver    |   exec . query . transaction . script
            +---+--------+---+
                |        |
        node:sqlite   PostgreSQL
```

## Conventions

These hold across every table.

**Timestamps are `INTEGER` milliseconds since the Unix epoch, UTC.** Local time
exists only in the presentation layer. Timezone is a display setting, never a
storage format.

**Camera time and node time are separate columns and are never reconciled.**
`events.occurred_at` is when the observing node says it happened;
`events.recorded_at` is when the control node durably accepted it. The difference
is evidence about the deployment — a camera with a drifting clock is a fact worth
keeping, not noise to smooth away.

**No table holds a credential.** `cameras.credentials_ref` is an opaque handle
into the OS keychain. A test walks every column in the schema and fails on
anything credential-shaped; `users.password_hash` is the single audited exception,
and it is a one-way hash of a local operator password, never a device credential.

**A position is never stored without its uncertainty.** Any table with
`latitude`/`longitude` also carries `uncertainty_meters` and `position_source`. A
test enforces it. Storing a coordinate without its error is how false precision
gets into a map.

**Append-only tables:** `audit_logs`, `incident_notes`, `ai_inferences`. No code
path updates or deletes them.

**Foreign keys are enabled** (`PRAGMA foreign_keys = ON`). SQLite has them off by
default, which silently permits orphaned evidence and dangling incident
references.

## Entities

| Group | Tables |
|---|---|
| Identity | `users`, `nodes` |
| Places | `locations`, `zones`, `camera_zone_links` |
| Cameras | `cameras`, `camera_profiles`, `camera_topology` |
| Observation | `tracks`, `track_observations`, `track_associations` |
| Analysis | `rules`, `events`, `event_zones`, `event_tracks` |
| Workflow | `incidents`, `incident_events`, `incident_notes`, `alerts` |
| Evidence | `recordings`, `evidence`, `incident_evidence` |
| AI | `ai_inferences`, `model_registry` |
| Platform | `system_settings`, `map_packages`, `retention_policies`, `audit_logs` |

The ERD is in [ARCHITECTURE.md](../ARCHITECTURE.md#7-domain-model-erd).

### A note on `incidents.distinct_object_count`

`incident.trackIds` lists every track *segment* that contributed. One person
crossing three cameras appears there three times, because each segment is separate
evidence. `distinct_object_count` is how many objects those segments are believed
to represent after cross-camera association.

They are separate columns because conflating them produces an incident titled
"6 people" when three walked past — a visible error that would undermine trust in
everything else on the screen.

## Migrations

```bash
npm run db status     # applied, pending, and any integrity problems
npm run db migrate    # apply everything pending
npm run db rollback   # undo the most recent migration
```

Rules:

- **Never modify a live schema by hand.** An operator upgrading an air-gapped
  deployment must get a deterministic result.
- **Every migration is numbered and checksummed.** Editing one that has already
  been applied somewhere is how two deployments silently diverge, so it is
  detected and raised as an *unrecoverable* error rather than a warning.
- **Every migration carries a `down`.** An upgrade that cannot be undone on a
  machine with no Internet and no spare hardware is a gamble, not an upgrade.
- **Migrations run in version order**, not declaration order, so one added on a
  branch and merged out of sequence still applies deterministically.
- **Each runs in its own transaction.** A failure leaves nothing partially
  applied, and nothing recorded as applied.

## Transactions

`driver.transaction(fn)` is synchronous by design. SQLite transactions are
connection-scoped, and an `await` inside one would interleave unrelated work into
the same transaction.

Nesting uses **savepoints**, because repositories compose — writing an incident
also writes its events — and an inner failure must be able to roll back without
abandoning the outer unit of work.

## Retention

Retention is tiered and applied per data class:

| Class | Default | Protected |
|---|---|---|
| `NORMAL_RECORDING` | 7 days | no |
| `EVENT_RECORDING` | 30 days | no |
| `INCIDENT_EVIDENCE` | 10 years | **yes** |
| `AUDIT_LOG` | 10 years | **yes** |

Protected classes are never removed by routine cleanup. The moment a recording
becomes incident evidence it stops being routine footage, and a retention job that
deletes it because it is eight days old has destroyed the only reason the system
was installed.

## Performance notes

- WAL is enabled for file-backed databases so the recorder and API can read while
  the event engine writes, which is the normal state of this system.
- `synchronous = NORMAL` with WAL: durable across process crashes, at risk only in
  a power loss. A recorder tolerates that far better than it tolerates an fsync per
  frame of metadata.
- Indexes cover the queries the UI actually makes: events by time, by camera and
  time, by type and time; incidents by status and time.
- Result rows have a **null prototype** (a `node:sqlite` property, deliberately
  retained), which closes a prototype-pollution path.
