# Database

> **Status:** brought back into line with `engine/sentinel/store.py` on
> 2026-09-06. What is built: SQLite in WAL with `synchronous = NORMAL`, eleven
> numbered forward migrations each with a `down`, foreign keys on, an
> integrity check and a newer-schema gate at open, backup and restore through
> SQLite's backup API, and the tables listed under *Entities*. What is
> **not**: PostgreSQL or any `SqlDriver`, migration checksums, and the
> `users`, `nodes`, `tracks`, `alerts`, `ai_inferences` and other tables the
> original design named — those are `PLAN` and are marked so below.

## Choice of engine

**Standalone mode uses SQLite in WAL mode.** It is in Python's standard
library, so a single-machine
installation needs no database service, no daemon to supervise, and no
third-party dependency in the supply chain. All three matter for something
expected to run unattended on an isolated network for months.

**Multi-node deployments were to use PostgreSQL** behind a `SqlDriver`
interface. `PLAN`: nothing of it exists in this codebase. There is one engine,
SQLite, opened by `sentinel.store.Store`.

```
        repositories (typed, domain-shaped)
                    |
            +-------v--------+
            |   SqlDriver    |   exec . query . transaction . script
            +---+--------+---+
                |        |
          sqlite3     PostgreSQL
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

**Append-only tables:** `audit_logs`. No code path updates or deletes a row,
and a test fails if one is added. `incident_notes` and `ai_inferences` are
`PLAN` and do not exist yet.

**Foreign keys are enabled** (`PRAGMA foreign_keys = ON`). SQLite has them off by
default, which silently permits orphaned evidence and dangling incident
references.

## Entities

| Group | Tables that exist (`store.py`, migrations 1–12) | Designed, `PLAN` |
|---|---|---|
| Places | `zones`, `sites` | `locations`, `camera_zone_links` |
| Cameras | `cameras` (with placement, `credentials_ref`, `record`) | `camera_profiles`, `camera_topology` |
| Observation | — | `tracks`, `track_observations`, `track_associations` |
| Analysis | `events` | `rules`, `event_zones`, `event_tracks` |
| Workflow | `incidents`, `incident_events` | `incident_notes`, `alerts` |
| Accounts | `users` (name, salted scrypt hash, role, active) | sessions |
| Evidence | `recordings`, `plate_reads` | `evidence`, `incident_evidence` |
| Identity register | `register_subjects`, `register_identifiers`, `register_sightings` | — |
| Identity and AI | — | `users`, `nodes`, `ai_inferences`, `model_registry` |
| Platform | `audit_logs`, `schema_migrations` | `system_settings`, `map_packages`, `retention_policies` |

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
python tasks.py db            # applied, pending, and the counts
python tasks.py db-migrate    # apply everything pending
python tasks.py db-rollback   # undo the most recent migration
```

Rules:

- **Never modify a live schema by hand.** An operator upgrading an air-gapped
  deployment must get a deterministic result.
- **Every migration is numbered.** Checksums are `PLAN`: editing one that has
  already been applied somewhere is how two deployments silently diverge, and
  today nothing detects it — the rule is a rule of the repository, not of the
  code. A database carrying a migration number this build does not know is
  refused at open with both numbers, so an older build never reads a newer
  schema.
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
- `synchronous = NORMAL` with WAL, set explicitly on every open: durable across
  process crashes, at risk only in a power loss between checkpoints. A recorder
  tolerates that far better than it tolerates an fsync per frame of metadata.
- `PRAGMA quick_check` on every open of a file database; a file that fails it
  is refused with the restore command named, never repaired silently.

## Backup and restore

```bash
sentinel backup                       # data directory/backups/sentinel-<stamp>.db + .sha256
sentinel backup --to E:/site-backups
sentinel restore E:/site-backups/sentinel-20260906-002713.db            # refuses if a database exists
sentinel restore E:/site-backups/sentinel-20260906-002713.db --replace  # moves the current one aside
```

A backup is taken with SQLite's backup API — a consistent snapshot while the
console keeps writing — never with a file copy, because a WAL database in use
is two files and a copy of one of them is a corrupt database that opens. The
`.sha256` beside it is what `restore` checks first; then `quick_check`; then
that the schema is one this build knows. `--replace` moves the current
database, its WAL and its shared-memory file aside as `<name>.replaced-<stamp>`
rather than deleting them. A database the console holds open cannot be moved
on Windows, which is the refusal a live site gets: stop the console first.
Recordings and evidence are not in the backup; they are files, hashed in the
index, and are copied like any other files.
- Indexes cover the queries the UI actually makes: events by time, by camera and
  time, by type and time; incidents by status and time.
- Result rows are returned as plain mappings, deliberately
  retained), which closes a prototype-pollution path.
