"""Commands that run the analysis and read what it concluded.

Each function takes the one `_Context` a command runs in — settings, store,
principal — and returns an exit code. They are here rather than in `cli.py`
so that the parser, the dispatch and the work are three things and not one.
"""

from __future__ import annotations

import argparse
import getpass
import json
import sys
import time
from pathlib import Path

from ..adapters.decode import redacted, split_password
from ..domain.geo import CameraPose, LatLon
from ..domain.zones import Schedule
from ..logs import get as _get_logger
from ..service.alerts import Alerts
from ..service.auth import AuthError
from ..service.evidence import export_incident, verify_package
from ..service.maintenance import StoreError, backup, restore_backup, verify_backup
from ..service.runtime import RetentionPolicy, Runtime, apply_retention
from ..service.site import SiteError
from ..service.supervise import clear_stop, install_service, request_stop, service_definition, stop_file, supervise
from ..version import build_info, describe
from .site_commands import _adhoc_id, _pose, _read_secret

_log = _get_logger(__name__)


def _detector_factory(model: Path | None, watch: str | None, confidence: float | None):
    from ..adapters.detectors import WATCHED_LABELS, detector_for, model_info

    classes = frozenset(w.strip().lower() for w in watch.split(",") if w.strip()) if watch else WATCHED_LABELS
    if model is not None:
        model_info(model, classes=classes)  # validates the watch list once, before any thread

    class _Factory:
        def __call__(self):
            return detector_for(model, classes=classes if model is not None else None, confidence=confidence)

    return _Factory()

def _query(args) -> "Query":
    from ..service.search import Query

    return Query(since=args.since, until=args.until, camera=args.camera, zone=args.zone,
                 severity=args.severity, state=getattr(args, "state", None), contains=args.contains,
                 limit=args.limit)

def _run(ctx: _Context) -> int:
    """Analyse the stored cameras (or ad-hoc sources) for a while; print what was concluded."""
    args, by = ctx.args, ctx.principal
    from ..adapters.detectors import DetectionError

    if args.stop:
        path = request_stop(ctx.settings.data_dir)
        print(f"asked the running analysis to stop ({path})")
        return 0
    # A stale request from a run that was killed before it could clear its own
    # file would stop this one on its first poll. Cleared here, before start.
    if clear_stop(ctx.settings.data_dir):
        _log.info("cleared a stop request left by an earlier run")

    model = Path(args.model) if args.model else (None if args.no_model else ctx.settings.default_model())
    try:
        factory = _detector_factory(model, args.watch, args.confidence)
    except DetectionError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    chosen: list[str] | None = None
    if args.source:
        chosen = []
        stored = {c.source: c for c in ctx.site.cameras(by)}
        for source in args.source:
            existing = stored.get(source)
            if existing is not None:
                # The site already knows this source. Running it as a second,
                # unplaced camera — which is what this used to do — means the
                # placement and the zones are silently not applied.
                print(f"using the stored camera {existing.id} for {source}")
                chosen.append(existing.id)
                continue
            cid = _adhoc_id(source, ctx.site.cameras(by))
            ctx.site.add_camera(cid, source, pose=_pose(args.place) if args.place else None, record=args.record, by=by)
            print(f"added {cid} for {source}; remove it later with `vigil cameras remove {cid}`")
            if not args.place:
                print(f"  {cid} is unplaced, so nothing it sees can be located or act on a zone", file=sys.stderr)
            chosen.append(cid)
    runtime = Runtime(ctx.site, detector_factory=factory, record_to=ctx.settings.recordings if args.record else None,
                      realtime=args.realtime, alerts=Alerts.from_settings(ctx.settings, store=ctx.store),
                      record_every_camera=args.record)
    try:
        started = runtime.start(by, cameras=chosen)
    except AuthError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if started == 0:
        print("nothing to run: add a camera first, or name a source", file=sys.stderr)
        return 2
    print(f"running {started} camera(s){' for ' + str(args.seconds) + ' s' if args.seconds else ''}; Ctrl-C stops")
    deadline = time.monotonic() + args.seconds if args.seconds else None
    stop = stop_file(ctx.settings.data_dir)
    last_line = 0.0
    asked_to_stop = False
    try:
        while runtime.running and (deadline is None or time.monotonic() < deadline):
            runtime.poll()
            if stop.exists():
                # The one way to stop a run on every platform, including a
                # scheduled task with no terminal. Cleared here so the
                # supervisor sees it too, then does not restart.
                asked_to_stop = True
                print("a stop was requested")
                break
            if time.monotonic() - last_line >= 5:
                last_line = time.monotonic()
                for cid, h in runtime.health().items():
                    print(f"  {cid:<12} {h.describe()}")
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("stopping")
    finally:
        runtime.stop(by)
        if asked_to_stop:
            clear_stop(ctx.settings.data_dir)
    incidents = runtime.incidents
    print(f"{ctx.store.event_count()} event(s) -> {len(incidents)} incident(s)")
    for inc in incidents:
        print(f"  {inc.describe()}")
    for alert in runtime.alerts.active():
        print(f"  ALERT {alert.kind} {alert.subject}: {alert.detail}")
    return 0

def _incidents(ctx: _Context) -> int:
    from ..service.search import Search, SearchError

    query = _query(ctx.args)
    try:
        incidents = Search(ctx.store).incidents(query, by=ctx.principal)
    except SearchError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if not incidents:
        print(f"no incidents matching {query.describe()}")
    for inc in incidents:
        print(f"{inc.id}  {inc.opened_at:%Y-%m-%d %H:%M:%S}  {inc.describe()}")
        print(f"    {inc.review.describe()}")
    return 0

def _events(ctx: _Context) -> int:
    """The raw assertions, for a question an incident summary cannot answer."""
    from ..service.search import Search, SearchError

    query = _query(ctx.args)
    try:
        events = Search(ctx.store).events(query, by=ctx.principal)
    except SearchError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if not events:
        print(f"no events matching {query.describe()}")
    for e in events:
        # The summary already names the zone; repeating it read as
        # "A person entered Forecourt in Forecourt".
        print(f"{e.occurred_at:%Y-%m-%d %H:%M:%S}  [{e.severity}] {e.summary}")
        print(f"    {e.evidence.camera_id} track {e.evidence.track_id}, rule {e.rule_id}, "
              f"confidence {e.confidence:.2f}, drawn by {e.evidence.detector.name}")
    return 0

def _review(ctx: _Context) -> int:
    """Acknowledge an incident, or dismiss it with a reason."""
    from ..service.review import IncidentReview, ReviewError

    review = IncidentReview(ctx.store)
    try:
        if ctx.args.judgement == "ack":
            incident = review.acknowledge(ctx.args.incident, by=ctx.principal, note=ctx.args.note)
        elif ctx.args.judgement == "dismiss":
            incident = review.dismiss(ctx.args.incident, by=ctx.principal, note=ctx.args.note or "")
        else:
            incident = review.reopen(ctx.args.incident, by=ctx.principal, note=ctx.args.note)
    except (ReviewError, AuthError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"{incident.id}: {incident.review.describe()}")
    return 0

def _export(ctx: _Context) -> int:
    incident = ctx.store.incident(ctx.args.incident)
    if incident is None:
        print(f"error: no incident {ctx.args.incident}", file=sys.stderr)
        return 1
    try:
        folder = export_incident(ctx.store, incident, Path(ctx.args.to or ctx.settings.evidence), by=ctx.principal)
    except AuthError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"exported to {folder}")
    return 0

def _verify(ctx: _Context) -> int:
    problems = verify_package(Path(ctx.args.folder))
    if problems:
        for p in problems:
            print(f"  {p}")
        return 1
    print("every file matches the manifest")
    return 0

def _audit(ctx: _Context) -> int:
    from ..service.auth import AUDIT_READ

    if ctx.principal.origin == "user" and not ctx.principal.may(AUDIT_READ):
        print("error: reading the audit trail needs an analyst or administrator account", file=sys.stderr)
        return 1
    for row in reversed(ctx.store.audit_trail(limit=ctx.args.limit)):
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(row["at"] / 1000))
        print(f"{stamp}  {row['principal']:<20} {row['action']:<24} {row['subject'] or '':<20} {row['detail'] or ''}")
    return 0

def _alerts(ctx: _Context) -> int:
    alerts = Alerts.from_settings(ctx.settings, store=ctx.store)
    print(alerts.describe())
    for sink in alerts.sinks:
        print(f"  {type(sink).__name__}: {getattr(sink, 'path', None) or getattr(sink, 'url', None) or getattr(sink, 'argv', None)}")
    if ctx.args.test:
        alerts._synchronous = True
        alerts.raise_("test", "operator", "a test alert from `vigil alerts --test`")
        alerts.clear("test", "operator")
        print("test alert raised and cleared through every sink")
    open_alerts = ctx.store.open_alerts()
    print(f"{len(open_alerts)} open alert(s)")
    for row in open_alerts:
        print(f"  {row['kind']} {row['subject']}: {row['detail']}")
    return 0

def _backup(ctx: _Context) -> int:
    target = Path(ctx.args.to) if ctx.args.to else ctx.settings.data_dir / "backups" / time.strftime("vigil-%Y%m%d-%H%M%S.db")
    path = backup(ctx.store, target)
    print(f"backed up to {path} (checksum beside it)")
    return 0

def _restore(ctx: _Context) -> int:
    ctx.store.close()
    try:
        verify_backup(ctx.args.backup)
        restore_backup(ctx.args.backup, ctx.settings.database)
    except StoreError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"restored {ctx.settings.database} from {ctx.args.backup}")
    return 0

def _supervise(ctx: _Context) -> int:
    """Run the product as a child and start it again when it dies."""
    arguments = list(ctx.args.child or ["run"])
    if "--data-dir" not in arguments:
        arguments = ["--data-dir", str(ctx.settings.data_dir), *arguments]
    from ..service.supervise import child_command

    ctx.store.audit(ctx.principal.actor, "supervisor.started", None, " ".join(arguments))
    ctx.close()
    return supervise(child_command(arguments), data_dir=ctx.settings.data_dir, max_restarts=ctx.args.max_restarts)

def _service(ctx: _Context) -> int:
    """Register the supervisor with this operating system, or print what would be."""
    import sys as _sys

    arguments = list(ctx.args.child or ["run"])
    if "--data-dir" not in arguments:
        arguments = ["--data-dir", str(ctx.settings.data_dir), *arguments]
    if ctx.args.service_command == "print":
        definition = service_definition(_sys.platform, arguments)
        print(f"{definition['kind']} for {_sys.platform}")
        if definition["path"] is not None:
            print(f"file: {definition['path']}")
            print(definition["content"])
        print("install:   " + " ".join(str(p) for p in definition["install"]))
        print("uninstall: " + " ".join(str(p) for p in definition["uninstall"]))
        print(definition["note"])
        return 0
    from ..service.auth import SITE_CONFIGURE

    if not ctx.principal.may(SITE_CONFIGURE):
        print("error: installing a service needs an operator or administrator account", file=sys.stderr)
        return 1
    if ctx.args.service_command == "install":
        code, definition = install_service(arguments)
        ctx.store.audit(ctx.principal.actor, "service.installed", definition["kind"], " ".join(arguments))
    else:
        from ..service.supervise import uninstall_service

        code, definition = uninstall_service()
        ctx.store.audit(ctx.principal.actor, "service.uninstalled", definition["kind"])
    print(f"{'installed' if ctx.args.service_command == 'install' else 'removed'} the {definition['kind']}"
          f"{'' if code == 0 else f' (the platform command exited {code})'}")
    print(definition["note"])
    return code

def _retention(ctx: _Context) -> int:
    """Sweep now, with the policy given, and say what could not be met."""
    from ..service.auth import SITE_CONFIGURE

    if not ctx.principal.may(SITE_CONFIGURE):
        print("error: retention needs an operator or administrator account", file=sys.stderr)
        return 1
    policy = RetentionPolicy(max_age_days=ctx.args.max_age_days, max_bytes=None,
                             min_free_bytes=int(ctx.args.min_free_gib * 1024**3) if ctx.args.min_free_gib is not None else None)
    shortfall = apply_retention(ctx.store, policy, principal=ctx.principal.actor)
    print(f"{ctx.store.recorded_bytes() / 1024**3:.2f} GiB recorded after the sweep")
    if shortfall:
        print(shortfall)
        return 1
    return 0

def _doctor(ctx: _Context) -> int:
    """Is this installation going to work? The last line of an install script."""
    from ..service.diagnostics import State, run_checks, worst

    from .cli import _keychain

    checks = run_checks(ctx.settings, ctx.store, _keychain(), probe=ctx.args.probe)
    for check in checks:
        print(check.describe())
    verdict = worst(checks)
    print()
    print(f"{verdict}: {sum(1 for c in checks if c.state is State.FAIL)} failing, "
          f"{sum(1 for c in checks if c.state is State.WARN)} to look at, "
          f"{sum(1 for c in checks if c.state is State.OK)} fine")
    return 1 if verdict is State.FAIL else 0

def _health(ctx: _Context) -> int:
    runtime = Runtime(ctx.site)
    for cid, h in runtime.health().items():
        print(f"{cid:<16} {h.describe()}")
    print(json.dumps(build_info()))
    return 0
