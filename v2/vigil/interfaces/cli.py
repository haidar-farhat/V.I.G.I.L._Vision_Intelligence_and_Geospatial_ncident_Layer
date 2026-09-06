"""The `vigil` command. Every service method has a command; nothing else is reachable."""

from __future__ import annotations

import argparse
import getpass
import json
import sys
import time
from pathlib import Path
from typing import Sequence

from .. import logs
from ..adapters.decode import is_live_source, redacted, split_password
from ..config import Settings
from ..domain.geo import CameraPose, LatLon
from ..domain.zones import Schedule
from ..service.alerts import Alerts
from ..service.auth import Accounts, AuthError, Principal, Role
from ..service.evidence import export_incident, verify_package
from ..service.runtime import Runtime
from ..service.site import SiteError, SiteService
from ..service.supervise import clear_stop, install_service, request_stop, service_definition, stop_file, supervise
from ..service.maintenance import StoreError, backup, open_store, restore_backup, verify_backup
from ..service.runtime import RetentionPolicy, apply_retention
from ..version import build_info, describe

_log = logs.get(__name__)


class _Context:
    """One command's world: settings, store, principal. Opened late, closed always."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.settings = Settings.from_environment()
        if args.data_dir:
            self.settings = Settings(Path(args.data_dir), self.settings.alert_file, self.settings.alert_command,
                                     self.settings.alert_webhook, self.settings.allow_public_sources)
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        logs.configure(self.settings.logs, level="DEBUG" if args.verbose else None)
        self.store = open_store(self.settings.database)
        self.accounts = Accounts(self.store)
        self.site = SiteService(self.store, _keychain())
        self.principal = self._sign_in()

    def _sign_in(self) -> Principal:
        secret = None
        if self.args.as_user:
            secret = sys.stdin.readline().rstrip("\r\n") if self.args.password_stdin else getpass.getpass(f"password for {self.args.as_user}: ")
        principal = self.accounts.principal_for(self.args.as_user, secret)
        if principal.origin == "open":
            print("no account exists yet: nothing is gated and the audit trail names the OS account. "
                  "Create one with `vigil users add NAME --role ADMIN`.", file=sys.stderr)
        return principal

    def close(self) -> None:
        self.store.close()


def _keychain():
    from ..adapters.keychain import Keychain

    return Keychain.system()


# ------------------------------------------------------------------ commands


def _console_placeholder(ctx: _Context) -> int:  # pragma: no cover - `main` intercepts it
    from .console.main import run as run_console

    ctx.close()
    return run_console(list(ctx.args.rest))


def _where(ctx: _Context) -> int:
    s = ctx.settings
    print(describe())
    for name, value in (("data", s.data_dir), ("database", s.database), ("logs", s.logs), ("recordings", s.recordings),
                        ("evidence", s.evidence), ("models", s.models), ("default model", s.default_model() or "none")):
        print(f"{name:<14} {value}")
    print(f"{'principal':<14} {ctx.principal.actor}")
    print(f"{'alerts':<14} {Alerts.from_settings(s).describe()}")
    return 0


def _users(ctx: _Context) -> int:
    a, args = ctx.accounts, ctx.args
    try:
        if args.users_command == "add":
            secret = _read_secret(f"password for {args.name}: ", args.stdin)
            if not args.stdin and _read_secret("again: ", False) != secret:
                print("error: the two passwords differ", file=sys.stderr)
                return 1
            user = a.add(args.name, secret, args.role, by=ctx.principal)
            print(f"added {user.name} ({user.role})")
        elif args.users_command == "list":
            users = a.users()
            if not users:
                print("no accounts. Until one exists nothing is gated.")
            for u in users:
                print(f"{u.name:<24} {u.role:<9} {'active' if u.active else 'disabled'}")
        elif args.users_command == "passwd":
            a.set_password(args.name, _read_secret(f"new password for {args.name}: ", args.stdin), by=ctx.principal)
            print(f"changed {args.name}'s password")
        elif args.users_command in ("disable", "enable"):
            a.set_active(args.name, args.users_command == "enable", by=ctx.principal)
            print(f"{args.users_command}d {args.name}")
        elif args.users_command == "role":
            a.set_role(args.name, args.role, by=ctx.principal)
            print(f"{args.name} is now {args.role}")
    except AuthError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


def _cameras(ctx: _Context) -> int:
    site, args, by = ctx.site, ctx.args, ctx.principal
    try:
        if args.cameras_command == "add":
            if split_password(args.source)[1]:
                print("warning: a password on the command line is readable by every process; prefer `vigil cameras password`", file=sys.stderr)
            pose = _pose(args.place) if args.place else None
            camera = site.add_camera(args.id, args.source, name=args.name, pose=pose, record=args.record, by=by)
            print(f"added {camera.id} ({redacted(camera.source)}){' placed' if camera.placed else ''}")
        elif args.cameras_command == "list":
            cameras = site.cameras(by)
            if not cameras:
                print("no cameras")
            for c in cameras:
                print(f"{c.id:<16} {redacted(c.source):<40} {'placed' if c.placed else 'unplaced':<9} {'record' if c.record else ''}")
        elif args.cameras_command == "place":
            camera = site.place_camera(args.id, _pose(args.place), by=by)
            print(f"placed {camera.id} at {camera.pose.position.lat:.6f},{camera.pose.position.lon:.6f}")
        elif args.cameras_command == "record":
            camera = site.set_recording(args.id, args.state == "on", by=by)
            print(f"{camera.id} recording {'on' if camera.record else 'off'}")
        elif args.cameras_command == "password":
            site.set_password(args.id, _read_secret(f"password for {args.id}: ", args.stdin), by=by)
            print(f"stored {args.id}'s password in the keychain")
        elif args.cameras_command == "remove":
            site.remove_camera(args.id, by=by)
            print(f"removed {args.id}")
    except (SiteError, AuthError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


def _zones(ctx: _Context) -> int:
    site, args, by = ctx.site, ctx.args, ctx.principal
    try:
        if args.zones_command == "add":
            ring = [LatLon(float(a), float(b)) for a, b in (p.split(",") for p in args.ring.split(";"))]
            schedule = Schedule(*[int(h) for h in args.closed.split("-")]) if args.closed else None
            zone = site.add_zone(args.id, args.name or args.id, args.kind, ring, watch=[w for w in (args.watch or "").split(",") if w],
                                 enter_after_millis=args.enter_after, exit_after_millis=args.exit_after, schedule=schedule, by=by)
            print(f"added zone {zone.id} ({zone.kind}, {len(zone.ring)} points, watching {', '.join(sorted(zone.watch)) or 'everything'})")
        elif args.zones_command == "list":
            zones = site.zones(by)
            if not zones:
                print("no zones")
            for z in zones:
                print(f"{z.id:<16} {z.name:<20} {z.kind:<11} {len(z.ring)} pts  watch {', '.join(sorted(z.watch)) or '*'}")
        elif args.zones_command == "remove":
            site.remove_zone(args.id, by=by)
            print(f"removed zone {args.id}")
    except (SiteError, AuthError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


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
    if args.source:
        for index, source in enumerate(args.source):
            cid = f"adhoc-{index}" if len(args.source) > 1 else "adhoc"
            if ctx.store.camera(cid) is None:
                ctx.site.add_camera(cid, source, pose=_pose(args.place) if args.place else None, record=args.record, by=by)
    runtime = Runtime(ctx.site, detector_factory=factory, record_to=ctx.settings.recordings if args.record else None,
                      realtime=args.realtime, alerts=Alerts.from_settings(ctx.settings, store=ctx.store),
                      record_every_camera=args.record)
    try:
        started = runtime.start(by, cameras=[f"adhoc-{i}" for i in range(len(args.source))] + (["adhoc"] if len(args.source) == 1 else []) if args.source else None)
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
    incidents = ctx.store.incidents(limit=ctx.args.limit)
    if not incidents:
        print("no incidents")
    for inc in incidents:
        print(f"{inc.id}  {inc.opened_at:%Y-%m-%d %H:%M:%S}  {inc.describe()}")
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


def _site(ctx: _Context) -> int:
    if ctx.args.site_command == "name":
        try:
            ctx.site.name_site(ctx.args.name, ctx.args.timezone, by=ctx.principal)
        except AuthError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        print(f"site is {ctx.args.name} ({ctx.args.timezone})")
    else:
        site = ctx.store.site()
        print(f"{site['name']} ({site['timezone']}); {len(ctx.store.cameras())} camera(s), {len(ctx.store.zones())} zone(s)")
    return 0


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


def _health(ctx: _Context) -> int:
    runtime = Runtime(ctx.site)
    for cid, h in runtime.health().items():
        print(f"{cid:<16} {h.describe()}")
    print(json.dumps(build_info()))
    return 0


# ------------------------------------------------------------------ helpers


def _read_secret(prompt: str, from_stdin: bool) -> str:
    if from_stdin:
        return sys.stdin.readline().rstrip("\r\n")
    return getpass.getpass(prompt)


def _pose(text: str) -> CameraPose:
    """lat,lon,height,heading,pitch[,hfov,vfov,range]"""
    parts = [float(p) for p in text.split(",")]
    if len(parts) < 5:
        raise ValueError("--place is lat,lon,height,heading,pitch[,hfov,vfov,range]")
    pose = CameraPose(LatLon(parts[0], parts[1]), parts[2], parts[3], parts[4], 0.0,
                      parts[5] if len(parts) > 5 else 62.0, parts[6] if len(parts) > 6 else 36.0, parts[7] if len(parts) > 7 else 60.0)
    pose.validate()
    return pose


def _detector_factory(model: Path | None, watch: str | None, confidence: float | None):
    from ..adapters.detectors import WATCHED_LABELS, detector_for, model_info

    classes = frozenset(w.strip().lower() for w in watch.split(",") if w.strip()) if watch else WATCHED_LABELS
    if model is not None:
        model_info(model, classes=classes)  # validates the watch list once, before any thread

    class _Factory:
        def __call__(self):
            return detector_for(model, classes=classes if model is not None else None, confidence=confidence)

    return _Factory()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vigil", description="Sentinel Vision v2 — local-first multi-camera incident intelligence.")
    parser.add_argument("--data-dir", default=None, help="where everything lives (default: VIGIL_DATA_DIR or the OS app-data folder)")
    parser.add_argument("--as", dest="as_user", default=None, metavar="NAME", help="run as this account (password prompted, or on stdin with --password-stdin)")
    parser.add_argument("--password-stdin", action="store_true", help="read the sign-in password from standard input")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--version", action="version", version=describe())
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("where", help="paths, principal, alert sinks").set_defaults(handler=_where)

    users = commands.add_parser("users", help="local accounts")
    uc = users.add_subparsers(dest="users_command", required=True)
    add = uc.add_parser("add"); add.add_argument("name"); add.add_argument("--role", default="OPERATOR", choices=[r.value for r in Role]); add.add_argument("--stdin", action="store_true")
    uc.add_parser("list")
    pw = uc.add_parser("passwd"); pw.add_argument("name"); pw.add_argument("--stdin", action="store_true")
    uc.add_parser("disable").add_argument("name")
    uc.add_parser("enable").add_argument("name")
    role = uc.add_parser("role"); role.add_argument("name"); role.add_argument("role", choices=[r.value for r in Role])
    users.set_defaults(handler=_users)

    cameras = commands.add_parser("cameras", help="the site's cameras")
    cc = cameras.add_subparsers(dest="cameras_command", required=True)
    cadd = cc.add_parser("add"); cadd.add_argument("id"); cadd.add_argument("source", help="a file, device:N, or an RTSP address (user@host, password via `cameras password`)")
    cadd.add_argument("--name"); cadd.add_argument("--place", help="lat,lon,height,heading,pitch[,hfov,vfov,range]"); cadd.add_argument("--record", action="store_true")
    cc.add_parser("list")
    cplace = cc.add_parser("place"); cplace.add_argument("id"); cplace.add_argument("place")
    crec = cc.add_parser("record"); crec.add_argument("id"); crec.add_argument("state", choices=["on", "off"])
    cpw = cc.add_parser("password"); cpw.add_argument("id"); cpw.add_argument("--stdin", action="store_true")
    cc.add_parser("remove").add_argument("id")
    cameras.set_defaults(handler=_cameras)

    zones = commands.add_parser("zones", help="named areas on the ground")
    zc = zones.add_subparsers(dest="zones_command", required=True)
    zadd = zc.add_parser("add"); zadd.add_argument("id"); zadd.add_argument("ring", help="lat,lon;lat,lon;lat,lon[;…]")
    zadd.add_argument("--name"); zadd.add_argument("--kind", default="RESTRICTED", choices=["RESTRICTED", "PERIMETER", "INTEREST"])
    zadd.add_argument("--watch", help="labels, comma-separated (default: every label)")
    zadd.add_argument("--enter-after", type=int, default=600); zadd.add_argument("--exit-after", type=int, default=2000)
    zadd.add_argument("--closed", help="HH-HH hours during which presence is after-hours, e.g. 22-6")
    zc.add_parser("list")
    zc.add_parser("remove").add_argument("id")
    zones.set_defaults(handler=_zones)

    run = commands.add_parser("run", help="analyse the stored cameras, or ad-hoc sources")
    run.add_argument("source", nargs="*", help="files, device:N or RTSP URLs; omit to run the stored cameras")
    run.add_argument("--for", dest="seconds", type=float, default=None, help="stop after this many seconds")
    run.add_argument("--place", help="pose for ad-hoc sources")
    run.add_argument("--model", help="an ONNX model (default: the newest *-seg.onnx in the models folder)")
    run.add_argument("--no-model", action="store_true", help="motion only")
    run.add_argument("--watch", help="labels to track (default: person, bicycle, car, motorcycle, bus, truck)")
    run.add_argument("--confidence", type=float, default=None)
    run.add_argument("--record", action="store_true",
                     help="record every camera for this run, whatever each camera's stored Record flag says")
    run.add_argument("--realtime", action="store_true", help="pace a file to its own frame rate")
    run.add_argument("--stop", action="store_true", help="ask a running analysis to stop, and exit")
    run.set_defaults(handler=_run)

    sup = commands.add_parser("supervise", help="run the product as a child and restart it when it dies")
    sup.add_argument("--max-restarts", type=int, default=None, help="give up after this many (default: never)")
    sup.add_argument("child", nargs="*", help="what to supervise, after `--` (default: run)")
    sup.set_defaults(handler=_supervise)

    svc = commands.add_parser("service", help="register the supervisor with this operating system")
    vc = svc.add_subparsers(dest="service_command", required=True)
    for name in ("install", "print"):
        sub = vc.add_parser(name)
        sub.add_argument("child", nargs="*", help="what the service runs, after `--` (default: run)")
    vc.add_parser("uninstall").set_defaults(child=[])
    svc.set_defaults(handler=_service)

    inc = commands.add_parser("incidents", help="what was concluded"); inc.add_argument("--limit", type=int, default=50); inc.set_defaults(handler=_incidents)
    exp = commands.add_parser("export", help="an incident as an evidence package"); exp.add_argument("incident"); exp.add_argument("--to"); exp.set_defaults(handler=_export)
    ver = commands.add_parser("verify", help="check an evidence package against its manifest"); ver.add_argument("folder"); ver.set_defaults(handler=_verify)
    aud = commands.add_parser("audit", help="the audit trail"); aud.add_argument("--limit", type=int, default=100); aud.set_defaults(handler=_audit)
    al = commands.add_parser("alerts", help="where alerts go, and what is open"); al.add_argument("--test", action="store_true"); al.set_defaults(handler=_alerts)
    bk = commands.add_parser("backup", help="copy the database with a checksum"); bk.add_argument("--to"); bk.set_defaults(handler=_backup)
    rs = commands.add_parser("restore", help="replace the database with a verified backup"); rs.add_argument("backup"); rs.set_defaults(handler=_restore)
    commands.add_parser("health", help="every camera's state").set_defaults(handler=_health)
    # Listed so `--help` names it; `main` hands it the rest of the line.
    console = commands.add_parser("console", help="open the operator console (a window)")
    console.add_argument("rest", nargs="*", help="--as NAME, --for SECONDS, --start, --record, --screenshots DIR")
    console.set_defaults(handler=_console_placeholder)
    site = commands.add_parser("site", help="the site's name and clock")
    sc = site.add_subparsers(dest="site_command", required=True)
    sname = sc.add_parser("name"); sname.add_argument("name"); sname.add_argument("--timezone", default="UTC")
    sc.add_parser("show")
    site.set_defaults(handler=_site)
    ret = commands.add_parser("retention", help="delete old unpreserved clips now")
    ret.add_argument("--max-age-days", type=float, default=14.0); ret.add_argument("--min-free-gib", type=float, default=5.0)
    ret.set_defaults(handler=_retention)
    return parser


#: Flags that may appear before the command, and whether each takes a value.
GLOBAL_FLAGS = {"--data-dir": True, "--as": True, "--password-stdin": False, "--verbose": False, "--version": False}


def _command_of(argv: Sequence[str]) -> tuple[str | None, int]:
    """The command and where it sits, skipping the global flags before it.

    Written out rather than guessed at: `vigil --data-dir X console --for 20`
    put the command third, and a check for `argv[0] == "console"` sent the
    whole line to argparse, which refused it. A camera *called* console is
    still just a value, because this stops at the first non-flag token.
    """
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in GLOBAL_FLAGS:
            index += 2 if GLOBAL_FLAGS[token] else 1
            continue
        if token.startswith("--") and "=" in token and token.split("=", 1)[0] in GLOBAL_FLAGS:
            index += 1
            continue
        if token.startswith("-"):
            return None, index
        return token, index
    return None, index


def main(argv: list[str] | None = None) -> int:
    # The console owns its own store, window and event loop, so it is handed
    # the rest of the command line whole rather than parsed here.
    argv = list(sys.argv[1:] if argv is None else argv)
    command, position = _command_of(argv)
    if command == "console":
        from .console.main import run as run_console

        return run_console(argv[:position] + argv[position + 1:])
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(errors="replace")
        except Exception:  # noqa: BLE001
            pass
    args = build_parser().parse_args(argv)
    try:
        ctx = _Context(args)
    except (StoreError, AuthError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    try:
        return int(args.handler(ctx))
    finally:
        try:
            ctx.close()
        except Exception:  # noqa: BLE001 - already closed by restore
            pass
