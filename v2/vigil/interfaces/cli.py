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
from ..adapters.decode import redacted, split_password
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

from .map_commands import add_arguments as _add_map_arguments
from .site_commands import _cameras, _pose, _read_secret, _site, _users, _where, _zones
from .work_commands import (
    _alerts, _audit, _backup, _doctor, _events, _export, _health, _incidents, _restore, _retention,
    _review, _run, _service, _supervise, _verify,
)

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












































# ------------------------------------------------------------------ helpers










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
    ccal = cc.add_parser("calibrate", help="measure a placed camera's pose from points you can find on both the picture and a map")
    ccal.add_argument("id")
    ccal.add_argument("--points", required=True,
                      help="u,v,lat,lon[,label]; … — u and v are fractions of the frame, 0,0 top-left. At least four, spread across the frame and across the range")
    ccal.add_argument("--solve-position", action="store_true",
                      help="also solve where the camera is. Off by default: a click on a map is worth about a metre and the orientation is worth two degrees, and at 40 m the second matters more")
    ccal.add_argument("--dry-run", action="store_true", help="report the fit without saving it")
    crec = cc.add_parser("record"); crec.add_argument("id"); crec.add_argument("state", choices=["on", "off"])
    cpw = cc.add_parser("password"); cpw.add_argument("id"); cpw.add_argument("--stdin", action="store_true")
    csrc = cc.add_parser("source", help="the camera moved to a new address; its placement stays")
    csrc.add_argument("id"); csrc.add_argument("source")
    cren = cc.add_parser("rename"); cren.add_argument("id"); cren.add_argument("name")
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
    zedit = zc.add_parser("edit", help="change what a zone means, keeping the ring it was drawn with")
    zedit.add_argument("id")
    zedit.add_argument("--name"); zedit.add_argument("--kind", choices=["RESTRICTED", "PERIMETER", "INTEREST"])
    zedit.add_argument("--watch", help="labels, comma-separated; empty for every label")
    zedit.add_argument("--enter-after", type=int); zedit.add_argument("--exit-after", type=int)
    zedit.add_argument("--closed", help="HH-HH, or `none` to clear the schedule")
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

    def _filters(parser, *, default_limit: int) -> None:
        parser.add_argument("--limit", type=int, default=default_limit)
        parser.add_argument("--since", help="2h, 3d, 2026-09-01, or a full ISO moment")
        parser.add_argument("--until", help="the same forms")
        parser.add_argument("--camera", help="only this camera")
        parser.add_argument("--zone", help="only this zone")
        parser.add_argument("--severity", help="this severity and worse")
        parser.add_argument("--contains", help="text in the summary")

    inc = commands.add_parser("incidents", help="what was concluded, and what is still waiting on a person")
    inc.add_argument("--state", default="queue", choices=["queue", "all", "new", "acknowledged", "dismissed"],
                     help="queue (new and acknowledged, the default), all, or one state")
    _filters(inc, default_limit=50)
    inc.set_defaults(handler=_incidents)
    evt = commands.add_parser("events", help="the raw assertions behind the incidents")
    _filters(evt, default_limit=100)
    evt.set_defaults(handler=_events)
    rev = commands.add_parser("review", help="acknowledge an incident, or dismiss it with a reason")
    rev.add_argument("incident")
    rev.add_argument("judgement", choices=["ack", "dismiss", "reopen"])
    rev.add_argument("--note", default=None, help="required to dismiss: why it is not worth acting on")
    rev.set_defaults(handler=_review)
    exp = commands.add_parser("export", help="an incident as an evidence package"); exp.add_argument("incident"); exp.add_argument("--to"); exp.set_defaults(handler=_export)
    ver = commands.add_parser("verify", help="check an evidence package against its manifest"); ver.add_argument("folder"); ver.set_defaults(handler=_verify)
    aud = commands.add_parser("audit", help="the audit trail"); aud.add_argument("--limit", type=int, default=100); aud.set_defaults(handler=_audit)
    al = commands.add_parser("alerts", help="where alerts go, and what is open"); al.add_argument("--test", action="store_true"); al.set_defaults(handler=_alerts)
    bk = commands.add_parser("backup", help="copy the database with a checksum"); bk.add_argument("--to"); bk.set_defaults(handler=_backup)
    rs = commands.add_parser("restore", help="replace the database with a verified backup"); rs.add_argument("backup"); rs.set_defaults(handler=_restore)
    commands.add_parser("health", help="every camera's state").set_defaults(handler=_health)
    _add_map_arguments(commands)
    doctor = commands.add_parser("doctor", help="check this installation before leaving site")
    doctor.add_argument("--probe", action="store_true", help="also open every camera, which is slow")
    doctor.set_defaults(handler=_doctor)
    # Listed so `--help` names it; `main` hands it the rest of the line.
    console = commands.add_parser("console", help="open the operator console (a window)")
    console.add_argument("rest", nargs="*", help="--as NAME, --for SECONDS, --start, --record, --screenshots DIR")
    console.set_defaults(handler=_console_placeholder)
    site = commands.add_parser("site", help="the site's name and clock")
    sc = site.add_subparsers(dest="site_command", required=True)
    sname = sc.add_parser("name"); sname.add_argument("name"); sname.add_argument("--timezone", default="UTC")
    sc.add_parser("show")
    detection = sc.add_parser("detection", help="what this site watches for, for every run")
    detection.add_argument("--watch", help="comma-separated labels, replacing what is there")
    detection.add_argument("--confidence", type=float, help="how sure the detector must be, 0.05 to 0.95")
    detection.add_argument("--detect-every", type=int, dest="detect_every", metavar="N",
                           help="run the detector on one frame in N and track through the rest. "
                                "Measured at 3.0x less detection for 0.008 box heights of lag at N=3; "
                                "the lag grows with how fast things move, so measure it on your own "
                                "footage with `tools/detector_options.py`")
    detection.add_argument("--tile", choices=["on", "off", "auto"],
                           help="also run the detector over crops of the far ground, where a person is "
                                "a couple of dozen pixels once the frame has been letterboxed. Costs one "
                                "inference per tile. `auto`, the default, turns it on where inference runs "
                                "on a GPU and off on CPU, where it would take a camera under 6 fps")
    detection.add_argument("--clear", action="store_true", help="go back to the built-in list and threshold")
    threats = sc.add_parser("threats", help="which labels this site treats as dangerous")
    threats.add_argument("--set", help="comma-separated labels, replacing what is there")
    threats.add_argument("--clear", action="store_true", help="treat nothing as a threat")
    threats.add_argument("--suggest", action="store_true", help="print a starting point to choose from")
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

    # No arguments at all means somebody double-clicked the executable, and
    # what they want is the window. argparse's answer was `error: the
    # following arguments are required: command` and exit 2 — a usage message
    # flashed in a console that closes before it can be read, from a product
    # whose README calls this file "both the command line and the console".
    #
    # `--help` and `--version` still print, because somebody who typed those
    # asked for text. Only the empty command line opens a window.
    if not argv:
        from .console.main import run as run_console

        return run_console([])

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
