"""Commands that describe or change the site: paths, accounts, cameras, zones.

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
from ..service.site import KEEP, SiteError
from ..service.supervise import clear_stop, install_service, request_stop, service_definition, stop_file, supervise
from ..version import build_info, describe

_log = _get_logger(__name__)


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

def _correspondences(text: str):
    """`u,v,lat,lon[,label]; …` — a point in the picture and where it is.

    `u` and `v` are fractions of the frame rather than pixels, because that is
    what a detection uses and what the projection is written in, so the same
    numbers mean the same thing at any resolution. Refused outside `[0, 1]`
    rather than clamped: a value of 540 is somebody typing pixels, and
    silently reading it as a fraction would produce a confident wrong pose.
    """
    from ..service.calibration import Correspondence
    from ..domain.geo import Vec2

    out = []
    for n, chunk in enumerate((c.strip() for c in text.split(";")), start=1):
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split(",")]
        if len(parts) < 4:
            raise ValueError(f"point {n} is '{chunk}'; each is u,v,lat,lon[,label]")
        try:
            u, v, lat, lon = (float(p) for p in parts[:4])
        except ValueError:
            raise ValueError(f"point {n} is '{chunk}'; u, v, lat and lon must all be numbers") from None
        if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
            raise ValueError(
                f"point {n} has u,v = {u},{v}, which is off the frame. These are fractions of the "
                f"picture, not pixels: the centre is 0.5,0.5 and the bottom-left corner is 0,1"
            )
        out.append(Correspondence(Vec2(u, v), LatLon(lat, lon), ",".join(parts[4:]) or f"point {n}"))
    return out


def _adhoc_id(source: str, existing) -> str:
    """A readable id for a source named on the command line, unique in this site."""
    from ..adapters.recorder import file_safe

    stem = Path(source).stem if not source.startswith("device:") else source.replace(":", "-")
    base = file_safe(stem)[:40] or "camera"
    taken = {c.id for c in existing}
    if base not in taken:
        return base
    for number in range(2, 100):
        if f"{base}-{number}" not in taken:
            return f"{base}-{number}"
    return f"{base}-{len(taken)}"

def _where(ctx: _Context) -> int:
    s = ctx.settings
    print(describe())
    for name, value in (("data", s.data_dir), ("database", s.database), ("logs", s.logs), ("recordings", s.recordings),
                        ("evidence", s.evidence), ("models", s.models), ("default model", s.default_model() or "none")):
        print(f"{name:<14} {value}")
    site = ctx.store.site()
    print(f"{'site':<14} {site['name']} — schedules are read in {site['timezone']}")
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
                # A measured heading is the number that decides how good this
                # camera's positions are, so it is on the line rather than
                # behind another command.
                how = "unplaced"
                if c.placed:
                    how = (f"+/-{c.pose.uncertainty.heading_deg:.2f}deg" if c.calibrated
                           else "placed")
                print(f"{c.id:<16} {redacted(c.source):<40} {how:<12} {'record' if c.record else ''}")
        elif args.cameras_command == "place":
            camera = site.place_camera(args.id, _pose(args.place), by=by)
            print(f"placed {camera.id} at {camera.pose.position.lat:.6f},{camera.pose.position.lon:.6f}")
        elif args.cameras_command == "calibrate":
            points = _correspondences(args.points)
            if args.dry_run:
                from ..service.calibration import CalibrationError, calibrate_pose

                camera = site.camera(args.id, by)
                if camera.pose is None:
                    print(f"error: {args.id} has no placement to refine", file=sys.stderr)
                    return 1
                try:
                    result = calibrate_pose(camera.pose, points, solve_position=args.solve_position)
                except CalibrationError as error:
                    print(f"error: {error}", file=sys.stderr)
                    return 1
                print(result.describe())
                print("not saved (--dry-run)"
                      if result.better_than(camera.pose.uncertainty)
                      else "not saved: this is worse than what the camera already assumes")
                return 0
            camera, result = site.calibrate_camera(args.id, points, solve_position=args.solve_position, by=by)
            print(f"measured {camera.id} from {len(points)} points")
            print(result.describe())
        elif args.cameras_command == "record":
            camera = site.set_recording(args.id, args.state == "on", by=by)
            print(f"{camera.id} recording {'on' if camera.record else 'off'}")
        elif args.cameras_command == "password":
            site.set_password(args.id, _read_secret(f"password for {args.id}: ", args.stdin), by=by)
            print(f"stored {args.id}'s password in the keychain")
        elif args.cameras_command == "source":
            if split_password(args.source)[1]:
                print("warning: a password on the command line is readable by every process; prefer `vigil cameras password`", file=sys.stderr)
            camera = site.set_source(args.id, args.source, by=by)
            print(f"{camera.id} now reads {redacted(camera.source)}"
                  f"{'; its placement and zones are unchanged' if camera.placed else ''}")
        elif args.cameras_command == "rename":
            camera = site.rename_camera(args.id, args.name, by=by)
            print(f"{camera.id} is called {camera.name}")
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
        elif args.zones_command == "edit":
            from ..service.site import KEEP

            schedule = KEEP
            if args.closed is not None:
                schedule = None if args.closed.lower() in ("none", "off") else Schedule(*[int(h) for h in args.closed.split("-")])
            zone = site.edit_zone(
                args.id,
                name=args.name if args.name is not None else KEEP,
                kind=args.kind if args.kind is not None else KEEP,
                watch=[w for w in args.watch.split(",") if w] if args.watch is not None else KEEP,
                enter_after_millis=args.enter_after if args.enter_after is not None else KEEP,
                exit_after_millis=args.exit_after if args.exit_after is not None else KEEP,
                schedule=schedule, by=by)
            print(f"{zone.id}: {zone.name} ({zone.kind}, {len(zone.ring)} points, "
                  f"watching {', '.join(sorted(zone.watch)) or 'everything'})")
        elif args.zones_command == "remove":
            site.remove_zone(args.id, by=by)
            print(f"removed zone {args.id}")
    except (SiteError, AuthError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0

def _site(ctx: _Context) -> int:
    if ctx.args.site_command == "detection":
        from ..service.detection import DetectionError

        try:
            every = getattr(ctx.args, "detect_every", None)
            tiling = getattr(ctx.args, "tile", None)
            tile = {"on": True, "off": False, "auto": None}.get(tiling, KEEP)
            if ctx.args.clear:
                chosen = ctx.site.set_detection([], None, by=ctx.principal, detect_every=1, tile=None)
            elif (ctx.args.watch is not None or ctx.args.confidence is not None
                  or every is not None or tiling is not None):
                current = ctx.site.detection()
                labels = (current.labels if ctx.args.watch is None
                          else [l for l in ctx.args.watch.split(",") if l.strip()])
                confidence = current.confidence if ctx.args.confidence is None else ctx.args.confidence
                chosen = ctx.site.set_detection(labels, confidence, by=ctx.principal,
                                                detect_every=every, tile=tile)
            else:
                chosen = ctx.site.detection()
        except (DetectionError, AuthError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        print(chosen.describe())
        return 0
    if ctx.args.site_command == "threats":
        from ..domain.threats import ThreatVocabulary

        try:
            if ctx.args.suggest:
                print("A starting point, not a default. Adopt only what applies to this site, and only "
                      "for a model that can name it:")
                for threat in sorted(ThreatVocabulary.suggested()._by_label.values(), key=lambda t: t.label):
                    print(f"  {threat.label:<14} {threat.severity}")
                return 0
            if ctx.args.clear:
                vocabulary = ctx.site.set_threats([], by=ctx.principal)
            elif ctx.args.set is not None:
                vocabulary = ctx.site.set_threats([l for l in ctx.args.set.split(",") if l.strip()], by=ctx.principal)
            else:
                vocabulary = ctx.site.threats()
        except (SiteError, AuthError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        print(vocabulary.describe())
        return 0
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
