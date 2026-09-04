#!/usr/bin/env python3
"""Run the packaged console on this machine's camera, and keep the evidence.

The rule this tool exists for: **the product is tested through the real camera
and the shipped binary**, never through a prerecorded file and never through
`python tasks.py console`. The unit suites use a rendered scene because they
must be deterministic; that is the right tool for a regression and the wrong
one for the question "does the thing in `dist/` work on the thing it ships
for". Three defects in this repository were found only by a person running the
packaged binary on the laptop camera — models looked for in the wrong folder,
three cameras fighting one webcam, a couch raised as an incident — and each had
a green suite behind it.

So this drives `dist/SentinelVision/SentinelVision-dev.exe` the way an
operator would, except that it cannot click: the console takes the same flags
`sentinel run` does, and this passes them. One invocation:

- isolates the run in a temporary data directory and settings file, so the
  operator's own database, log and watch list are untouched;
- seeds a camera (`device:0` by default), a placement, and a person-only
  restricted zone starting two metres in front of it — the arrangement the
  harsh camera battery used, because it is the one that found the seated
  person "entering" a zone they never left their chair for;
- starts, runs for N seconds, photographs the window and every panel, prints
  the summary, and closes;
- copies the run's log beside the pictures and stdout, and reads the summary
  back to say, in one table, what the binary concluded: frames analysed, each
  track with its class, events, incidents, and whether a person was seen.

It needs a camera, and it needs the bundle built (`python tasks.py package`).
It is not part of CI: CI has no camera and no person. It is what a person runs
before saying a build works, and what `python tasks.py exetest` runs.

Nothing here reaches the network. The camera is a device on this machine.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "dist" / "SentinelVision"
OUT = ROOT / "dist" / "exetest"

sys.path[:0] = [str(ROOT / "engine")]

# Consoles on this machine are cp1252; a glyph in a progress line must not be
# what kills a test run.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

#: A laptop on a desk: 1.2 m up, level-ish, tilted a little down. Anything a
#: webcam can see the ground of is within a few metres; the zone below sits
#: inside that. Latitude and longitude are the reference site's.
DEFAULT_PLACE = "33.8938,35.5018,1.2,180,-15"

#: How far in front of the camera the default zone begins, and how wide it is.
ZONE_AHEAD_METRES = 2.0
ZONE_HALF_WIDTH_METRES = 2.0


def executable() -> Path:
    name = "SentinelVision-dev.exe" if sys.platform == "win32" else "SentinelVision-dev"
    return BUNDLE / name


def default_zone(place: str) -> str:
    """A person-only restricted square just in front of the placed camera.

    Computed with the engine's own geodesy, because a hand-typed ring is
    wrong in the fourth decimal and the tester spends the evening finding out.
    """
    from sentinel.cli import _pose
    from sentinel.core import destination_point

    pose = _pose(place)
    near = ZONE_AHEAD_METRES + ZONE_HALF_WIDTH_METRES
    centre = destination_point(pose.position, pose.heading, near)
    corners = [
        destination_point(centre, pose.heading + bearing, ZONE_HALF_WIDTH_METRES * 1.4142)
        for bearing in (45.0, 135.0, 225.0, 315.0)
    ]
    ring = ";".join(f"{point.lat:.6f},{point.lon:.6f}" for point in corners)
    return f"Room:{ring}"


def redacted(text: str) -> str:
    try:
        from sentinel.redact import redact_text

        return redact_text(text)
    except Exception:  # noqa: BLE001 - the engine may not import from a bare checkout
        return text


def read_back(stdout: str) -> dict:
    """What the binary said it concluded, from the summary it printed."""
    facts: dict = {
        "frames": 0, "detections": 0, "tracks": [], "events": 0, "incidents": 0,
        "screenshots": 0, "starved": "STARVED" in stdout,
    }
    for match in re.finditer(r"frames\s+(\d+),\s+(\d+) detection\(s\)", stdout):
        facts["frames"] += int(match.group(1))
        facts["detections"] += int(match.group(2))
    for match in re.finditer(r"^\s+events\s+(\d+)\s*$", stdout, re.MULTILINE):
        facts["events"] += int(match.group(1))
    match = re.search(r"^\s+incidents\s+(\d+)\s*$", stdout, re.MULTILINE)
    if match:
        facts["incidents"] = int(match.group(1))
    for match in re.finditer(
        r"track (\d+)\s+(\S*)\s+observed in\s+(\d+) frames, spanning ([\d.]+)s", stdout
    ):
        facts["tracks"].append(
            (int(match.group(1)), match.group(2), int(match.group(3)), float(match.group(4)))
        )
    facts["screenshots"] = len(re.findall(r"^screenshot\s+", stdout, re.MULTILINE))
    return facts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="exe-camera-test",
        description="Run the packaged console on this machine's camera and keep the evidence.",
    )
    parser.add_argument("--seconds", type=float, default=30.0, help="how long to run (default 30)")
    parser.add_argument("--camera", default="device:0", help="the camera (default device:0)")
    parser.add_argument("--place", default=DEFAULT_PLACE, help=f"lat,lon,height,heading,pitch (default {DEFAULT_PLACE})")
    parser.add_argument("--zone", default=None, help="name:lat,lon;… (default: a 4 m square 2 m ahead, named Room)")
    parser.add_argument("--zone-classes", default="Room=person", help="which labels the zone acts on (default Room=person)")
    parser.add_argument("--watch", default=None, help="classes to track this run (default: the console's security default)")
    parser.add_argument("--confidence", default=None, help="minimum confidence this run (default: the console's, 0.50)")
    parser.add_argument("--model", default=None, help="an ONNX model; default: the bundle's models/*-seg.onnx")
    parser.add_argument("--no-model", action="store_true", help="motion only")
    parser.add_argument("--keep", action="store_true", help="keep the temporary data directory")
    args = parser.parse_args(argv)

    exe = executable()
    if not exe.is_file():
        print(f"{exe} is not built. Run `python tasks.py package` first: the point of this "
              "tool is the binary in dist/, not a checkout.", file=sys.stderr)
        return 2

    zone = args.zone or default_zone(args.place)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    shots = OUT / stamp
    shots.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix="sentinel-exetest-"))

    command = [
        str(exe),
        "--settings", str(workspace / "console.ini"),
        "--camera", args.camera,
        "--place", args.place,
        "--zone", zone,
        "--zone-classes", args.zone_classes,
        "--start",
        "--for", f"{args.seconds:g}",
        "--screenshots", str(shots),
    ]
    if args.watch:
        command += ["--watch", args.watch]
    if args.confidence:
        command += ["--confidence", args.confidence]
    if args.model:
        command += ["--model", args.model]
    if args.no_model:
        command += ["--no-model"]

    env = {**os.environ, "SENTINEL_DATA_DIR": str(workspace)}
    print(f"binary      {exe}")
    print(f"data        {workspace}  (isolated; the operator's data is untouched)")
    print(f"evidence    {shots}")
    print("command     " + redacted(" ".join(command[1:])))
    print(f"running for {args.seconds:g}s — a real person in frame is what this is for")
    print()

    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command, env=env, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=args.seconds + 120.0,
        )
    except subprocess.TimeoutExpired as expired:
        (shots / "stdout.txt").write_text(expired.stdout or "", encoding="utf-8")
        (shots / "stderr.txt").write_text(expired.stderr or "", encoding="utf-8")
        print(f"FAIL: the binary did not exit within {args.seconds + 120:.0f}s and was killed.", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - started

    (shots / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (shots / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
    log = workspace / "logs" / "sentinel.log"
    if log.is_file():
        shutil.copy2(log, shots / "sentinel.log")

    facts = read_back(completed.stdout)
    pictures = sorted(shots.glob("*.png"))
    people = [track for track in facts["tracks"] if track[1] == "person"]
    longest = max((track[3] for track in people), default=0.0)

    print(f"exit        {completed.returncode}   ({elapsed:.1f}s wall)")
    print(f"frames      {facts['frames']} analysed, {facts['detections']} detections")
    print(f"tracks      {len(facts['tracks'])} ids issued"
          + (f"; person tracks {len(people)}, longest {longest:.1f}s" if facts["tracks"] else ""))
    for track_id, label, seen, span in facts["tracks"]:
        print(f"  track {track_id:<3} {label or 'unclassified':<12} {seen:>4} frames  {span:>5.1f}s")
    print(f"events      {facts['events']}")
    print(f"incidents   {facts['incidents']}")
    print(f"pictures    {len(pictures)}  in {shots}")
    if completed.stderr.strip():
        tail = completed.stderr.strip().splitlines()[-5:]
        print("stderr tail:")
        for line in tail:
            print(f"  {redacted(line)}")

    problems = []
    if completed.returncode != 0:
        problems.append(f"the binary exited {completed.returncode}")
    if facts["starved"]:
        problems.append("the run was starved of frames — is another program using the camera?")
    if facts["frames"] == 0:
        problems.append("no frame was analysed; the camera did not deliver or the run never started")
    if not pictures:
        problems.append("no picture was written; the run ended before it could photograph itself")

    if not args.keep:
        shutil.rmtree(workspace, ignore_errors=True)
    else:
        print(f"kept        {workspace}")

    print()
    if problems:
        for problem in problems:
            print(f"FAIL: {problem}", file=sys.stderr)
        return 1
    if not people:
        print("PASS, with a caveat: no person was tracked. Frames flowed and pictures were "
              "taken, but nobody was in front of the camera — look at the pictures.")
        return 0
    print(f"PASS: a person was tracked for {longest:.1f}s; look at the pictures before believing it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
