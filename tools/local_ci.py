#!/usr/bin/env python3
"""The CI pipeline, run here, on this machine.

Waiting on remote CI to find out whether something works is a slow loop and,
until somebody looks at the result, an unread one. This runs every stage the
workflow runs that *can* run locally, in the same order, and says plainly which
stages it could not.

What runs here and what does not:

| Stage | Locally | Why |
|---|---|---|
| Static guards | yes | pure Python |
| Rust format, lint, test, build | yes | the toolchain is installed |
| Engine and console suites | yes | |
| Offline acceptance | **partly** | see below |
| Three platforms | **no** | this machine is one of them |
| Packaged executables | yes | `--package` |
| The running application | yes | `--screenshots` |

**The offline stage is the honest gap.** The CI job drops outbound traffic with
`iptables` and *proves* the drop before running anything. Neither is available
here — Windows has no iptables, and this process cannot firewall itself in a way
it could not also undo. What runs instead is the whole suite with a poisoned
proxy environment and a blocked DNS resolver, which catches a library that
politely reads `HTTP_PROXY` and misses one that opens a raw socket. That is a
weaker proof and is reported as one. The container in `docker-compose.yml` runs
the real thing with `network_mode: none` when Docker is available.

`--package` builds the three executables and then *runs* each one, because a
bundle that builds and does not launch is the failure this project has already
had once.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# This machine's console is cp1252, and an arrow glyph in a progress line is
# enough to kill the whole run with a UnicodeEncodeError. A pipeline runner that
# dies on its own decoration is a runner nobody uses, so stdout is reconfigured
# rather than the characters being guessed at.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "core"
ENGINE = ROOT / "engine"
CONSOLE = ROOT / "apps" / "console"


@dataclass
class Stage:
    name: str
    command: list[str]
    cwd: Path
    #: Extra environment. Never replaces the inherited one.
    env: dict[str, str] = field(default_factory=dict)
    #: A stage that reports rather than fails. Used for the checks this machine
    #: can only approximate, so the summary never claims a proof it did not run.
    advisory: bool = False
    note: str = ""


@dataclass
class Result:
    stage: Stage
    code: int
    seconds: float

    @property
    def ok(self) -> bool:
        return self.code == 0

    @property
    def state(self) -> str:
        if self.ok:
            return "pass"
        return "ADVISORY" if self.stage.advisory else "FAIL"


def python_path() -> dict[str, str]:
    parts = [str(ENGINE), str(ENGINE / "tests"), str(CONSOLE)]
    existing = os.environ.get("PYTHONPATH")
    if existing:
        parts.append(existing)
    return {"PYTHONPATH": os.pathsep.join(parts)}


def offline_environment() -> dict[str, str]:
    """As close to "no network" as a process can put itself on Windows.

    Poisons every proxy variable the ecosystem reads and points DNS at a black
    hole. A library that asks politely — urllib, requests, most SDKs — is
    stopped. One that opens a raw socket to a literal address is not, which is
    exactly why this stage is advisory and the container is the real proof.
    """
    poison = "http://127.0.0.1:9"
    return {
        "HTTP_PROXY": poison, "HTTPS_PROXY": poison,
        "http_proxy": poison, "https_proxy": poison,
        "ALL_PROXY": poison, "all_proxy": poison,
        "NO_PROXY": "", "no_proxy": "",
        # Anything that honours it will refuse to reach a network at all.
        "SENTINEL_OFFLINE": "1",
        **python_path(),
    }


def stages(*, package: bool, quick: bool) -> list[Stage]:
    found = [
        Stage("static · source audit", [sys.executable, "tools/offline_audit.py"], ROOT),
        Stage("static · binary audit", [sys.executable, "tools/binary_audit.py"], ROOT),
        Stage("static · docs lint", [sys.executable, "tools/docs_lint.py"], ROOT),
        Stage("rust · format", ["cargo", "fmt", "--check"], CORE),
        Stage("rust · clippy", ["cargo", "clippy", "--all-targets", "--", "-D", "warnings"], CORE),
        Stage("rust · test", ["cargo", "test"], CORE),
        Stage("rust · build", ["cargo", "build", "--release"], CORE),
        Stage("engine · tests", [sys.executable, "-m", "pytest", "-q"], ENGINE, python_path()),
        Stage(
            "console · tests", [sys.executable, "-m", "pytest", "-q"], CONSOLE,
            {**python_path(), "QT_QPA_PLATFORM": "offscreen"},
        ),
    ]

    if not quick:
        found.append(
            Stage(
                "offline · engine suite with the network poisoned",
                [sys.executable, "-m", "pytest", "-q"], ENGINE,
                offline_environment(),
                advisory=True,
                note=(
                    "Proxy variables poisoned and DNS blackholed. A library that "
                    "opens a raw socket is NOT stopped by this — "
                    "`docker compose run --rm verify` is the real proof."
                ),
            )
        )

    if package:
        found.append(Stage("package · three executables", [sys.executable, "tasks.py", "package"], ROOT))

    return found


def run(stage: Stage) -> Result:
    print(f"\n\033[1m▶ {stage.name}\033[0m", flush=True)
    started = time.perf_counter()
    completed = subprocess.run(
        stage.command, cwd=stage.cwd, env={**os.environ, **stage.env}
    )
    return Result(stage, completed.returncode, time.perf_counter() - started)


def launch_check() -> list[Result]:
    """Run each packaged executable. A bundle that builds is not a bundle that runs.

    This project has already shipped an executable nobody could launch. The
    build succeeding says nothing about that, so each one is started here and
    has to answer.
    """
    bundle = ROOT / "dist" / "SentinelVision"
    suffix = ".exe" if sys.platform == "win32" else ""
    checks = [
        # The CLI can answer a question and exit, which is the strongest of the
        # three: it loads the engine, the Rust core and every dependency.
        Stage(f"launch · sentinel{suffix} where", [str(bundle / f"sentinel{suffix}"), "where"], ROOT),
        Stage(
            f"launch · sentinel{suffix} coverage",
            [str(bundle / f"sentinel{suffix}"), "coverage",
             "--place", "33.8938,35.5018,6,180,-22"], ROOT,
        ),
    ]

    results = []
    for stage in checks:
        if not Path(stage.command[0]).is_file():
            print(f"\n\033[33m▶ {stage.name} — not built, skipped\033[0m")
            continue
        results.append(run(stage))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="local-ci",
        description="Run every CI stage this machine can run, in CI's order.",
    )
    parser.add_argument("--package", action="store_true",
                        help="also build the three executables and launch each one")
    parser.add_argument("--quick", action="store_true",
                        help="skip the slower advisory stages")
    parser.add_argument("--screenshots", action="store_true",
                        help="drive the real console and capture PNGs of it")
    args = parser.parse_args(argv)

    results: list[Result] = []
    for stage in stages(package=args.package, quick=args.quick):
        result = run(stage)
        results.append(result)
        if not result.ok and not stage.advisory:
            # Stop at the first real failure. Continuing reports results for a
            # binary that did not build, which is worse than no result.
            break

    if args.package and all(r.ok or r.stage.advisory for r in results):
        results.extend(launch_check())

    if args.screenshots:
        results.append(
            run(Stage(
                "screenshots · the real console",
                [sys.executable, "tools/screenshot_console.py"], ROOT,
                {**python_path(), "QT_QPA_PLATFORM": "offscreen"},
            ))
        )

    print("\n" + "=" * 72)
    width = max(len(r.stage.name) for r in results)
    failed = advisory = 0
    for result in results:
        colour = "\033[32m" if result.ok else ("\033[33m" if result.stage.advisory else "\033[31m")
        print(f"  {colour}{result.state:<9}\033[0m {result.stage.name:<{width}}  {result.seconds:6.1f}s")
        if not result.ok:
            if result.stage.advisory:
                advisory += 1
                if result.stage.note:
                    print(f"            {result.stage.note}")
            else:
                failed += 1
    print("=" * 72)

    # Said every time, because a green local run is not a green CI run and the
    # difference is three operating systems and a real network block.
    print(
        "\n  Not covered here: Linux and macOS, Python 3.12, and the offline\n"
        "  acceptance job's real `iptables` block. Those need remote CI or the\n"
        "  container. A pass here is necessary, not sufficient."
    )

    if failed:
        print(f"\n\033[31m{failed} stage(s) failed\033[0m")
        return 1
    if advisory:
        print(f"\n\033[33mgreen, with {advisory} advisory stage(s) not passing\033[0m")
        return 0
    print("\n\033[32mgreen\033[0m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
