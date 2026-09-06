"""Fail if shipped source names any destination off the site.

Scans `vigil/` and `tools/` for URLs and hostnames. Tests and docs are not
scanned: a test that names an address to *refuse* is doing its job.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCANNED = [ROOT / "vigil", ROOT / "tools", ROOT / "tasks.py"]
URL = re.compile(r"\b(?:https?|rtsps?|ftp|ws|wss)://[^\s'\"<>]+", re.I)
HOST = re.compile(r"\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:com|net|org|io|dev|cloud|ai|co|gov|edu)\b", re.I)
ALLOWED_URL_PREFIXES = ()  # nothing


def scan() -> list[str]:
    problems = []
    count = 0
    for entry in SCANNED:
        paths = [entry] if entry.is_file() else sorted(entry.rglob("*.py"))
        for path in paths:
            count += 1
            text = path.read_text(encoding="utf-8", errors="replace")
            for number, line in enumerate(text.splitlines(), 1):
                for match in URL.finditer(line):
                    problems.append(f"{path.relative_to(ROOT)}:{number}: url {match.group(0)}")
                for match in HOST.finditer(line):
                    problems.append(f"{path.relative_to(ROOT)}:{number}: hostname {match.group(0)}")
    if count == 0:
        problems.append("nothing was scanned")
    return problems


def main() -> int:
    problems = scan()
    if problems:
        print("\n".join(problems))
        print(f"offline audit: {len(problems)} problem(s)")
        return 1
    print("offline audit: no route out")
    return 0


if __name__ == "__main__":
    sys.exit(main())
