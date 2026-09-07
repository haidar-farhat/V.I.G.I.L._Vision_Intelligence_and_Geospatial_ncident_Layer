"""Render the public page from the repository, so its numbers cannot drift.

`python tasks.py site` reads the templates in `site/`, fills every
`{{placeholder}}` from the capability manifest, the version, the commit, the
line counts and pytest's own count of the suite, and writes the finished page
to `dist/site/`. The GitHub Pages workflow runs exactly this. A hand-kept page
would be a second description of the product, and the copy that drifts is the
one a stranger reads.

Nothing here names an address: the offline audit scans this file. The links
the page carries live in the templates, which are not shipped source.
"""

from __future__ import annotations

import html
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
OUT = ROOT / "dist" / "site"
PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")


def pytest_count(root: Path = ROOT) -> int:
    """pytest's own number, which counts parametrised cases; falls back to a
    count of test functions when the suite cannot be collected here."""
    try:
        proc = subprocess.run([sys.executable, "-m", "pytest", "tests", "--collect-only", "-q", "-p", "no:cacheprovider"],
                              cwd=str(root), capture_output=True, text=True, timeout=300)
        match = re.search(r"(\d+) tests? collected", proc.stdout + proc.stderr)
        if match:
            return int(match.group(1))
    except (OSError, subprocess.SubprocessError):
        pass
    return sum(text.count("def test_") for text in _texts(root / "tests", "*.py"))


def _texts(folder: Path, pattern: str) -> list[str]:
    return [p.read_text(encoding="utf-8") for p in sorted(folder.rglob(pattern)) if "__pycache__" not in p.parts]


def _commit(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(root), text=True).strip()
    except (OSError, subprocess.SubprocessError):
        return "unstamped"


def note_html(note: str) -> str:
    """The manifest notes are written in the two markdown marks they use."""
    text = html.escape(note)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    return text


def gather(root: Path = ROOT, *, tests: int | None = None) -> dict[str, str]:
    sys.path.insert(0, str(root))
    from vigil.capabilities import MANIFEST, State
    from vigil.version import __version__

    counts = Counter(c.state for c in MANIFEST)
    python_lines = sum(len(t.splitlines()) for t in _texts(root / "vigil", "*.py"))
    rust_texts = _texts(root / "core" / "src", "*.rs")
    rows, index_items = [], []
    for c in MANIFEST:
        state = c.state.value
        symbols = ", ".join(f"<code>{html.escape(s)}</code>" for s in c.symbols) or "—"
        tests_ = ", ".join(f"<code>{html.escape(t)}</code>" for t in c.tests) or "—"
        rows.append(
            f'<article class="cap" id="cap-{c.id}">\n'
            f'  <header><span class="id">{c.id}</span><h3>{html.escape(c.title)}</h3>'
            f'<span class="chip {state.lower()}">{state}</span></header>\n'
            f'  <dl><dt>Symbols</dt><dd>{symbols}</dd><dt>Tests</dt><dd>{tests_}</dd></dl>\n'
            + (f'  <p>{note_html(c.note)}</p>\n' if c.note else "")
            + '</article>')
        index_items.append(f'<li><a href="capabilities.html#cap-{c.id}"><span class="id">{c.id}</span>'
                           f'{html.escape(c.title)}</a><span class="state {state}">{state}</span></li>')
    return {
        "version": __version__,
        "commit": _commit(root),
        "built": time.strftime("%Y-%m-%d", time.gmtime()),
        "tested": str(counts.get(State.TESTED, 0)),
        "impl": str(counts.get(State.IMPL, 0)),
        "plan": str(counts.get(State.PLAN, 0)),
        "capability_count": str(len(MANIFEST)),
        "python_tests": str(tests if tests is not None else pytest_count(root)),
        "rust_tests": str(sum(t.count("#[test]") for t in rust_texts)),
        "python_lines": f"{python_lines:,}",
        "rust_lines": f"{sum(len(t.splitlines()) for t in rust_texts):,}",
        "capability_rows": "\n".join(rows),
        "capability_index": "\n".join(index_items),
    }


def render(template: str, data: dict[str, str]) -> str:
    missing = sorted({m.group(1) for m in PLACEHOLDER.finditer(template)} - set(data))
    if missing:
        raise KeyError(f"the template names {missing} and the build supplies nothing for them")
    return PLACEHOLDER.sub(lambda m: data[m.group(1)], template)


def build(site: Path = SITE, out: Path = OUT, data: dict[str, str] | None = None) -> list[Path]:
    data = data or gather()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    written = []
    for path in sorted(site.iterdir()):
        if path.name.startswith(".") and path.name != ".nojekyll":
            continue
        target = out / path.name
        if path.suffix == ".html":
            target.write_text(render(path.read_text(encoding="utf-8"), data), encoding="utf-8")
        elif path.is_file():
            shutil.copy2(path, target)
        else:
            continue
        written.append(target)
    # GitHub Pages runs Jekyll by default, which drops files it does not
    # understand and folders that start with an underscore. This turns it off.
    (out / ".nojekyll").write_text("", encoding="utf-8")
    return written


def main() -> int:
    written = build()
    for path in written:
        print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
