#!/usr/bin/env python3
"""Check the diagrams in the documentation actually parse.

Every architectural claim in this repository is accompanied by a diagram, and a
mermaid diagram that fails to parse renders as a block of raw text — or, on some
viewers, as nothing at all. That failure is silent: the markdown is still valid,
the file still opens, and the explanation the reader needed is simply gone.

This is not a mermaid implementation. It catches the mistakes that are actually
made when writing these by hand, each of which has happened here:

1. **A label split across lines.** Writing ``\\n`` inside a node label instead of
   ``<br/>``. In markdown the ``\\n`` may survive as two characters, or — if the
   text passed through a language that interprets escapes on the way in — become
   a real newline, which splits the label and breaks the diagram.
2. **Unbalanced brackets or quotes** in a node definition.
3. **An empty fence**, or one whose first line is not a diagram type.
4. **A style directive naming a node that does not exist**, which is how a
   diagram loses the colour that carried half its meaning.

Run it directly, or as ``python tasks.py check``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Every markdown file in the repository, wherever it lives.
SKIPPED_DIRS = {".git", "target", "__pycache__", "node_modules", ".venv"}

#: The first word of a mermaid block. Anything else is a typo or a diagram type
#: this repository does not use, and both are worth being told about.
DIAGRAM_TYPES = {
    "flowchart", "graph", "sequenceDiagram", "stateDiagram", "stateDiagram-v2",
    "erDiagram", "classDiagram", "gantt", "pie", "journey", "gitGraph",
    "timeline", "mindmap", "quadrantChart", "sankey-beta", "block-beta",
    "xychart-beta", "C4Context",
}

#: `style <id> fill:...` and `class <id> ...` — both name a node that must exist.
STYLE_PATTERN = re.compile(r"^\s*(?:style|class)\s+([A-Za-z0-9_,\s]+?)\s+[a-z]")

#: A node id at the point it is defined: `ID[...`, `ID(...`, `ID{...`, `ID>...`.
DEFINITION_PATTERN = re.compile(r"(?<![\w.\-])([A-Za-z_][\w-]*)\s*[\[\({>]")

#: A node id used bare in an edge: `A --> B`, `A -.-> B`, `A ==> B`.
EDGE_PATTERN = re.compile(r"(?<![\w.\-])([A-Za-z_][\w-]*)(?=\s*(?:-{2,}|-\.|={2,}))")

#: Keywords that appear where a node id would and are not nodes.
NOT_NODES = {
    "subgraph", "end", "direction", "style", "class", "classDef", "click",
    "linkStyle", "flowchart", "graph", "TB", "TD", "BT", "RL", "LR",
}


def _markdown_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*.md")
        if not SKIPPED_DIRS & set(path.relative_to(ROOT).parts)
    )


def _blocks(text: str) -> list[tuple[int, list[str]]]:
    """Every mermaid fence, as (line number of the fence, its lines)."""
    blocks: list[tuple[int, list[str]]] = []
    current: list[str] | None = None
    start = 0

    for number, line in enumerate(text.split("\n"), start=1):
        stripped = line.strip()
        if current is None:
            if stripped.startswith("```mermaid"):
                current, start = [], number
            continue
        if stripped.startswith("```"):
            blocks.append((start, current))
            current = None
            continue
        current.append(line)

    if current is not None:
        blocks.append((start, current))
    return blocks


def check_block(name: str, start: int, lines: list[str]) -> list[str]:
    problems: list[str] = []
    body = [line for line in lines if line.strip() and not line.strip().startswith("%%")]

    if not body:
        return [f"{name}:{start}: empty mermaid block"]

    first = body[0].strip().split()[0].rstrip(";")
    if first not in DIAGRAM_TYPES:
        problems.append(
            f"{name}:{start}: '{first}' is not a diagram type; the block will "
            "render as raw text"
        )

    # Flow-style diagrams are the only ones whose node ids can be checked this
    # way. A sequence or ER diagram has different syntax entirely.
    flow = first in {"flowchart", "graph"}
    defined: set[str] = set()

    for offset, line in enumerate(lines):
        number = start + offset + 1
        stripped = line.strip()
        if not stripped or stripped.startswith("%%"):
            continue

        if stripped.count('"') % 2 == 1:
            problems.append(
                f"{name}:{number}: unbalanced quote — a label is left open, "
                "which usually means a line break inside it should be <br/>"
            )
        if "\\n" in stripped:
            problems.append(
                f"{name}:{number}: literal '\\n' in a label; mermaid needs <br/>"
            )

        # Bracket balance is a flow-diagram property only. An ER diagram's
        # cardinality is written `||--o{`, and a state diagram opens a composite
        # state with a `{` that closes lines later — both are unbalanced per
        # line and both are correct.
        if flow:
            for opener, closer in (("[", "]"), ("{", "}")):
                if stripped.count(opener) != stripped.count(closer):
                    problems.append(f"{name}:{number}: unbalanced '{opener}{closer}'")

        if flow:
            for match in DEFINITION_PATTERN.finditer(stripped):
                if match.group(1) not in NOT_NODES:
                    defined.add(match.group(1))
            for match in EDGE_PATTERN.finditer(stripped):
                if match.group(1) not in NOT_NODES:
                    defined.add(match.group(1))
            if stripped.startswith("subgraph"):
                # `subgraph name["Label"]` — the name is a node for styling.
                rest = stripped[len("subgraph"):].strip()
                identifier = re.match(r"([A-Za-z_][\w-]*)", rest)
                if identifier:
                    defined.add(identifier.group(1))

    if flow:
        for offset, line in enumerate(lines):
            match = STYLE_PATTERN.match(line)
            if not match:
                continue
            for target in (part.strip() for part in match.group(1).split(",")):
                if target and target not in defined:
                    problems.append(
                        f"{name}:{start + offset + 1}: style names '{target}', "
                        "which is not a node in this diagram — the styling is "
                        "silently dropped"
                    )

    return problems


def audit() -> tuple[list[str], int, int]:
    problems: list[str] = []
    files = _markdown_files()
    diagrams = 0

    for path in files:
        name = path.relative_to(ROOT).as_posix()
        for start, lines in _blocks(path.read_text(encoding="utf-8", errors="replace")):
            diagrams += 1
            problems.extend(check_block(name, start, lines))

    return problems, len(files), diagrams


def main() -> int:
    problems, files, diagrams = audit()

    if problems:
        print(f"\ndocs lint: {len(problems)} problem(s) in {diagrams} diagrams\n")
        for problem in problems:
            print(f"  {problem}")
        return 1

    print(f"docs lint: {diagrams} diagrams across {files} files, all parse")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
