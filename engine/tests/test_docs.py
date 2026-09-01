"""Tests for the documentation lint.

Every architectural claim in this repository is accompanied by a diagram, and a
mermaid diagram that fails to parse renders as a block of raw text — or, on some
viewers, as nothing at all. Nothing errors, nothing warns, and the explanation
the reader needed is simply gone.

Each case below is a mistake that was actually made writing these, not a
hypothetical. The last of them — a label split across lines — went undetected
through a whole documentation pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import docs_lint  # noqa: E402


def check(source: str) -> list[str]:
    return docs_lint.check_block("fake.md", 1, source.strip("\n").split("\n"))


# --------------------------------------------------- the diagrams that exist


def test_every_diagram_in_the_repository_parses():
    problems, _, diagrams = docs_lint.audit()

    assert problems == []
    # A lint that finds nothing because it looked at nothing passes vacuously.
    assert diagrams > 20, "the diagrams were not found, so nothing was checked"


def test_the_lint_reads_the_documents_it_is_supposed_to():
    names = {path.name for path in docs_lint._markdown_files()}

    assert {"README.md", "STATUS.md", "ARCHITECTURE.md", "OVERVIEW.md"} <= names


# ------------------------------------------------------- what it must catch


def test_a_label_split_across_lines_is_caught():
    # The mistake this lint exists for. Writing `\n` inside a node label instead
    # of `<br/>` — and then having something interpret the escape on the way in,
    # so the label really does end mid-line.
    problems = check(
        '''
flowchart LR
    A["first line
second line"] --> B["fine"]
'''
    )

    assert problems, "a label split across lines went undetected"
    assert "unbalanced quote" in problems[0]


def test_a_literal_escape_in_a_label_is_caught():
    problems = check(r'''
flowchart LR
    A["first line\nsecond line"] --> B["fine"]
''')

    assert any("<br/>" in problem for problem in problems)


def test_an_unbalanced_bracket_is_caught():
    problems = check(
        '''
flowchart LR
    A["missing the close" --> B["fine"]
'''
    )

    assert problems


def test_a_style_naming_a_node_that_does_not_exist_is_caught():
    # This one loses the colour that carried half the diagram's meaning, and
    # loses it silently: mermaid renders the diagram, just without the styling.
    problems = check(
        '''
flowchart LR
    A["one"] --> B["two"]

    style A fill:#1e3f2f,color:#fff
    style TYPO fill:#8a1f1f,color:#fff
'''
    )

    assert any("TYPO" in problem for problem in problems)


def test_a_block_that_is_not_a_diagram_is_caught():
    problems = check(
        '''
flowcart LR
    A --> B
'''
    )

    assert any("not a diagram type" in problem for problem in problems)


def test_an_empty_block_is_caught():
    assert check("\n")


# --------------------------------------------------- what it must NOT flag


def test_an_er_diagram_is_not_flagged_for_its_cardinality():
    # `||--o{` is unbalanced braces and entirely correct. A lint that refused it
    # would be a lint somebody deletes.
    assert (
        check(
            '''
erDiagram
    cameras ||--o{ events : "observed by"
    events ||--o{ incident_events : "belongs to"
'''
        )
        == []
    )


def test_a_state_diagram_with_a_composite_state_is_not_flagged():
    assert (
        check(
            '''
stateDiagram-v2
    [*] --> Outside
    state Inside {
        Entering --> Present
    }
'''
        )
        == []
    )


def test_subgraphs_and_styled_subgraphs_are_understood():
    # A subgraph id is a legitimate style target, and treating it as undefined
    # would flag most of the diagrams in this repository.
    assert (
        check(
            '''
flowchart TB
    subgraph core["the core"]
        direction LR
        A["one"] --> B["two"]
    end
    subgraph shell["the shell"]
        C["three"]
    end
    core --> shell

    style core fill:#1e3a5f,color:#fff
    style shell fill:#1e3f2f,color:#fff
'''
        )
        == []
    )


def test_edge_labels_and_dotted_edges_are_understood():
    assert (
        check(
            '''
flowchart LR
    A["one"] -->|"because"| B["two"]
    B -.->|"sometimes"| C["three"]
    C ==> D["four"]

    style D fill:#4c1d24,color:#fff
'''
        )
        == []
    )
