"""The public page is rendered from the tree, and a placeholder left standing is a page that lies by omission."""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _builder():
    spec = importlib.util.spec_from_file_location("vigil_build_site", ROOT / "tools" / "build_site.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_every_page_renders_with_nothing_left_unfilled(tmp_path):
    from vigil.capabilities import MANIFEST

    builder = _builder()
    data = builder.gather(tests=442)  # pytest's own count is not re-run inside pytest
    written = builder.build(out=tmp_path / "site", data=data)
    names = {p.name for p in written}
    assert {"index.html", "capabilities.html", "style.css"} <= names
    assert (tmp_path / "site" / ".nojekyll").is_file(), "Jekyll would otherwise drop what it does not understand"
    for page in written:
        if page.suffix == ".html":
            text = page.read_text(encoding="utf-8")
            assert "{{" not in text, f"{page.name} still holds a placeholder"
            assert "<title>" in text and "</html>" in text
    capabilities = (tmp_path / "site" / "capabilities.html").read_text(encoding="utf-8")
    index = (tmp_path / "site" / "index.html").read_text(encoding="utf-8")
    for capability in MANIFEST:
        assert f'id="cap-{capability.id}"' in capabilities, capability.id
        assert f"#cap-{capability.id}" in index, capability.id
    assert data["tested"] in index and data["version"] in index


def test_a_placeholder_the_build_cannot_fill_is_an_error_not_a_blank():
    builder = _builder()
    with pytest.raises(KeyError):
        builder.render("<p>{{nothing_supplies_this}}</p>", {"version": "1"})
    assert builder.render("v{{version}}", {"version": "1"}) == "v1"


def test_manifest_notes_keep_their_two_marks_and_nothing_else_is_html():
    builder = _builder()
    assert builder.note_html("a `code` **bold** <b>") == "a <code>code</code> <strong>bold</strong> &lt;b&gt;"
