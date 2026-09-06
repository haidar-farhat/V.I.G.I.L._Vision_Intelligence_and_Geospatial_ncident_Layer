"""Tested code nothing calls is a red test here, not a discovery six months later."""

import ast
import importlib
import inspect
from pathlib import Path

from vigil.capabilities import MANIFEST, State

ROOT = Path(__file__).resolve().parent.parent


def _resolve(dotted: str):
    """Import the longest module prefix, then walk the attributes.

    A manifest entry may name a method (`...Runtime.metrics`), so splitting
    at the last dot and importing the rest is not enough.
    """
    parts = dotted.split(".")
    module, index = None, 0
    for size in range(len(parts), 0, -1):
        try:
            module = importlib.import_module(".".join(parts[:size]))
            index = size
            break
        except ModuleNotFoundError:
            continue
    assert module is not None, f"no module in {dotted}"
    thing = module
    for name in parts[index:]:
        thing = getattr(thing, name)
    return thing


def test_every_symbol_in_the_manifest_exists_and_its_tests_reference_it():
    for capability in MANIFEST:
        for dotted in capability.symbols:
            _resolve(dotted)
        if capability.state is State.TESTED:
            assert capability.symbols and capability.tests, capability.id
            text = "\n".join((ROOT / t).read_text(encoding="utf-8") for t in capability.tests)
            for dotted in capability.symbols:
                name = dotted.rpartition(".")[2]
                assert name in text, f"{capability.id}: {name} is claimed TESTED but {capability.tests} never mention it"
        if capability.state is State.PLAN:
            assert not capability.symbols, f"{capability.id} is PLAN but names symbols"


def test_every_domain_symbol_claimed_tested_is_used_outside_its_own_module():
    """A measurement nothing reads is the v1 defect: correct, tested, and dead.

    `distance_from_camera` was exactly that until the track table and the
    evidence report started using it.
    """
    sources = {path: path.read_text(encoding="utf-8") for path in (ROOT / "vigil").rglob("*.py")}
    for capability in MANIFEST:
        if capability.state is not State.TESTED:
            continue
        for dotted in capability.symbols:
            module_path = ROOT / (dotted.rsplit(".", 1)[0].replace(".", "/") + ".py")
            name = dotted.rpartition(".")[2]
            elsewhere = [p for p, text in sources.items() if p != module_path and name in text]
            assert elsewhere, f"{dotted} is claimed TESTED but nothing outside {module_path.name} uses it"


def test_every_public_service_method_is_called_by_an_interface():
    """A service method no interface calls is the recurring v1 defect. Interfaces: the CLI (and tests do not count)."""
    from vigil.service import evidence, runtime, site
    from vigil.service.auth import Accounts

    public = set()
    for cls in (site.SiteService, runtime.Runtime, Accounts):
        for name, _ in inspect.getmembers(cls, inspect.isfunction):
            if not name.startswith("_") and name not in ("close", "poll", "correlate", "principal_for", "seconds_until_allowed", "bind"):
                public.add(name)
    public |= {"export_incident", "verify_package", "apply_retention"}
    called = set()
    # `rglob`, not `glob`: the console is an interface too, and a method only
    # it reached read as unreachable until this was widened.
    for path in (ROOT / "vigil" / "interfaces").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                called.add(node.attr)
            elif isinstance(node, ast.Name):
                called.add(node.id)
    # Methods the runtime calls on itself in the poll loop are reachable through `run`.
    runtime_internal = {"health", "start", "stop", "source_with_credentials", "seconds_since_frame", "seconds_since_started"}
    for path in (ROOT / "vigil" / "service").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                runtime_internal.add(node.attr)
    unreachable = sorted(m for m in public if m not in called and m not in runtime_internal)
    assert not unreachable, f"service methods no interface calls: {unreachable}"


def test_the_generated_capabilities_page_is_current():
    import subprocess
    import sys

    page = ROOT / "CAPABILITIES.md"
    before = page.read_text(encoding="utf-8") if page.is_file() else ""
    subprocess.check_call([sys.executable, str(ROOT / "tasks.py"), "capabilities"], cwd=str(ROOT), stdout=subprocess.DEVNULL)
    assert page.read_text(encoding="utf-8") == before, "CAPABILITIES.md is stale: run `python tasks.py capabilities`"
