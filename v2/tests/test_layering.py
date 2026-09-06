"""The import direction and the module size budget, enforced."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "vigil"
# `kernel` is below the domain, not beside it. The domain must be able to say
# "predict this track" without knowing a shared library exists, and a kernel
# that could import the domain would let the arithmetic start depending on
# what the arithmetic is for.
#
# `perception` is beside the adapters: it reads pixels, so it may use OpenCV,
# and it produces domain types, so it may import the domain. Nothing in the
# domain may import it back.
LAYERS = {"kernel": -1, "domain": 0, "adapters": 1, "perception": 1, "storage": 1,
          "service": 2, "interfaces": 3}
FORBIDDEN = {
    "kernel": {"domain", "adapters", "perception", "storage", "service", "interfaces", "cv2", "sqlite3"},
    "domain": {"adapters", "perception", "storage", "service", "interfaces", "cv2", "sqlite3",
               "threading", "socket", "os"},
    "adapters": {"perception", "service", "interfaces", "storage"},
    "perception": {"adapters", "service", "interfaces", "storage"},
    "storage": {"service", "interfaces"},
    "service": {"interfaces"},
    "interfaces": {"storage.", "adapters.decode.VideoSource"},
}
MAX_LINES = 800


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                module = ("." * node.level) + module
            names.add(module)
            names.update(f"{module}.{a.name}" for a in node.names)
    return names


def test_the_import_direction_holds():
    for path in ROOT.rglob("*.py"):
        layer = path.relative_to(ROOT).parts[0]
        if layer not in FORBIDDEN:
            continue
        imports = _imports(path)
        for name in imports:
            bare = name.lstrip(".")
            for banned in FORBIDDEN[layer]:
                if bare == banned or bare.startswith(banned + ".") or (banned.endswith(".") and bare.startswith(banned)):
                    if layer == "domain" and banned in ("os", "threading") and name != banned:
                        continue
                    raise AssertionError(f"{path.relative_to(ROOT)} ({layer}) imports {name}")


def test_no_module_is_over_budget():
    over = [(p.relative_to(ROOT), n) for p in ROOT.rglob("*.py") if (n := len(p.read_text(encoding='utf-8').splitlines())) > MAX_LINES]
    assert not over, f"split these: {over}"


def test_the_domain_does_no_io():
    for path in (ROOT / "domain").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        for word in ("open(", "socket", "subprocess", "sqlite3", "time.time", "datetime.now"):
            assert word not in source, f"{path.name} reaches outside: {word}"
