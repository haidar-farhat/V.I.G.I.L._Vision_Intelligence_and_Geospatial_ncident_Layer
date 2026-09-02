# PyInstaller build. Run it with `python tasks.py package`, never directly —
# the task builds the Rust core first, and packaging a stale library is how a
# green build ships broken geometry.
#
# Three executables come out of one analysis. `entry.py` decides which
# application starts from the name it was launched as, because analysing PySide6
# three times costs minutes and produces the same answer:
#
#   SentinelVision.exe        the operator console. No terminal window.
#   SentinelVision-dev.exe    the same console with a terminal attached and
#                             --verbose forced on. Not a debug build — the same
#                             code with its output visible. It exists because a
#                             packaged Qt application on Windows has nowhere to
#                             print, so an exception before the window appears
#                             is invisible and "it just closes" is the least
#                             actionable bug report there is.
#   sentinel.exe              the headless analyser, for a scheduled job or a
#                             developer with one file to look at.
#
# `onedir`, not `onefile`. A onefile build unpacks ~200 MB of Qt to a temporary
# directory on every launch: slow, leaves debris when it is killed, and on a
# locked-down machine can be blocked outright. What ships is a folder with
# executables in it, which is what every real Qt application ships.

import sys
from pathlib import Path

# PyInstaller sets SPECPATH to the directory holding this file.
ROOT = Path(SPECPATH).parent
ENGINE = ROOT / "engine"
CONSOLE = ROOT / "apps" / "console"


def core_library():
    """The built Rust core, and where it must land inside the bundle.

    Beside the `sentinel` package, because `core.py` looks in its own directory
    first — so the packaged application finds its core exactly the way a
    checkout does, with no frozen-build special case in the loader.
    """
    names = {"win32": "sentinel_core.dll", "darwin": "libsentinel_core.dylib"}
    name = names.get(sys.platform, "libsentinel_core.so")
    built = ROOT / "core" / "target" / "release" / name

    if not built.is_file():
        raise SystemExit(
            f"The engine core is not built: {built} does not exist.\n"
            "Run `python tasks.py package`, which builds it first. Packaging "
            "without it produces an application that cannot start."
        )
    return (str(built), "sentinel")


# Nothing here reaches the network at runtime and nothing may be added that
# does. These are excluded because they pull in a network stack, or ship a
# browser engine, or bloat the bundle with something a security appliance has no
# use for. QtWebEngine in particular: shipping a browser in a control-room
# console means inheriting its update cadence and its network assumptions.
EXCLUDED = [
    "tkinter",
    "test",
    "unittest",
    "pydoc_data",
    "setuptools",
    "pip",
    "onnx",  # the test-fixture model builder. onnxruntime stays; onnx does not.
    "matplotlib",
    "IPython",
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebEngineQuick",
    "PySide6.QtQuick",
    "PySide6.QtQml",
    "PySide6.Qt3DCore",
    "PySide6.QtMultimedia",
    "PySide6.QtCharts",
    "PySide6.QtDataVisualization",
]

analysis = Analysis(
    [str(Path(SPECPATH) / "entry.py")],
    pathex=[str(ENGINE), str(CONSOLE)],
    binaries=[core_library()],
    datas=[],
    hiddenimports=[
        # Imported inside functions, so static analysis cannot see them.
        "sentinel.cli",
        "sentinel_console.app",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=EXCLUDED,
    noarchive=False,
)

pyz = PYZ(analysis.pure)


def executable(name: str, *, terminal: bool):
    return EXE(
        pyz,
        analysis.scripts,
        [],
        exclude_binaries=True,
        name=name,
        console=terminal,
        debug=False,
        strip=False,
        # UPX is off deliberately. A packed binary looks exactly like malware to
        # every endpoint product an operator runs, and a security appliance that
        # trips the antivirus is a security appliance that gets uninstalled.
        upx=False,
        icon=None,
    )


COLLECT(
    executable("SentinelVision", terminal=False),
    executable("SentinelVision-dev", terminal=True),
    executable("sentinel", terminal=True),
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    name="SentinelVision",
)
