"""The installers, read back rather than trusted.

A package that builds and carries the wrong tree is the failure nobody sees
until a site does, so every format that can be opened without its platform's
tools is opened here: the WiX source as XML, the `.deb` as the `ar` and `tar`
it is, the archive as itself. What needs `light`, `pkgbuild` or `dpkg` is
proven on the CI runner that has it, and nowhere else.
"""

from __future__ import annotations

import importlib.util
import io
import os
import plistlib
import sys
import tarfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VERSION = "2.0.0a1"


def _installers():
    spec = importlib.util.spec_from_file_location("vigil_installers", ROOT / "packaging" / "installers.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bundle(tmp_path):
    """A stand-in for dist/vigil with the shapes that break packagers: an empty
    folder, a shared library, a file name with an ampersand and an apostrophe."""
    payload = tmp_path / "vigil"
    (payload / "_internal" / "deep").mkdir(parents=True)
    (payload / "models").mkdir()
    (payload / "empty").mkdir()
    (payload / "vigil.exe").write_bytes(b"MZ-cli")
    (payload / "vigil").write_bytes(b"ELF-cli")
    (payload / "vigil-console.exe").write_bytes(b"MZ-gui")
    (payload / "vigil-console").write_bytes(b"ELF-gui")
    (payload / "_internal" / "deep" / "odd & name's.txt").write_bytes(b"xyz")
    (payload / "_internal" / "libvigil_core.so").write_bytes(b"so")
    (payload / "models" / "yolov8n-seg.onnx").write_bytes(b"onnx")
    return payload


def _files(payload: Path) -> set[str]:
    return {p.relative_to(payload).as_posix() for p in payload.rglob("*") if p.is_file()}


def test_versions_are_translated_for_formats_that_want_numbers_only():
    installers = _installers()
    assert installers.numeric_version("2.0.0a1") == "2.0.0"
    assert installers.numeric_version("2.1") == "2.1.0"
    assert installers.numeric_version("3.4.5.6+local") == "3.4.5"
    assert installers.debian_version("2.0.0a1") == "2.0.0~a1", "a tilde sorts before the release it precedes"
    assert installers.debian_version("2.0.0-rc1") == "2.0.0~rc1"
    assert installers.debian_version("2.0.0") == "2.0.0"


def test_the_wix_source_carries_every_file_once_and_names_what_it_is(bundle):
    installers = _installers()
    text = installers.wix_source(bundle, VERSION)
    root = ET.fromstring(text)
    ns = {"w": installers.WIX_NAMESPACE}
    product = root.find("w:Product", ns)
    assert product.get("Version") == "2.0.0" and VERSION in product.get("Name")
    assert product.get("UpgradeCode") == installers.UPGRADE_CODE
    assert "UNSIGNED" in product.find("w:Package", ns).get("Comments")

    sources = [Path(f.get("Source")) for f in root.iter(f"{{{installers.WIX_NAMESPACE}}}File")]
    assert {s.relative_to(bundle).as_posix() for s in sources} == _files(bundle)
    assert len(sources) == len(set(sources)), "a file listed twice installs twice"
    names = {f.get("Name") for f in root.iter(f"{{{installers.WIX_NAMESPACE}}}File")}
    assert "odd & name's.txt" in names, "the awkward name survives XML"

    components = list(root.iter(f"{{{installers.WIX_NAMESPACE}}}Component"))
    # One per file, one for the empty folder, one for the shortcut, one for PATH.
    assert len(components) == len(sources) + 1 + 2
    assert all(c.get("Win64") == "yes" for c in components), "a 32-bit component in a 64-bit folder is an ICE80 error"
    ids = [c.get("Id") for c in components]
    guids = [c.get("Guid") for c in components]
    assert len(ids) == len(set(ids)) and len(guids) == len(set(guids))
    assert all(len(i) <= 72 and i[0].isalpha() for i in ids)
    referenced = {r.get("Id") for r in root.iter(f"{{{installers.WIX_NAMESPACE}}}ComponentRef")}
    assert referenced == set(ids), "a component nothing references is never installed"
    assert root.find(".//w:CreateFolder", ns) is not None, "the empty folder is still created"


def test_component_guids_are_the_same_in_every_build(bundle):
    """An upgrade replaces a file by its GUID; a fresh GUID per build would install a second copy beside it."""
    installers = _installers()
    assert installers.component_guid("_internal/a.dll") == installers.component_guid("_internal/a.dll")
    assert installers.component_guid("_internal/a.dll") != installers.component_guid("_internal/b.dll")
    assert installers.wix_source(bundle, VERSION) == installers.wix_source(bundle, VERSION)


def test_the_licence_survives_rtf():
    installers = _installers()
    rtf = installers.license_rtf("a{b}\\c\né")
    assert "\\{" in rtf and "\\}" in rtf and "\\\\" in rtf and "\\par" in rtf and "\\u233?" in rtf
    assert rtf.startswith("{\\rtf1") and rtf.endswith("}")


def test_the_deb_reads_back_as_what_dpkg_expects(bundle, tmp_path):
    installers = _installers()
    artefact = installers.build_deb(bundle, VERSION, "vigil-test-linux-x64", tmp_path, arch="amd64")
    members = installers.read_ar(artefact.path.read_bytes())
    assert [name for name, _ in members] == ["debian-binary", "control.tar.gz", "data.tar.gz"], "order is the format"
    assert members[0][1] == b"2.0\n"

    with tarfile.open(fileobj=io.BytesIO(members[1][1]), mode="r:gz") as control:
        text = control.extractfile("control").read().decode()
    for line in ("Package: vigil", "Version: 2.0.0~a1", "Architecture: amd64", "Depends: libc6"):
        assert line in text, text
    assert "UNSIGNED" in text

    with tarfile.open(fileobj=io.BytesIO(members[2][1]), mode="r:gz") as data:
        entries = {m.name: m for m in data.getmembers()}
        for relative in _files(bundle):
            assert f"opt/vigil/{relative}" in entries, relative
        assert entries["opt/vigil/vigil"].mode == 0o755
        assert entries["opt/vigil/vigil-console"].mode == 0o755
        assert entries["opt/vigil/_internal/libvigil_core.so"].mode == 0o755
        assert entries["opt/vigil/models/yolov8n-seg.onnx"].mode == 0o644
        assert "opt/vigil/empty" in entries and entries["opt/vigil/empty"].isdir()
        link = entries["usr/bin/vigil"]
        assert link.issym() and link.linkname == "/opt/vigil/vigil"
        assert entries["usr/bin/vigil-console"].linkname == "/opt/vigil/vigil-console"
        desktop = data.extractfile("usr/share/applications/vigil.desktop").read().decode()
        assert "Exec=/opt/vigil/vigil-console" in desktop
        assert all(m.uid == 0 and m.gid == 0 for m in entries.values()), "root owns every entry"
        # `ar` pads odd-sized members; the tar inside must still open cleanly,
        # which the reads above already proved for both.


def test_the_macos_launcher_is_a_launcher_and_not_a_second_copy():
    installers = _installers()
    plist = plistlib.loads(installers.info_plist(VERSION))
    assert plist["CFBundleShortVersionString"] == VERSION and plist["CFBundleVersion"] == "2.0.0"
    assert plist["CFBundleExecutable"] == "Vigil" and plist["CFBundlePackageType"] == "APPL"
    script = installers.launcher_script()
    assert script.startswith("#!/bin/sh") and "/usr/local/vigil/vigil-console" in script


@pytest.mark.skipif(os.name == "nt", reason="staging writes symlinks, which need a privilege on Windows")
def test_the_macos_staging_tree_installs_at_root(bundle, tmp_path):
    installers = _installers()
    staging = installers.stage_macos(bundle, VERSION, tmp_path / "root")
    assert (staging / "usr" / "local" / "vigil" / "vigil").is_file()
    assert os.readlink(staging / "usr" / "local" / "bin" / "vigil") == "/usr/local/vigil/vigil"
    launcher = staging / "Applications" / "Vigil.app" / "Contents" / "MacOS" / "Vigil"
    assert launcher.stat().st_mode & 0o111, "the launcher must be executable or the app will not open"


def test_the_archive_unpacks_to_one_folder_holding_everything(bundle, tmp_path):
    installers = _installers()
    artefact = installers.build_archive(bundle, "vigil-test", tmp_path)
    if artefact.path.suffix == ".zip":
        with zipfile.ZipFile(artefact.path) as archive:
            names = archive.namelist()
    else:
        with tarfile.open(artefact.path) as archive:
            names = [m.name for m in archive.getmembers() if m.isfile()]
    assert all(n.startswith("vigil-test/") for n in names)
    assert {n.removeprefix("vigil-test/") for n in names} == _files(bundle)


def test_without_the_platform_tools_the_archive_is_still_built_and_summed(bundle, tmp_path, monkeypatch):
    """No WiX, no NSIS, no pkgbuild: the deliverable that always works is still delivered, and listed."""
    installers = _installers()
    monkeypatch.setattr(installers.shutil, "which", lambda *_: None)
    monkeypatch.setattr(installers, "find_wix", lambda: None)
    artefacts, problems = installers.build_all(bundle, VERSION, tmp_path / "out", tmp_path / "build", "MIT", verify=False)
    kinds = [a.kind for a in artefacts]
    assert kinds[0] == "archive" and not problems
    if sys.platform not in ("win32", "darwin"):
        assert "deb" in kinds, "the .deb needs no tool and is always built on Linux"
    sums = next((tmp_path / "out").glob("SHA256SUMS-*.txt")).read_text(encoding="utf-8").splitlines()
    assert len(sums) == len(artefacts)
    assert all(len(line.split("  ")[0]) == 64 for line in sums)
