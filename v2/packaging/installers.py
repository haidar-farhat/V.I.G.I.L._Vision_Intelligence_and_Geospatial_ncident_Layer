"""Installers for the packaged bundle: one format per platform, none of them signed.

Driven by `python tasks.py installer`, which builds `dist/vigil` first. Each
builder takes that folder and produces something an operator can install
without a Python, a compiler or an Internet connection:

| Platform | Always | When the tool is present |
|---|---|---|
| Windows | `.zip` | `.msi` (WiX 3: `candle` and `light`), `-setup.exe` (NSIS: `makensis`) |
| Linux | `.tar.gz`, `.deb` | — the `.deb` is written here, no `dpkg` needed |
| macOS | `.tar.gz` | `.pkg` (`pkgbuild`), `.dmg` (`hdiutil`) |

The archive is the deliverable that always works: it needs nothing installed,
nothing elevated, and installs by unpacking. Everything else is the
convenience on top, and each one records in its own metadata that it is
**unsigned**. Signing needs a certificate this repository does not have; an
unsigned installer is honest about what it is, and a self-signed one would be
worse than none because it teaches an operator to click through the warning
that is supposed to protect them.

# Where each was proven

The Windows builders run on the machine this was written on and the MSI is
checked after building by extracting it (`msiexec /a`) and comparing every
file against the bundle. The Linux `.deb` writer is pure Python and its
output is read back by `tests/test_installers.py` on any platform; the
package is *installed* only on the Linux CI runner. The macOS `.pkg` and
`.dmg` call Apple's own tools and are built and installed only on the macOS
CI runner. Nothing here claims a platform it has not run on.

This file lives in `packaging/` rather than `tools/` because the WiX schema
namespace is a URL, and the offline audit is right to refuse a URL in shipped
source — this is build tooling, not the product, and none of it ships.
"""

from __future__ import annotations

import hashlib
import io
import os
import platform
import plistlib
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import time
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

PRODUCT = "Vigil"
PACKAGE = "vigil"            # the Debian package name, and the bundle folder
MANUFACTURER = "Romisys"
IDENTIFIER = "com.romisys.vigil"
#: Fixed for the life of the product. Windows Installer uses it to recognise
#: an earlier version of the same product and replace it; a new one here would
#: leave two Vigils installed side by side.
UPGRADE_CODE = "8F4C2B6E-3A1D-4C7E-9B2A-5D6E7F8A9B0C"
#: Component GUIDs are derived from the file's path under this namespace, so
#: the same file gets the same GUID in every build, which is what lets a major
#: upgrade replace it rather than install a second copy beside it.
_GUID_NAMESPACE = uuid.UUID("2c1a3b4d-5e6f-4a7b-8c9d-0e1f2a3b4c5d")

WIX_NAMESPACE = "http://schemas.microsoft.com/wix/2006/wi"

#: What the Qt console needs from a Debian-family system. The same list the CI
#: runner installs before running the suite headless.
DEB_DEPENDS = (
    "libc6 (>= 2.31)", "libgl1", "libegl1", "libglib2.0-0", "libfontconfig1", "libdbus-1-3",
    "libxkbcommon0", "libxkbcommon-x11-0", "libxcb-cursor0", "libxcb-icccm4", "libxcb-keysyms1",
    "libxcb-shape0", "libxcb-xinerama0", "libxcb-render-util0", "libxcb-image0",
)


@dataclass(frozen=True, slots=True)
class Artefact:
    path: Path
    kind: str          # "archive", "msi", "nsis", "deb", "pkg", "dmg"
    note: str = ""


# --------------------------------------------------------------------- shared

def platform_tag() -> str:
    """`windows-x64`, `linux-x64`, `macos-arm64` — what goes in a file name."""
    system = {"win32": "windows", "darwin": "macos"}.get(sys.platform, "linux")
    machine = platform.machine().lower()
    arch = "arm64" if machine in ("arm64", "aarch64") else "x64"
    return f"{system}-{arch}"


def exe_name(stem: str) -> str:
    return f"{stem}.exe" if os.name == "nt" else stem


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_sums(out_dir: Path, artefacts: list[Artefact]) -> Path:
    """One sums file per platform beside the artefacts, in the format `sha256sum -c`
    reads. Named by platform so the three CI runners' files can sit in one
    release without overwriting each other."""
    lines = [f"{sha256_file(a.path)}  {a.path.name}" for a in artefacts if a.path.is_file()]
    sums = out_dir / f"SHA256SUMS-{platform_tag()}.txt"
    sums.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sums


def build_archive(payload: Path, stem: str, out_dir: Path) -> Artefact:
    """A zip on Windows, a tar.gz elsewhere, each unpacking to one folder named `stem`."""
    if os.name == "nt":
        archive = out_dir / f"{stem}.zip"
        if archive.exists():
            archive.unlink()
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
            for path in sorted(payload.rglob("*")):
                if path.is_file():
                    bundle.write(path, Path(stem) / path.relative_to(payload))
    else:
        archive = out_dir / f"{stem}.tar.gz"
        if archive.exists():
            archive.unlink()
        with tarfile.open(archive, "w:gz", compresslevel=6) as bundle:
            bundle.add(payload, arcname=stem, filter=_root_owned)
    return Artefact(archive, "archive", "installs by unpacking; needs nothing")


def _root_owned(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = info.gid = 0
    info.uname = info.gname = "root"
    return info


def numeric_version(version: str, parts: int = 3) -> str:
    """The leading numeric components of a version, for formats that accept nothing else.

    Windows Installer wants `major.minor.build`, all integers; `2.0.0a1` is not
    a version to it. The pre-release tag is kept in the product's display name
    and comments instead, so it is still visible where a person looks.
    """
    numbers = re.findall(r"\d+", version.split("+")[0])[:parts]
    while len(numbers) < parts:
        numbers.append("0")
    return ".".join(numbers)


def debian_version(version: str) -> str:
    """`2.0.0a1` -> `2.0.0~a1`: a tilde sorts *before* the release it precedes,
    which is what makes `2.0.0` an upgrade from its own alpha."""
    match = re.match(r"^(\d+(?:\.\d+)*)(.*)$", version)
    if not match:
        return version
    numbers, rest = match.groups()
    rest = rest.lstrip("-.")
    return f"{numbers}~{rest}" if rest else numbers


# -------------------------------------------------------------------- Windows

def find_wix() -> Path | None:
    """The WiX 3 `bin` folder: on PATH, under `%WIX%`, under Program Files, or in
    the local build cache — the last is where a developer machine keeps the
    portable binaries so nothing is installed system-wide."""
    candle = shutil.which("candle")
    if candle:
        return Path(candle).parent
    candidates = []
    if os.environ.get("WIX"):
        candidates.append(Path(os.environ["WIX"]) / "bin")
    for base in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles")):
        if base:
            candidates += sorted(Path(base).glob("WiX Toolset v3*/bin"), reverse=True)
    if os.environ.get("LOCALAPPDATA"):
        candidates.append(Path(os.environ["LOCALAPPDATA"]) / "vigil-build" / "wix3")
    for folder in candidates:
        if (folder / "candle.exe").is_file() and (folder / "light.exe").is_file():
            return folder
    return None


def _ident(prefix: str, relative: str) -> str:
    """A WiX identifier: letters, digits, underscores, at most 72 characters,
    and the same for the same path in every build."""
    return prefix + hashlib.sha1(relative.encode("utf-8")).hexdigest()[:20]


def component_guid(relative: str) -> str:
    return str(uuid.uuid5(_GUID_NAMESPACE, relative)).upper()


def license_rtf(text: str) -> str:
    """The plainest RTF that Windows Installer's licence page will display."""
    out = []
    for char in text:
        if char in "\\{}":
            out.append("\\" + char)
        elif char == "\n":
            out.append("\\par\n")
        elif ord(char) > 127:
            out.append(f"\\u{ord(char)}?")
        else:
            out.append(char)
    return "{\\rtf1\\ansi\\deff0{\\fonttbl{\\f0\\fmodern Consolas;}}\\f0\\fs18\n" + "".join(out) + "}"


def wix_source(payload: Path, version: str, *, license_rtf_path: Path | None = None) -> str:
    """The whole installer as WiX source, generated from the bundle folder.

    Generated rather than kept as a file for the reason the NSIS script is: a
    hand-maintained file list is a second description of the build, and the
    copy that drifts is the one nobody runs. One component per file, each with
    a GUID derived from its path, so an upgrade replaces what it should.
    """
    display = f"{PRODUCT} {version}"
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<Wix xmlns={quoteattr(WIX_NAMESPACE)}>',
        f'  <Product Id="*" Name={quoteattr(display)} Language="1033" Version="{numeric_version(version)}"',
        f'           Manufacturer={quoteattr(MANUFACTURER)} UpgradeCode="{UPGRADE_CODE}">',
        f'    <Package InstallerVersion="500" Compressed="yes" InstallScope="perMachine" Platform="x64"',
        f'             Description={quoteattr(display + " — local-first incident intelligence")}',
        '             Comments="UNSIGNED. Built by tasks.py installer; signing needs a certificate this build does not have." />',
        '    <MajorUpgrade AllowSameVersionUpgrades="yes"',
        '                  DowngradeErrorMessage="A newer version of [ProductName] is already installed." />',
        '    <MediaTemplate EmbedCab="yes" CompressionLevel="high" />',
        '    <Directory Id="TARGETDIR" Name="SourceDir">',
        '      <Directory Id="ProgramFiles64Folder">',
        f'        <Directory Id="INSTALLDIR" Name={quoteattr(PRODUCT)}>',
    ]
    refs: list[str] = []
    lines += _wix_tree(payload, payload, refs, depth=5)
    lines += [
        '        </Directory>',
        '      </Directory>',
        '      <Directory Id="ProgramMenuFolder">',
        f'        <Directory Id="MenuDir" Name={quoteattr(PRODUCT)} />',
        '      </Directory>',
        '    </Directory>',
        # The Start-menu entry opens the window; the command line is what PATH is for.
        '    <DirectoryRef Id="MenuDir">',
        f'      <Component Id="Shortcuts" Guid="{component_guid("//shortcuts")}" Win64="yes">',
        f'        <Shortcut Id="ConsoleShortcut" Name={quoteattr(PRODUCT)} Description="Open the operator console"',
        '                  Target="[INSTALLDIR]vigil-console.exe" WorkingDirectory="INSTALLDIR" />',
        '        <RemoveFolder Id="RemoveMenuDir" On="uninstall" />',
        f'        <RegistryValue Root="HKCU" Key="Software\\{PRODUCT}" Name="installed" Type="integer" Value="1" KeyPath="yes" />',
        '      </Component>',
        '    </DirectoryRef>',
        '    <DirectoryRef Id="INSTALLDIR">',
        f'      <Component Id="PathEntry" Guid="{component_guid("//path")}" Win64="yes">',
        f'        <RegistryValue Root="HKLM" Key="Software\\{PRODUCT}" Name="InstallDir" Type="string" Value="[INSTALLDIR]" KeyPath="yes" />',
        '        <Environment Id="PathEntry" Name="PATH" Value="[INSTALLDIR]" Permanent="no" Part="last" Action="set" System="yes" />',
        '      </Component>',
        '    </DirectoryRef>',
        '    <ComponentGroup Id="Bundle">',
    ]
    lines += [f'      <ComponentRef Id="{ref}" />' for ref in refs]
    lines += [
        '    </ComponentGroup>',
        f'    <Feature Id="Main" Title={quoteattr(display)} Level="1" Absent="disallow">',
        '      <ComponentGroupRef Id="Bundle" />',
        '      <ComponentRef Id="Shortcuts" />',
        '      <ComponentRef Id="PathEntry" />',
        '    </Feature>',
        '    <Property Id="WIXUI_INSTALLDIR" Value="INSTALLDIR" />',
    ]
    if license_rtf_path is not None:
        lines.append(f'    <WixVariable Id="WixUILicenseRtf" Value={quoteattr(str(license_rtf_path))} />')
    lines += [
        '    <UIRef Id="WixUI_InstallDir" />',
        '  </Product>',
        '</Wix>',
    ]
    return "\n".join(lines) + "\n"


def _wix_tree(folder: Path, root: Path, refs: list[str], *, depth: int) -> list[str]:
    pad = "  " * depth
    lines: list[str] = []
    entries = sorted(folder.iterdir(), key=lambda p: (p.is_dir(), p.name.lower()))
    if not entries:
        # An empty folder is still part of the layout; without a component
        # Windows Installer would never create it.
        relative = folder.relative_to(root).as_posix()
        cid = _ident("c_", relative + "/")
        refs.append(cid)
        lines.append(f'{pad}<Component Id="{cid}" Guid="{component_guid(relative + "/")}" Win64="yes"><CreateFolder /></Component>')
        return lines
    for path in entries:
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            lines.append(f'{pad}<Directory Id="{_ident("d_", relative)}" Name={quoteattr(path.name)}>')
            lines += _wix_tree(path, root, refs, depth=depth + 1)
            lines.append(f'{pad}</Directory>')
        else:
            cid = _ident("c_", relative)
            refs.append(cid)
            lines.append(f'{pad}<Component Id="{cid}" Guid="{component_guid(relative)}" Win64="yes">')
            lines.append(f'{pad}  <File Id="{_ident("f_", relative)}" Name={quoteattr(path.name)} '
                         f'Source={quoteattr(str(path))} KeyPath="yes" />')
            lines.append(f'{pad}</Component>')
    return lines


def build_msi(payload: Path, version: str, stem: str, out_dir: Path, build_dir: Path,
              license_text: str | None, run=subprocess.call) -> Artefact | None:
    """`candle` then `light`. Returns `None`, having said why, when WiX is absent."""
    wix = find_wix()
    if wix is None:
        print("WiX 3 (candle.exe and light.exe) is not on this machine, so no .msi was built. The archive "
              "installs by unpacking. Put the WiX 3 binaries on PATH, or under "
              f"{Path(os.environ.get('LOCALAPPDATA', '~')) / 'vigil-build' / 'wix3'}, and re-run.")
        return None
    build_dir.mkdir(parents=True, exist_ok=True)
    rtf = None
    if license_text:
        rtf = build_dir / "license.rtf"
        rtf.write_text(license_rtf(license_text), encoding="utf-8")
    source = build_dir / f"{stem}.wxs"
    source.write_text(wix_source(payload, version, license_rtf_path=rtf), encoding="utf-8")
    obj = build_dir / f"{stem}.wixobj"
    msi = out_dir / f"{stem}.msi"
    if run([str(wix / "candle.exe"), "-nologo", "-arch", "x64", "-out", str(obj), str(source)]) != 0:
        return None
    # `-spdb`: no .wixpdb beside the installer. `-sw1076`: every component here
    # has a real KeyPath, and the warning about ones that do not is noise.
    if run([str(wix / "light.exe"), "-nologo", "-ext", "WixUIExtension", "-cultures:en-us", "-spdb",
            "-sw1076", "-out", str(msi), str(obj)]) != 0:
        return None
    return Artefact(msi, "msi", "UNSIGNED; Windows will warn, and that warning is correct")


def verify_msi(msi: Path, scratch: Path, payload: Path) -> list[str]:
    """Extract the MSI with an administrative install and compare it file for
    file against the bundle. This is the check that the installer carries what
    was built, run on the machine that built it."""
    if shutil.which("msiexec") is None:
        return ["msiexec is not available, so the .msi was not extracted for checking"]
    target = scratch / "extract"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    # A string, not a list: msiexec wants `TARGETDIR="C:\with space"` and refuses
    # the whole-argument quoting a list gets on Windows with error 1639.
    code = subprocess.call(f'msiexec /a "{msi}" /qn TARGETDIR="{target}"')
    if code != 0:
        return [f"msiexec /a exited {code}; the installer could not be extracted for checking"]
    roots = [p for p in target.rglob(exe_name("vigil")) if p.is_file()]
    if not roots:
        return ["the extracted installer holds no vigil.exe"]
    installed = roots[0].parent
    expected = {p.relative_to(payload).as_posix(): p.stat().st_size for p in payload.rglob("*") if p.is_file()}
    found = {p.relative_to(installed).as_posix(): p.stat().st_size for p in installed.rglob("*") if p.is_file()}
    problems = []
    missing = sorted(set(expected) - set(found))
    extra = sorted(set(found) - set(expected))
    if missing:
        problems.append(f"{len(missing)} file(s) in the bundle are not in the installer, starting with {missing[0]}")
    if extra:
        problems.append(f"{len(extra)} file(s) in the installer are not in the bundle, starting with {extra[0]}")
    wrong = [name for name in set(expected) & set(found) if expected[name] != found[name]]
    if wrong:
        problems.append(f"{len(wrong)} file(s) differ in size, starting with {sorted(wrong)[0]}")
    return problems


#: The NSIS script, generated for the same reason the WiX source is.
NSIS_TEMPLATE = """\
; Generated by `python tasks.py installer`. Do not edit; edit packaging/installers.py.
Unicode true
Name "{name} {version}"
OutFile "{output}"
InstallDir "$PROGRAMFILES64\\{name}"
InstallDirRegKey HKLM "Software\\{name}" "InstallDir"
RequestExecutionLevel admin
ShowInstDetails show

Page directory
Page instfiles
UninstPage uninstConfirm
UninstPage instfiles

Section "Install"
  SetOutPath "$INSTDIR"
  File /r "{payload}\\*.*"
  WriteRegStr HKLM "Software\\{name}" "InstallDir" "$INSTDIR"
  WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\{name}" \
      "DisplayName" "{name} {version}"
  WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\{name}" \
      "UninstallString" "$INSTDIR\\uninstall.exe"
  WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\{name}" \
      "DisplayVersion" "{version}"
  WriteUninstaller "$INSTDIR\\uninstall.exe"
  CreateDirectory "$SMPROGRAMS\\{name}"
  CreateShortcut "$SMPROGRAMS\\{name}\\{name}.lnk" "$INSTDIR\\vigil-console.exe"
SectionEnd

Section "Uninstall"
  ; The application only. A site's database, recordings and evidence live in
  ; the data directory and are NOT removed: uninstalling a program must not
  ; destroy the footage of an incident, and somebody who wants that gone can
  ; be told where it is rather than have it deleted from under them.
  Delete "$SMPROGRAMS\\{name}\\{name}.lnk"
  RMDir "$SMPROGRAMS\\{name}"
  Delete "$INSTDIR\\uninstall.exe"
  RMDir /r "$INSTDIR\\_internal"
  Delete "$INSTDIR\\*.exe"
  Delete "$INSTDIR\\*.dll"
  RMDir /r "$INSTDIR\\models"
  DeleteRegKey HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\{name}"
  DeleteRegKey HKLM "Software\\{name}"
  RMDir "$INSTDIR"
SectionEnd
"""


def build_nsis(payload: Path, version: str, stem: str, out_dir: Path, run=subprocess.call) -> Artefact | None:
    script = out_dir / f"{stem}.nsi"
    setup = out_dir / f"{stem}-setup.exe"
    script.write_text(NSIS_TEMPLATE.format(name=PRODUCT, version=version, output=str(setup),
                                           payload=str(payload)), encoding="utf-8")
    makensis = shutil.which("makensis")
    if makensis is None:
        print(f"script   {script}")
        print("makensis is not on this machine, so no -setup.exe was built. Install NSIS and re-run for it.")
        return None
    if run([makensis, str(script)]) != 0:
        return None
    return Artefact(setup, "nsis", "UNSIGNED; Windows will warn, and that warning is correct")


# ---------------------------------------------------------------------- Linux

def desktop_entry(version: str) -> str:
    return "\n".join([
        "[Desktop Entry]",
        "Type=Application",
        f"Name={PRODUCT}",
        f"Version={version}",
        "Comment=Local-first multi-camera incident intelligence",
        f"Exec=/opt/{PACKAGE}/vigil-console",
        "Terminal=false",
        "Categories=Video;Security;",
    ]) + "\n"


def deb_control(version: str, arch: str, installed_kib: int) -> str:
    return "\n".join([
        f"Package: {PACKAGE}",
        f"Version: {debian_version(version)}",
        f"Architecture: {arch}",
        f"Maintainer: {MANUFACTURER} <swteam@romisys.com>",
        f"Installed-Size: {installed_kib}",
        f"Depends: {', '.join(DEB_DEPENDS)}",
        "Section: video",
        "Priority: optional",
        f"Description: {PRODUCT} — local-first multi-camera incident intelligence",
        " Ordinary cameras in, a small number of reviewable incidents out, with the",
        " evidence that produced each one, and no route to the Internet at any point.",
        " .",
        " UNSIGNED: this package carries no signature and is not from a repository.",
    ]) + "\n"


def _ar(members: list[tuple[str, bytes]]) -> bytes:
    """A `.deb` is an `ar` archive of exactly three members, in this order.

    Written by hand because `dpkg-deb` exists only on Debian systems and the
    format is sixty bytes of header per member; a dependency on the host's
    packaging tools would mean the package could only be tested where it is
    installed.
    """
    out = io.BytesIO()
    out.write(b"!<arch>\n")
    now = str(int(time.time())).encode()
    for name, data in members:
        header = (name.encode().ljust(16) + now.ljust(12) + b"0".ljust(6) + b"0".ljust(6)
                  + b"100644".ljust(8) + str(len(data)).encode().ljust(10) + b"`\n")
        assert len(header) == 60, header
        out.write(header)
        out.write(data)
        if len(data) % 2:
            out.write(b"\n")
    return out.getvalue()


def read_ar(data: bytes) -> list[tuple[str, bytes]]:
    """The inverse of `_ar`, so a test can open what was written."""
    assert data[:8] == b"!<arch>\n", "not an ar archive"
    members, offset = [], 8
    while offset + 60 <= len(data):
        header = data[offset:offset + 60]
        name = header[:16].decode().strip().rstrip("/")
        size = int(header[48:58].decode().strip())
        body = data[offset + 60:offset + 60 + size]
        members.append((name, body))
        offset += 60 + size + (size % 2)
    return members


def _tar_bytes(add) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz", compresslevel=6) as tar:
        add(tar)
    return buffer.getvalue()


def build_deb(payload: Path, version: str, stem: str, out_dir: Path, arch: str | None = None) -> Artefact:
    """`/opt/vigil` holds the bundle, `/usr/bin` gets two symlinks, the desktop
    gets a launcher. The data tree is owned by root and the two executables
    are the only thing marked executable beyond what the bundle already was."""
    arch = arch or ("arm64" if platform.machine().lower() in ("arm64", "aarch64") else "amd64")
    prefix = f"opt/{PACKAGE}"
    size = sum(p.stat().st_size for p in payload.rglob("*") if p.is_file())

    def data(tar: tarfile.TarFile) -> None:
        for name in ("opt", prefix, "usr", "usr/bin", "usr/share", "usr/share/applications",
                     "usr/share/doc", f"usr/share/doc/{PACKAGE}"):
            _add_dir(tar, name)
        for path in sorted(payload.rglob("*")):
            relative = path.relative_to(payload).as_posix()
            if path.is_dir():
                _add_dir(tar, f"{prefix}/{relative}")
                continue
            info = tar.gettarinfo(str(path), arcname=f"{prefix}/{relative}")
            info.uid = info.gid = 0
            info.uname = info.gname = "root"
            executable = relative in ("vigil", "vigil-console") or path.suffix in (".so",) or ".so." in path.name
            info.mode = 0o755 if executable or (info.mode & stat.S_IXUSR) else 0o644
            with path.open("rb") as handle:
                tar.addfile(info, handle)
        for stem_name in ("vigil", "vigil-console"):
            link = tarfile.TarInfo(f"usr/bin/{stem_name}")
            link.type = tarfile.SYMTYPE
            link.linkname = f"/{prefix}/{stem_name}"
            link.uname = link.gname = "root"
            tar.addfile(link)
        _add_text(tar, f"usr/share/applications/{PACKAGE}.desktop", desktop_entry(version))
        _add_text(tar, f"usr/share/doc/{PACKAGE}/README", f"{PRODUCT} {version}. Run `vigil doctor` first.\n")

    def control(tar: tarfile.TarFile) -> None:
        _add_text(tar, "control", deb_control(version, arch, installed_kib=(size + 1023) // 1024))

    deb = out_dir / f"{stem}.deb"
    deb.write_bytes(_ar([("debian-binary", b"2.0\n"),
                         ("control.tar.gz", _tar_bytes(control)),
                         ("data.tar.gz", _tar_bytes(data))]))
    return Artefact(deb, "deb", "UNSIGNED; not from a repository, install with `apt install ./…deb`")


def _add_dir(tar: tarfile.TarFile, name: str) -> None:
    info = tarfile.TarInfo(name)
    info.type = tarfile.DIRTYPE
    info.mode = 0o755
    info.mtime = int(time.time())
    info.uname = info.gname = "root"
    tar.addfile(info)


def _add_text(tar: tarfile.TarFile, name: str, text: str, mode: int = 0o644) -> None:
    data = text.encode("utf-8")
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = mode
    info.mtime = int(time.time())
    info.uname = info.gname = "root"
    tar.addfile(info, io.BytesIO(data))


# ---------------------------------------------------------------------- macOS

def info_plist(version: str) -> bytes:
    return plistlib.dumps({
        "CFBundleName": PRODUCT,
        "CFBundleDisplayName": PRODUCT,
        "CFBundleIdentifier": f"{IDENTIFIER}.console",
        "CFBundleVersion": numeric_version(version),
        "CFBundleShortVersionString": version,
        "CFBundleExecutable": PRODUCT,
        "CFBundlePackageType": "APPL",
        "LSMinimumSystemVersion": "12.0",
        "NSHighResolutionCapable": True,
    })


def launcher_script(install_dir: str = f"/usr/local/{PACKAGE}") -> str:
    """`/Applications/Vigil.app` is a launcher for the console installed under
    `/usr/local/vigil`, not a second copy of the 800 MB bundle."""
    return f'#!/bin/sh\nexec "{install_dir}/vigil-console" "$@"\n'


def stage_macos(payload: Path, version: str, staging: Path) -> Path:
    """The tree `pkgbuild` installs at `/`."""
    if staging.exists():
        shutil.rmtree(staging)
    install = staging / "usr" / "local" / PACKAGE
    shutil.copytree(payload, install, symlinks=True)
    bins = staging / "usr" / "local" / "bin"
    bins.mkdir(parents=True)
    for stem_name in ("vigil", "vigil-console"):
        os.symlink(f"/usr/local/{PACKAGE}/{stem_name}", bins / stem_name)
    app = staging / "Applications" / f"{PRODUCT}.app" / "Contents"
    (app / "MacOS").mkdir(parents=True)
    (app / "Info.plist").write_bytes(info_plist(version))
    launcher = app / "MacOS" / PRODUCT
    launcher.write_text(launcher_script(), encoding="utf-8")
    launcher.chmod(0o755)
    return staging


def build_pkg(payload: Path, version: str, stem: str, out_dir: Path, build_dir: Path,
              run=subprocess.call) -> list[Artefact]:
    out: list[Artefact] = []
    if shutil.which("pkgbuild") is None:
        print("pkgbuild is not on this machine, so no .pkg was built. The archive installs by unpacking.")
        return out
    staging = stage_macos(payload, version, build_dir / "pkgroot")
    pkg = out_dir / f"{stem}.pkg"
    if run(["pkgbuild", "--root", str(staging), "--identifier", IDENTIFIER, "--version", numeric_version(version),
            "--install-location", "/", str(pkg)]) != 0:
        return out
    out.append(Artefact(pkg, "pkg", "UNSIGNED; Gatekeeper will refuse a double-click — open it with right-click, Open"))
    if shutil.which("hdiutil") is None:
        return out
    dmg_root = build_dir / "dmgroot"
    if dmg_root.exists():
        shutil.rmtree(dmg_root)
    dmg_root.mkdir(parents=True)
    shutil.copy2(pkg, dmg_root / pkg.name)
    (dmg_root / "READ ME FIRST.txt").write_text(
        f"{PRODUCT} {version}\n\nThis package is UNSIGNED. macOS will refuse a double-click; right-click the "
        f".pkg and choose Open. It installs to /usr/local/{PACKAGE}, puts `vigil` and `vigil-console` in "
        "/usr/local/bin, and adds Vigil to Applications. Run `vigil doctor` first.\n", encoding="utf-8")
    dmg = out_dir / f"{stem}.dmg"
    if dmg.exists():
        dmg.unlink()
    if run(["hdiutil", "create", "-volname", f"{PRODUCT} {version}", "-srcfolder", str(dmg_root), "-ov",
            "-format", "UDZO", str(dmg)]) == 0:
        out.append(Artefact(dmg, "dmg", "the .pkg above, on a disk image"))
    return out


# ------------------------------------------------------------------ the whole

def build_all(payload: Path, version: str, out_dir: Path, build_dir: Path, license_text: str | None,
              *, verify: bool = True) -> tuple[list[Artefact], list[str]]:
    """Every installer this platform can produce, and every problem found checking them."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{PACKAGE}-{version}-{platform_tag()}"
    # Yesterday's files beside today's would be listed by nothing and trusted
    # by somebody; only this platform's are removed, so a folder shared by
    # several runners keeps the others' work.
    for old in list(out_dir.glob(f"{stem}*")) + list(out_dir.glob(f"SHA256SUMS-{platform_tag()}*")):
        old.unlink()
    artefacts = [build_archive(payload, stem, out_dir)]
    problems: list[str] = []
    if sys.platform == "win32":
        msi = build_msi(payload, version, stem, out_dir, build_dir, license_text)
        if msi is not None:
            artefacts.append(msi)
            if verify:
                problems += verify_msi(msi.path, build_dir / "verify", payload)
        nsis = build_nsis(payload, version, stem, out_dir)
        if nsis is not None:
            artefacts.append(nsis)
    elif sys.platform == "darwin":
        artefacts += build_pkg(payload, version, stem, out_dir, build_dir)
    else:
        artefacts.append(build_deb(payload, version, stem, out_dir))
    write_sums(out_dir, artefacts)
    return artefacts, problems


__all__ = [name for name in dir() if not name.startswith("_")]
