#!/usr/bin/env python3
"""Look for phone-home endpoints inside compiled dependencies.

`offline_audit.py` reads this project's own source and would have passed the
following forever, because the destination is not in any file it scans:

    libonnxruntime.so.1.29.0
      https://mobile.events.data.microsoft.com/OneCollector/1.0/   x3
      OneCollector                                                 x2
      mbedtls                                                    x127
      onnxruntime.db, osDescription, cpuModel, totalMemoryMB

That is a dependency this project already ships, on by default in the official
builds, with a statically linked TLS stack, a persistent device id and a machine
fingerprint payload. It was invisible to a guard whose docstring says *"a
dependency that phones home does so whether or not this code asked it to"*.
This is the guard that can see it.

**What it does.** Walks a directory of shipped artifacts, reads every compiled
file, and looks for two things: a curated list of known telemetry and analytics
endpoints, and any URL-shaped string whose host is not obviously inert. The
first list fails the build. The second is reported for a person to look at,
because failing on it is not possible — see below.

**Why it cannot simply fail on every URL.** Signed Windows binaries embed the
certificate chain of whoever signed them, and that chain contains CRL and
issuer-certificate URLs — `crl.microsoft.com`, `www.microsoft.com/pkiops/…`.
Those are inert data inside a signature blob, not something the code fetches,
and every signed DLL on the machine has them. A guard that failed on those would
fail on everything and be switched off within a day. So known-bad fails, and
unknown is reported.

**What it cannot do.** A host assembled at runtime from parts, or obfuscated,
is invisible to it. It raises the cost of hiding a phone-home; it does not make
it impossible. The offline CI job — the whole suite with outbound traffic
dropped — is what covers behaviour rather than strings, along every path the
tests actually reach.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Extensions worth reading. Everything here is machine code or an archive of it.
BINARY_SUFFIXES = {".so", ".dll", ".dylib", ".pyd", ".exe", ".bin", ".node"}

#: A versioned shared object is `libfoo.so.1.29.0`, whose `Path.suffix` is
#: `.0`. That is the *normal* name for the real library on Linux, and matching
#: on the final suffix skipped it: this audit's first run read the Python
#: extension module and walked straight past `libonnxruntime.so.1.29.0` — the
#: 28 MB file that actually contains the uploader.
VERSIONED_SO = re.compile(r"\.so(\.\d+)+$")

#: Endpoints that exist to receive data about the machine running the software.
#: Finding one inside a shipped binary fails the build. Each entry is a host or
#: a distinctive path, lowercase, matched as a substring of the raw bytes.
TELEMETRY_ENDPOINTS = [
    # Microsoft 1DS / OneCollector — the one that was actually found.
    b"events.data.microsoft.com",
    b"onecollector",
    b"vortex.data.microsoft.com",
    b"dc.services.visualstudio.com",
    # The usual analytics and crash-reporting collectors.
    b"google-analytics.com",
    b"analytics.google.com",
    b"app.posthog.com",
    b"api.segment.io",
    b"api.amplitude.com",
    b"api.mixpanel.com",
    b"sentry.io",
    b"bugsnag.com",
    b"api.rollbar.com",
    b"datadoghq.com",
    b"newrelic.com",
    b"in.appcenter.ms",
    b"firebaseinstallations.googleapis.com",
    b"crashlytics.com",
    # Model and asset fetchers, which break the promise just as thoroughly.
    b"huggingface.co",
    b"cdn-lfs.huggingface.co",
    b"storage.googleapis.com/tfhub",
    b"cdn.jsdelivr.net",
    b"unpkg.com",
    b"fonts.googleapis.com",
    b"fonts.gstatic.com",
]

#: Hosts that appear inside signed binaries as part of a certificate chain, or
#: as documentation references in a comment string. Inert: nothing fetches them
#: at runtime. Reported at most once, never a failure.
INERT_HOST_PATTERNS = [
    re.compile(rb"crl\.[a-z0-9.-]+", re.IGNORECASE),
    re.compile(rb"ocsp\.[a-z0-9.-]+", re.IGNORECASE),
    re.compile(rb"[a-z0-9.-]*\.?microsoft\.com/pki", re.IGNORECASE),
    re.compile(rb"[a-z0-9.-]*\.?digicert\.com", re.IGNORECASE),
    re.compile(rb"[a-z0-9.-]*\.?verisign\.com", re.IGNORECASE),
    re.compile(rb"[a-z0-9.-]*\.?sectigo\.com", re.IGNORECASE),
    re.compile(rb"[a-z0-9.-]*\.?globalsign\.com", re.IGNORECASE),
    re.compile(rb"[a-z0-9.-]*\.?entrust\.net", re.IGNORECASE),
    re.compile(rb"[a-z0-9.-]*\.?symantec\.com", re.IGNORECASE),
    re.compile(rb"[a-z0-9.-]*\.?letsencrypt\.org", re.IGNORECASE),
    # Documentation and source references. A URL in a help string is not a fetch.
    re.compile(rb"github\.com", re.IGNORECASE),
    re.compile(rb"gitlab\.com", re.IGNORECASE),
    re.compile(rb"www\.microsoft\.com/en-us/research", re.IGNORECASE),
    re.compile(rb"opensource\.org", re.IGNORECASE),
    re.compile(rb"www\.apache\.org", re.IGNORECASE),
    re.compile(rb"www\.gnu\.org", re.IGNORECASE),
    re.compile(rb"creativecommons\.org", re.IGNORECASE),
    re.compile(rb"w3\.org", re.IGNORECASE),
    re.compile(rb"xmlpull\.org", re.IGNORECASE),
    re.compile(rb"iana\.org", re.IGNORECASE),
    re.compile(rb"ietf\.org", re.IGNORECASE),
    re.compile(rb"unicode\.org", re.IGNORECASE),
    re.compile(rb"python\.org", re.IGNORECASE),
    re.compile(rb"numpy\.org", re.IGNORECASE),
    re.compile(rb"scipy\.org", re.IGNORECASE),
    re.compile(rb"opencv\.org", re.IGNORECASE),
    re.compile(rb"onnx\.ai", re.IGNORECASE),
    re.compile(rb"khronos\.org", re.IGNORECASE),
    re.compile(rb"nvidia\.com", re.IGNORECASE),
    re.compile(rb"intel\.com", re.IGNORECASE),
    re.compile(rb"qt\.io", re.IGNORECASE),
    re.compile(rb"zlib\.net", re.IGNORECASE),
    re.compile(rb"libpng\.org", re.IGNORECASE),
    re.compile(rb"ffmpeg\.org", re.IGNORECASE),
    re.compile(rb"videolan\.org", re.IGNORECASE),
    re.compile(rb"example\.(com|org|net)", re.IGNORECASE),
    re.compile(rb"localhost", re.IGNORECASE),
]

#: Findings that have been read in context. Each entry is a claim somebody
#: checked, with the evidence and — where the thing is real — what is done about
#: it. This is not a way to make the audit quiet: anything absent fails the
#: build, and an entry that says `mitigation` is admitting the endpoint is
#: genuinely in the shipped bytes.
#:
#: The distinction matters. `inert` means the string is data nothing resolves.
#: `mitigation` means the uploader is real, present, and switched off by
#: configuration — which is a weaker guarantee, and is stated as one.
ACKNOWLEDGED = {
    "huggingface.co": {
        "inert": (
            "onnxruntime: one occurrence, in the help text of a mixture-of-"
            "experts kernel, citing a blog post about the SwiGLU activation. "
            "Read the surrounding bytes: documentation, not a fetch."
        ),
    },
    "events.data.microsoft.com": {
        "what": (
            "onnxruntime's Microsoft 1DS / OneCollector uploader. Present in "
            "the manylinux and macOS wheels — verified by scanning the shipped "
            ".so, which also carries a statically linked mbedTLS stack (127 "
            "symbols), a persistent device-id database (onnxruntime.db) and the "
            "payload fields osDescription, cpuModel and totalMemoryMB. "
            "Microsoft's own privacy documentation states telemetry is ON by "
            "default in the official builds, and PyPI wheels are those builds. "
            "The Windows wheels carry the ETW provider instead, which goes to "
            "the operating system's diagnostics pipeline rather than a socket."
        ),
        "mitigation": (
            "ORT_DISABLE_TELEMETRY=1, set by sentinel.telemetry.silence() at "
            "every entry point *before* the native library initialises, plus "
            "onnxruntime.disable_telemetry_events() for the runtime half. Both "
            "are used because neither is sufficient alone: the variable cannot "
            "reach a library already loaded, and the API cannot un-send an "
            "initialisation event."
        ),
        "residual": (
            "The uploader remains in the binary. Only a source build with "
            "--no_telemetry removes it, which means giving up PyPI wheels for "
            "onnxruntime. The offline CI job is what covers behaviour: the whole "
            "suite runs with outbound traffic dropped and the drop proven first."
        ),
    },
    "onecollector": {
        "what": "The path component of the endpoint above; the same finding.",
        "mitigation": "See events.data.microsoft.com.",
        "residual": "See events.data.microsoft.com.",
    },
}

URL = re.compile(rb"https?://([A-Za-z0-9._~-]{3,120})", re.IGNORECASE)

#: Read in chunks with an overlap, so a needle straddling a boundary is still
#: found. Binaries here are tens of megabytes; reading them whole is fine but
#: this keeps the memory flat for a bundle that is a gigabyte.
CHUNK = 8 * 1024 * 1024
OVERLAP = 256


class Finding:
    __slots__ = ("path", "needle", "kind")

    def __init__(self, path: Path, needle: str, kind: str):
        self.path, self.needle, self.kind = path, needle, kind

    def __str__(self) -> str:
        return f"{self.path}: {self.kind} '{self.needle}'"


def is_binary(path: Path) -> bool:
    return (
        path.suffix.lower() in BINARY_SUFFIXES
        or VERSIONED_SO.search(path.name) is not None
    )


def binaries_under(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*") if path.is_file() and is_binary(path)
    )


def _is_inert(host: bytes) -> bool:
    return any(pattern.search(host) for pattern in INERT_HOST_PATTERNS)


def scan_bytes(data: bytes, path: Path) -> tuple[list[Finding], set[str]]:
    """Known-bad endpoints, and unknown hosts worth a look."""
    lowered = data.lower()
    findings = [
        Finding(path, endpoint.decode(), "telemetry endpoint")
        for endpoint in TELEMETRY_ENDPOINTS
        if endpoint in lowered
    ]

    unknown: set[str] = set()
    for match in URL.finditer(data):
        host = match.group(1)
        if _is_inert(host):
            continue
        try:
            unknown.add(host.decode("ascii").rstrip(".").lower())
        except UnicodeDecodeError:
            continue

    return findings, unknown


def scan(path: Path) -> tuple[list[Finding], set[str]]:
    findings: list[Finding] = []
    unknown: set[str] = set()
    seen: set[str] = set()

    with path.open("rb") as handle:
        tail = b""
        while True:
            block = handle.read(CHUNK)
            if not block:
                break
            found, hosts = scan_bytes(tail + block, path)
            for finding in found:
                if finding.needle not in seen:
                    seen.add(finding.needle)
                    findings.append(finding)
            unknown |= hosts
            tail = block[-OVERLAP:]

    return findings, unknown


def audit(roots: list[Path]) -> tuple[list[Finding], dict[str, set[str]], int]:
    findings: list[Finding] = []
    unknown: dict[str, set[str]] = {}
    scanned = 0

    for root in roots:
        if not root.exists():
            continue
        for path in binaries_under(root):
            scanned += 1
            found, hosts = scan(path)
            findings.extend(found)
            if hosts:
                unknown[str(path.relative_to(root.parent) if root.parent in path.parents
                            else path)] = hosts

    return findings, unknown, scanned


def default_roots() -> list[Path]:
    """What to scan when nobody says.

    The shipped bundle if it exists, because that is the artifact an operator
    actually receives. Otherwise the installed packages this project depends on,
    which is what a developer is running.
    """
    bundle = ROOT / "dist" / "SentinelVision"
    if bundle.is_dir():
        return [bundle]

    roots: list[Path] = []
    for name in ("onnxruntime", "cv2", "numpy"):
        try:
            module = __import__(name)
        except Exception:  # noqa: BLE001
            continue
        location = getattr(module, "__file__", None)
        if location:
            roots.append(Path(location).resolve().parent)
    return roots


def main(argv: list[str] | None = None) -> int:
    arguments = list(argv if argv is not None else sys.argv[1:])
    verbose = "--verbose" in arguments
    arguments = [a for a in arguments if not a.startswith("--")]

    roots = [Path(a).resolve() for a in arguments] or default_roots()
    if not roots:
        print("binary audit: nothing to scan (no bundle, no dependencies found)")
        return 0

    findings, unknown, scanned = audit(roots)

    acknowledged = [f for f in findings if f.needle in ACKNOWLEDGED]
    findings = [f for f in findings if f.needle not in ACKNOWLEDGED]

    managed = sorted({
        f.needle for f in acknowledged if "mitigation" in ACKNOWLEDGED[f.needle]
    })
    inert = sorted({
        f.needle for f in acknowledged if "inert" in ACKNOWLEDGED[f.needle]
    })

    if managed:
        # Loudly, and never folded in with the inert ones. These are real
        # uploaders in the shipped bytes, switched off by configuration — which
        # is a weaker thing than not being there, and reads as one.
        print("PRESENT AND DISARMED — real uploaders, off by configuration:")
        for needle in managed:
            entry = ACKNOWLEDGED[needle]
            print(f"\n  {needle}")
            print(f"    what      {entry['what']}")
            print(f"    disarmed  {entry['mitigation']}")
            print(f"    residual  {entry['residual']}")
        print()

    if inert:
        print("acknowledged — read in context, inert:")
        for needle in inert:
            print(f"  {needle} — {ACKNOWLEDGED[needle]['inert']}")
        print()

    if findings:
        print(f"\nbinary audit: {len(findings)} phone-home endpoint(s) in {scanned} binaries\n")
        for finding in findings:
            print(f"  {finding}")
        print(
            "\nA dependency that phones home does so whether or not this code\n"
            "asked it to. Either configure it off *before it loads* and say so\n"
            "at the point of use — the way ORT_DISABLE_TELEMETRY is set in\n"
            "sentinel/telemetry.py — or do not ship it."
        )
        return 1

    if managed:
        # "No phone-home endpoint" would be false while two are listed above.
        print(
            f"binary audit: {scanned} binaries, nothing new — "
            f"{len(managed)} known endpoint(s) present and disarmed"
        )
    else:
        print(f"binary audit: {scanned} binaries, no known phone-home endpoint")

    if unknown and verbose:
        print("\nhosts found but not recognised as inert — worth a look, not a failure:")
        for name, hosts in sorted(unknown.items()):
            for host in sorted(hosts)[:8]:
                print(f"  {name}: {host}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
