"""Tests for disarming dependency telemetry, and for the guard that finds it.

These exist because of a live defect in the shipped product, found by scanning
a wheel rather than by reading documentation: onnxruntime's manylinux and macOS
builds contain a Microsoft 1DS uploader — a collector endpoint with an ingestion
token, a statically linked mbedTLS stack, a persistent device-id database, and a
payload carrying `osDescription`, `cpuModel` and `totalMemoryMB`. Telemetry is
on by default in the official builds, and PyPI wheels are the official builds.

`tools/offline_audit.py` could never have found it. Its own docstring says *"a
dependency that phones home does so whether or not this code asked it to"*, and
it reads `.py`, `.rs` and `.toml` — so a hostname inside a 28 MB `.so` was
invisible to it for as long as it existed.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from sentinel import telemetry

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import binary_audit  # noqa: E402


# ------------------------------------------------------------------ disarming


def test_the_variable_is_set_before_onnxruntime_could_read_it():
    # The native library reads ORT_DISABLE_TELEMETRY when it initialises, which
    # is earlier than any Python call can reach. Microsoft's documentation is
    # explicit that an initialisation event may already have been emitted before
    # `disable_telemetry_events()` becomes reachable, so the variable is the
    # control that matters and the API call is the second half.
    code = (
        "import os, sys; sys.path.insert(0, r'%s');"
        "from sentinel import telemetry; telemetry.silence();"
        "print(os.environ.get('ORT_DISABLE_TELEMETRY'))"
    ) % (ROOT / "engine")
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )

    assert result.stdout.strip() == "1", result.stderr


def test_the_cli_disarms_before_it_does_anything_else():
    # Not in a library function somewhere — at the entry point, before the
    # imports that matter. A worker node started from the CLI must never have a
    # window where the uploader is live.
    code = (
        "import os, sys; sys.path.insert(0, r'%s');"
        "from sentinel.cli import main; main(['where']);"
        "print('ORT=' + str(os.environ.get('ORT_DISABLE_TELEMETRY')))"
    ) % (ROOT / "engine")
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )

    assert "ORT=1" in result.stdout, result.stderr


def test_an_operator_s_own_setting_is_not_overwritten(monkeypatch):
    # Quietly reversing somebody's deliberate choice is exactly the kind of
    # thing this module exists to prevent.
    monkeypatch.setenv(telemetry.ORT_DISABLE, "0")
    monkeypatch.delenv("SENTINEL_TELEMETRY_SILENCED", raising=False)

    telemetry.silence()

    assert os.environ[telemetry.ORT_DISABLE] == "0"


def test_silencing_twice_is_harmless(monkeypatch):
    # The console, the CLI and the node each want to call it and none of them
    # knows about the others.
    monkeypatch.delenv("SENTINEL_TELEMETRY_SILENCED", raising=False)
    monkeypatch.delenv(telemetry.ORT_DISABLE, raising=False)

    telemetry.silence()
    telemetry.silence()

    assert os.environ[telemetry.ORT_DISABLE] == "1"


def test_the_runtime_half_survives_a_build_without_the_api():
    # A build where the call is absent, or present and useless, is not a reason
    # to fail to start.
    telemetry.silence_runtime_apis()


# ------------------------------------------------------- the guard that sees it


def test_a_versioned_shared_object_is_scanned(tmp_path: Path):
    # `libfoo.so.1.29.0` has suffix `.0`, and that is the *normal* name for the
    # real library on Linux. The first version of this audit read the Python
    # extension module and walked straight past `libonnxruntime.so.1.29.0` — the
    # 28 MB file that actually contains the uploader.
    (tmp_path / "libfoo.so.1.29.0").write_bytes(b"\x00" * 16)
    (tmp_path / "libbar.so").write_bytes(b"\x00" * 16)
    (tmp_path / "notes.txt").write_bytes(b"not a binary")

    found = {path.name for path in binary_audit.binaries_under(tmp_path)}

    assert found == {"libfoo.so.1.29.0", "libbar.so"}


def test_a_collector_endpoint_inside_a_binary_is_found(tmp_path: Path):
    blob = tmp_path / "libthing.so.2.0.0"
    blob.write_bytes(
        b"\x00" * 4096
        + b"https://mobile.events.data.microsoft.com/OneCollector/1.0/"
        + b"\x00" * 4096
    )

    findings, _ = binary_audit.scan(blob)

    assert {f.needle for f in findings} >= {"events.data.microsoft.com", "onecollector"}


def test_an_endpoint_straddling_a_read_boundary_is_still_found(tmp_path: Path):
    # Binaries are read in chunks. A needle landing across the seam would be
    # invisible without the overlap, and it would be invisible *silently*.
    blob = tmp_path / "libseam.so"
    filler = binary_audit.CHUNK - 10
    blob.write_bytes(b"\x00" * filler + b"https://sentry.io/api/1/store/" + b"\x00" * 32)

    findings, _ = binary_audit.scan(blob)

    assert any(f.needle == "sentry.io" for f in findings)


@pytest.mark.parametrize(
    "host",
    [b"https://crl.microsoft.com/pki/crl/products/x.crl",
     b"http://www.microsoft.com/pkiops/certs/y.crt",
     b"https://github.com/some/repo",
     b"https://www.apache.org/licenses/LICENSE-2.0"],
)
def test_certificate_and_documentation_urls_are_not_findings(tmp_path: Path, host: bytes):
    # Every signed Windows binary embeds its certificate chain, and that chain
    # carries CRL and issuer URLs. A guard that failed on those would fail on
    # everything and be switched off within a day.
    blob = tmp_path / "libsigned.so"
    blob.write_bytes(b"\x00" * 256 + host + b"\x00" * 256)

    findings, unknown = binary_audit.scan(blob)

    # No *finding* is the contract. An unrecognised host lands in `unknown`,
    # which is reported for a person to read and never fails a build — because
    # the URL pattern captures a host, and some hosts are only inert in the
    # context of the path they appear with. `www.microsoft.com/pkiops/…` is a
    # certificate reference; `www.microsoft.com` alone is not obviously
    # anything. Failing on that ambiguity would make the guard unusable.
    assert findings == []


def test_a_certificate_revocation_host_is_recognised_outright(tmp_path: Path):
    # The commonest false positive by a wide margin: every signed binary on the
    # machine carries one, so this has to be recognised rather than tolerated.
    blob = tmp_path / "libsigned.so"
    blob.write_bytes(
        b"\x00" * 64 + b"http://crl.microsoft.com/pki/crl/x.crl" + b"\x00" * 64
    )

    findings, unknown = binary_audit.scan(blob)

    assert findings == []
    assert unknown == set()


def test_the_known_uploader_is_acknowledged_with_its_mitigation():
    # Acknowledging it is not the same as calling it clean. The entry has to say
    # what it is, what disarms it, and what is left over — because "present and
    # switched off" is a weaker guarantee than "not there", and the difference
    # is what an operator is entitled to know.
    entry = binary_audit.ACKNOWLEDGED["events.data.microsoft.com"]

    assert "mitigation" in entry and "residual" in entry
    assert telemetry.ORT_DISABLE in entry["mitigation"]
    assert "remains in the binary" in entry["residual"]


def test_an_unknown_collector_still_fails(tmp_path: Path):
    # The acknowledged list must not become a way to make the audit quiet.
    blob = tmp_path / "libnew.so"
    blob.write_bytes(b"\x00" * 64 + b"https://api.amplitude.com/2/httpapi" + b"\x00" * 64)

    assert binary_audit.main([str(tmp_path)]) == 1


def test_the_shipped_dependencies_hold(capsys):
    # The assertion this whole file exists to make, over whatever is actually
    # installed. Anything new fails; the known one is reported as disarmed.
    assert binary_audit.main([]) == 0
