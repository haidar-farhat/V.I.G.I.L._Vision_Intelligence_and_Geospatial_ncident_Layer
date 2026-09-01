"""Tests for the static zero-WAN guard.

The offline CI job proves the tests need no network by taking the network away.
That is a real proof and it is not enough: it only covers the code the tests
happen to run, and the first time an operator exercises an untested path is not
when a phone-home should be discovered.

`tools/offline_audit.py` scans the shipped source instead. These tests exist
because a guard nobody has watched fail is a guard nobody knows works — each one
feeds it source that is deliberately bad and requires it to say so.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import offline_audit  # noqa: E402


# ------------------------------------------------------ the guard on real source


def test_the_shipped_source_has_no_route_off_the_site():
    # The assertion this whole file exists to make. If it ever fails, the
    # message names the file and line, and the answer is to remove the
    # dependency — not to widen the allow list.
    assert offline_audit.audit() == []


def test_the_scan_actually_covers_the_engine_and_the_console():
    # A scanner pointed at nothing passes everything. These are the two paths
    # most likely to acquire a dependency, so their presence is asserted rather
    # than assumed.
    scanned = {path.relative_to(ROOT).as_posix() for path in offline_audit._files()}

    assert "engine/sentinel/decode.py" in scanned
    assert "engine/sentinel/detect.py" in scanned
    assert "apps/console/sentinel_console/app.py" in scanned
    assert "core/src/ffi.rs" in scanned


def test_the_scanner_does_not_scan_itself():
    # It has to spell out every name it forbids, so scanning it finds all of
    # them. Excluded by path, not by an opt-out marker any other file could use.
    scanned = {path.resolve() for path in offline_audit._files()}

    assert offline_audit.SELF not in scanned


# ------------------------------------------------------------- cloud SDKs


@pytest.mark.parametrize(
    "line",
    [
        "import boto3",
        "from google.cloud import storage",
        "import openai",
        "from azure.storage.blob import BlobServiceClient",
        "huggingface_hub = '*'",
    ],
)
def test_a_cloud_sdk_is_refused(line: str):
    problems = offline_audit.scan_text("fake.py", line)

    assert problems, f"a cloud SDK slipped through: {line}"
    assert "cloud SDK" in problems[0]


# ------------------------------------------------------------- telemetry


@pytest.mark.parametrize(
    "line",
    [
        "import sentry_sdk",
        "from posthog import Posthog",
        "import ddtrace",
        "from opentelemetry.exporter.otlp import Exporter",
    ],
)
def test_a_telemetry_package_is_refused(line: str):
    problems = offline_audit.scan_text("fake.py", line)

    assert problems, f"a telemetry package slipped through: {line}"
    assert "telemetry" in problems[0]


# ------------------------------------------------------------ external hosts


@pytest.mark.parametrize(
    "line",
    [
        'URL = "https://api.some-vendor.com/v1/detect"',
        'MODEL = "http://models.example-cdn.net/yolo.onnx"',
        "# see https://github.com/some/repo for the algorithm",
        'STREAM = "rtsp://camera.remote-site.io:554/live"',
        'BROKER = "mqtts://broker.hivemq.com:8883"',
        'FEED = "wss://8.8.8.8/events"',
    ],
)
def test_an_external_host_is_refused(line: str):
    problems = offline_audit.scan_text("fake.py", line)

    assert problems, f"an external host slipped through: {line}"
    assert "external host" in problems[0]


@pytest.mark.parametrize(
    "line",
    [
        'CAMERA = "rtsp://admin:secret@192.168.1.64:554/Streaming/Channels/101"',
        'CAMERA = "rtsp://10.20.30.40/stream"',
        'CAMERA = "rtsp://172.16.4.4/stream"',
        'NODE = "https://sentinel-node-2.local:8443/api"',
        'LOCAL = "http://127.0.0.1:9000/health"',
        'LOOPBACK = "http://[::1]:9000/health"',
        'V6_PRIVATE = "https://[fd00::1]:8443/api"',
        'LINK_LOCAL = "http://169.254.1.1/"',
    ],
)
def test_a_host_inside_the_site_is_allowed(line: str):
    # These are the URLs the product legitimately contains. A guard that refused
    # them would be a guard somebody switches off.
    assert offline_audit.scan_text("fake.py", line) == []


def test_a_credential_in_front_of_a_host_does_not_hide_it():
    # `user:pass@host` — the host is what gets judged, so a credential must not
    # be mistaken for one.
    problems = offline_audit.scan_text(
        "fake.py", 'URL = "https://admin:hunter2@analytics.vendor.com/collect"'
    )

    assert problems
    assert "analytics.vendor.com" in problems[0]


def test_a_bare_name_is_refused_rather_than_resolved():
    # Resolving it to find out whether it is local would need the Internet,
    # which is the thing being guaranteed away.
    problems = offline_audit.scan_text("fake.py", 'HOST = "https://updates/latest"')

    assert problems, "an unresolvable name was treated as safe"


def test_a_substring_is_not_a_dependency():
    # `azure_sky` is a theme colour, not Azure. A guard that cries wolf is a
    # guard that gets suppressed.
    clean = [
        "AZURE_SKY = '#4a90d9'",
        "def strip_ecosystem(name): ...",
        "stripe_width = 4  # map hatching",
        "self._datadog_style = False",
    ]

    for line in clean:
        assert offline_audit.scan_text("fake.py", line) == [], line
