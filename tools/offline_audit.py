#!/usr/bin/env python3
"""Static proof that the source contains no way out.

The offline CI job proves the *tests* need no network by taking the network
away. That is necessary and not sufficient: a code path nobody tested can still
call home the first time an operator exercises it, and the suite would never
have noticed. This scans the shipped source instead, so a cloud SDK, an
analytics package or a hard-coded external host fails the build the moment it is
committed rather than the moment a customer runs it.

Three checks, each a separate failure with its own reason:

1. **Cloud and remote-service SDKs.** Importing one is not the same as calling
   it, but this system has no legitimate use for any of them, so an import is
   treated as the finding. There is no configuration that turns one on.
2. **Analytics, telemetry and crash reporting.** §132 forbids hidden telemetry.
   A dependency that phones home does so whether or not this code asked it to.
3. **Hard-coded external hosts.** Anything routable that is not loopback, not
   RFC1918, not link-local and not a documentation-reserved name. A URL in a
   string is a destination somebody can reach, whether or not a socket opens
   today.

What is deliberately *not* scanned: the documentation, the tests and the CI
definition. All three name external hosts on purpose — the offline job proves it
cannot reach `example.com`, and a scanner that forbade writing that down would
be forbidding the proof.

Run it directly, or as ``python tasks.py check``.
"""

from __future__ import annotations

import ipaddress
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Only shipped source. See the module docstring for why tests and docs are not
#: included.
SCANNED = [
    ROOT / "engine" / "sentinel",
    ROOT / "apps" / "console" / "sentinel_console",
    ROOT / "apps" / "console" / "main.py",
    ROOT / "core" / "src",
    ROOT / "tasks.py",
    ROOT / "tools",
    # The packaged build's entry point and spec ship inside the executable and
    # decide what it loads, so they are shipped source like any other.
    ROOT / "packaging",
]

SUFFIXES = {".py", ".rs", ".toml"}

#: A guard has to spell out every name it forbids, so scanning one finds all of
#: them. Excluded by name rather than by a marker comment, because a marker
#: comment is a mechanism any other file could also use to opt out.
#:
#: `binary_audit.py` is here for exactly the same reason: it names the telemetry
#: endpoints it looks for inside compiled dependencies, and naming them is its
#: whole job.
SELF = Path(__file__).resolve()
GUARDS = {SELF, SELF.parent / "binary_audit.py"}

#: Distribution names and import roots. Matched as whole words so `azure` does
#: not fire on `azure_sky_theme`.
CLOUD_SDKS = [
    "boto3", "botocore", "awscli", "aiobotocore",
    "google.cloud", "google_cloud", "googleapiclient", "gcloud",
    "azure.storage", "azure.identity", "azure.core", "msal",
    "firebase_admin", "pyrebase",
    "openai", "anthropic", "cohere", "replicate", "huggingface_hub",
    "dropbox", "paramiko_cloud", "supabase", "pymongo.srv",
    "twilio", "sendgrid", "stripe",
]

ANALYTICS = [
    "sentry_sdk", "posthog", "mixpanel", "amplitude", "segment_analytics",
    "analytics_python", "bugsnag", "rollbar", "datadog", "ddtrace",
    "newrelic", "elasticapm", "opencensus", "applicationinsights",
    "google_analytics", "ga4mp", "statsig", "launchdarkly",
    # OpenTelemetry's OTLP exporters ship a network transport by default. The
    # API and SDK alone do not, so only the exporters are refused.
    "opentelemetry.exporter",
]

#: Hosts a URL may legitimately name in shipped source.
ALLOWED_HOSTS = {
    "localhost", "127.0.0.1", "0.0.0.0", "::1", "[::1]",
    # Reserved by RFC 2606 / RFC 6761 precisely so that documentation and code
    # samples have somewhere safe to point.
    "example.com", "example.org", "example.net", "example.invalid",
}

#: Suffixes that can never resolve off the site.
ALLOWED_SUFFIXES = (".local", ".lan", ".internal", ".invalid", ".localdomain")

#: Schemes worth checking. `file:` and `data:` go nowhere.
URL_PATTERN = re.compile(
    r"\b(?:https?|ftp|ftps|ws|wss|rtsp|rtsps|rtmp|mqtt|mqtts)://([^\s\"'`)<>\,]+)",
    re.IGNORECASE,
)


class Finding(Exception):
    pass


def _is_private_host(host: str) -> bool:
    """Whether a host is unreachable from outside the site."""
    host = host.strip("[]")
    if host.lower() in ALLOWED_HOSTS:
        return True
    if host.lower().endswith(ALLOWED_SUFFIXES):
        return True

    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        # A name, not an address. It is refused rather than resolved: resolving
        # it would need the Internet, which is the thing being guaranteed away.
        # A placeholder is still a placeholder that somebody will fill in.
        return False

    return (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_unspecified
        or address.is_multicast
    )


def _host_of(url: str) -> str:
    netloc = url.split("/", 1)[0]
    # A credential in front of the host is out of scope here; strip it so the
    # host is what gets judged.
    netloc = netloc.rpartition("@")[2] or netloc
    if netloc.startswith("["):
        return netloc.partition("]")[0] + "]"
    return netloc.rsplit(":", 1)[0] if netloc.count(":") == 1 else netloc


def _files() -> list[Path]:
    found: list[Path] = []
    for entry in SCANNED:
        if entry.is_file():
            found.append(entry)
        elif entry.is_dir():
            found.extend(
                path
                for path in sorted(entry.rglob("*"))
                if path.suffix in SUFFIXES and "__pycache__" not in path.parts
            )
    return [path for path in found if path.resolve() not in GUARDS]


def _word_present(text: str, needle: str) -> bool:
    pattern = re.compile(r"(?<![\w.])" + re.escape(needle) + r"(?![\w])")
    return bool(pattern.search(text))


def scan_text(name: str, text: str) -> list[str]:
    """The whole check, over one file's content.

    Separated from the walk so the checker can be tested against source that is
    deliberately bad. A guard nobody has watched fail is a guard nobody knows
    works.
    """
    problems: list[str] = []

    for number, line in enumerate(text.splitlines(), start=1):
        for sdk in CLOUD_SDKS:
            if _word_present(line, sdk):
                problems.append(
                    f"{name}:{number}: cloud SDK '{sdk}'. "
                    "§132: no cloud dependency, in any form."
                )
        for package in ANALYTICS:
            if _word_present(line, package):
                problems.append(
                    f"{name}:{number}: telemetry package '{package}'. "
                    "§132: no hidden telemetry."
                )
        for match in URL_PATTERN.finditer(line):
            host = _host_of(match.group(1))
            if not _is_private_host(host):
                problems.append(
                    f"{name}:{number}: external host '{host}'. "
                    "Nothing in shipped source may name a destination "
                    "outside the site."
                )

    return problems


def audit() -> list[str]:
    scanned = _files()
    if not scanned:
        return ["nothing was scanned; the paths in SCANNED are wrong"]

    problems: list[str] = []
    for path in scanned:
        problems.extend(
            scan_text(
                path.relative_to(ROOT).as_posix(),
                path.read_text(encoding="utf-8", errors="replace"),
            )
        )
    return problems


def main() -> int:
    problems = audit()
    count = len(_files())

    if problems:
        print(f"\noffline audit: {len(problems)} problem(s) in {count} files\n")
        for problem in problems:
            print(f"  {problem}")
        print(
            "\nIf one of these is genuinely offline — a scheme this does not "
            "understand, say — widen the allow list in tools/offline_audit.py "
            "with the reason, rather than deleting the check."
        )
        return 1

    print(f"offline audit: {count} files, no route out")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
