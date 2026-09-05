"""Something leaves the process when a camera goes dark.

Until this module existed a dark camera, a recording that stopped early, a
retention sweep that could not reach its target and a stuck analysis thread
all reached the log and, sometimes, a label — and nobody is watching the log
(OBS-01). An alert is the same fact made *outward*: written to a file a
monitoring agent can tail, handed to a command the operator chose, posted to
an endpoint on the local network, and shown and sounded by the console.

What an alert is here:

- **Raised once.** The same condition on the same subject is one alert
  until it clears; a camera that has been dark for an hour does not produce
  three thousand six hundred alerts. Clearing is an event too, and audited.
- **Audited.** `alert.raised` and `alert.cleared` rows carry the kind, the
  subject and the detail, so the evidence trail says when the system knew.
- **Never fatal.** A sink that fails is logged and the others still run. An
  alerting path that can take the analysis down is worse than none.
- **Local.** The webhook sink refuses an address outside the local network
  for the same reason `decode` refuses a camera there; the product does not
  reach the Internet, and an alert is not an exception.

Configuration is by environment variable, read once by `Alerts.from_environment`,
because the console, the node and a service all share it without an argument
list to remember: `SENTINEL_ALERT_FILE` (default `alerts.log` beside the
log; empty disables), `SENTINEL_ALERT_COMMAND` (run per alert with the
alert in its environment) and `SENTINEL_ALERT_WEBHOOK` (a local-network
address, POSTed a JSON body).
"""

from __future__ import annotations

import ipaddress
import json
import os
import shlex
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Protocol
from urllib.parse import urlsplit

from .logs import get as _get_logger

_log = _get_logger(__name__)

FILE_VARIABLE = "SENTINEL_ALERT_FILE"
COMMAND_VARIABLE = "SENTINEL_ALERT_COMMAND"
WEBHOOK_VARIABLE = "SENTINEL_ALERT_WEBHOOK"

#: The conditions the node raises. Named here so a sink, a test and the
#: documentation agree on the words.
CAMERA_DARK = "camera.dark"
RECORDING_STOPPED = "recording.stopped"
RETENTION_SHORTFALL = "retention.shortfall"
THREAD_STUCK = "analysis.thread_stuck"
DISK_LOW = "disk.low"
KINDS = frozenset({CAMERA_DARK, RECORDING_STOPPED, RETENTION_SHORTFALL, THREAD_STUCK, DISK_LOW})

#: How long a sink may take before it is abandoned. A sink runs on its own
#: thread; the analysis loop never waits on one.
SINK_TIMEOUT_SECONDS = 10.0


@dataclass(frozen=True, slots=True)
class Alert:
    kind: str
    subject: str
    detail: str
    #: Wall-clock seconds; what a person reading the file wants.
    raised_at: float

    @property
    def key(self) -> tuple[str, str]:
        return (self.kind, self.subject)

    def line(self, *, cleared: bool = False) -> str:
        stamp = datetime.fromtimestamp(self.raised_at, timezone.utc).isoformat(timespec="seconds")
        state = "CLEARED" if cleared else "RAISED"
        return f"{stamp} {state} {self.kind} {self.subject} — {self.detail}"

    def as_dict(self, *, cleared: bool = False) -> dict:
        return {
            "kind": self.kind, "subject": self.subject, "detail": self.detail,
            "raised_at": self.raised_at, "state": "cleared" if cleared else "raised",
        }


class Sink(Protocol):
    def deliver(self, alert: Alert, *, cleared: bool = False) -> None: ...


class FileSink:
    """One line per alert, appended. The simplest thing a monitor can tail."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def deliver(self, alert: Alert, *, cleared: bool = False) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(alert.line(cleared=cleared) + "\n")


class CommandSink:
    """Run a command the operator chose, with the alert in its environment.

    ``SENTINEL_ALERT_KIND``, ``_SUBJECT``, ``_DETAIL`` and ``_STATE`` are set
    for the child, and the same four are appended as arguments, so a shell
    script, a pager tool or a siren driver can be pointed at without a
    wrapper. The command is not waited on beyond `SINK_TIMEOUT_SECONDS`.
    """

    def __init__(self, command: str | Iterable[str]):
        self.argv = shlex.split(command, posix=os.name != "nt") if isinstance(command, str) else list(command)
        if not self.argv:
            raise ValueError("an alert command cannot be empty")

    def deliver(self, alert: Alert, *, cleared: bool = False) -> None:
        state = "cleared" if cleared else "raised"
        environment = dict(os.environ)
        environment.update({
            "SENTINEL_ALERT_KIND": alert.kind, "SENTINEL_ALERT_SUBJECT": alert.subject,
            "SENTINEL_ALERT_DETAIL": alert.detail, "SENTINEL_ALERT_STATE": state,
        })
        child = subprocess.Popen(
            [*self.argv, alert.kind, alert.subject, alert.detail, state],
            env=environment, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            child.wait(timeout=SINK_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            _log.warning("alert command still running after %.0f s; left to itself", SINK_TIMEOUT_SECONDS)


class WebhookSink:
    """POST the alert as JSON to an address on the local network.

    The host is resolved and every address checked to be private, loopback
    or link-local before anything is sent — the same rule the decoder applies
    to a camera, for the same reason. A public address is refused at
    construction, so a misconfiguration fails at start and not at the first
    alert, which is the moment nobody can afford it to.
    """

    def __init__(self, url: str, *, allow_public: bool = False):
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError("the alert webhook must be a web address with a host")
        self.url = url
        self.host = parts.hostname
        if not allow_public:
            _require_private(self.host)

    def deliver(self, alert: Alert, *, cleared: bool = False) -> None:
        from urllib.request import Request, urlopen

        body = json.dumps(alert.as_dict(cleared=cleared)).encode("utf-8")
        request = Request(self.url, data=body, method="POST")
        request.add_header("Content-Type", "application/json")
        with urlopen(request, timeout=SINK_TIMEOUT_SECONDS) as response:  # noqa: S310 - private host only
            response.read(0)


def _require_private(host: str) -> None:
    try:
        addresses = {info[4][0] for info in socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)}
    except OSError as error:
        raise ValueError(f"the alert webhook host {host!r} cannot be resolved: {error}") from error
    public = []
    for address in addresses:
        try:
            parsed = ipaddress.ip_address(address)
        except ValueError:
            continue
        if not (parsed.is_private or parsed.is_loopback or parsed.is_link_local):
            public.append(address)
    if public:
        raise ValueError(
            f"the alert webhook host {host!r} resolves to {', '.join(sorted(public))}, "
            "which is outside the local network; this system does not reach the Internet"
        )


class Alerts:
    """The conditions that are currently true, and the sinks told about them.

    Owned by a node. ``store`` is where the audit rows go and may be ``None``
    for a hub used outside one; ``clock`` is the wall clock and is a
    parameter so a test can pin it.
    """

    def __init__(
        self,
        sinks: Iterable[Sink] = (),
        *,
        store=None,
        actor: str = "node",
        clock: Callable[[], float] = time.time,
        synchronous: bool = False,
    ):
        self._sinks = list(sinks)
        self._store = store
        self._actor = actor
        self._clock = clock
        #: Sinks run on their own thread unless a test asks otherwise.
        self._synchronous = synchronous
        self._active: dict[tuple[str, str], Alert] = {}
        self._unseen: list[Alert] = []
        self._raised_count = 0
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- facts

    def active(self) -> tuple[Alert, ...]:
        with self._lock:
            return tuple(sorted(self._active.values(), key=lambda alert: alert.raised_at))

    def take_new(self) -> tuple[Alert, ...]:
        """Alerts raised since the last call. For a console that sounds once each."""
        with self._lock:
            fresh, self._unseen = tuple(self._unseen), []
        return fresh

    @property
    def raised_count(self) -> int:
        return self._raised_count

    @property
    def sinks(self) -> tuple[Sink, ...]:
        return tuple(self._sinks)

    def bind(self, store, actor: str) -> None:
        """Give a hub built without a store the node's, so its rows are audited."""
        if self._store is None:
            self._store = store
            self._actor = actor

    # -------------------------------------------------------------- raising

    def raise_(self, kind: str, subject: str, detail: str) -> bool:
        """Say that ``kind`` is true of ``subject``. True when newly raised."""
        with self._lock:
            if (kind, subject) in self._active:
                return False
            alert = Alert(kind, subject, detail, self._clock())
            self._active[alert.key] = alert
            self._unseen.append(alert)
            self._raised_count += 1
        _log.error("ALERT %s %s — %s", kind, subject, detail)
        self._audit("alert.raised", alert)
        self._fan_out(alert, cleared=False)
        return True

    def clear(self, kind: str, subject: str) -> bool:
        """Say that ``kind`` is no longer true of ``subject``. True when it was."""
        with self._lock:
            alert = self._active.pop((kind, subject), None)
        if alert is None:
            return False
        _log.warning("alert cleared: %s %s", kind, subject)
        self._audit("alert.cleared", alert)
        self._fan_out(alert, cleared=True)
        return True

    def _audit(self, action: str, alert: Alert) -> None:
        if self._store is None:
            return
        try:
            self._store.audit(self._actor, action, alert.subject, f"{alert.kind}: {alert.detail}")
        except Exception:  # noqa: BLE001 - the trail must not take the alert down
            _log.exception("could not audit %s", action)

    def _fan_out(self, alert: Alert, *, cleared: bool) -> None:
        if not self._sinks:
            return
        if self._synchronous:
            self._deliver(alert, cleared)
            return
        worker = threading.Thread(
            target=self._deliver, args=(alert, cleared), name="sentinel-alert", daemon=True,
        )
        worker.start()

    def _deliver(self, alert: Alert, cleared: bool) -> None:
        for sink in self._sinks:
            try:
                sink.deliver(alert, cleared=cleared)
            except Exception:  # noqa: BLE001 - one sink failing must not silence the rest
                _log.exception("alert sink %s failed", type(sink).__name__)

    # --------------------------------------------------------- construction

    @classmethod
    def from_environment(cls, *, store=None, actor: str = "node", environ=None) -> "Alerts":
        """The sinks the environment asks for. A bad value is logged, not fatal."""
        environ = os.environ if environ is None else environ
        sinks: list[Sink] = []

        file = environ.get(FILE_VARIABLE)
        if file is None:
            from .paths import log_directory

            sinks.append(FileSink(log_directory() / "alerts.log"))
        elif file != "":
            sinks.append(FileSink(file))

        command = environ.get(COMMAND_VARIABLE)
        if command:
            try:
                sinks.append(CommandSink(command))
            except ValueError as error:
                _log.error("%s ignored: %s", COMMAND_VARIABLE, error)

        webhook = environ.get(WEBHOOK_VARIABLE)
        if webhook:
            from .decode import _ALLOW_PUBLIC_SOURCES

            try:
                sinks.append(WebhookSink(webhook, allow_public=_ALLOW_PUBLIC_SOURCES))
            except ValueError as error:
                _log.error("%s ignored: %s", WEBHOOK_VARIABLE, error)

        return cls(sinks, store=store, actor=actor)

    def describe(self) -> str:
        names = [type(sink).__name__ for sink in self._sinks]
        return "alerts go to " + (", ".join(names) if names else "the log only")
