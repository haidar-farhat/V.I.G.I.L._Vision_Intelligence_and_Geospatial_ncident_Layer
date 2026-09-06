"""Something leaves the process when a camera goes dark."""

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

from ..logs import get as _get_logger

_log = _get_logger(__name__)

CAMERA_DARK = "camera.dark"
#: A camera that is producing frames nobody could detect anything in:
#: out of focus, blown out, black, or repeating the same frame. v1 and
#: v2 both had "dark" — no frames at all — and nothing between that and
#: "working", so every one of those failed silently while the frame
#: counter climbed and the fps looked healthy.
CAMERA_DEGRADED = "camera.degraded"
RECORDING_STOPPED = "recording.stopped"
RETENTION_SHORTFALL = "retention.shortfall"
THREAD_STUCK = "analysis.thread_stuck"
DISK_LOW = "disk.low"
KINDS = frozenset({CAMERA_DARK, CAMERA_DEGRADED, RECORDING_STOPPED, RETENTION_SHORTFALL,
                  THREAD_STUCK, DISK_LOW})
SINK_TIMEOUT_SECONDS = 10.0


@dataclass(frozen=True, slots=True)
class Alert:
    kind: str
    subject: str
    detail: str
    raised_at: float

    @property
    def key(self) -> tuple[str, str]:
        return (self.kind, self.subject)

    def line(self, *, cleared: bool = False) -> str:
        stamp = datetime.fromtimestamp(self.raised_at, timezone.utc).isoformat(timespec="seconds")
        return f"{stamp} {'CLEARED' if cleared else 'RAISED'} {self.kind} {self.subject} - {self.detail}"

    def as_dict(self, *, cleared: bool = False) -> dict:
        return {"kind": self.kind, "subject": self.subject, "detail": self.detail, "raised_at": self.raised_at,
                "state": "cleared" if cleared else "raised"}


class Sink(Protocol):
    def deliver(self, alert: Alert, *, cleared: bool = False) -> None: ...


class FileSink:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def deliver(self, alert: Alert, *, cleared: bool = False) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(alert.line(cleared=cleared) + "\n")


class CommandSink:
    def __init__(self, command: str | Iterable[str]):
        self.argv = shlex.split(command, posix=os.name != "nt") if isinstance(command, str) else list(command)
        if not self.argv:
            raise ValueError("an alert command cannot be empty")

    def deliver(self, alert: Alert, *, cleared: bool = False) -> None:
        state = "cleared" if cleared else "raised"
        env = dict(os.environ, VIGIL_ALERT_KIND=alert.kind, VIGIL_ALERT_SUBJECT=alert.subject,
                   VIGIL_ALERT_DETAIL=alert.detail, VIGIL_ALERT_STATE=state)
        child = subprocess.Popen([*self.argv, alert.kind, alert.subject, alert.detail, state], env=env,
                                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            child.wait(timeout=SINK_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            _log.warning("alert command still running after %.0f s; left to itself", SINK_TIMEOUT_SECONDS)


class WebhookSink:
    def __init__(self, url: str, *, allow_public: bool = False):
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError("the alert webhook must be a web address with a host")
        self.url, self.host = url, parts.hostname
        if not allow_public:
            _require_private(self.host)

    def deliver(self, alert: Alert, *, cleared: bool = False) -> None:
        from urllib.request import Request, urlopen

        request = Request(self.url, data=json.dumps(alert.as_dict(cleared=cleared)).encode("utf-8"), method="POST")
        request.add_header("Content-Type", "application/json")
        with urlopen(request, timeout=SINK_TIMEOUT_SECONDS) as response:  # noqa: S310
            response.read(0)


def _require_private(host: str) -> None:
    try:
        addresses = {info[4][0] for info in socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)}
    except OSError as error:
        raise ValueError(f"the alert webhook host {host!r} cannot be resolved: {error}") from error
    public = [a for a in addresses if _is_public(a)]
    if public:
        raise ValueError(f"the alert webhook host {host!r} resolves to {', '.join(sorted(public))}, outside the local network")


def _is_public(address: str) -> bool:
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError:
        return False
    return not (parsed.is_private or parsed.is_loopback or parsed.is_link_local)


class Alerts:
    """Raised once per (kind, subject) until cleared; audited; fanned out off-thread."""

    def __init__(self, sinks: Iterable[Sink] = (), *, store=None, principal: str = "system",
                 clock: Callable[[], float] = time.time, synchronous: bool = False):
        self._sinks = list(sinks)
        self._store = store
        self._principal = principal
        self._clock = clock
        self._synchronous = synchronous
        self._active: dict[tuple[str, str], Alert] = {}
        self._unseen: list[Alert] = []
        self._lock = threading.Lock()
        self.raised_count = 0

    @property
    def sinks(self) -> tuple[Sink, ...]:
        return tuple(self._sinks)

    def bind(self, store, principal: str) -> None:
        if self._store is None:
            self._store, self._principal = store, principal

    def active(self) -> tuple[Alert, ...]:
        with self._lock:
            return tuple(sorted(self._active.values(), key=lambda a: a.raised_at))

    def take_new(self) -> tuple[Alert, ...]:
        with self._lock:
            fresh, self._unseen = tuple(self._unseen), []
        return fresh

    def raise_(self, kind: str, subject: str, detail: str) -> bool:
        with self._lock:
            if (kind, subject) in self._active:
                return False
            alert = Alert(kind, subject, detail, self._clock())
            self._active[alert.key] = alert
            self._unseen.append(alert)
            self.raised_count += 1
        _log.error("ALERT %s %s - %s", kind, subject, detail)
        self._persist("alert.raised", alert, cleared=False)
        self._fan_out(alert, False)
        return True

    def clear(self, kind: str, subject: str) -> bool:
        with self._lock:
            alert = self._active.pop((kind, subject), None)
        if alert is None:
            return False
        _log.warning("alert cleared: %s %s", kind, subject)
        self._persist("alert.cleared", alert, cleared=True)
        self._fan_out(alert, True)
        return True

    def _persist(self, action: str, alert: Alert, *, cleared: bool) -> None:
        if self._store is None:
            return
        try:
            self._store.audit(self._principal, action, alert.subject, f"{alert.kind}: {alert.detail}")
            now = int(self._clock() * 1000)
            if cleared:
                self._store.close_alert(alert.kind, alert.subject, now)
            else:
                self._store.open_alert(alert.kind, alert.subject, alert.detail, now)
        except Exception:  # noqa: BLE001
            _log.exception("could not persist %s", action)

    def _fan_out(self, alert: Alert, cleared: bool) -> None:
        if not self._sinks:
            return
        if self._synchronous:
            self._deliver(alert, cleared)
        else:
            threading.Thread(target=self._deliver, args=(alert, cleared), name="vigil-alert", daemon=True).start()

    def _deliver(self, alert: Alert, cleared: bool) -> None:
        for sink in self._sinks:
            try:
                sink.deliver(alert, cleared=cleared)
            except Exception:  # noqa: BLE001
                _log.exception("alert sink %s failed", type(sink).__name__)

    @classmethod
    def from_settings(cls, settings, *, store=None, principal: str = "system") -> "Alerts":
        sinks: list[Sink] = []
        if settings.alert_file is None:
            sinks.append(FileSink(settings.logs / "alerts.log"))
        elif settings.alert_file != "":
            sinks.append(FileSink(settings.alert_file))
        if settings.alert_command:
            try:
                sinks.append(CommandSink(settings.alert_command))
            except ValueError as error:
                _log.error("VIGIL_ALERT_COMMAND ignored: %s", error)
        if settings.alert_webhook:
            try:
                sinks.append(WebhookSink(settings.alert_webhook, allow_public=settings.allow_public_sources))
            except ValueError as error:
                _log.error("VIGIL_ALERT_WEBHOOK ignored: %s", error)
        return cls(sinks, store=store, principal=principal)

    def describe(self) -> str:
        names = [type(s).__name__ for s in self._sinks]
        return "alerts go to " + (", ".join(names) if names else "the log only")
