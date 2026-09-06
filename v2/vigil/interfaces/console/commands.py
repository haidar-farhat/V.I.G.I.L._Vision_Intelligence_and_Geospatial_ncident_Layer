"""The one path from the console to the service.

Every mutation the window can make is a method here. Each carries the
principal, so the audit trail names a person; each returns an `Outcome`
rather than raising, so a refusal becomes a sentence on the status bar
instead of a traceback swallowed by a Qt slot — v1's defect.

Nothing in this module knows about widgets, and nothing in the window talks
to `SiteService` or `Runtime` directly. A structural test enforces both.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ...domain.geo import CameraPose, LatLon
from ...domain.zones import Schedule, ZoneKind
from ...service.auth import ANALYSIS_CONTROL, INCIDENT_EXPORT, SITE_CONFIGURE, AuthError, Principal
from ...service.evidence import export_incident
from ...service.runtime import Runtime
from ...service.site import SiteError, SiteService
from ...logs import get as _get_logger

_log = _get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Outcome:
    """What happened, in a sentence a person can read, and the thing if any."""

    ok: bool
    message: str
    value: Any = None

    def __bool__(self) -> bool:
        return self.ok


class Commands:
    def __init__(self, site: SiteService, runtime: Runtime, principal: Principal, *, evidence_dir: Path,
                 model: Path | None = None, model_places: Sequence[Path] = ()):
        self._site = site
        self._runtime = runtime
        self._principal = principal
        self._evidence_dir = Path(evidence_dir)
        #: The model this console will use, and everywhere it was looked for.
        #: The window says both, so a silent fall-back to motion is impossible.
        self.model = model
        self.model_places = tuple(model_places)

    # ------------------------------------------------------------ who, what

    @property
    def principal(self) -> Principal:
        return self._principal

    @property
    def runtime(self) -> Runtime:
        return self._runtime

    def note(self, action: str, detail: str | None = None, subject: str | None = None) -> None:
        """Record something the console itself did, under the principal's name."""
        self._site.store.audit(self._principal.actor, action, subject, detail)

    def detector_info(self, camera_id: str):
        """What is drawing the conclusions on one camera, or ``None``."""
        return self._runtime.detector_info(camera_id)

    def may(self, permission: str) -> bool:
        return self._principal.may(permission)

    def may_configure(self) -> bool:
        return self.may(SITE_CONFIGURE)

    def refusal(self, permission: str) -> str:
        """Why a control is refused, named for the person and the permission."""
        who = self._principal.name
        need = {SITE_CONFIGURE: "an operator or administrator", INCIDENT_EXPORT: "an operator, analyst or administrator",
                ANALYSIS_CONTROL: "an operator or administrator"}.get(permission, "a different")
        return f"{who} may not do that: it needs {need} account."

    # ------------------------------------------------------------- reading

    def cameras(self):
        return self._site.cameras(self._principal)

    def zones(self):
        return self._site.zones(self._principal)

    def health(self):
        return self._runtime.health()

    def incidents(self):
        return self._runtime.incidents or tuple(self._site.store.incidents(limit=200))

    def audit_rows(self, limit: int = 200):
        return self._site.store.audit_trail(limit=limit)

    def alerts(self):
        return self._runtime.alerts

    # ------------------------------------------------------------- writing

    def _guard(self, permission: str, what, *args, **kwargs) -> Outcome:
        """Run a service call, turn its refusals into sentences, and audit nothing extra."""
        try:
            value = what(*args, **kwargs)
        except AuthError:
            self._site.store.audit(self._principal.actor, "console.refused", getattr(what, "__name__", "?"), permission)
            return Outcome(False, self.refusal(permission))
        except (SiteError, ValueError) as error:
            return Outcome(False, str(error))
        except Exception as error:  # noqa: BLE001 - a slot must never die silently
            _log.exception("console: %s failed", getattr(what, "__name__", "?"))
            return Outcome(False, f"{type(error).__name__}: {error}")
        return Outcome(True, "", value)

    def add_camera(self, camera_id: str, source: str, *, name: str | None = None, pose: CameraPose | None = None,
                   record: bool = False) -> Outcome:
        outcome = self._guard(SITE_CONFIGURE, self._site.add_camera, camera_id, source, name=name, pose=pose,
                              record=record, by=self._principal)
        return Outcome(True, f"Added {camera_id}.", outcome.value) if outcome else outcome

    def place_camera(self, camera_id: str, pose: CameraPose) -> Outcome:
        outcome = self._guard(SITE_CONFIGURE, self._site.place_camera, camera_id, pose, by=self._principal)
        return Outcome(True, f"Placed {camera_id}.", outcome.value) if outcome else outcome

    def set_recording(self, camera_id: str, on: bool) -> Outcome:
        outcome = self._guard(SITE_CONFIGURE, self._site.set_recording, camera_id, on, by=self._principal)
        return Outcome(True, f"{camera_id} recording {'on' if on else 'off'}.", outcome.value) if outcome else outcome

    def set_password(self, camera_id: str, password: str) -> Outcome:
        outcome = self._guard(SITE_CONFIGURE, self._site.set_password, camera_id, password, by=self._principal)
        return Outcome(True, f"Stored {camera_id}'s password in the keychain.", outcome.value) if outcome else outcome

    def remove_camera(self, camera_id: str) -> Outcome:
        outcome = self._guard(SITE_CONFIGURE, self._site.remove_camera, camera_id, by=self._principal)
        return Outcome(True, f"Removed {camera_id}.") if outcome else outcome

    def add_zone(self, zone_id: str, name: str, kind: ZoneKind | str, ring: Sequence[LatLon], *,
                 watch: Sequence[str] = (), schedule: Schedule | None = None) -> Outcome:
        outcome = self._guard(SITE_CONFIGURE, self._site.add_zone, zone_id, name, kind, ring, watch=watch,
                              schedule=schedule, by=self._principal)
        return Outcome(True, f"Added zone {name}.", outcome.value) if outcome else outcome

    def remove_zone(self, zone_id: str) -> Outcome:
        outcome = self._guard(SITE_CONFIGURE, self._site.remove_zone, zone_id, by=self._principal)
        return Outcome(True, "Removed the zone.") if outcome else outcome

    # ----------------------------------------------------------- analysis

    def start(self) -> Outcome:
        outcome = self._guard(ANALYSIS_CONTROL, self._runtime.start, self._principal)
        if not outcome:
            return outcome
        started = int(outcome.value or 0)
        if started == 0:
            return Outcome(False, "Nothing to run: add a camera first.")
        return Outcome(True, f"Running {started} camera(s).", started)

    def stop(self) -> Outcome:
        outcome = self._guard(ANALYSIS_CONTROL, self._runtime.stop, self._principal)
        if not outcome:
            return outcome
        return Outcome(True, "Stopped." if outcome.value else "Stopped; one thread would not end and was left running.")

    def poll(self):
        return self._runtime.poll()

    def export(self, incident) -> Outcome:
        outcome = self._guard(INCIDENT_EXPORT, export_incident, self._site.store, incident, self._evidence_dir,
                              by=self._principal)
        if not outcome:
            return outcome
        folder = Path(outcome.value)
        clips = len(list((folder / "clips").glob("*"))) if (folder / "clips").is_dir() else 0
        return Outcome(True, f"Exported to {folder} with {clips} clip(s).", folder)
