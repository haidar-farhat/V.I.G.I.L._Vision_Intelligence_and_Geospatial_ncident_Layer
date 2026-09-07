"""Settings, read once from the environment. Nothing else reads `os.environ`."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

DATA_VARIABLE = "VIGIL_DATA_DIR"


def _default_data_directory() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "data"
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "SentinelVision" / "v2"
    return Path.home() / ".local" / "share" / "sentinel-vision" / "v2"


@dataclass(frozen=True, slots=True)
class Settings:
    data_dir: Path
    alert_file: str | None
    alert_command: str | None
    alert_webhook: str | None
    allow_public_sources: bool

    @classmethod
    def from_environment(cls, environ=None) -> "Settings":
        environ = os.environ if environ is None else environ
        data = Path(environ[DATA_VARIABLE]) if environ.get(DATA_VARIABLE) else _default_data_directory()
        return cls(
            data_dir=data,
            alert_file=environ.get("VIGIL_ALERT_FILE"),
            alert_command=environ.get("VIGIL_ALERT_COMMAND"),
            alert_webhook=environ.get("VIGIL_ALERT_WEBHOOK"),
            allow_public_sources=environ.get("VIGIL_ALLOW_PUBLIC_SOURCES", "").strip().lower() in ("1", "true", "yes"),
        )

    @property
    def database(self) -> Path:
        return self.data_dir / "vigil.db"

    @property
    def logs(self) -> Path:
        return self.data_dir / "logs"

    @property
    def recordings(self) -> Path:
        return self.data_dir / "recordings"

    @property
    def evidence(self) -> Path:
        return self.data_dir / "evidence"

    @property
    def models(self) -> Path:
        """Where a model *should* be put. See `model_directories` for where one is looked for."""
        return self.model_directories()[0]

    def model_directories(self) -> list[Path]:
        """Every place a model may be, in order.

        More than one on purpose: a packaged build ships them beside the
        executable, a checkout keeps them in the repository, and a deployment
        may put them with its data. Looking in only one of those is how a run
        silently falls back to motion detection — which happened, and the
        photograph of the console is what caught it.
        """
        places = []
        if getattr(sys, "frozen", False):
            places.append(Path(sys.executable).resolve().parent / "models")
        places.append(self.data_dir / "models")
        if not getattr(sys, "frozen", False):
            places.append(Path(__file__).resolve().parent.parent / "models")
        seen, ordered = set(), []
        for place in places:
            if place not in seen:
                seen.add(place)
                ordered.append(place)
        return ordered

    def default_model(self) -> Path | None:
        """The newest segmentation model found, or any model, or ``None``."""
        for directory in self.model_directories():
            if not directory.is_dir():
                continue
            candidates = sorted(directory.glob("*.onnx"))
            segment = [c for c in candidates if "seg" in c.name.lower()]
            if segment or candidates:
                return (segment or candidates)[0]
        return None
