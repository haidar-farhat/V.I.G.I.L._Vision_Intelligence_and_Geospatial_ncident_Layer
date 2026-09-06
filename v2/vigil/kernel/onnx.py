"""Opening an ONNX model, once, with the same discipline everywhere.

# Why this is in the kernel

Three parts of this product load a model: the object detector in `adapters`,
and faces and plates in `perception`. Those two layers may not import each
other, so without somewhere below both, the loading rules would exist in
duplicate — and the rules are exactly the kind that rot when duplicated.
They are:

- **Nothing is downloaded, ever.** A missing model is an error naming the path
  that was looked at, not a fetch.
- **Telemetry is disarmed** before the runtime is imported and again after,
  because a security appliance that phones home is not one.
- **The provider is read back off the session**, never assumed from what was
  asked for. `onnxruntime` falls back silently when a provider fails to
  initialise — a CUDA build against the wrong driver runs on the CPU and says
  nothing — and "why is this slow" is answered by that one string more often
  than by anything else.
- **The file's digest is recorded**, so an evidence package can say which
  weights produced a conclusion.

# What it deliberately does not do

It does not interpret outputs. Every model has its own output shape, and
guessing at one is how a loader ends up quietly producing plausible garbage
from a model it does not understand. Callers read their own tensors.
"""

from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path

#: Execution providers to prefer, best first. Only those the installed runtime
#: reports are used, and the one actually chosen is read back.
#:
#: `AzureExecutionProvider` is deliberately absent: it is a remote endpoint,
#: and this product does not send frames anywhere.
PREFERRED_PROVIDERS = (
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
    "ROCMExecutionProvider",
    "CoreMLExecutionProvider",
    "OpenVINOExecutionProvider",
    "CPUExecutionProvider",
)

#: Threads per session. Two rather than every core: several cameras each get a
#: session, and a session that takes the whole machine starves the others.
DEFAULT_INTRA_OP_THREADS = 2
THREADS_VARIABLE = "VIGIL_ORT_THREADS"

_LOCK = threading.Lock()
_DIGESTS: dict[tuple[str, int, int], str] = {}


class ModelError(RuntimeError):
    """A model that is missing, unreadable, or not what it claims to be."""


def silence_telemetry() -> None:
    os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")


def model_key(path: str | Path) -> tuple[str, int, int]:
    """Path, size and mtime. A model edited in place is a different model."""
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ModelError(
            f"no model at {resolved}; models are supplied by the operator and nothing is downloaded"
        )
    stat = resolved.stat()
    return (str(resolved), stat.st_size, stat.st_mtime_ns)


def available_providers() -> list[str]:
    """The providers the installed runtime offers, best first, or empty."""
    silence_telemetry()
    try:
        import onnxruntime as ort
    except ImportError:
        return []
    offered = set(ort.get_available_providers())
    return [p for p in PREFERRED_PROVIDERS if p in offered]


def threads() -> int:
    raw = os.environ.get(THREADS_VARIABLE, "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return DEFAULT_INTRA_OP_THREADS


def open_session(path: str | Path) -> tuple[object, str]:
    """`(session, provider)`. The provider is what it *got*, not what it asked."""
    resolved = Path(path).resolve()
    model_key(resolved)
    silence_telemetry()
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise ModelError(
            f"onnxruntime is not installed, so {resolved.name} cannot be run"
        ) from error

    try:
        ort.disable_telemetry_events()
    except Exception:  # noqa: BLE001 - older runtimes have no such call
        pass
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = threads()
    options.inter_op_num_threads = 1
    providers = available_providers() or ["CPUExecutionProvider"]
    try:
        session = ort.InferenceSession(str(resolved), sess_options=options, providers=providers)
    except Exception as error:  # noqa: BLE001 - onnxruntime raises many types
        raise ModelError(f"could not load the model at {resolved}: {error}") from error
    active = session.get_providers()
    return session, (active[0] if active else "unknown")


def digest(path: str | Path) -> str:
    """The file's SHA-256, cached by path/size/mtime.

    Cached because a digest over a 50 MB model is tens of milliseconds and
    several workers open the same file at start-up; keyed on size and mtime so
    a model replaced on disk is hashed again rather than remembered wrongly.
    """
    key = model_key(path)
    with _LOCK:
        known = _DIGESTS.get(key)
    if known is not None:
        return known
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    value = hasher.hexdigest()
    with _LOCK:
        _DIGESTS[key] = value
    return value


def forget() -> None:
    with _LOCK:
        _DIGESTS.clear()
