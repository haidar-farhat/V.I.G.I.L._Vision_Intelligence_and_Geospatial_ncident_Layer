"""Cameras attached to this machine, through the operating system's own API.

A USB or built-in camera is not a file and not a network stream. It is a device
the operating system owns, and the only honest way to find one is to ask the
operating system — which is what this module does, using the native interface on
each platform and nothing else:

| Platform | Enumerated by | Opened through |
|---|---|---|
| Windows | `Win32_PnPEntity`, the PnP device registry | Media Foundation (`CAP_MSMF`), falling back to DirectShow |
| Linux | `/sys/class/video4linux`, the V4L2 device tree | Video4Linux2 (`CAP_V4L2`) |
| macOS | `system_profiler SPCameraDataType` | AVFoundation (`CAP_AVFOUNDATION`) |

No third-party dependency is added for this. Each of the three is a query the
platform already answers — a CIM query, a sysfs read, a system profile — and all
three are local: nothing here opens a socket, and the offline audit covers this
file like any other.

**The index problem, stated rather than hidden.** OpenCV opens a camera by a
small integer. The operating system identifies one by a name and a stable
hardware id. There is no supported way to map between them, so this module
*assumes* they enumerate in the same order and says so: every result carries
`index_confirmed`, which is only true once something has actually opened that
index and seen a frame. On Linux the assumption is not needed — `/dev/video2` is
index 2 by construction — and the flag is set accordingly.

That matters because two identical webcams are indistinguishable by name, and a
USB bus can enumerate in a different order after a reboot. The interface
therefore shows the operator a frame and lets them confirm which camera they are
looking at. A system that guessed, and was wrong, would attribute an intrusion to
the wrong side of a building.

**Nothing here opens a camera unless asked.** Enumeration reads metadata and
captures nothing. :func:`probe` is the only function that opens a device, it is
never called on start-up, and on macOS it is what triggers the operating
system's own permission prompt — which is the correct place for that to happen.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from .logs import get as _get_logger

_log = _get_logger(__name__)

#: The scheme that names a local device. `device:0` is the canonical spelling and
#: is what is stored, displayed and handed to `VideoSource`.
SCHEME = "device"

#: How long any platform query may take. These are local calls that normally
#: answer in milliseconds; a hang means something is wrong with the device
#: subsystem, and a security appliance must not block its start-up on that.
QUERY_TIMEOUT_SECONDS = 8.0

#: How long a probe may spend trying to pull one frame. A camera that has not
#: produced an image in this long is not going to.
PROBE_TIMEOUT_SECONDS = 5.0

#: Highest index worth probing when the platform gives no list at all. Cameras
#: are enumerated from zero and consecutively; ten is far past any real machine
#: and keeps a blind scan bounded.
MAX_BLIND_INDEX = 10


@dataclass(frozen=True, slots=True)
class LocalCamera:
    """A camera the operating system says is attached.

    Existing is not the same as working: a device can be listed and still be in
    use by another application, disabled by policy, or refused by the operating
    system's privacy controls. :func:`probe` is what turns "listed" into
    "opened, and here is the resolution".
    """

    #: What OpenCV opens. See the module docstring on why this is separate from
    #: the identity below.
    index: int
    #: What the operating system calls it. Shown to the operator.
    name: str
    #: A stable hardware identifier, when the platform provides one — a PnP
    #: instance path, a device node, a USB unique id. `None` when it does not.
    #: This is what survives a reboot; the index is not.
    identifier: str | None
    #: The capture API this will be opened through, named rather than left to
    #: OpenCV's "any", so provenance can record which one actually worked.
    backend: str
    #: Whether `index` has been shown to open this camera, rather than assumed
    #: from enumeration order. False until something has looked.
    index_confirmed: bool = False
    #: Filled in by :func:`probe`.
    width: int | None = None
    height: int | None = None

    @property
    def source(self) -> str:
        """What to hand :class:`~sentinel.decode.VideoSource`."""
        return f"{SCHEME}:{self.index}"

    @property
    def label(self) -> str:
        """A one-line description for an operator."""
        size = f" — {self.width}x{self.height}" if self.width and self.height else ""
        certainty = "" if self.index_confirmed else "  (index assumed)"
        return f"{self.index}: {self.name}{size}{certainty}"


class DeviceError(RuntimeError):
    """A device could not be enumerated or opened."""


# ---------------------------------------------------------------- the scheme


def is_device_source(url: str) -> bool:
    return str(url).strip().lower().startswith(f"{SCHEME}:")


def device_index(url: str) -> int:
    """The index in ``device:N``.

    Raises rather than defaulting to zero. A typo that silently opened the
    built-in webcam instead of the one the operator meant is precisely the sort
    of quiet substitution this system must not make.
    """
    text = str(url).strip()
    if not is_device_source(text):
        raise DeviceError(f"{text!r} is not a device source; expected 'device:N'")

    _, _, remainder = text.partition(":")
    remainder = remainder.strip()
    if not remainder.isdigit():
        raise DeviceError(
            f"{text!r} is not a device source. The form is 'device:N', where N "
            "is the index from `sentinel devices`."
        )
    return int(remainder)


def preferred_backend() -> str:
    """The name of this platform's native capture API."""
    return {
        "win32": "Media Foundation",
        "darwin": "AVFoundation",
    }.get(sys.platform, "Video4Linux2")


def backend_constants() -> list[tuple[int, str]]:
    """OpenCV backend ids to try, in order, with the name of each.

    A list rather than one value because Windows genuinely needs two: Media
    Foundation is the modern interface and the right default, and DirectShow
    still works with older devices that Media Foundation will not open at all.
    Trying the second is a fallback, not a preference, and which one succeeded
    is recorded rather than forgotten.
    """
    import cv2

    if sys.platform == "win32":
        return [(cv2.CAP_MSMF, "Media Foundation"), (cv2.CAP_DSHOW, "DirectShow")]
    if sys.platform == "darwin":
        return [(cv2.CAP_AVFOUNDATION, "AVFoundation")]
    return [(cv2.CAP_V4L2, "Video4Linux2")]


# ------------------------------------------------------------- enumeration


def _run(command: list[str]) -> str:
    """Run a local platform query, or return nothing.

    Never raises. A machine with no camera subsystem, a locked-down PowerShell
    policy or a missing `system_profiler` are all ordinary states, and none of
    them is a reason for the application to fail to start — the answer is simply
    an empty list, which the caller already has to handle.
    """
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=QUERY_TIMEOUT_SECONDS,
            # No shell. The arguments are fixed literals here, and keeping it
            # that way means they cannot become an injection point if a caller
            # ever passes something through.
            shell=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        _log.debug("device query %s failed: %s", command[0], type(error).__name__)
        return ""

    if result.returncode != 0:
        _log.debug("device query %s exited %d", command[0], result.returncode)
        return ""
    return result.stdout


def _windows_cameras() -> list[LocalCamera]:
    """Ask the PnP device registry, through CIM.

    `Camera` is what Media Foundation devices register as on Windows 10 and 11;
    `Image` is the older class that many USB webcams still use. Both are
    included, because excluding one silently loses half the cameras in the
    field.
    """
    script = (
        "Get-CimInstance Win32_PnPEntity "
        "| Where-Object { $_.PNPClass -in 'Camera','Image' -and $_.Status -eq 'OK' } "
        "| Select-Object Name, DeviceID "
        "| ConvertTo-Json -Compress"
    )
    output = _run(
        [
            "powershell", "-NoProfile", "-NonInteractive",
            "-ExecutionPolicy", "Bypass", "-Command", script,
        ]
    )
    if not output.strip():
        return []

    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        _log.debug("the PnP query returned something that is not JSON")
        return []

    # ConvertTo-Json emits an object rather than an array for a single result.
    entries = parsed if isinstance(parsed, list) else [parsed]
    backend = preferred_backend()

    return [
        LocalCamera(
            index=index,
            name=str(entry.get("Name") or "Unnamed camera"),
            identifier=str(entry["DeviceID"]) if entry.get("DeviceID") else None,
            backend=backend,
            # Assumed. Windows gives no supported mapping from a PnP instance to
            # a Media Foundation index; see the module docstring.
            index_confirmed=False,
        )
        for index, entry in enumerate(entries)
        if isinstance(entry, dict)
    ]


def _linux_cameras() -> list[LocalCamera]:
    """Read the V4L2 device tree.

    The one platform where the index is a fact rather than an assumption:
    `/dev/video2` *is* index 2, and that is what V4L2 and OpenCV both use.

    Modern UVC cameras expose more than one node — a capture node and a metadata
    node, sometimes several. The metadata nodes open and produce no image, so
    they are filtered out by the device's own reported capabilities rather than
    by guessing from the name.
    """
    root = Path("/sys/class/video4linux")
    if not root.is_dir():
        return []

    cameras: list[LocalCamera] = []
    for node in sorted(root.iterdir(), key=lambda path: path.name):
        if not node.name.startswith("video"):
            continue
        digits = node.name[len("video"):]
        if not digits.isdigit():
            continue

        try:
            name = (node / "name").read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue

        if not _linux_node_captures(node):
            _log.debug("%s is not a capture node; skipping", node.name)
            continue

        cameras.append(
            LocalCamera(
                index=int(digits),
                name=name or node.name,
                identifier=f"/dev/{node.name}",
                backend=preferred_backend(),
                # `/dev/videoN` is index N. Nothing is assumed here.
                index_confirmed=True,
            )
        )

    return cameras


def _linux_node_captures(node: Path) -> bool:
    """Whether a V4L2 node is a video capture device.

    Read from `device_caps` when the kernel exposes it, which it has since 4.x.
    When it does not, the node is kept: excluding a real camera because its
    capabilities could not be read is the worse of the two mistakes.
    """
    for filename in ("device_caps", "capabilities"):
        try:
            raw = (node / filename).read_text(encoding="utf-8").strip()
        except OSError:
            continue
        try:
            caps = int(raw, 16 if raw.lower().startswith("0x") else 16)
        except ValueError:
            continue
        # V4L2_CAP_VIDEO_CAPTURE. A metadata or output-only node does not have
        # it, and is exactly what should be filtered here.
        return bool(caps & 0x00000001)
    return True


def _macos_cameras() -> list[LocalCamera]:
    """Ask the system profiler.

    `SPCameraDataType` is the same source the operating system's own settings
    panel reads. It reports built-in, USB and Continuity cameras alike.
    """
    output = _run(["system_profiler", "-json", "SPCameraDataType"])
    if not output.strip():
        return []

    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        return []

    entries = parsed.get("SPCameraDataType", []) if isinstance(parsed, dict) else []
    backend = preferred_backend()
    cameras: list[LocalCamera] = []

    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        cameras.append(
            LocalCamera(
                index=index,
                name=str(entry.get("_name") or "Unnamed camera"),
                identifier=(
                    str(entry.get("spcamera_unique-id"))
                    if entry.get("spcamera_unique-id")
                    else None
                ),
                backend=backend,
                index_confirmed=False,
            )
        )

    return cameras


def list_cameras() -> list[LocalCamera]:
    """Every camera the operating system reports, without opening any of them.

    Captures nothing. Enumeration is metadata only, which is why it is safe to
    call from an interface: listing the cameras on a machine must not switch one
    on, and on macOS it must not trigger a permission prompt for a camera the
    operator has not asked to use.

    Returns an empty list rather than raising when the platform has nothing to
    say. A machine with no camera is an ordinary machine.
    """
    if sys.platform == "win32":
        cameras = _windows_cameras()
    elif sys.platform == "darwin":
        cameras = _macos_cameras()
    else:
        cameras = _linux_cameras()

    _log.debug("%d local camera(s) reported by the operating system", len(cameras))
    return cameras


# ------------------------------------------------------------------ probing


class _QuietOpenCV:
    """Silence OpenCV's own logger for the duration of a probe.

    A probe *expects* failures — that is what it is for. OpenCV writes its
    attempts to stderr at WARN, so without this an ordinary "index 1 is a
    Windows Hello infrared sensor and will not open" prints a backend warning at
    an operator who can do nothing with it, next to this module's own line
    saying the same thing in words.

    Scoped to the probe and restored afterwards, because a genuine decode
    failure later on is exactly the noise that should be heard.
    """

    __slots__ = ("_previous",)

    def __enter__(self) -> "_QuietOpenCV":
        import cv2

        self._previous = None
        try:
            self._previous = cv2.utils.logging.getLogLevel()
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
        except (AttributeError, cv2.error):
            # An OpenCV build without the logging module. Noise is the cost.
            self._previous = None
        return self

    def __exit__(self, *_: object) -> None:
        import cv2

        if self._previous is None:
            return
        try:
            cv2.utils.logging.setLogLevel(self._previous)
        except (AttributeError, cv2.error):
            pass


def probe(index: int) -> LocalCamera | None:
    """Open one camera briefly and report what it actually is.

    This is the only function here that switches a camera on, and it is never
    called on start-up. It opens the device, pulls a single frame, and closes
    it — because a device that opens and produces nothing is a common and
    confusing state, and reporting it as working would be a lie the operator
    finds out about at the worst moment.

    Returns ``None`` when nothing usable is at that index.
    """
    import cv2

    with _QuietOpenCV():
        return _probe(index)


def _probe(index: int) -> LocalCamera | None:
    import cv2

    for constant, backend in backend_constants():
        capture = None
        try:
            capture = cv2.VideoCapture(index, constant)
            if not capture.isOpened():
                continue

            ok, frame = capture.read()
            if not ok or frame is None or frame.size == 0:
                _log.debug("index %d opened on %s but produced no frame", index, backend)
                continue

            height, width = frame.shape[:2]
            _log.info(
                "camera index %d opened through %s at %dx%d", index, backend, width, height
            )
            return LocalCamera(
                index=index,
                name=f"Camera {index}",
                identifier=None,
                backend=backend,
                index_confirmed=True,
                width=int(width),
                height=int(height),
            )
        except cv2.error as error:
            _log.debug("index %d failed on %s: %s", index, backend, error)
        finally:
            if capture is not None:
                capture.release()

    return None


def discover(*, probe_indices: bool = True) -> list[LocalCamera]:
    """The operating system's list, confirmed against what actually opens.

    The two halves answer different questions and neither is sufficient. The
    platform knows the *names* — "Integrated Webcam", "Logitech C920" — and an
    operator picks a camera by name. OpenCV knows which *indices* open. Pairing
    them positionally is an assumption, and the result says so: a camera whose
    index has been confirmed by opening it carries `index_confirmed=True`, and
    one that has not does not.

    With ``probe_indices=False`` nothing is opened at all, which is what an
    interface wants for a first listing.
    """
    named = list_cameras()
    if not probe_indices:
        return named

    confirmed: list[LocalCamera] = []
    # A blind scan when the platform said nothing — a machine can have a working
    # camera the device registry does not describe, and refusing to look for it
    # would make the feature useless exactly where it is needed most.
    limit = len(named) if named else MAX_BLIND_INDEX

    for index in range(limit):
        opened = probe(index)
        if opened is None:
            continue

        if index < len(named):
            # Keep the operating system's name and identity, take the measured
            # size and backend, and record that the index is now a fact.
            confirmed.append(
                replace(
                    named[index],
                    backend=opened.backend,
                    index_confirmed=True,
                    width=opened.width,
                    height=opened.height,
                )
            )
        else:
            confirmed.append(opened)

    if named and not confirmed:
        _log.warning(
            "%d camera(s) are listed by the operating system but none would open. "
            "They may be in use by another application, or blocked by a privacy "
            "setting.",
            len(named),
        )

    return confirmed
