"""Tests for local camera enumeration.

**No test here switches a camera on.** Enumeration is metadata only, and the one
test that opens a real device is skipped unless `SENTINEL_TEST_CAMERA=1` is set —
because a test suite that turns on the developer's webcam, or that fails on a CI
runner with no camera, is a test suite people stop running.

Each platform is exercised against captured output from that platform's own
query: a PowerShell CIM result, a sysfs tree, a `system_profiler` document. That
is what makes these runnable on any machine and still capable of failing: the
parsing is the part that breaks, and the parsing does not need a camera.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from sentinel import devices, logs
from sentinel.devices import DeviceError, LocalCamera


@pytest.fixture(autouse=True)
def clean_logging():
    logs.reset()
    yield
    logs.reset()


# ------------------------------------------------------------------ the scheme


@pytest.mark.parametrize(
    "text, index",
    [("device:0", 0), ("device:1", 1), ("device:12", 12), ("  device:3  ", 3), ("DEVICE:2", 2)],
)
def test_a_device_source_is_recognised_and_parsed(text: str, index: int):
    assert devices.is_device_source(text)
    assert devices.device_index(text) == index


@pytest.mark.parametrize(
    "text",
    [
        "device:",
        "device:front",
        "device:-1",
        "device:0.5",
        "device: 0 1",
    ],
)
def test_a_malformed_device_source_is_refused_rather_than_defaulted(text: str):
    # Refused, never defaulted to zero. A typo that quietly opened the built-in
    # webcam instead of the one an operator meant would point a camera at
    # somewhere nobody chose, and every position it reported would be wrong.
    with pytest.raises(DeviceError):
        devices.device_index(text)


@pytest.mark.parametrize(
    "text",
    ["gate.mp4", "rtsp://10.0.0.5/s", "/dev/video0", "C:\\media\\clip.mp4", ""],
)
def test_things_that_are_not_devices_are_not_mistaken_for_one(text: str):
    assert not devices.is_device_source(text)


def test_the_source_string_round_trips():
    camera = LocalCamera(index=3, name="Logitech C920", identifier="usb-x", backend="V4L2")

    assert camera.source == "device:3"
    assert devices.device_index(camera.source) == 3


def test_an_unconfirmed_index_says_so_where_an_operator_will_read_it():
    # Two identical webcams are indistinguishable by name, and a USB bus can
    # enumerate differently after a reboot. The label must not imply certainty
    # the system does not have.
    assumed = LocalCamera(0, "Integrated Camera", None, "Media Foundation")
    confirmed = LocalCamera(0, "Integrated Camera", None, "DirectShow",
                            index_confirmed=True, width=640, height=480)

    assert "assumed" in assumed.label
    assert "assumed" not in confirmed.label
    assert "640x480" in confirmed.label


# -------------------------------------------------------------------- Windows


WINDOWS_PNP = json.dumps([
    {"Name": "Integrated Camera",
     "DeviceID": "USB\\VID_5986&PID_2174&MI_00\\7&3B5246B5&1&0000"},
    {"Name": "Integrated IR Camera",
     "DeviceID": "USB\\VID_5986&PID_2174&MI_02\\7&3B5246B5&1&0002"},
])


def test_the_windows_device_registry_is_parsed(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(devices, "_run", lambda command: WINDOWS_PNP)

    found = devices.list_cameras()

    assert [camera.name for camera in found] == [
        "Integrated Camera", "Integrated IR Camera",
    ]
    assert found[0].identifier.startswith("USB\\VID_5986")
    # Windows gives no supported mapping from a PnP instance to a capture index.
    assert all(camera.index_confirmed is False for camera in found)


def test_a_single_windows_camera_is_not_mistaken_for_a_broken_list(monkeypatch):
    # PowerShell's ConvertTo-Json emits an object rather than an array when
    # there is exactly one result — the single most common shape on a laptop,
    # and the one a naive parser drops entirely.
    single = json.dumps({"Name": "Integrated Camera", "DeviceID": "USB\\ONE"})
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(devices, "_run", lambda command: single)

    found = devices.list_cameras()

    assert len(found) == 1
    assert found[0].name == "Integrated Camera"


def test_a_machine_with_no_camera_is_an_ordinary_machine(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(devices, "_run", lambda command: "")

    assert devices.list_cameras() == []


def test_a_platform_query_that_returns_rubbish_does_not_crash(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(devices, "_run", lambda command: "not json at all {{{")

    assert devices.list_cameras() == []


# ---------------------------------------------------------------------- Linux


def build_v4l2_tree(root: Path, nodes: dict[str, tuple[str, str]]) -> Path:
    """A fake `/sys/class/video4linux`. `nodes` maps name -> (label, caps)."""
    root.mkdir(parents=True, exist_ok=True)
    for node, (label, caps) in nodes.items():
        directory = root / node
        directory.mkdir()
        (directory / "name").write_text(label + "\n", encoding="utf-8")
        (directory / "device_caps").write_text(caps + "\n", encoding="utf-8")
    return root


def test_the_v4l2_tree_is_read_and_the_index_is_a_fact(monkeypatch, tmp_path: Path):
    # The one platform where the index is not an assumption: /dev/video2 *is*
    # index 2, and that is what V4L2 and OpenCV both use.
    tree = build_v4l2_tree(tmp_path / "v4l", {
        "video0": ("Integrated Camera: Integrated C", "0x84a00001"),
        "video2": ("Logitech C920", "0x84a00001"),
    })
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(devices, "Path", lambda p: tree if "video4linux" in str(p) else Path(p))

    found = devices.list_cameras()

    assert [camera.index for camera in found] == [0, 2]
    assert [camera.name for camera in found] == [
        "Integrated Camera: Integrated C", "Logitech C920",
    ]
    assert all(camera.index_confirmed for camera in found)
    assert found[1].identifier == "/dev/video2"


def test_a_metadata_node_is_not_offered_as_a_camera(monkeypatch, tmp_path: Path):
    # A modern UVC camera exposes a capture node *and* a metadata node. The
    # metadata node opens happily and produces no image, so offering it to an
    # operator gives them a camera that appears to work and shows nothing.
    tree = build_v4l2_tree(tmp_path / "v4l", {
        "video0": ("Integrated Camera", "0x84a00001"),   # capture
        "video1": ("Integrated Camera", "0x00a00000"),   # metadata only
    })
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(devices, "Path", lambda p: tree if "video4linux" in str(p) else Path(p))

    found = devices.list_cameras()

    assert [camera.index for camera in found] == [0]


def test_a_node_whose_capabilities_cannot_be_read_is_kept(monkeypatch, tmp_path: Path):
    # Excluding a real camera because its capabilities were unreadable is the
    # worse of the two mistakes: the operator can see that a listed camera shows
    # nothing, but cannot see one that was never listed.
    tree = tmp_path / "v4l"
    (tree / "video0").mkdir(parents=True)
    (tree / "video0" / "name").write_text("Odd Camera\n", encoding="utf-8")

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(devices, "Path", lambda p: tree if "video4linux" in str(p) else Path(p))

    found = devices.list_cameras()

    assert [camera.name for camera in found] == ["Odd Camera"]


def test_a_linux_machine_with_no_v4l2_at_all(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        devices, "Path", lambda p: tmp_path / "absent" if "video4linux" in str(p) else Path(p)
    )

    assert devices.list_cameras() == []


# ---------------------------------------------------------------------- macOS


MACOS_PROFILE = json.dumps({
    "SPCameraDataType": [
        {"_name": "FaceTime HD Camera", "spcamera_unique-id": "0x8020000005ac8514"},
        {"_name": "iPhone Camera", "spcamera_unique-id": "0x0000000012345678"},
    ]
})


def test_the_macos_system_profile_is_parsed(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(devices, "_run", lambda command: MACOS_PROFILE)

    found = devices.list_cameras()

    assert [camera.name for camera in found] == ["FaceTime HD Camera", "iPhone Camera"]
    assert found[0].identifier == "0x8020000005ac8514"
    assert all(camera.backend == "AVFoundation" for camera in found)


# ------------------------------------------------------------------- backends


def test_each_platform_names_its_own_capture_interface(monkeypatch):
    # Named, never left to OpenCV's "any". An operator whose camera works on one
    # machine and not another needs to know which interface each one took.
    import cv2

    for platform, expected, constant in (
        ("win32", "Media Foundation", cv2.CAP_MSMF),
        ("darwin", "AVFoundation", cv2.CAP_AVFOUNDATION),
        ("linux", "Video4Linux2", cv2.CAP_V4L2),
    ):
        monkeypatch.setattr(sys, "platform", platform)
        assert devices.preferred_backend() == expected
        assert devices.backend_constants()[0] == (constant, expected)


def test_windows_keeps_directshow_as_a_fallback(monkeypatch):
    # Measured on this machine: the integrated camera opens on DirectShow and
    # not on Media Foundation. Without the fallback there would be no camera.
    import cv2

    monkeypatch.setattr(sys, "platform", "win32")
    order = devices.backend_constants()

    assert [name for _, name in order] == ["Media Foundation", "DirectShow"]
    assert order[1][0] == cv2.CAP_DSHOW


def test_no_other_platform_pretends_to_have_a_fallback(monkeypatch):
    for platform in ("linux", "darwin"):
        monkeypatch.setattr(sys, "platform", platform)
        assert len(devices.backend_constants()) == 1


# --------------------------------------------------------------- what it runs


def test_the_platform_queries_never_use_a_shell():
    # The arguments are fixed literals, and keeping `shell=False` means they
    # cannot become an injection point if a caller ever passes something in.
    source = Path(devices.__file__).read_text(encoding="utf-8")

    assert "shell=False" in source
    assert "shell=True" not in source


def test_a_platform_query_that_hangs_does_not_hang_the_application(monkeypatch):
    # A hung device subsystem is a real state, and a security appliance must not
    # block its start-up on one.
    import subprocess

    def hang(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="powershell", timeout=8.0)

    monkeypatch.setattr(subprocess, "run", hang)

    assert devices._run(["powershell", "-Command", "x"]) == ""


def test_enumeration_opens_nothing(monkeypatch):
    # The property that makes it safe to call from an interface: listing the
    # cameras on a machine must not switch one on, and on macOS must not trigger
    # a permission prompt for a camera nobody asked to use.
    import cv2

    def forbidden(*args, **kwargs):
        raise AssertionError("enumeration opened a camera")

    monkeypatch.setattr(cv2, "VideoCapture", forbidden)
    monkeypatch.setattr(devices, "_run", lambda command: "")

    devices.list_cameras()
    devices.discover(probe_indices=False)


# ------------------------------------------------------- against real hardware


needs_camera = pytest.mark.skipif(
    os.environ.get("SENTINEL_TEST_CAMERA", "") not in ("1", "true", "yes", "on"),
    reason=(
        "opens a real camera. Set SENTINEL_TEST_CAMERA=1 to run it. Off by "
        "default because a suite that switches on the developer's webcam, or "
        "that fails on a runner with no camera, is a suite people stop running."
    ),
)


@needs_camera
def test_a_real_camera_opens_and_produces_a_frame():
    from sentinel.decode import VideoSource

    found = devices.discover()
    if not found:
        pytest.skip("no camera on this machine would open")

    camera = found[0]
    assert camera.index_confirmed
    assert camera.width and camera.height

    source = VideoSource(camera.source, source_id="test-camera")
    try:
        info = source.open()
        assert info.is_live is True
        assert info.width > 0 and info.height > 0
        # Provenance: which interface actually opened it, not which one was
        # preferred.
        assert info.backend in {name for _, name in devices.backend_constants()}

        frame = next(iter(source))
        assert frame.image.shape[:2] == (info.height, info.width)
        # A live source timestamps with the wall clock, so this is a real epoch.
        assert frame.timestamp_millis > 1_600_000_000_000
    finally:
        source.close()
