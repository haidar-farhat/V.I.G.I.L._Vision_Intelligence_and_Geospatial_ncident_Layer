"""Tests for video decode.

Two things are being protected here. One is ordinary: frames come out, with
sensible timestamps, and a missing file fails cleanly.

The other is not ordinary, and is the reason several of these tests exist at
all: **a camera password must never escape**. An RTSP URL carries one, and the
number of places a URL gets printed — a log line, an exception, a repr in a
traceback, a status field in the interface — means the only defence that works is
for the credential to be unreachable from anything except the moment of
connection.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import scene
from sentinel.decode import (
    DecodeError,
    LiveStream,
    VideoSource,
    redact_url,
)

#: Used only in tests, and asserted never to appear in output.
SECRET = "hunter2-not-a-real-password"
CAMERA_URL = f"rtsp://admin:{SECRET}@10.20.30.40:554/Streaming/Channels/101"


# ------------------------------------------------------------------ redaction


def test_a_password_is_removed_from_a_url():
    redacted = redact_url(CAMERA_URL)

    assert SECRET not in redacted
    assert "admin" in redacted, "the username is not a secret and identifies the camera"
    assert "10.20.30.40:554" in redacted
    assert "/Streaming/Channels/101" in redacted


def test_redaction_does_not_leak_the_password_length():
    # A fixed marker, not one asterisk per character. Length is information.
    short = redact_url("rtsp://user:a@host/s")
    long = redact_url("rtsp://user:" + "a" * 64 + "@host/s")

    assert short == long


def test_a_url_without_a_password_is_unchanged():
    plain = "rtsp://10.20.30.40:554/Streaming/Channels/101"
    assert redact_url(plain) == plain


def test_a_file_path_is_unchanged():
    assert redact_url("C:/media/evidence.mp4") == "C:/media/evidence.mp4"


def test_an_unparseable_url_does_not_return_itself():
    # If the URL cannot be parsed it cannot be redacted, so it must not be
    # echoed back — an unparseable string may still contain a credential.
    mangled = redact_url("rtsp://user:pw@[::bad::]:554/x")
    assert "pw" not in mangled


def test_a_source_never_reveals_its_password_anywhere():
    source = VideoSource(CAMERA_URL, source_id="cam-07")

    # Every route by which a URL normally escapes.
    for text in (repr(source), str(source), source.display_url, source.source_id,
                 f"{source}", "{}".format(source), f"{source!r}"):
        assert SECRET not in text


def test_a_failure_to_open_a_camera_does_not_name_the_password():
    # 203.0.113.x is TEST-NET-3: reserved for documentation, routed nowhere.
    url = "rtsp://admin:s3cr3t-value@203.0.113.99:554/none"
    source = VideoSource(url)

    with pytest.raises(DecodeError) as caught:
        source.open()

    assert "s3cr3t-value" not in str(caught.value)
    assert "203.0.113.99" in str(caught.value), "the operator still needs to know which camera"


# ---------------------------------------------------------------------- files


def test_a_missing_file_fails_before_opening_anything():
    with pytest.raises(DecodeError, match="No such video file"):
        VideoSource("does-not-exist.mp4").open()


def test_a_directory_is_not_a_video(tmp_path: Path):
    with pytest.raises(DecodeError, match="Not a file"):
        VideoSource(tmp_path).open()


def test_reading_before_opening_is_refused():
    with pytest.raises(DecodeError, match="not open"):
        VideoSource("anything.mp4").read()


def test_the_reference_video_decodes_completely(reference_video: Path):
    with VideoSource(reference_video, source_id="cam-01") as source:
        frames = list(source)

    assert len(frames) == scene.FRAME_COUNT
    assert all(frame.source_id == "cam-01" for frame in frames)
    assert frames[0].width == scene.WIDTH
    assert frames[0].height == scene.HEIGHT


def test_frame_indices_are_contiguous_from_zero(reference_video: Path):
    with VideoSource(reference_video) as source:
        indices = [frame.index for frame in source]

    assert indices == list(range(scene.FRAME_COUNT))


def test_timestamps_come_from_the_container_and_increase(reference_video: Path):
    # Everything downstream measures time as a difference between these. A
    # timestamp invented from a nominal frame rate is wrong exactly when a
    # stream stutters, which is when it matters.
    with VideoSource(reference_video) as source:
        stamps = [frame.timestamp_millis for frame in source]

    assert stamps[0] == 0
    assert all(later > earlier for earlier, later in zip(stamps, stamps[1:]))

    expected_span = 1000 * (scene.FRAME_COUNT - 1) / scene.FPS
    assert stamps[-1] == pytest.approx(expected_span, rel=0.02)


def test_source_info_reports_what_the_container_claims(reference_video: Path):
    with VideoSource(reference_video) as source:
        info = source.info

    assert (info.width, info.height) == (scene.WIDTH, scene.HEIGHT)
    assert info.fps == pytest.approx(scene.FPS, abs=0.5)
    assert info.frame_count == scene.FRAME_COUNT
    assert info.is_live is False


def test_a_file_is_not_treated_as_live(reference_video: Path):
    assert VideoSource(reference_video).is_live is False


def test_decoding_is_repeatable(reference_video: Path):
    # Replay must reproduce the original result, so the same file must decode to
    # the same bytes every time.
    def digest() -> bytes:
        import hashlib

        running = hashlib.sha256()
        with VideoSource(reference_video) as source:
            for frame in source:
                running.update(frame.image.tobytes())
        return running.digest()

    assert digest() == digest()


# ----------------------------------------------------------------- live rules


def test_an_rtsp_url_is_recognised_as_live():
    assert VideoSource(CAMERA_URL).is_live is True


def test_the_live_guess_can_be_overridden(reference_video: Path):
    assert VideoSource(reference_video, live=True).is_live is True


def test_a_file_cannot_be_wrapped_in_a_dropping_stream(reference_video: Path):
    # LiveStream discards frames to stay current. Doing that to a file would
    # make replay non-deterministic, and a replay that does not reproduce the
    # original result is not evidence.
    with pytest.raises(DecodeError, match="non-deterministic"):
        LiveStream(VideoSource(reference_video))


def test_a_live_stream_drops_rather_than_queues(reference_video: Path):
    # The file is replayed as if live: decode runs far ahead of this test, which
    # never reads, so the queue must stay at one frame and the rest must be
    # counted as dropped rather than accumulated.
    source = VideoSource(reference_video, live=True)
    with LiveStream(source) as stream:
        first = stream.read(timeout=5.0)
        assert first is not None

        deadline = 5.0
        step = 0.05
        waited = 0.0
        while stream.dropped_frames == 0 and waited < deadline:
            import time

            time.sleep(step)
            waited += step

        assert stream.dropped_frames > 0, "a backlog was allowed to build"
        assert stream._queue.qsize() <= 1
