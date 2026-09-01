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
    contains_credential,
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


# ------------------------------------------- redaction against awkward URLs
#
# Each of these was a real leak. They are asserted with `contains_credential`,
# which extracts the secrets from *that* URL rather than matching one hard-coded
# sentinel — so a leak through a path nobody anticipated still fails the test.

AWKWARD = [
    ("a password containing an @", "rtsp://admin:p@ss:w0rd@10.0.0.5/s"),
    ("a password but no username", "rtsp://:onlypass@10.0.0.5/s"),
    ("an IPv6 literal", "rtsp://user:pw0rd@[2001:db8::1]:554/s"),
    ("a non-numeric port", "rtsp://user:pw0rd@host:notaport/s"),
    ("a credential in the query", "http://cam/stream?user=admin&password=hunter2xyz&x=1"),
    ("a token in the query", "rtsp://cam/s?token=abc123def"),
    ("an uppercase query key", "rtsp://cam/s?Password=abc123def"),
    ("a scheme decode does not know", "srt://admin:pw0rd@10.0.0.5:9000"),
    ("no scheme at all, with an @", "admin:pw0rd@10.0.0.5/s"),
]


@pytest.mark.parametrize("description,url", AWKWARD, ids=[d for d, _ in AWKWARD])
def test_redaction_survives(description: str, url: str):
    redacted = redact_url(url)
    assert not contains_credential(redacted, url), (
        f"{description}: {redacted!r} still carries a secret from {url!r}"
    )


def test_redaction_keeps_the_url_useful():
    # A redactor that returns "<redacted>" for everything leaks nothing and
    # helps nobody. The host, port, path and non-secret query must survive.
    redacted = redact_url("rtsp://admin:hunter2@10.20.30.40:554/Streaming/Channels/101?x=1")

    assert "10.20.30.40:554" in redacted
    assert "/Streaming/Channels/101" in redacted
    assert "admin" in redacted
    assert "x=1" in redacted


def test_an_ipv6_literal_keeps_its_brackets():
    # Rebuilding the host from urlsplit().hostname drops them and produces a
    # display URL nobody can paste back.
    assert "[2001:db8::1]:554" in redact_url("rtsp://user:pw@[2001:db8::1]:554/s")


def test_a_malformed_port_does_not_raise():
    # urlsplit().port raises ValueError here. An earlier version let that
    # escape out of VideoSource.__init__, outside any caller's error handling.
    VideoSource("rtsp://user:pw@host:notaport/s")


def test_a_windows_path_is_left_alone():
    assert redact_url(r"C:\media\clip.mp4") == r"C:\media\clip.mp4"


def test_an_unknown_shape_fails_closed(monkeypatch):
    # When redaction cannot be sure, it must return a marker rather than echo
    # the input. Echoing a string that might hold a password is the one outcome
    # that must never happen.
    assert redact_url("") == "<no source>"
    assert "@" not in redact_url("some/path/with@sign")


# ------------------------------------------------- the missing-file leak


def test_a_missing_file_error_never_carries_the_credential():
    """The path an earlier version leaked through.

    `live` is a caller override and `_looks_live` only knows six schemes, so a
    credentialed srt:// or rtmp:// URL lands in the file branch — where the
    error was built from the raw path and shown to the operator.
    """
    url = "srt://admin:hunter2secret@10.0.0.5:9000/live"
    source = VideoSource(url, live=False)

    with pytest.raises(DecodeError) as caught:
        source.open()

    assert not contains_credential(str(caught.value), url)
    assert "10.0.0.5" in str(caught.value), "the operator still needs to know which source"


def test_a_directory_error_never_carries_the_credential(tmp_path: Path):
    url = f"{tmp_path}?password=hunter2secret"
    with pytest.raises(DecodeError) as caught:
        VideoSource(url, live=False).open()

    assert "hunter2secret" not in str(caught.value)


# ------------------------------------------------------------- egress guard


def test_a_camera_outside_the_local_network_is_refused():
    """The zero-WAN promise, enforced where it can actually be broken.

    A camera URL is the one string an operator types that the software then
    connects to. 8.8.8.8 is routable and public; refusing it is what stops a
    typo or a poisoned DNS entry from turning this system into one that reaches
    the Internet.
    """
    with pytest.raises(DecodeError) as caught:
        VideoSource("rtsp://8.8.8.8:554/stream").open()

    message = str(caught.value)
    assert "outside the local network" in message
    assert "8.8.8.8" in message, "an operator must be told what was refused"
    assert "SENTINEL_ALLOW_PUBLIC_SOURCES" in message, "and how to override it deliberately"


def test_a_private_address_is_allowed_through_to_the_connect():
    # 10.x is RFC 1918. It must fail on reachability, not on the egress guard —
    # otherwise the guard would block every real deployment.
    with pytest.raises(DecodeError) as caught:
        VideoSource("rtsp://10.255.255.1:554/stream").open()

    assert "outside the local network" not in str(caught.value)


def test_loopback_is_allowed():
    with pytest.raises(DecodeError) as caught:
        VideoSource("rtsp://127.0.0.1:1/stream").open()

    assert "outside the local network" not in str(caught.value)
