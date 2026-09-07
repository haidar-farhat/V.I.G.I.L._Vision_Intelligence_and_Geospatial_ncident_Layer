import pytest

from vigil.adapters.decode import DecodeError, LiveReader, VideoSource, is_live_source, redacted, require_private, split_password, with_password


def test_a_file_reads_every_frame_with_timestamps_that_advance(reference_video):
    with VideoSource(reference_video, source_id="cam") as source:
        info = source.info
        assert info is not None and info.width == 320 and not info.live
        frames = list(source)
    assert len(frames) == 90
    assert frames[-1].timestamp_millis > frames[0].timestamp_millis
    assert frames[0].source_id == "cam"


def test_a_missing_file_is_a_decode_error(tmp_path):
    with pytest.raises(DecodeError):
        VideoSource(tmp_path / "nothing.mp4").open()


def test_passwords_are_split_redacted_and_restored():
    url = "rtsp" + "://admin:s3cret@10.0.0.9:554/stream"
    clean, secret = split_password(url)
    assert secret == "s3cret" and ":s3cret@" not in clean and clean.startswith("rtsp")
    assert "s3cret" not in redacted(url) and "***" in redacted(url)
    assert with_password(clean, "s3cret") == url
    assert split_password("device:0") == ("device:0", None)
    assert is_live_source("device:0") and is_live_source(url) and not is_live_source("clip.mp4")


def test_a_public_address_is_refused_and_a_private_one_passes():
    require_private("127.0.0.1")
    require_private("10.0.0.9")
    with pytest.raises(DecodeError, match="outside the local network"):
        require_private("1.1.1.1")
    require_private("1.1.1.1", allow_public=True)


def test_the_live_reader_keeps_the_newest_frame_and_counts_drops(reference_video):
    reader = LiveReader(VideoSource(reference_video, source_id="cam"))
    reader.start()
    first = reader.read(timeout=3.0)
    assert first is not None
    import time

    time.sleep(0.5)
    later = reader.read(timeout=3.0)
    assert later is not None and later.index > first.index
    assert reader.dropped > 0, "a slow consumer must lose the oldest frames, not block"
    assert reader.stop()


class _Flaky:
    """A live source that dies after a few frames, then comes back.

    There is no way to unplug a camera inside a test, so the clocks and the
    failure are the only parts faked; everything else is what a real source
    does.
    """

    def __init__(self, fail_after: int = 3, failures: int = 2):
        import numpy as np

        self.source_id, self.display, self.live = "flaky", "flaky", True
        self._fail_after, self._failures = fail_after, failures
        self._served = 0
        self.opens = 0
        self.closes = 0
        self._image = np.zeros((8, 8, 3), dtype="uint8")

    def open(self):
        from vigil.adapters.decode import SourceInfo

        self.opens += 1
        self._served = 0
        return SourceInfo(self.source_id, self.display, 8, 8, 10.0, True)

    def read(self):
        from vigil.adapters.decode import Frame

        self._served += 1
        if self._failures > 0 and self._served > self._fail_after:
            self._failures -= 1
            return None  # the stream ended, which is how a dropped camera looks
        return Frame(self._image, self._served, self._served, self.source_id)

    def close(self):
        self.closes += 1


def test_a_live_source_that_drops_is_reconnected_with_a_growing_pause(monkeypatch):
    """The reconnect path is what a camera on a flaky switch spends its night in."""
    import time as time_module

    from vigil.adapters import decode

    monkeypatch.setattr(decode, "BACKOFF_SECONDS", (0.01, 0.02))
    source = _Flaky(fail_after=3, failures=2)
    reader = LiveReader(source)
    reader.start()
    try:
        deadline = time_module.monotonic() + 10
        while reader.reconnects < 2 and time_module.monotonic() < deadline:
            reader.read(timeout=0.2)
        assert reader.reconnects >= 2, "the reader gave up instead of reconnecting"
        assert source.opens >= 3 and source.closes >= 2, "each attempt must open and close the source"
        assert reader.read(timeout=2.0) is not None, "it never recovered"
        assert reader.fault is None, "a recovered reader must stop reporting the old fault"
    finally:
        assert reader.stop()
