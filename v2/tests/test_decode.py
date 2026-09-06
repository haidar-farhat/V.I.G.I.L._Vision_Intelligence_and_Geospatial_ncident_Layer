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
