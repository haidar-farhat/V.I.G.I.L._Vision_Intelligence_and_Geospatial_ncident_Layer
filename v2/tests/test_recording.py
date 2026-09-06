from pathlib import Path

import numpy as np
import pytest

from vigil.adapters.recorder import Recorder, Segment, file_safe
from vigil.service.runtime import RetentionPolicy, apply_retention
from vigil.storage.store import Store


def test_file_safe_names_keep_a_device_id_off_an_ntfs_stream():
    assert file_safe("device:0") == "device-0"
    assert file_safe("north gate/07") == "north-gate-07"
    assert file_safe("") == "camera"


def test_frames_become_hashed_segments(tmp_path):
    recorder = Recorder("device:0", tmp_path, fps=10.0, segment_seconds=1.0)
    recorder.start()
    for i in range(25):
        recorder.write(np.full((120, 160, 3), i * 10, dtype=np.uint8), i * 100)
    closed = recorder.take_closed()
    assert len(closed) == 2, "two full seconds closed while the third is open"
    final = recorder.close()
    assert len(final) == 1
    every = closed + final
    assert all(s.path.is_file() and s.size_bytes > 0 and len(s.sha256) == 64 for s in every)
    assert sum(s.frames for s in every) == 25
    assert recorder.directory.name == "device-0"


def _segment(path: Path, camera="cam", *, start: int, size=1000, preserved=False) -> Segment:
    path.write_bytes(b"\0" * size)
    return Segment(camera, path, start, start + 60_000, 900, 160, 120, 15.0, size, "ab" * 32)


def test_retention_deletes_the_oldest_unpreserved_clips_and_reports_a_shortfall(tmp_path, monkeypatch):
    import shutil
    from collections import namedtuple

    usage = namedtuple("usage", "total used free")
    free = {"value": 100 * 1024**2}
    monkeypatch.setattr(shutil, "disk_usage", lambda p: usage(10**12, 0, free["value"]))
    with Store(":memory:") as store:
        old = _segment(tmp_path / "old.mp4", start=1_000)
        keep = _segment(tmp_path / "keep.mp4", start=2_000)
        new = _segment(tmp_path / "new.mp4", start=3_000)
        for s in (old, keep, new):
            store.save_segment(s)
        store.preserve_segments([keep.path])
        shortfall = apply_retention(store, RetentionPolicy(max_age_days=None, max_bytes=2500, min_free_bytes=None), now_millis=4_000)
        assert not old.path.exists() and keep.path.exists() and new.path.exists(), "oldest first, and only as much as needed"
        assert shortfall is None
        actions = [r["action"] for r in store.audit_trail()]
        assert actions.count("recording.deleted") == 1
        # Everything left is preserved or needed, and the disk is still short.
        store.preserve_segments([new.path])
        shortfall = apply_retention(store, RetentionPolicy(max_age_days=None, max_bytes=None, min_free_bytes=1024**3), now_millis=4_000)
        assert shortfall is not None and "preserved" in shortfall
        assert keep.path.exists() and new.path.exists()
