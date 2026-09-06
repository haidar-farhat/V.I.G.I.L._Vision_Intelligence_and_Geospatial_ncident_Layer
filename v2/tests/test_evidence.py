import json

import pytest

from test_incidents import event
from vigil.adapters.recorder import Segment
from vigil.domain.incidents import Correlator
from vigil.service.auth import Forbidden, Principal, Role
from vigil.service.evidence import export_incident, verify_package
from vigil.storage.store import Store

ANALYST = Principal("ann", Role.ANALYST, "user")
VIEWER = Principal("vic", Role.VIEWER, "user")


def test_an_export_carries_the_report_the_clips_and_a_manifest_that_verifies(tmp_path):
    with Store(":memory:") as store:
        events = [event("a", 1, 10_000), event("a", 2, 12_000)]
        store.save_events(events)
        incident = Correlator().correlate(events)[0]
        store.save_incidents([incident])
        clip = tmp_path / "a-clip.mp4"
        clip.write_bytes(b"video bytes")
        import hashlib

        store.save_segment(Segment("a", clip, 5_000, 20_000, 100, 160, 120, 15.0, clip.stat().st_size, hashlib.sha256(clip.read_bytes()).hexdigest()))
        folder = export_incident(store, incident, tmp_path / "evidence", by=ANALYST)
        assert (folder / "report.json").is_file() and (folder / "report.txt").is_file()
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        assert len(manifest["clips"]) == 1 and (folder / manifest["clips"][0]["file"]).is_file()
        assert verify_package(folder) == []
        assert str(clip) in store.preserved_paths(), "an exported clip is preserved from retention"
        assert "incident.exported" in [r["action"] for r in store.audit_trail()]
        (folder / "report.txt").write_text("tampered", encoding="utf-8")
        assert verify_package(folder) == ["report.txt: digest differs"]
        with pytest.raises(Forbidden):
            export_incident(store, incident, tmp_path / "e2", by=VIEWER)
