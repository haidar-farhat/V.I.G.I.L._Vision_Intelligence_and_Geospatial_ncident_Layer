from vigil.domain.detection import BoundingBox, Detection
from vigil.domain.tracking import Tracker, TrackerConfig


def _box(x, y=0.6, w=0.1, h=0.2):
    return BoundingBox(x, y, w, h)


def test_a_track_is_confirmed_after_cumulative_hits_even_with_a_miss_between():
    tracker = Tracker(TrackerConfig(min_hits_to_confirm=2))
    tracker.update([Detection(_box(0.1), 0.9, 1)], 0)
    assert tracker.tracks() == [], "one sighting is not a track"
    tracker.update([], 66)
    tracker.update([Detection(_box(0.12), 0.9, 1)], 133)
    tracks = tracker.tracks()
    assert len(tracks) == 1 and tracks[0].hits == 2 and tracks[0].confirmed


def test_one_object_keeps_one_id_across_frames_and_classes_never_merge():
    tracker = Tracker(TrackerConfig(min_hits_to_confirm=1))
    ids = set()
    for i in range(10):
        tracker.update([Detection(_box(0.1 + i * 0.02), 0.8, 1), Detection(_box(0.6, y=0.3), 0.8, 2)], i * 66)
        ids.update(t.id for t in tracker.tracks())
    assert ids == {1, 2}


def test_a_lost_track_coasts_then_ends_and_never_records_coasted_speed(pose):
    tracker = Tracker(TrackerConfig(min_hits_to_confirm=1, max_gap_millis=500), pose)
    tracker.update([Detection(_box(0.4, y=0.7), 0.9, 1)], 0)
    tracker.update([Detection(_box(0.42, y=0.7), 0.9, 1)], 100)
    history = len(tracker.tracks()[0].ground_history)
    update = tracker.update([], 400)
    assert update.ended == () and tracker.tracks()[0].coasting
    assert len(tracker.tracks()[0].ground_history) == history, "a coasted box is extrapolation, not a measurement"
    update = tracker.update([], 1000)
    assert update.ended == (1,) and tracker.tracks() == []


def test_speed_is_none_until_enough_span_and_then_measured(pose):
    tracker = Tracker(TrackerConfig(min_hits_to_confirm=1, min_motion_span_millis=1000), pose)
    for i in range(20):
        tracker.update([Detection(_box(0.3 + i * 0.01, y=0.7), 0.9, 1)], i * 100)
        track = tracker.tracks()[0]
        if i * 100 < 1000:
            assert track.speed_mps is None
    assert track.speed_mps is not None and track.speed_mps > 0
    assert track.position is not None and track.position.is_projected


def test_reset_ends_everything():
    tracker = Tracker(TrackerConfig(min_hits_to_confirm=1))
    tracker.update([Detection(_box(0.1), 0.9, 1), Detection(_box(0.5), 0.9, 1)], 0)
    assert sorted(tracker.reset()) == [1, 2] and tracker.tracks() == []
