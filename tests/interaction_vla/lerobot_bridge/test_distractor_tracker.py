from interaction_vla.lerobot_bridge.teacher_schema import DistractorTracker


def test_challenger_needs_point_one_margin_for_three_frames() -> None:
    tracker = DistractorTracker(
        count=2,
        replacement_margin=0.10,
        replacement_frames=3,
        dropout_frames=3,
    )
    assert tracker.update({"a": 0.8, "b": 0.6, "c": 0.0}) == ("a", "b")
    assert tracker.update({"a": 0.8, "b": 0.6, "c": 0.71}) == ("a", "b")
    assert tracker.update({"a": 0.8, "b": 0.6, "c": 0.71}) == ("a", "b")
    assert tracker.update({"a": 0.8, "b": 0.6, "c": 0.71}) == ("a", "c")


def test_retained_track_survives_three_missing_frames_then_expires() -> None:
    tracker = DistractorTracker(
        count=2,
        replacement_margin=0.10,
        replacement_frames=3,
        dropout_frames=3,
    )
    tracker.update({"a": 0.8, "b": 0.6})

    assert tracker.update({"a": 0.8}) == ("a", "b")
    assert tracker.update({"a": 0.8}) == ("a", "b")
    assert tracker.update({"a": 0.8}) == ("a", "b")
    assert tracker.update({"a": 0.8}) == ("a", None)


def test_ties_are_resolved_by_stable_track_name() -> None:
    tracker = DistractorTracker(
        count=2,
        replacement_margin=0.10,
        replacement_frames=3,
        dropout_frames=3,
    )
    assert tracker.update({"c": 0.5, "a": 0.5, "b": 0.5}) == ("a", "b")
