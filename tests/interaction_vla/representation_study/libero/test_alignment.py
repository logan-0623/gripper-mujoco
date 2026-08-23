import numpy as np
import pytest

from interaction_vla.representation_study.libero.alignment import (
    EpisodeDescriptor,
    align_episode_sources,
)


def _episode(kind: str, episode_id: str, offset: float = 0.0) -> EpisodeDescriptor:
    actions = np.arange(21, dtype=np.float64).reshape(3, 7) / 20.0 + offset
    return EpisodeDescriptor(
        source_kind=kind,
        suite="libero_spatial",
        task_id=0,
        episode_id=episode_id,
        actions=actions,
        robot_states=np.zeros((3, 8), dtype=np.float64) if kind == "lerobot" else None,
        relative_path=f"task/{episode_id}.hdf5" if kind == "raw" else None,
        demo_key=episode_id if kind == "raw" else None,
        model_xml_sha256="a" * 64 if kind == "raw" else None,
    )


def test_episode_alignment_is_unique_and_content_bound() -> None:
    raw = (_episode("raw", "demo_a"), _episode("raw", "demo_b", 1.0))
    lerobot = (
        _episode("lerobot", "episode_7", 1.0),
        _episode("lerobot", "episode_2"),
    )
    manifest = align_episode_sources(raw, lerobot, action_atol=1e-8)
    assert [(row.raw_episode_id, row.lerobot_episode_id) for row in manifest.rows] == [
        ("demo_a", "episode_2"),
        ("demo_b", "episode_7"),
    ]
    assert len(manifest.semantic_sha256) == 64


def test_episode_alignment_rejects_ambiguous_action_sequences() -> None:
    raw = (_episode("raw", "demo_a"),)
    lerobot = (
        _episode("lerobot", "episode_1"),
        _episode("lerobot", "episode_2"),
    )
    with pytest.raises(ValueError, match="ambiguous"):
        align_episode_sources(raw, lerobot, action_atol=1e-8)


def test_episode_alignment_rejects_unmatched_episode() -> None:
    with pytest.raises(ValueError, match="no matching"):
        align_episode_sources(
            (_episode("raw", "demo_a"),),
            (_episode("lerobot", "episode_1", 0.1),),
            action_atol=1e-8,
        )
