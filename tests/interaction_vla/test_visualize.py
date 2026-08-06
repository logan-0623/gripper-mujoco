from __future__ import annotations

import os

import numpy as np
import pytest
import torch
import mujoco
from PIL import Image

from interaction_vla.data import collect_episode, save_episode
from interaction_vla.env import KinematicTabletopEnv
from interaction_vla.expert import ScriptedExpert
from interaction_vla.models.policy import build_action_policy
from interaction_vla.train import TrainingStatistics
from interaction_vla.visualize import (
    ANCHOR_RGBA,
    TARGET_RGBA,
    VisualizationSession,
    apply_semantic_colors,
    build_parser,
    compose_comparison_frame,
    export_comparison_gif,
    render_rgb,
    run_native_viewer,
    save_animated_gif,
)


def object_positions(session: VisualizationSession) -> np.ndarray:
    return np.stack([entity.position for entity in session.snapshot.objects])


def tiny_checkpoint(tmp_path, representation: str):
    episode = collect_episode(
        KinematicTabletopEnv(max_objects=5),
        ScriptedExpert(),
        seed=9,
        object_count=2,
    )
    episode_path = save_episode(episode, tmp_path / "episode.npz")
    statistics = TrainingStatistics.fit((episode_path,))
    model_kwargs = {
        "max_nodes": 8,
        "max_edges": 56,
        "node_feature_dim": 23,
        "edge_feature_dim": 10,
        "graph_hidden_dim": 16,
        "embedding_dim": 16,
        "policy_hidden_dim": 16,
        "message_rounds": 1,
    }
    policy = build_action_policy(representation=representation, **model_kwargs)
    checkpoint = tmp_path / f"{representation}.pt"
    torch.save(
        {
            "representation": representation,
            "model_kwargs": model_kwargs,
            "model_state": policy.state_dict(),
            "statistics": statistics.checkpoint_state(),
        },
        checkpoint,
    )
    return checkpoint


def test_expert_visualization_session_is_deterministic() -> None:
    first = VisualizationSession.create(
        controller="expert",
        seed=71,
        object_count=5,
        layout_mode="crowded",
        max_steps=5,
    )
    second = VisualizationSession.create(
        controller="expert",
        seed=71,
        object_count=5,
        layout_mode="crowded",
        max_steps=5,
    )

    np.testing.assert_array_equal(first.snapshot.gripper.position, second.snapshot.gripper.position)
    np.testing.assert_array_equal(object_positions(first), object_positions(second))

    first.advance()
    second.advance()

    np.testing.assert_array_equal(first.snapshot.gripper.position, second.snapshot.gripper.position)
    np.testing.assert_array_equal(object_positions(first), object_positions(second))


def test_learned_visualization_session_requires_a_checkpoint() -> None:
    with pytest.raises(ValueError, match="checkpoint"):
        VisualizationSession.create(
            controller="graph",
            seed=71,
            object_count=5,
            layout_mode="crowded",
            max_steps=5,
        )


def test_learned_visualization_session_loads_and_validates_representation(tmp_path) -> None:
    checkpoint = tiny_checkpoint(tmp_path, "graph")
    session = VisualizationSession.create(
        controller="graph",
        checkpoint=checkpoint,
        seed=71,
        object_count=5,
        layout_mode="crowded",
        max_steps=5,
    )

    session.advance()

    assert session.env.backend.step_count == 1
    assert session.policy is not None
    assert session.statistics is not None

    with pytest.raises(ValueError, match="representation"):
        VisualizationSession.create(
            controller="flat",
            checkpoint=checkpoint,
            seed=71,
            object_count=5,
            layout_mode="crowded",
            max_steps=5,
        )


def test_expert_recovery_injection_gets_its_own_visual_frame() -> None:
    session = VisualizationSession.create(
        controller="expert",
        seed=42,
        object_count=2,
        layout_mode="normal",
        max_steps=120,
        recovery_kind="align_offset",
    )

    for _ in range(120):
        step_count_before = session.env.backend.step_count
        session.advance()
        if session.injected:
            break

    assert session.injected
    assert session.env.backend.step_count == step_count_before

    session.advance()

    assert session.env.backend.step_count == step_count_before + 1


def test_learned_controller_rejects_recovery_injection(tmp_path) -> None:
    checkpoint = tiny_checkpoint(tmp_path, "graph")

    with pytest.raises(ValueError, match="recovery"):
        VisualizationSession.create(
            controller="graph",
            checkpoint=checkpoint,
            seed=42,
            object_count=2,
            layout_mode="normal",
            max_steps=120,
            recovery_kind="align_offset",
        )


def test_semantic_colors_mark_target_and_nearest_distractor() -> None:
    session = VisualizationSession.create(
        controller="expert",
        seed=71,
        object_count=5,
        layout_mode="crowded",
        max_steps=5,
    )

    assignment = apply_semantic_colors(session.env, session.snapshot)

    target = session.snapshot.target_object
    expected_anchor = min(
        (entity for entity in session.snapshot.objects if entity.name != target.name),
        key=lambda entity: np.linalg.norm(entity.position[:2] - target.position[:2]),
    )
    assert assignment.target_name == target.name
    assert assignment.anchor_name == expected_anchor.name
    np.testing.assert_allclose(
        session.env.model.geom(f"{assignment.target_name}_geom").rgba,
        TARGET_RGBA,
    )
    np.testing.assert_allclose(
        session.env.model.geom(f"{assignment.anchor_name}_geom").rgba,
        ANCHOR_RGBA,
    )


@pytest.mark.skipif(
    bool(os.environ.get("CODEX_SANDBOX")),
    reason="macOS CoreGraphics is unavailable inside the Codex sandbox",
)
def test_offscreen_renderer_returns_rgb_frame() -> None:
    session = VisualizationSession.create(
        controller="expert",
        seed=71,
        object_count=5,
        layout_mode="crowded",
        max_steps=5,
    )
    apply_semantic_colors(session.env, session.snapshot)
    renderer = mujoco.Renderer(session.env.model, height=64, width=80)
    try:
        frame = render_rgb(session.env, renderer)
    finally:
        renderer.close()

    assert frame.shape == (64, 80, 3)
    assert frame.dtype == np.uint8
    assert frame.max() > frame.min()


def test_native_viewer_validates_fps_before_opening_a_window() -> None:
    session = VisualizationSession.create(
        controller="expert",
        seed=71,
        object_count=2,
        layout_mode="normal",
        max_steps=5,
    )

    with pytest.raises(ValueError, match="fps"):
        run_native_viewer(session, fps=0)


def test_comparison_frame_places_two_panels_under_labels() -> None:
    left = np.zeros((32, 48, 3), dtype=np.uint8)
    right = np.full((32, 48, 3), 255, dtype=np.uint8)

    combined = compose_comparison_frame(
        left,
        right,
        left_label="Flat · step 0",
        right_label="Graph · step 0",
    )

    assert combined.size == (96, 56)
    np.testing.assert_array_equal(np.asarray(combined)[24:, :48], left)
    np.testing.assert_array_equal(np.asarray(combined)[24:, 48:], right)


def test_animated_gif_contains_every_frame(tmp_path) -> None:
    left = np.zeros((32, 48, 3), dtype=np.uint8)
    right = np.full((32, 48, 3), 255, dtype=np.uint8)
    first = compose_comparison_frame(left, right, left_label="Flat", right_label="Graph")
    second = compose_comparison_frame(right, left, left_label="Flat", right_label="Graph")
    path = tmp_path / "comparison.gif"

    save_animated_gif((first, second), path, fps=5)

    with Image.open(path) as image:
        assert image.n_frames == 2
        assert image.size == first.size


def test_comparison_export_rejects_non_gif_output_before_loading_checkpoints(tmp_path) -> None:
    with pytest.raises(ValueError, match="gif"):
        export_comparison_gif(
            flat_checkpoint=tmp_path / "missing_flat.pt",
            graph_checkpoint=tmp_path / "missing_graph.pt",
            seed=71,
            object_count=5,
            layout_mode="crowded",
            max_steps=1,
            fps=5,
            width=64,
            height=64,
            output=tmp_path / "comparison.mp4",
        )


@pytest.mark.skipif(
    bool(os.environ.get("CODEX_SANDBOX")),
    reason="macOS CoreGraphics is unavailable inside the Codex sandbox",
)
def test_real_checkpoint_comparison_exports_multiple_frames(tmp_path) -> None:
    flat = tiny_checkpoint(tmp_path, "flat")
    graph = tiny_checkpoint(tmp_path, "graph")

    output = export_comparison_gif(
        flat_checkpoint=flat,
        graph_checkpoint=graph,
        seed=71,
        object_count=5,
        layout_mode="crowded",
        max_steps=1,
        fps=5,
        width=64,
        height=64,
        output=tmp_path / "comparison.gif",
    )

    with Image.open(output) as image:
        assert image.n_frames >= 2
        assert image.size == (128, 88)


def test_visualization_cli_has_viewer_and_gif_subcommands() -> None:
    parser = build_parser()

    viewer = parser.parse_args(["viewer", "--controller", "expert"])
    exporter = parser.parse_args(
        [
            "export-gif",
            "--flat-checkpoint",
            "flat.pt",
            "--graph-checkpoint",
            "graph.pt",
        ]
    )

    assert viewer.command == "viewer"
    assert viewer.layout == "crowded"
    assert viewer.seed == 2_140_042
    assert exporter.command == "export-gif"
    assert exporter.output == "docs/media/flat_vs_graph_crowded.gif"
