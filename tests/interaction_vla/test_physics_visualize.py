from __future__ import annotations

import numpy as np
import pytest

import interaction_vla.physics_visualize as physics_visualize_module
from interaction_vla.chunked_controller import ChunkControllerDiagnostics
from interaction_vla.physics_recording import VIEW_LABELS, VIEW_TO_CAMERA
from interaction_vla.physics_visualize import (
    PhysicsVisualizationSession,
    build_parser,
    export_physics_gif,
    run_dashboard,
)


def test_parser_exposes_physical_dashboard_native_teleop_and_gif_commands() -> None:
    parser = build_parser()
    dashboard = parser.parse_args(
        "dashboard --controller expert --layout crowded --object-count 4 --seed 2140049".split()
    )
    teleop = parser.parse_args(
        "teleop --layout normal --object-count 3 --seed 2140049 --record demo.npz".split()
    )
    native = parser.parse_args(
        "native --controller graph --checkpoint graph.pt --layout crowded".split()
    )
    exporter = parser.parse_args(
        "export-gif --controller expert --output demo.gif".split()
    )
    comparison = parser.parse_args(
        "export-comparison-gif --flat-checkpoint flat.pt "
        "--graph-checkpoint graph.pt --output comparison.gif".split()
    )

    assert dashboard.command == "dashboard"
    assert teleop.command == "teleop" and teleop.record == "demo.npz"
    assert native.command == "native" and native.controller == "graph"
    assert exporter.command == "export-gif"
    assert comparison.command == "export-comparison-gif"


def test_four_panel_order_and_labels_match_reference_layout() -> None:
    assert tuple(VIEW_TO_CAMERA) == ("agent", "wrist", "side", "top")
    assert tuple(VIEW_LABELS.values()) == (
        "Agent View",
        "Wrist / Egocentric View",
        "Side View",
        "Top View",
    )


def test_expert_session_advances_one_20hz_physical_step() -> None:
    session = PhysicsVisualizationSession.create(
        config_path="configs/physics_smoke_macos.yaml",
        controller="expert",
        seed=11,
        object_count=2,
        layout_mode="normal",
        max_steps=3,
    )

    assert session.env.step_count == 0
    session.advance()
    assert session.env.step_count == 1
    assert session.last_action.shape == (7,)
    assert session.env.physics.policy_hz == 20
    assert "IK" in session.overlay()
    assert "IK scale" not in session.overlay()


def test_learned_session_projects_action_and_displays_scale(monkeypatch) -> None:
    session = PhysicsVisualizationSession.create(
        config_path="configs/physics_smoke_macos.yaml",
        controller="expert",
        seed=11,
        object_count=2,
        layout_mode="normal",
        max_steps=1,
    )
    session.controller_name = "graph"
    session.expert = None
    raw_action = np.asarray((1, 0, 0, 0, 0, 0, 1), dtype=np.float32)
    session._policy_action = lambda: raw_action.copy()  # type: ignore[method-assign]

    class Projection:
        action = np.asarray((0.25, 0, 0, 0, 0, 0, 1), dtype=np.float32)
        scale = 0.25

    monkeypatch.setattr(
        physics_visualize_module,
        "project_cartesian_action",
        lambda controller, action: Projection(),
        raising=False,
    )

    session.advance()

    np.testing.assert_array_equal(session.last_action, Projection.action)
    assert session.last_ik_projection_scale == 0.25
    assert "IK scale 0.25" in session.overlay()


def test_sequence_session_advances_only_through_shared_chunk_controller() -> None:
    session = PhysicsVisualizationSession.create(
        config_path="configs/physics_smoke_macos.yaml",
        controller="expert",
        seed=11,
        object_count=2,
        layout_mode="normal",
        max_steps=1,
    )
    session.controller_name = "graph"
    session.expert = None
    executed = np.asarray((0.1, 0, 0, 0, 0, 0, 1), dtype=np.float32)

    class FakeChunkController:
        horizon = 8

        def act(self, env):
            return executed.copy(), ChunkControllerDiagnostics(
                ensemble_size=3,
                raw_first_action=executed.copy(),
                aggregated_action=executed.copy(),
                raw_gripper_score=0.9,
                gripper_command=1.0,
                gripper_switch_count=1,
                smoothing_delta_norm=0.0,
                ik_projection_scale=0.5,
            )

    session.learned_controller = FakeChunkController()  # type: ignore[assignment]
    session._policy_action = lambda: (_ for _ in ()).throw(  # type: ignore[method-assign]
        AssertionError("v3 visualization must not use one-step inference")
    )

    session.advance()

    np.testing.assert_array_equal(session.last_action, executed)
    assert session.last_chunk_diagnostics is not None
    assert session.last_chunk_diagnostics.ensemble_size == 3
    assert "H8" in session.overlay()
    assert "ensemble 3" in session.overlay()


@pytest.mark.parametrize("controller", ["expert", "teleop"])
def test_nonlearned_sessions_do_not_use_ik_projection(
    monkeypatch,
    controller: str,
) -> None:
    session = PhysicsVisualizationSession.create(
        config_path="configs/physics_smoke_macos.yaml",
        controller=controller,
        seed=11,
        object_count=2,
        layout_mode="normal",
        max_steps=1,
    )
    monkeypatch.setattr(
        physics_visualize_module,
        "project_cartesian_action",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("expert and teleop must execute their original actions")
        ),
    )

    session.advance()

    assert session.last_ik_projection_scale == 1.0
    assert "IK scale" not in session.overlay()


def test_non_gif_output_is_rejected_before_recorder_creation(monkeypatch, tmp_path) -> None:
    def fail_create(*args, **kwargs):
        raise AssertionError("session must not be created for an invalid output suffix")

    monkeypatch.setattr(PhysicsVisualizationSession, "create", fail_create)
    with pytest.raises(ValueError, match=".gif"):
        export_physics_gif(
            config_path="configs/physics_smoke_macos.yaml",
            controller="expert",
            checkpoint=None,
            seed=11,
            object_count=2,
            layout_mode="normal",
            max_steps=3,
            fps=20,
            width=64,
            height=64,
            output=tmp_path / "bad.mp4",
        )


def test_dashboard_rejects_mjpython_before_glfw_window_creation(monkeypatch) -> None:
    monkeypatch.setattr(
        "interaction_vla.physics_visualize._macos_process_name", lambda: "mjpython"
    )
    monkeypatch.setattr(
        "interaction_vla.physics_visualize.glfw.init",
        lambda: (_ for _ in ()).throw(AssertionError("GLFW must not start")),
    )
    with pytest.raises(RuntimeError, match=".venv/bin/python"):
        run_dashboard(object())  # type: ignore[arg-type]
