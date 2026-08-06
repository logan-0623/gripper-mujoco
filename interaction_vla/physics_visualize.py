from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass, field
from pathlib import Path
import sys
import time

import glfw
import mujoco
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from .chunked_controller import ChunkControllerDiagnostics, ChunkedPolicyController
from .config import ExperimentConfig, load_config
from .env import TerminationReason
from .graph.builder import SceneGraphBuilder
from .models.encoders import SceneBatch
from .models.policy import ActionPolicy
from .physics_action_safety import project_cartesian_action
from .physics_env import FrankaContactEnv
from .physics_evaluate import (
    validate_physics_checkpoint,
    validate_temporal_checkpoint,
)
from .physics_expert import PhysicsScriptedExpert
from .physics_recording import (
    MultiViewFrame,
    MultiViewRecorder,
    VIEW_LABELS,
    VIEW_TO_CAMERA,
    compose_dashboard_frame,
)
from .teleop import TeleopController
from .train import (
    TrainingStatistics,
    build_sequence_provenance_fields,
    build_training_provenance,
    load_training_checkpoint,
    resolve_training_data,
)


def _macos_process_name() -> str:
    if sys.platform != "darwin":
        return Path(sys.executable).name
    libc = ctypes.CDLL(None)
    getprogname = libc.getprogname
    getprogname.restype = ctypes.c_char_p
    value = getprogname()
    return "" if value is None else value.decode(errors="replace")


def _make_env(config: ExperimentConfig, *, max_steps: int) -> FrankaContactEnv:
    return FrankaContactEnv(
        max_objects=config.max_objects,
        max_steps=max_steps,
        min_object_distance=config.environment.min_object_distance,
        workspace_low=config.environment.workspace_low,
        workspace_high=config.environment.workspace_high,
        crowded_anchor_min_distance=config.environment.crowded_anchor_min_distance,
        crowded_anchor_max_distance=config.environment.crowded_anchor_max_distance,
        physics=config.physics,
    )


@dataclass
class PhysicsVisualizationSession:
    config: ExperimentConfig
    env: FrankaContactEnv
    controller_name: str
    seed: int
    object_count: int
    layout_mode: str
    builder: SceneGraphBuilder
    expert: PhysicsScriptedExpert | None = None
    policy: ActionPolicy | None = None
    statistics: TrainingStatistics | None = None
    learned_controller: ChunkedPolicyController | None = None
    last_chunk_diagnostics: ChunkControllerDiagnostics | None = None
    teleop: TeleopController | None = None
    done: bool = False
    reason: TerminationReason = TerminationReason.RUNNING
    last_action: np.ndarray = field(
        default_factory=lambda: np.asarray((0, 0, 0, 0, 0, 0, 1), dtype=np.float32)
    )
    last_ik_projection_scale: float = 1.0
    last_info: dict[str, object] = field(default_factory=dict)
    executed_actions: list[np.ndarray] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        *,
        config_path: str | Path,
        controller: str,
        seed: int,
        object_count: int,
        layout_mode: str,
        max_steps: int,
        checkpoint: str | Path | None = None,
    ) -> "PhysicsVisualizationSession":
        if controller not in {"expert", "flat", "graph", "teleop"}:
            raise ValueError("controller must be expert, flat, graph, or teleop")
        if controller in {"flat", "graph"} and checkpoint is None:
            raise ValueError(f"{controller} controller requires --checkpoint")
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        config = load_config(config_path)
        if config.backend != "franka_contact":
            raise ValueError("physics visualization requires backend=franka_contact")
        env = _make_env(config, max_steps=max_steps)
        env.reset(seed=seed, object_count=object_count, layout_mode=layout_mode)
        expert = PhysicsScriptedExpert(config.physics) if controller == "expert" else None
        if expert is not None:
            expert.reset(seed=seed)
        policy = None
        statistics = None
        learned_controller = None
        builder = SceneGraphBuilder(
            max_objects=config.max_objects,
            feature_schema="physics_v2",
        )
        if checkpoint is not None:
            from .physics_data import expert_gate_provenance

            policy, statistics, payload = load_training_checkpoint(checkpoint, "cpu")
            physical_hashes = expert_gate_provenance(
                config_path,
                Path(config.output_dir) / "expert_gate.json",
            )
            validate_physics_checkpoint(
                payload,
                expected_provenance=physical_hashes,
            )
            validate_temporal_checkpoint(payload, config)
            if payload.get("representation") != controller:
                raise ValueError(
                    f"checkpoint representation is {payload.get('representation')!r}, "
                    f"not requested controller {controller!r}"
                )
            policy.eval()
            if config.sequence.enabled:
                data_selection = resolve_training_data(
                    config.data_dir,
                    split_seed=config.seed,
                    include_recovery=config.recovery.enabled,
                )
                expected_training_provenance, _ = build_training_provenance(
                    config,
                    data_selection,
                    expert_gate_hash=physical_hashes["expert_gate_hash"],
                )
                expected_training_provenance.update(
                    build_sequence_provenance_fields(
                        config,
                        data_selection,
                        model_seed=int(payload["model_seed"]),
                    )
                )
                if dict(payload.get("training_provenance", {})) != dict(
                    expected_training_provenance
                ):
                    raise ValueError(
                        "checkpoint training provenance does not match the current dataset"
                    )
                learned_controller = ChunkedPolicyController(
                    policy=policy,
                    statistics=statistics,
                    builder=builder,
                    horizon=config.sequence.horizon,
                    temporal_decay=config.sequence.temporal_decay,
                    gripper_close_threshold=(
                        config.sequence.gripper_close_threshold
                    ),
                    gripper_open_threshold=config.sequence.gripper_open_threshold,
                    device=config.sequence.rollout_device,
                    edge_shuffle=False,
                    edge_shuffle_seed=seed,
                )
                learned_controller.reset(env)
        return cls(
            config=config,
            env=env,
            controller_name=controller,
            seed=int(seed),
            object_count=int(object_count),
            layout_mode=layout_mode,
            builder=builder,
            expert=expert,
            policy=policy,
            statistics=statistics,
            learned_controller=learned_controller,
            teleop=TeleopController() if controller == "teleop" else None,
        )

    def reset(self) -> None:
        self.env.reset(
            seed=self.seed,
            object_count=self.object_count,
            layout_mode=self.layout_mode,
        )
        if self.expert is not None:
            self.expert.reset(seed=self.seed)
        if self.teleop is not None:
            self.teleop.clear_reset_request()
        if self.learned_controller is not None:
            self.learned_controller.reset(self.env)
        self.done = False
        self.reason = TerminationReason.RUNNING
        self.last_ik_projection_scale = 1.0
        self.last_chunk_diagnostics = None
        self.last_info = {}
        self.executed_actions.clear()

    def _policy_action(self) -> np.ndarray:
        if self.policy is None or self.statistics is None:
            raise RuntimeError("learned physical controller is not loaded")
        graph = self.builder.build(self.env.snapshot())
        scene = SceneBatch(
            node_features=torch.from_numpy(graph.node_features[None]).float(),
            edge_index=torch.from_numpy(graph.edge_index).long(),
            edge_features=torch.from_numpy(graph.edge_features[None]).float(),
            node_mask=torch.from_numpy(graph.node_mask[None]).bool(),
            edge_mask=torch.from_numpy(graph.edge_mask[None]).bool(),
        )
        scene = self.statistics.normalize_scene(scene)
        proprioception = self.statistics.normalize_proprioception(
            torch.from_numpy(self.env.proprioception()[None]).float()
        )
        with torch.no_grad():
            action = self.policy(scene, proprioception)[0].cpu().numpy()
        action[:6] = np.clip(action[:6], -1.0, 1.0)
        action[6] = np.clip(action[6], 0.0, 1.0)
        return action.astype(np.float32)

    def advance(self) -> None:
        if self.done:
            return
        snapshot = self.env.snapshot()
        if self.expert is not None:
            action = self.expert.act(
                snapshot, self.env.contact_diagnostics, self.env.grasp_state
            )
        elif self.teleop is not None:
            action = self.teleop.action()
        elif self.learned_controller is not None:
            action, self.last_chunk_diagnostics = self.learned_controller.act(
                self.env
            )
            self.last_ik_projection_scale = (
                self.last_chunk_diagnostics.ik_projection_scale
            )
        else:
            projection = project_cartesian_action(
                self.env.controller,
                self._policy_action(),
            )
            action = projection.action
            self.last_ik_projection_scale = projection.scale
        self.last_action = np.asarray(action, dtype=np.float32).copy()
        self.executed_actions.append(self.last_action.copy())
        transition = self.env.step(self.last_action)
        self.done = transition.done
        self.reason = transition.reason
        self.last_info = dict(transition.info)

    def overlay(self) -> str:
        phase = self.expert.phase.value if self.expert is not None else self.controller_name
        contact = self.env.contact_diagnostics
        stable = self.env.grasp_state.stable_object or "-"
        reason = self.reason.value if self.done else "running"
        ik_status = "limited" if self.last_info.get("ik_limited", False) else "ok"
        chunk_status = ""
        if self.last_chunk_diagnostics is not None:
            chunk_status = (
                f" | H{self.learned_controller.horizon}"
                f" ensemble {self.last_chunk_diagnostics.ensemble_size}"
                f" grip {self.last_chunk_diagnostics.gripper_command:.0f}"
            )
        placement = self.env.last_placement
        strict_status = (
            f" | target-bi {int(self.env.grasp_state.bilateral_object == self.env.target_name)}"
            f" stable {int(self.env.grasp_state.stable_object == self.env.target_name)}"
            f" margin ({placement.containment_margin[0]:+.3f},"
            f"{placement.containment_margin[1]:+.3f})"
            f" base/wall {int(placement.base_contact)}/{int(placement.wall_contact)}"
            f" strict {int(placement.strict_stable)}"
        )
        projection_status = (
            f" | IK scale {self.last_ik_projection_scale:.2f}{chunk_status}"
            if self.controller_name in {"flat", "graph"}
            else ""
        )
        return (
            f"{self.controller_name}/{phase} | step {self.env.step_count} | "
            f"target {self.env.target_name} | L {','.join(sorted(contact.left_objects)) or '-'} "
            f"R {','.join(sorted(contact.right_objects)) or '-'} | stable {stable} | "
            f"IK {ik_status}{projection_status}{strict_status} | {reason}"
        )


def _save_gif(frames: list[Image.Image], output: Path, *, fps: int) -> Path:
    if len(frames) < 2:
        raise ValueError("an animated GIF requires at least two frames")
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=round(1000 / fps),
        loop=0,
        optimize=False,
    )
    return output


def export_physics_gif(
    *,
    config_path: str | Path,
    controller: str,
    checkpoint: str | Path | None,
    seed: int,
    object_count: int,
    layout_mode: str,
    max_steps: int,
    fps: int,
    width: int,
    height: int,
    output: str | Path,
) -> Path:
    destination = Path(output)
    if destination.suffix.lower() != ".gif":
        raise ValueError("output path must use the .gif extension")
    if min(fps, width, height, max_steps) < 1:
        raise ValueError("fps, dimensions, and max_steps must be positive")
    session = PhysicsVisualizationSession.create(
        config_path=config_path,
        controller=controller,
        checkpoint=checkpoint,
        seed=seed,
        object_count=object_count,
        layout_mode=layout_mode,
        max_steps=max_steps,
    )
    recorder = MultiViewRecorder(session.env.model, width=width, height=height)
    progress = tqdm(
        total=max_steps,
        desc=f"GIF {controller}",
        unit="frame",
        dynamic_ncols=True,
    )
    try:
        frames = [compose_dashboard_frame(recorder.capture(session.env), overlay=session.overlay())]
        while not session.done:
            session.advance()
            progress.set_postfix(
                reason=session.reason.value,
                stable=int(
                    session.env.grasp_state.stable_object
                    == session.env.target_name
                ),
                strict=int(session.env.last_placement.strict_stable),
            )
            progress.update(1)
            frames.append(
                compose_dashboard_frame(recorder.capture(session.env), overlay=session.overlay())
            )
        return _save_gif(frames, destination, fps=fps)
    except Exception as error:
        if "CoreGraphics" in str(error) or "CGL" in type(error).__name__:
            raise RuntimeError(
                "MuJoCo rendering needs an active macOS graphical session; run this "
                "command from the normal Terminal with .venv/bin/mjpython."
            ) from error
        raise
    finally:
        progress.close()
        recorder.close()


def export_physics_comparison_gif(
    *,
    config_path: str | Path,
    flat_checkpoint: str | Path,
    graph_checkpoint: str | Path,
    seed: int,
    object_count: int,
    layout_mode: str,
    max_steps: int,
    fps: int,
    width: int,
    height: int,
    output: str | Path,
) -> Path:
    destination = Path(output)
    if destination.suffix.lower() != ".gif":
        raise ValueError("output path must use the .gif extension")
    sessions = [
        PhysicsVisualizationSession.create(
            config_path=config_path,
            controller=representation,
            checkpoint=checkpoint,
            seed=seed,
            object_count=object_count,
            layout_mode=layout_mode,
            max_steps=max_steps,
        )
        for representation, checkpoint in (
            ("flat", flat_checkpoint),
            ("graph", graph_checkpoint),
        )
    ]
    recorders = [
        MultiViewRecorder(session.env.model, width=width, height=height)
        for session in sessions
    ]

    def compose_pair() -> Image.Image:
        dashboards = [
            compose_dashboard_frame(recorder.capture(session.env), overlay=session.overlay())
            for recorder, session in zip(recorders, sessions, strict=True)
        ]
        pair = Image.new("RGB", (dashboards[0].width * 2, dashboards[0].height), "white")
        pair.paste(dashboards[0], (0, 0))
        pair.paste(dashboards[1], (dashboards[0].width, 0))
        return pair

    progress = tqdm(
        total=max_steps,
        desc="GIF flat-vs-graph",
        unit="frame",
        dynamic_ncols=True,
    )
    try:
        frames = [compose_pair()]
        while not all(session.done for session in sessions):
            for session in sessions:
                session.advance()
            progress.set_postfix(
                flat_reason=sessions[0].reason.value,
                graph_reason=sessions[1].reason.value,
                flat_stable=int(
                    sessions[0].env.grasp_state.stable_object
                    == sessions[0].env.target_name
                ),
                graph_stable=int(
                    sessions[1].env.grasp_state.stable_object
                    == sessions[1].env.target_name
                ),
                flat_strict=int(sessions[0].env.last_placement.strict_stable),
                graph_strict=int(sessions[1].env.last_placement.strict_stable),
            )
            progress.update(1)
            frames.append(compose_pair())
        return _save_gif(frames, destination, fps=fps)
    finally:
        progress.close()
        for recorder in recorders:
            recorder.close()


def _render_four_viewports(
    session: PhysicsVisualizationSession,
    window,
    *,
    scenes: dict[str, mujoco.MjvScene],
    cameras: dict[str, mujoco.MjvCamera],
    option: mujoco.MjvOption,
    context: mujoco.MjrContext,
) -> None:
    framebuffer_width, framebuffer_height = glfw.get_framebuffer_size(window)
    half_width = framebuffer_width // 2
    half_height = framebuffer_height // 2
    viewports = {
        "agent": mujoco.MjrRect(0, half_height, half_width, framebuffer_height - half_height),
        "wrist": mujoco.MjrRect(half_width, half_height, framebuffer_width - half_width, framebuffer_height - half_height),
        "side": mujoco.MjrRect(0, 0, half_width, half_height),
        "top": mujoco.MjrRect(half_width, 0, framebuffer_width - half_width, half_height),
    }
    for name in VIEW_TO_CAMERA:
        mujoco.mjv_updateScene(
            session.env.model,
            session.env.data,
            option,
            None,
            cameras[name],
            mujoco.mjtCatBit.mjCAT_ALL,
            scenes[name],
        )
        mujoco.mjr_render(viewports[name], scenes[name], context)
        diagnostics = session.overlay() if name == "agent" else ""
        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            viewports[name],
            VIEW_LABELS[name],
            diagnostics,
            context,
        )


def run_dashboard(
    session: PhysicsVisualizationSession,
    *,
    record: str | Path | None = None,
) -> Path | None:
    if _macos_process_name() == "mjpython":
        raise RuntimeError(
            "the custom GLFW dashboard must run on the macOS main thread; use "
            ".venv/bin/python -m interaction_vla.physics_visualize dashboard/teleop. "
            "Use mjpython only for the native subcommand."
        )
    if record is not None and Path(record).suffix.lower() != ".npz":
        raise ValueError("record path must use the .npz extension")
    if not glfw.init():
        raise RuntimeError("GLFW initialization failed")
    window = glfw.create_window(1024, 768, "Franka Contact · Agent/Wrist/Side/Top", None, None)
    if window is None:
        glfw.terminate()
        raise RuntimeError("GLFW window creation failed")
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    scenes = {name: mujoco.MjvScene(session.env.model, maxgeom=10000) for name in VIEW_TO_CAMERA}
    cameras: dict[str, mujoco.MjvCamera] = {}
    for name, camera_name in VIEW_TO_CAMERA.items():
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        camera.fixedcamid = session.env.model.camera(camera_name).id
        cameras[name] = camera
    option = mujoco.MjvOption()
    context = mujoco.MjrContext(session.env.model, mujoco.mjtFontScale.mjFONTSCALE_150)
    recorder = (
        MultiViewRecorder(session.env.model, width=256, height=256)
        if record is not None
        else None
    )
    frames: list[MultiViewFrame] = []
    if recorder is not None:
        frames.append(recorder.capture(session.env))
    if session.teleop is not None:
        glfw.set_key_callback(
            window,
            lambda _window, key, _scancode, action, _mods: session.teleop.handle_key(
                key, action
            ),
        )
    try:
        while not glfw.window_should_close(window):
            started = time.monotonic()
            glfw.poll_events()
            if session.teleop is not None and session.teleop.quit:
                break
            if session.teleop is not None and session.teleop.discard_and_reset:
                frames.clear()
                session.reset()
                if recorder is not None:
                    frames.append(recorder.capture(session.env))
            if not session.done:
                session.advance()
                if recorder is not None:
                    frames.append(recorder.capture(session.env))
            _render_four_viewports(
                session,
                window,
                scenes=scenes,
                cameras=cameras,
                option=option,
                context=context,
            )
            glfw.swap_buffers(window)
            if session.done and session.teleop is None:
                time.sleep(0.75)
                break
            time.sleep(max(0.0, 1.0 / session.env.physics.policy_hz - (time.monotonic() - started)))
    finally:
        if recorder is not None:
            recorder.close()
        context.free()
        glfw.destroy_window(window)
        glfw.terminate()
    if record is not None:
        if not frames:
            raise RuntimeError("teleop recording contains no retained frames")
        saving_recorder = MultiViewRecorder(session.env.model, width=256, height=256)
        try:
            return saving_recorder.save_episode(frames, record)
        finally:
            saving_recorder.close()
    return None


def run_native(session: PhysicsVisualizationSession) -> None:
    try:
        import mujoco.viewer

        with mujoco.viewer.launch_passive(session.env.model, session.env.data) as viewer:
            while viewer.is_running() and not session.done:
                started = time.monotonic()
                session.advance()
                viewer.sync()
                time.sleep(
                    max(
                        0.0,
                        1.0 / session.env.physics.policy_hz
                        - (time.monotonic() - started),
                    )
                )
            if viewer.is_running():
                time.sleep(0.75)
    except Exception as error:
        if sys.platform == "darwin":
            raise RuntimeError(
                "MuJoCo native viewer launch failed. Run exactly: .venv/bin/mjpython "
                "-m interaction_vla.physics_visualize native --controller expert "
                "--layout crowded --object-count 4 --seed 2140049"
            ) from error
        raise


def _common_arguments(parser: argparse.ArgumentParser, *, controller: bool = True) -> None:
    if controller:
        parser.add_argument(
            "--controller", choices=("expert", "flat", "graph"), required=True
        )
        parser.add_argument("--checkpoint")
    parser.add_argument("--config", default="configs/physics_smoke_macos.yaml")
    parser.add_argument("--layout", choices=("normal", "crowded"), default="crowded")
    parser.add_argument("--object-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2_140_049)
    parser.add_argument("--max-steps", type=int, default=180)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize the real Franka contact backend")
    commands = parser.add_subparsers(dest="command", required=True)
    dashboard = commands.add_parser("dashboard", help="four synchronized camera views")
    _common_arguments(dashboard)
    native = commands.add_parser("native", help="native interactive MuJoCo viewer")
    _common_arguments(native)
    teleop = commands.add_parser("teleop", help="four-view 7D keyboard teleoperation")
    _common_arguments(teleop, controller=False)
    teleop.add_argument("--record")
    exporter = commands.add_parser("export-gif", help="export a four-view physical GIF")
    _common_arguments(exporter)
    exporter.add_argument("--fps", type=int, default=20)
    exporter.add_argument("--width", type=int, default=256)
    exporter.add_argument("--height", type=int, default=256)
    exporter.add_argument("--output", default="docs/media/franka_contact_expert.gif")
    exporter.add_argument("--flat-checkpoint")
    exporter.add_argument("--graph-checkpoint")
    comparison = commands.add_parser(
        "export-comparison-gif",
        help="export synchronized Flat/Graph four-view GIFs",
    )
    _common_arguments(comparison, controller=False)
    comparison.add_argument("--fps", type=int, default=20)
    comparison.add_argument("--width", type=int, default=256)
    comparison.add_argument("--height", type=int, default=256)
    comparison.add_argument(
        "--output",
        default="docs/media/franka_contact_flat_vs_graph.gif",
    )
    comparison.add_argument("--flat-checkpoint", required=True)
    comparison.add_argument("--graph-checkpoint", required=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "teleop":
            session = PhysicsVisualizationSession.create(
                config_path=args.config,
                controller="teleop",
                seed=args.seed,
                object_count=args.object_count,
                layout_mode=args.layout,
                max_steps=args.max_steps,
            )
            output = run_dashboard(session, record=args.record)
            if output is not None:
                print(output)
        elif args.command in {"dashboard", "native"}:
            session = PhysicsVisualizationSession.create(
                config_path=args.config,
                controller=args.controller,
                checkpoint=args.checkpoint,
                seed=args.seed,
                object_count=args.object_count,
                layout_mode=args.layout,
                max_steps=args.max_steps,
            )
            if args.command == "dashboard":
                run_dashboard(session)
            else:
                run_native(session)
        elif args.flat_checkpoint or args.graph_checkpoint:
            if not (args.flat_checkpoint and args.graph_checkpoint):
                raise ValueError("comparison GIF requires both checkpoint paths")
            print(
                export_physics_comparison_gif(
                    config_path=args.config,
                    flat_checkpoint=args.flat_checkpoint,
                    graph_checkpoint=args.graph_checkpoint,
                    seed=args.seed,
                    object_count=args.object_count,
                    layout_mode=args.layout,
                    max_steps=args.max_steps,
                    fps=args.fps,
                    width=args.width,
                    height=args.height,
                    output=args.output,
                )
            )
        else:
            print(
                export_physics_gif(
                    config_path=args.config,
                    controller=args.controller,
                    checkpoint=args.checkpoint,
                    seed=args.seed,
                    object_count=args.object_count,
                    layout_mode=args.layout,
                    max_steps=args.max_steps,
                    fps=args.fps,
                    width=args.width,
                    height=args.height,
                    output=args.output,
                )
            )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
