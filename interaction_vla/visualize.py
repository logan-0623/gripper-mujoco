from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time

import numpy as np
import torch
import mujoco
from PIL import Image, ImageDraw

from .env import TerminationReason
from .expert import ScriptedExpert
from .graph.builder import SceneGraphBuilder
from .graph.schema import SceneSnapshot
from .models.encoders import scene_graphs_to_batch
from .models.policy import ActionPolicy
from .mujoco_env import MujocoTabletopEnv
from .recovery import (
    PerturbationKind,
    RecoverySpec,
    apply_recovery_spec,
    make_recovery_spec,
)
from .train import TrainingStatistics, load_training_checkpoint


TARGET_RGBA = np.asarray((0.10, 0.80, 0.25, 1.0), dtype=np.float32)
ANCHOR_RGBA = np.asarray((1.00, 0.50, 0.05, 1.0), dtype=np.float32)
OTHER_RGBA = np.asarray((0.20, 0.40, 0.90, 1.0), dtype=np.float32)
INACTIVE_RGBA = np.asarray((0.35, 0.35, 0.35, 0.25), dtype=np.float32)


@dataclass(frozen=True)
class SemanticColorAssignment:
    target_name: str
    anchor_name: str


def apply_semantic_colors(
    env: MujocoTabletopEnv,
    snapshot: SceneSnapshot,
) -> SemanticColorAssignment:
    target = snapshot.target_object
    anchor = min(
        (entity for entity in snapshot.objects if entity.name != target.name),
        key=lambda entity: np.linalg.norm(entity.position[:2] - target.position[:2]),
    )
    active_names = {entity.name for entity in snapshot.objects}
    for index in range(env.max_objects):
        name = f"object_{index}"
        color = INACTIVE_RGBA
        if name in active_names:
            color = TARGET_RGBA if name == target.name else OTHER_RGBA
        if name == anchor.name:
            color = ANCHOR_RGBA
        env.model.geom(f"{name}_geom").rgba[:] = color
    env.model.geom("gripper_geom").rgba[:] = (0.12, 0.12, 0.12, 1.0)
    env.model.geom("receptacle").rgba[:] = (0.10, 0.70, 0.75, 0.65)
    return SemanticColorAssignment(target_name=target.name, anchor_name=anchor.name)


def render_rgb(
    env: MujocoTabletopEnv,
    renderer: mujoco.Renderer,
    *,
    camera: str = "agentview",
) -> np.ndarray:
    renderer.update_scene(env.data, camera=camera)
    return np.asarray(renderer.render(), dtype=np.uint8).copy()


def run_native_viewer(session: "VisualizationSession", *, fps: int) -> None:
    if fps < 1:
        raise ValueError("fps must be positive")
    apply_semantic_colors(session.env, session.snapshot)
    try:
        import mujoco.viewer

        with mujoco.viewer.launch_passive(session.env.model, session.env.data) as viewer:
            viewer.cam.lookat[:] = (0.0, 0.0, 0.08)
            viewer.cam.distance = 1.1
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -35
            viewer.sync()
            while viewer.is_running() and not session.done:
                started = time.monotonic()
                session.advance()
                viewer.sync()
                time.sleep(max(0.0, 1.0 / fps - (time.monotonic() - started)))
            if viewer.is_running():
                time.sleep(1.0)
    except Exception as error:
        if sys.platform == "darwin":
            raise RuntimeError(
                "MuJoCo viewer launch failed. On macOS run this command through "
                ".venv/bin/mjpython -m interaction_vla.visualize viewer ..."
            ) from error
        raise


def compose_comparison_frame(
    left: np.ndarray,
    right: np.ndarray,
    *,
    left_label: str,
    right_label: str,
) -> Image.Image:
    left_array = np.asarray(left, dtype=np.uint8)
    right_array = np.asarray(right, dtype=np.uint8)
    if left_array.shape != right_array.shape or left_array.ndim != 3:
        raise ValueError("comparison panels must have identical HxWxC shapes")
    if left_array.shape[2] != 3:
        raise ValueError("comparison panels must be RGB")
    height, width, _ = left_array.shape
    combined = Image.new("RGB", (width * 2, height + 24), color="white")
    combined.paste(Image.fromarray(left_array, mode="RGB"), (0, 24))
    combined.paste(Image.fromarray(right_array, mode="RGB"), (width, 24))
    draw = ImageDraw.Draw(combined)
    draw.text((5, 5), left_label, fill="black")
    draw.text((width + 5, 5), right_label, fill="black")
    return combined


def save_animated_gif(
    frames: tuple[Image.Image, ...] | list[Image.Image],
    output: str | Path,
    *,
    fps: int,
) -> Path:
    values = tuple(frame.convert("RGB") for frame in frames)
    if len(values) < 2:
        raise ValueError("an animated GIF requires at least two frames")
    if fps < 1:
        raise ValueError("fps must be positive")
    destination = Path(output)
    if destination.suffix.lower() != ".gif":
        raise ValueError("output path must use the .gif extension")
    destination.parent.mkdir(parents=True, exist_ok=True)
    values[0].save(
        destination,
        save_all=True,
        append_images=list(values[1:]),
        duration=round(1000 / fps),
        loop=0,
        optimize=False,
    )
    return destination


def _session_label(display_name: str, session: "VisualizationSession") -> str:
    status = session.reason.value if session.done else "running"
    return f"{display_name} · step {session.env.backend.step_count} · {status}"


def export_comparison_gif(
    *,
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
    if min(max_steps, fps, width, height) < 1:
        raise ValueError("max_steps, fps, width, and height must be positive")

    flat = VisualizationSession.create(
        controller="flat",
        checkpoint=flat_checkpoint,
        seed=seed,
        object_count=object_count,
        layout_mode=layout_mode,
        max_steps=max_steps,
    )
    graph = VisualizationSession.create(
        controller="graph",
        checkpoint=graph_checkpoint,
        seed=seed,
        object_count=object_count,
        layout_mode=layout_mode,
        max_steps=max_steps,
    )
    apply_semantic_colors(flat.env, flat.snapshot)
    apply_semantic_colors(graph.env, graph.snapshot)

    flat_renderer = None
    graph_renderer = None
    try:
        flat_renderer = mujoco.Renderer(flat.env.model, height=height, width=width)
        graph_renderer = mujoco.Renderer(graph.env.model, height=height, width=width)
        frames = [
            compose_comparison_frame(
                render_rgb(flat.env, flat_renderer),
                render_rgb(graph.env, graph_renderer),
                left_label=_session_label("Flat", flat),
                right_label=_session_label("Graph", graph),
            )
        ]
        while not (flat.done and graph.done):
            flat.advance()
            graph.advance()
            frames.append(
                compose_comparison_frame(
                    render_rgb(flat.env, flat_renderer),
                    render_rgb(graph.env, graph_renderer),
                    left_label=_session_label("Flat", flat),
                    right_label=_session_label("Graph", graph),
                )
            )
        return save_animated_gif(frames, destination, fps=fps)
    except Exception as error:
        if "CoreGraphics" in str(error) or "CGL" in type(error).__name__:
            raise RuntimeError(
                "MuJoCo offscreen rendering needs an active macOS graphical session. "
                "Run the export command from your normal Terminal application."
            ) from error
        raise
    finally:
        if flat_renderer is not None:
            flat_renderer.close()
        if graph_renderer is not None:
            graph_renderer.close()


@dataclass
class VisualizationSession:
    env: MujocoTabletopEnv
    snapshot: SceneSnapshot
    controller_name: str
    max_steps: int
    expert: ScriptedExpert | None
    policy: ActionPolicy | None
    statistics: TrainingStatistics | None
    builder: SceneGraphBuilder
    done: bool = False
    reason: TerminationReason = TerminationReason.RUNNING
    injected: bool = False
    recovery_spec: RecoverySpec | None = None

    @classmethod
    def create(
        cls,
        *,
        controller: str,
        seed: int,
        object_count: int,
        layout_mode: str,
        max_steps: int,
        checkpoint: str | Path | None = None,
        recovery_kind: str | None = None,
    ) -> "VisualizationSession":
        if controller not in {"expert", "flat", "graph"}:
            raise ValueError("controller must be one of: expert, flat, graph")
        if controller != "expert" and checkpoint is None:
            raise ValueError(f"{controller} controller requires a checkpoint")
        if controller != "expert" and recovery_kind is not None:
            raise ValueError("recovery visualization is supported only for the expert controller")
        if max_steps < 1:
            raise ValueError("max_steps must be positive")

        env = MujocoTabletopEnv(max_objects=5, max_steps=max_steps)
        snapshot = env.reset(
            seed=seed,
            object_count=object_count,
            layout_mode=layout_mode,
        )
        expert = ScriptedExpert() if controller == "expert" else None
        recovery_spec = None
        if recovery_kind is not None:
            requested_kind = PerturbationKind(recovery_kind)
            recovery_spec = next(
                spec
                for variant_id in range(4)
                if (spec := make_recovery_spec(seed, variant_id)).kind is requested_kind
            )
        policy = None
        statistics = None
        if checkpoint is not None:
            policy, statistics, payload = load_training_checkpoint(checkpoint, "cpu")
            representation = str(payload["representation"])
            if representation != controller:
                raise ValueError(
                    f"checkpoint representation is {representation!r}, "
                    f"not requested controller {controller!r}"
                )
            policy.eval()
        return cls(
            env=env,
            snapshot=snapshot,
            controller_name=controller,
            max_steps=max_steps,
            expert=expert,
            policy=policy,
            statistics=statistics,
            builder=SceneGraphBuilder(max_objects=5),
            recovery_spec=recovery_spec,
        )

    def advance(self) -> None:
        if self.done:
            return
        if self.expert is not None:
            action = self.expert.act(self.snapshot)
            if (
                self.recovery_spec is not None
                and not self.injected
                and self.expert.phase is self.recovery_spec.injection_phase
            ):
                self.snapshot = apply_recovery_spec(self.env, self.recovery_spec)
                self.injected = True
                return
        else:
            if self.policy is None or self.statistics is None:
                raise RuntimeError("learned controller is not loaded")
            scene = scene_graphs_to_batch((self.builder.build(self.snapshot),))
            scene = self.statistics.normalize_scene(scene)
            proprioception = torch.from_numpy(self.env.proprioception()).float().unsqueeze(0)
            proprioception = self.statistics.normalize_proprioception(proprioception)
            with torch.no_grad():
                action = self.policy(scene, proprioception)[0].cpu().numpy()
        transition = self.env.step(action)
        self.snapshot = transition.snapshot
        self.done = transition.done
        self.reason = transition.reason


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize deterministic interaction-Graph rollouts in MuJoCo"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    viewer = subparsers.add_parser("viewer", help="open the native MuJoCo viewer")
    viewer.add_argument(
        "--controller", choices=("expert", "flat", "graph"), required=True
    )
    viewer.add_argument("--checkpoint")
    viewer.add_argument("--layout", choices=("normal", "crowded"), default="crowded")
    viewer.add_argument("--object-count", type=int, default=5)
    viewer.add_argument("--seed", type=int, default=2_140_042)
    viewer.add_argument("--max-steps", type=int, default=120)
    viewer.add_argument("--fps", type=int, default=12)
    viewer.add_argument(
        "--recovery-kind",
        choices=tuple(kind.value for kind in PerturbationKind),
    )

    exporter = subparsers.add_parser(
        "export-gif", help="export paired Flat and Graph rollouts"
    )
    exporter.add_argument("--flat-checkpoint", required=True)
    exporter.add_argument("--graph-checkpoint", required=True)
    exporter.add_argument(
        "--layout", choices=("normal", "crowded"), default="crowded"
    )
    exporter.add_argument("--object-count", type=int, default=5)
    exporter.add_argument("--seed", type=int, default=2_140_042)
    exporter.add_argument("--max-steps", type=int, default=120)
    exporter.add_argument("--fps", type=int, default=12)
    exporter.add_argument("--width", type=int, default=320)
    exporter.add_argument("--height", type=int, default=240)
    exporter.add_argument(
        "--output", default="docs/media/flat_vs_graph_crowded.gif"
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "viewer":
            session = VisualizationSession.create(
                controller=args.controller,
                checkpoint=args.checkpoint,
                seed=args.seed,
                object_count=args.object_count,
                layout_mode=args.layout,
                max_steps=args.max_steps,
                recovery_kind=args.recovery_kind,
            )
            run_native_viewer(session, fps=args.fps)
        else:
            output = export_comparison_gif(
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
            print(output)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
