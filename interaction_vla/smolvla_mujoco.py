"""Minimal SmolVLA-to-Franka MuJoCo runtime adapter.

This is a runtime smoke interface, not a LIBERO success-rate evaluator.  The
checkpoint must still be trained for the scene/task distribution being used.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .franka_controller import so3_log


def smolvla_state(env: Any) -> np.ndarray:
    """Map Franka state to LIBERO's [position, axis-angle, gripper qpos] format."""
    values = np.asarray(env.proprioception(), dtype=np.float32)
    if values.shape != (23,) or not np.isfinite(values).all():
        raise ValueError("Franka proprioception must be finite with shape (23,)")
    _, rotation = env.controller.tcp_pose()
    # LIBERO stores mirrored finger positions; the local Panda model uses positive qpos for both.
    fingers = values[13:15]
    result = np.concatenate(
        (values[:3], so3_log(rotation), (fingers[0], -fingers[1]))
    ).astype(np.float32)
    assert result.shape == (8,)
    return result


def _image_tensor(rgb: np.ndarray, *, device: torch.device) -> torch.Tensor:
    values = np.asarray(rgb)
    if values.ndim != 3 or values.shape[2] != 3 or values.dtype != np.uint8:
        raise ValueError("camera RGB must be uint8 HxWx3")
    if values.shape[:2] != (256, 256):
        raise ValueError("SmolVLA runtime requires 256x256 camera frames")
    return torch.from_numpy(values).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)


def observation_batch(
    *, image: np.ndarray, image2: np.ndarray, state: np.ndarray, task: str, device: torch.device
) -> dict[str, object]:
    state_values = np.asarray(state, dtype=np.float32)
    if state_values.shape != (8,) or not np.isfinite(state_values).all():
        raise ValueError("SmolVLA state must be finite with shape (8,)")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task must be a non-empty language instruction")
    # Match LeRobot's LIBERO observation processor, which rotates both camera frames 180 degrees.
    image = np.flip(np.asarray(image), axis=(0, 1)).copy()
    image2 = np.flip(np.asarray(image2), axis=(0, 1)).copy()
    return {
        "observation.images.image": _image_tensor(image, device=device),
        "observation.images.image2": _image_tensor(image2, device=device),
        "observation.state": torch.from_numpy(state_values).unsqueeze(0).to(device),
        "task": [task],
    }


class SmolVLAMuJoCo:
    """Frozen SmolVLA policy with a single-step Franka MuJoCo adapter."""

    def __init__(self, checkpoint: str | Path, *, device: str = "auto") -> None:
        from interaction_vla.device import resolve_device
        from lerobot.policies import make_pre_post_processors
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        self.device = resolve_device(device)
        checkpoint = Path(checkpoint)
        config = SmolVLAConfig.from_pretrained(str(checkpoint))
        config.device = self.device.type
        config.empty_cameras = 1
        config.validate_features()
        # Keep the runtime offline when the local VLM tokenizer/config is present.
        local_vlm = checkpoint.parent / "SmolVLM2-500M-Instruct"
        if local_vlm.is_dir():
            config.vlm_model_name = str(local_vlm.resolve())
            config.load_vlm_weights = False
        self.policy = SmolVLAPolicy.from_pretrained(
            str(checkpoint), config=config, local_files_only=True, strict=True
        ).to(self.device).eval().requires_grad_(False)
        overrides = {"device_processor": {"device": self.device.type}}
        if local_vlm.is_dir():
            overrides["tokenizer_processor"] = {"tokenizer_name": str(local_vlm.resolve())}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            config,
            pretrained_path=str(checkpoint),
            preprocessor_overrides=overrides,
            postprocessor_overrides={"device_processor": {"device": self.device.type}},
        )

    def action(self, batch: dict[str, object]) -> np.ndarray:
        return self.action_chunk(batch)[0]

    def action_chunk(self, batch: dict[str, object]) -> np.ndarray:
        processed = self.preprocessor(batch)
        self.policy.reset()
        with torch.inference_mode():
            value = self.postprocessor(self.policy.predict_action_chunk(processed))
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(1)
        if tensor.ndim != 3 or tensor.shape[0] != 1 or tensor.shape[-1] != 7:
            raise ValueError(f"SmolVLA returned unexpected action shape: {tuple(tensor.shape)}")
        actions = tensor[0].detach().cpu().numpy().astype(np.float64)
        if not np.isfinite(actions).all():
            raise ValueError("SmolVLA returned a non-finite action")
        return actions


def rollout(
    checkpoint: str | Path,
    *,
    task: str,
    seed: int = 0,
    object_count: int = 3,
    target_index: int = 0,
    steps: int = 10,
    device: str = "auto",
) -> None:
    from interaction_vla.lerobot_bridge.capture import DualViewCapture
    from interaction_vla.physics_env import FrankaContactEnv

    env = FrankaContactEnv(max_steps=steps)
    env.reset(seed=seed, object_count=object_count, target_index=target_index)
    capture = DualViewCapture(env.model, width=256, height=256)
    policy = SmolVLAMuJoCo(checkpoint, device=device)
    try:
        for index in range(steps):
            try:
                frame = capture.capture(env, include_teacher=False)
            except Exception as error:
                raise RuntimeError(
                    "MuJoCo RGB rendering is unavailable in this macOS session; "
                    "run from an active graphical session or Linux EGL, or use --dry-run."
                ) from error
            batch = observation_batch(
                image=frame.views["agent"].rgb,
                image2=frame.views["wrist"].rgb,
                state=smolvla_state(env),
                task=task,
                device=policy.device,
            )
            action = policy.action(batch)
            transition = env.step(action)
            print(
                f"step={index + 1} action={np.round(action, 4).tolist()} "
                f"done={transition.done} reason={transition.reason.value}",
                flush=True,
            )
            if transition.done:
                break
    finally:
        capture.close()


def dry_run(checkpoint: str | Path, *, task: str, device: str = "auto") -> None:
    policy = SmolVLAMuJoCo(checkpoint, device=device)
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    action = policy.action(
        observation_batch(
            image=image, image2=image, state=np.zeros(8, dtype=np.float32), task=task, device=policy.device
        )
    )
    print(f"dry-run passed: device={policy.device} action_shape={action.shape} action={np.round(action, 4).tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/pretrained/smolvla_libero"))
    parser.add_argument("--task", default="pick up the green object and place it in the receptacle")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true", help="load policy and test the tensor contract without rendering")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--object-count", type=int, default=3)
    parser.add_argument("--target-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {args.checkpoint}")
    if args.dry_run:
        dry_run(args.checkpoint, task=args.task, device=args.device)
    else:
        rollout(
            args.checkpoint,
            task=args.task,
            seed=args.seed,
            object_count=args.object_count,
            target_index=args.target_index,
            steps=args.steps,
            device=args.device,
        )


if __name__ == "__main__":
    main()
