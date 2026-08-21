from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

from interaction_vla.env import TerminationReason
from interaction_vla.lerobot_bridge.config import load_bridge_config
from interaction_vla.lerobot_bridge.provenance import sha256_file

from ..backends import make_backend
from ..config import RepresentationStudyConfig
from ..state_bank.io import write_json_atomic
from ..taps.registry import registered_taps
from .checkpoint import (
    capture_rng_state,
    load_training_checkpoint,
    restore_rng_state,
    save_training_checkpoint,
)
from .core import (
    ResidualActorCritic,
    clipped_ppo_loss,
    generalized_advantage_estimate,
    normalized_curve_auc,
)
from .environment import ResidualMujocoRuntime


RL_REPORT_SCHEMA_VERSION = "interaction_residual_rl_report_v1"
RESIDUAL_BUNDLE_SCHEMA_VERSION = "interaction_residual_policy_bundle_v1"


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pack_observation(batch: Mapping[str, object]) -> dict[str, object]:
    packed: dict[str, object] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu()
            if key.startswith("observation.images."):
                tensor = (tensor.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
            packed[key] = tensor
        elif key == "task":
            packed[key] = list(value)  # type: ignore[arg-type]
        else:
            raise ValueError(f"unsupported online observation field: {key}")
    return packed


def collate_packed_observations(
    rows: Sequence[Mapping[str, object]], *, device: torch.device
) -> dict[str, object]:
    if not rows:
        raise ValueError("cannot collate empty online observations")
    result: dict[str, object] = {}
    for key in rows[0]:
        if key == "task":
            result[key] = [str(row[key][0]) for row in rows]  # type: ignore[index]
            continue
        values = [row[key] for row in rows]
        if not all(isinstance(value, torch.Tensor) and value.shape[0] == 1 for value in values):
            raise ValueError(f"packed tensor field is invalid: {key}")
        tensor = torch.cat(values, dim=0)  # type: ignore[arg-type]
        if key.startswith("observation.images."):
            tensor = tensor.to(torch.float32).div_(255.0)
        result[key] = tensor.to(device)
    return result


def _seed_policy_noise(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def recompute_latents_with_policy_seeds(
    runtime: Any,
    observations: Sequence[Mapping[str, object]],
    policy_seeds: Sequence[int],
    *,
    tap_id: str,
) -> torch.Tensor:
    """Replay stochastic VLA inference exactly while retaining latent gradients.

    Flow-matching VLAs draw inference noise before reaching action-proximal taps.
    PPO therefore records one seed per transition and replays each sample separately;
    a fresh batched draw would not correspond to the behavior-policy latent.
    """
    if not observations or len(observations) != len(policy_seeds):
        raise ValueError("policy-noise replay inputs must be aligned and non-empty")
    rows: list[torch.Tensor] = []
    for observation, seed in zip(observations, policy_seeds, strict=True):
        _seed_policy_noise(int(seed))
        batch = collate_packed_observations([observation], device=runtime.device)
        _, latent = runtime.differentiable_action_and_latent(batch, tap_id=tap_id)
        if latent.ndim != 2 or latent.shape[0] != 1:
            raise ValueError("replayed policy latent must contain exactly one row")
        rows.append(latent)
    return torch.cat(rows, dim=0)


def _parent_stage(stage: str) -> str:
    if stage not in {"rl_head", "rl_representation"}:
        raise ValueError("residual RL stage must be rl_head or rl_representation")
    # continued_sft is a matched extra-imitation control branch. Both RL branches
    # start from the same SFT policy so SFT→RL deltas isolate online adaptation.
    return "sft"


def _tap_id(backend: str) -> str:
    matches = [tap.tap_id for tap in registered_taps(backend) if tap.role == "action_proximal"]
    if len(matches) != 1:
        raise RuntimeError("backend must define exactly one action-proximal tap")
    return matches[0]


def _config_binding(config: RepresentationStudyConfig, backend: str, stage: str) -> str:
    payload = {
        "config_sha256": sha256_file(config.config_path),
        "backend": backend,
        "stage": stage,
        "parent_checkpoint": config.stage_config(backend, _parent_stage(stage)).checkpoint,
        "rl": asdict(config.rl),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _output(config: RepresentationStudyConfig, backend: str, stage: str) -> Path:
    destination = config.rl.output_dir / backend / stage
    configured = Path(config.stage_config(backend, stage).checkpoint)
    if configured != destination / "checkpoint":
        raise ValueError(
            f"configured {backend}/{stage} checkpoint must be {destination / 'checkpoint'}"
        )
    return destination


def _parameter_groups(
    backend_runtime: Any,
    residual: ResidualActorCritic,
    *,
    adapt_representation: bool,
    head_lr: float,
    representation_lr: float,
) -> list[dict[str, object]]:
    groups: list[dict[str, object]] = [
        {"params": list(residual.parameters()), "lr": head_lr, "name": "residual_head"}
    ]
    if adapt_representation:
        backend_runtime.set_trainable_groups(("fusion",))
        parameters = [
            parameter
            for parameter in backend_runtime.policy.parameters()
            if parameter.requires_grad
        ]
        if not parameters:
            raise ValueError("RL representation stage selected no trainable VLA parameters")
        groups.append(
            {"params": parameters, "lr": representation_lr, "name": "late_representation"}
        )
    else:
        backend_runtime.set_trainable_groups(())
    return groups


def _trainable_policy_state(runtime: Any) -> dict[str, torch.Tensor] | None:
    values = {
        name: parameter.detach().cpu()
        for name, parameter in runtime.policy.named_parameters()
        if parameter.requires_grad
    }
    return values or None


def _evaluate_unseeded(
    config: RepresentationStudyConfig,
    *,
    backend_name: str,
    parent_checkpoint: str,
    tap_id: str,
    residual: ResidualActorCritic,
    environment_steps: int,
    policy_state: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, object]:
    backend = make_backend(backend_name, device=config.rl.device)
    backend.load_checkpoint(parent_checkpoint)
    if policy_state is not None:
        backend.policy.load_state_dict(policy_state, strict=False)
    residual.eval()
    bridge = load_bridge_config(config.dataset.bridge_config)
    reasons: dict[str, int] = {}
    returns: list[float] = []
    lengths: list[int] = []
    clipping_rates: list[float] = []
    projection_scales: list[float] = []
    with ResidualMujocoRuntime(
        bridge=bridge,
        backend=backend,
        tap_id=tap_id,
        residual_scale=config.rl.residual_scale,
        max_steps=config.rl.max_episode_steps,
        object_counts=config.rl.object_counts,
        layouts=config.rl.layouts,
        seed=config.rl.seed + 10_000,
        reward_mode=config.rl.reward_mode,
        progress_reward_scale=config.rl.progress_reward_scale,
    ) as environment:
        for episode in range(config.rl.eval_episodes):
            if episode:
                environment.reset()
            while True:
                base, latent = environment.policy_features()
                with torch.no_grad():
                    value = residual.sample(latent.to(next(residual.parameters()).device), deterministic=True)
                transition = environment.step(
                    base_action=base,
                    latent=latent,
                    residual=value.residual[0].cpu().numpy(),
                )
                if transition.done:
                    reasons[transition.reason] = reasons.get(transition.reason, 0) + 1
                    returns.append(transition.episode_return)
                    lengths.append(transition.episode_length)
                    clipping_rates.append(
                        transition.episode_action_clipping_rate
                    )
                    projection_scales.append(
                        transition.episode_mean_ik_projection_scale
                    )
                    break
    success = reasons.get(TerminationReason.SUCCESS.value, 0) / config.rl.eval_episodes
    return {
        "steps": int(environment_steps),
        "success_rate": float(success),
        "mean_return": float(np.mean(returns)),
        "mean_episode_length": float(np.mean(lengths)),
        "action_clipping_rate": float(np.mean(clipping_rates)),
        "mean_ik_projection_scale": float(np.mean(projection_scales)),
        "termination_counts": reasons,
        "episodes": config.rl.eval_episodes,
    }


def _evaluate(
    config: RepresentationStudyConfig,
    *,
    backend_name: str,
    parent_checkpoint: str,
    tap_id: str,
    residual: ResidualActorCritic,
    environment_steps: int,
    policy_state: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, object]:
    state = capture_rng_state()
    try:
        _seed_all(config.rl.seed + 20_000)
        return _evaluate_unseeded(
            config,
            backend_name=backend_name,
            parent_checkpoint=parent_checkpoint,
            tap_id=tap_id,
            residual=residual,
            environment_steps=environment_steps,
            policy_state=policy_state,
        )
    finally:
        restore_rng_state(state)


def _save_bundle(
    destination: Path,
    *,
    config: RepresentationStudyConfig,
    backend_name: str,
    stage: str,
    parent_checkpoint: str,
    tap_id: str,
    latent_dim: int,
    residual: ResidualActorCritic,
    runtime: Any,
) -> Path:
    checkpoint = destination / "checkpoint"
    checkpoint.mkdir(parents=True, exist_ok=True)
    policy_checkpoint = parent_checkpoint
    if stage == "rl_representation":
        policy_path = checkpoint / "adapted_policy"
        runtime.policy.save_pretrained(policy_path)
        runtime.preprocessor.save_pretrained(policy_path)
        runtime.postprocessor.save_pretrained(policy_path)
        policy_checkpoint = policy_path.as_posix()
    torch.save(
        {
            "latent_dim": latent_dim,
            "action_dim": 7,
            "adapt_representation": False,
            "state_dict": residual.state_dict(),
        },
        checkpoint / "residual.pt",
    )
    write_json_atomic(
        checkpoint / "residual_study.json",
        {
            "schema_version": RESIDUAL_BUNDLE_SCHEMA_VERSION,
            "backend": backend_name,
            "stage": stage,
            "base_checkpoint": parent_checkpoint,
            "policy_checkpoint": policy_checkpoint,
            "tap_id": tap_id,
            "latent_dim": latent_dim,
            "residual_scale": list(config.rl.residual_scale),
            "reward_mode": config.rl.reward_mode,
            "config_binding": _config_binding(config, backend_name, stage),
        },
    )
    return checkpoint


def train_residual_rl(
    config: RepresentationStudyConfig,
    *,
    backend: str,
    stage: str,
    resume: bool,
) -> dict[str, object]:
    parent = _parent_stage(stage)
    parent_checkpoint = config.stage_config(backend, parent).checkpoint
    destination = _output(config, backend, stage)
    state_path = destination / "training_state.pt"
    binding = _config_binding(config, backend, stage)
    if destination.exists() and not resume:
        raise FileExistsError(f"RL output already exists; pass --resume: {destination}")
    if resume and not state_path.is_file():
        raise FileNotFoundError(f"RL resume checkpoint not found: {state_path}")
    destination.mkdir(parents=True, exist_ok=True)

    _seed_all(config.rl.seed)
    runtime = make_backend(backend, device=config.rl.device)
    runtime.load_checkpoint(parent_checkpoint)
    tap_id = _tap_id(backend)
    bridge = load_bridge_config(config.dataset.bridge_config)
    with ResidualMujocoRuntime(
        bridge=bridge,
        backend=runtime,
        tap_id=tap_id,
        residual_scale=config.rl.residual_scale,
        max_steps=config.rl.max_episode_steps,
        object_counts=config.rl.object_counts,
        layouts=config.rl.layouts,
        seed=config.rl.seed,
        reward_mode=config.rl.reward_mode,
        progress_reward_scale=config.rl.progress_reward_scale,
    ) as environment:
        _, example_latent = environment.policy_features()
        latent_dim = int(example_latent.shape[1])
        residual = ResidualActorCritic(
            latent_dim, action_dim=7, adapt_representation=False
        ).to(runtime.device)
        adapt = stage == "rl_representation"
        optimizer = torch.optim.AdamW(
            _parameter_groups(
                runtime,
                residual,
                adapt_representation=adapt,
                head_lr=config.rl.learning_rate,
                representation_lr=config.rl.representation_learning_rate,
            )
        )
        steps = 0
        update = 0
        curve: list[dict[str, object]] = []
        if resume:
            loaded = load_training_checkpoint(state_path, map_location=runtime.device)
            if loaded["metadata"].get("config_binding") != binding:
                raise ValueError("RL resume checkpoint differs from the current config")
            residual.load_state_dict(loaded["residual_policy"])
            if loaded["base_policy"] is not None:
                runtime.policy.load_state_dict(loaded["base_policy"], strict=False)
            optimizer.load_state_dict(loaded["optimizer"])
            steps = int(loaded["environment_steps"])
            update = int(loaded["update"])
            curve = [dict(row) for row in loaded["curve"]]
            restore_rng_state(loaded["rng_state"])
            environment_state = loaded["rng_state"].get("environment")
            if environment_state is None:
                raise ValueError("RL resume checkpoint has no environment RNG state")
            environment.restore_rng_state(environment_state)
        if not curve:
            curve.append(
                _evaluate(
                    config,
                    backend_name=backend,
                    parent_checkpoint=parent_checkpoint,
                    tap_id=tap_id,
                    residual=residual,
                    environment_steps=0,
                    policy_state=_trainable_policy_state(runtime),
                )
            )
            initial_rng_state = capture_rng_state()
            initial_rng_state["environment"] = environment.rng_state()
            save_training_checkpoint(
                state_path,
                policy=residual,
                optimizer=optimizer,
                policy_state=_trainable_policy_state(runtime),
                environment_steps=0,
                update=0,
                curve=curve,
                rng_state=initial_rng_state,
                metadata={
                    "backend": backend,
                    "stage": stage,
                    "config_binding": binding,
                    "latent_dim": latent_dim,
                },
            )

        progress = tqdm(
            total=config.rl.total_steps,
            initial=steps,
            desc=f"{backend}/{stage} residual PPO",
            unit="step",
            dynamic_ncols=True,
        )
        while steps < config.rl.total_steps:
            environment.reset()
            count = min(config.rl.rollout_steps, config.rl.total_steps - steps)
            latents: list[torch.Tensor] = []
            observations: list[dict[str, object]] = []
            policy_seeds: list[int] = []
            residuals: list[torch.Tensor] = []
            log_probs: list[torch.Tensor] = []
            values: list[torch.Tensor] = []
            rewards: list[float] = []
            dones: list[bool] = []
            for index in range(count):
                policy_seed = int(
                    np.random.SeedSequence(
                        (config.rl.seed, 0x504F4C59, steps + index)
                    ).generate_state(1, dtype=np.uint32)[0]
                )
                _seed_policy_noise(policy_seed)
                base, latent_cpu = environment.policy_features()
                latent = latent_cpu.to(runtime.device)
                with torch.no_grad():
                    sampled = residual.sample(latent, deterministic=False)
                transition = environment.step(
                    base_action=base,
                    latent=latent_cpu,
                    residual=sampled.residual[0].cpu().numpy(),
                )
                latents.append(latent_cpu[0])
                observations.append(pack_observation(transition.observation))
                policy_seeds.append(policy_seed)
                residuals.append(sampled.residual[0].cpu())
                log_probs.append(sampled.log_prob[0].cpu())
                values.append(sampled.value[0].cpu())
                rewards.append(transition.reward)
                dones.append(transition.done)
                if transition.done and index + 1 < count:
                    environment.reset()
            if dones[-1]:
                last_value = 0.0
            else:
                _, final_latent = environment.policy_features()
                with torch.no_grad():
                    last_value = float(
                        residual.distribution_and_value(final_latent.to(runtime.device))[1][0].item()
                    )
            advantages_np, returns_np = generalized_advantage_estimate(
                rewards,
                torch.stack(values).numpy(),
                dones,
                last_value=last_value,
                gamma=config.rl.gamma,
                gae_lambda=config.rl.gae_lambda,
            )
            residual_tensor = torch.stack(residuals).to(runtime.device)
            old_log_prob = torch.stack(log_probs).to(runtime.device)
            advantages = torch.from_numpy(advantages_np).to(runtime.device)
            returns = torch.from_numpy(returns_np).to(runtime.device)
            indices = np.arange(count)
            residual.train()
            for _ in range(config.rl.update_epochs):
                np.random.shuffle(indices)
                for start in range(0, count, config.rl.minibatch_size):
                    choice = indices[start : start + config.rl.minibatch_size]
                    if adapt:
                        latent = recompute_latents_with_policy_seeds(
                            runtime,
                            [observations[int(index)] for index in choice],
                            [policy_seeds[int(index)] for index in choice],
                            tap_id=tap_id,
                        )
                    else:
                        latent = torch.stack(
                            [latents[int(index)] for index in choice]
                        ).to(runtime.device)
                    choice_tensor = torch.as_tensor(choice, device=runtime.device)
                    evaluation = residual.evaluate(latent, residual_tensor[choice_tensor])
                    loss, _ = clipped_ppo_loss(
                        evaluation,
                        old_log_prob=old_log_prob[choice_tensor],
                        advantages=advantages[choice_tensor],
                        returns=returns[choice_tensor],
                        clip_coef=config.rl.clip_coef,
                        value_coef=config.rl.value_coef,
                        entropy_coef=config.rl.entropy_coef,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    parameters = [
                        parameter
                        for group in optimizer.param_groups
                        for parameter in group["params"]
                    ]
                    torch.nn.utils.clip_grad_norm_(parameters, config.rl.max_grad_norm)
                    optimizer.step()
            steps += count
            update += 1
            progress.update(count)
            if steps % config.rl.eval_interval == 0 or steps == config.rl.total_steps:
                curve.append(
                    _evaluate(
                        config,
                        backend_name=backend,
                        parent_checkpoint=parent_checkpoint,
                        tap_id=tap_id,
                        residual=residual,
                        environment_steps=steps,
                        policy_state=_trainable_policy_state(runtime),
                    )
                )
            rng_state = capture_rng_state()
            rng_state["environment"] = environment.rng_state()
            save_training_checkpoint(
                state_path,
                policy=residual,
                optimizer=optimizer,
                policy_state=_trainable_policy_state(runtime),
                environment_steps=steps,
                update=update,
                curve=curve,
                rng_state=rng_state,
                metadata={
                    "backend": backend,
                    "stage": stage,
                    "config_binding": binding,
                    "latent_dim": latent_dim,
                },
            )
        progress.close()
        checkpoint = _save_bundle(
            destination,
            config=config,
            backend_name=backend,
            stage=stage,
            parent_checkpoint=parent_checkpoint,
            tap_id=tap_id,
            latent_dim=latent_dim,
            residual=residual,
            runtime=runtime,
        )
    auc = normalized_curve_auc(
        [int(row["steps"]) for row in curve],
        [float(row["success_rate"]) for row in curve],
        budget=config.rl.total_steps,
    )
    report = {
        "schema_version": RL_REPORT_SCHEMA_VERSION,
        "passed": True,
        "backend": backend,
        "stage": stage,
        "checkpoint": checkpoint.as_posix(),
        "environment_steps": steps,
        "updates": update,
        "normalized_learning_curve_auc": auc,
        "initial_success_rate": float(curve[0]["success_rate"]),
        "final_success_rate": float(curve[-1]["success_rate"]),
        "final_return_gain": float(curve[-1]["mean_return"]) - float(curve[0]["mean_return"]),
        "curve": curve,
    }
    hits = [int(row["steps"]) for row in curve if float(row["success_rate"]) >= config.rl.success_threshold]
    report["steps_to_fixed_threshold"] = min(hits) if hits else None
    write_json_atomic(destination / "report.json", report)
    return report


def evaluate_residual_rl(
    config: RepresentationStudyConfig, *, backend: str, stage: str
) -> dict[str, object]:
    checkpoint = _output(config, backend, stage) / "checkpoint"
    metadata_path = checkpoint / "residual_study.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"residual policy bundle not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("config_binding") != _config_binding(config, backend, stage):
        raise ValueError("residual policy bundle differs from current config")
    runtime = make_backend(backend, device=config.rl.device)
    runtime.load_checkpoint(str(metadata["policy_checkpoint"]))
    residual_payload = torch.load(
        checkpoint / "residual.pt", map_location=runtime.device, weights_only=False
    )
    residual = ResidualActorCritic(
        int(residual_payload["latent_dim"]), adapt_representation=False
    ).to(runtime.device)
    residual.load_state_dict(residual_payload["state_dict"])
    row = _evaluate(
        config,
        backend_name=backend,
        parent_checkpoint=str(metadata["policy_checkpoint"]),
        tap_id=str(metadata["tap_id"]),
        residual=residual,
        environment_steps=config.rl.total_steps,
    )
    report = {"schema_version": RL_REPORT_SCHEMA_VERSION, "passed": True, **row}
    write_json_atomic(_output(config, backend, stage) / "evaluation.json", report)
    return report
