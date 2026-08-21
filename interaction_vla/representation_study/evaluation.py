from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from typing import Mapping, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

from interaction_vla.env import TerminationReason
from interaction_vla.lerobot_bridge.config import load_bridge_config

from .backends import make_backend
from .config import RepresentationStudyConfig
from .extraction import build_stage_manifest
from .rl.environment import ResidualMujocoRuntime
from .state_bank.io import write_json_atomic


POLICY_EVALUATION_SCHEMA_VERSION = "interaction_stage_policy_evaluation_v2"


def evaluation_cases(config: RepresentationStudyConfig) -> tuple[dict[str, object], ...]:
    rng = np.random.default_rng(config.rl.seed + 30_000)
    return tuple(
        {
            "case_id": index,
            "environment_seed": int(rng.integers(0, 2**32 - 1)),
            "policy_seed": int(config.rl.seed + 40_000 + index),
            "object_count": int(config.rl.object_counts[index % len(config.rl.object_counts)]),
            "layout": config.rl.layouts[index % len(config.rl.layouts)],
        }
        for index in range(config.rl.eval_episodes)
    )


def aggregate_episode_outcomes(
    episodes: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not episodes:
        raise ValueError("policy evaluation requires episodes")
    reasons: dict[str, int] = {}
    for row in episodes:
        reason = str(row["termination_reason"])
        reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "episodes": len(episodes),
        "success_rate": reasons.get(TerminationReason.SUCCESS.value, 0) / len(episodes),
        "timeout_rate": reasons.get(TerminationReason.TIMEOUT.value, 0) / len(episodes),
        "drop_rate": reasons.get(TerminationReason.DROPPED.value, 0) / len(episodes),
        "wrong_object_rate": reasons.get(TerminationReason.WRONG_OBJECT.value, 0) / len(episodes),
        "mean_steps": float(np.mean([int(row["steps"]) for row in episodes])),
        "action_clipping_rate": float(
            np.mean([float(row.get("action_clipping_rate", 0.0)) for row in episodes])
        ),
        "mean_ik_projection_scale": float(
            np.mean(
                [float(row.get("mean_ik_projection_scale", 1.0)) for row in episodes]
            )
        ),
        "termination_counts": reasons,
    }


def evaluate_policy_stage(
    config: RepresentationStudyConfig,
    *,
    backend: str,
    stage: str,
    force: bool = False,
) -> dict[str, object]:
    stage_manifest, revision = build_stage_manifest(config, backend=backend, stage=stage)
    binding = hashlib.sha256(stage_manifest.to_json().encode("utf-8")).hexdigest()
    destination = config.analysis.output_dir / "policy_evaluation" / backend / stage
    report_path = destination / "report.json"
    existing = None
    if report_path.is_file():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if existing.get("stage_manifest_sha256") == binding and not force:
            return existing
        if not force:
            raise ValueError("existing stage policy evaluation is stale or incompatible")
    runtime = make_backend(backend, device=config.rl.device)
    runtime.load_checkpoint_for_dataset(
        config.stage_config(backend, stage).checkpoint,
        repo_id=config.dataset.repo_id,
        dataset_root=config.dataset.root,
    )
    from .taps.registry import registered_taps

    tap_id = next(value.tap_id for value in registered_taps(backend) if value.role == "action_proximal")
    random.seed(config.rl.seed + 30_000)
    np.random.seed(config.rl.seed + 30_000)
    torch.manual_seed(config.rl.seed + 30_000)
    bridge = load_bridge_config(config.dataset.bridge_config)
    episodes: list[dict[str, object]] = []
    with ResidualMujocoRuntime(
        bridge=bridge,
        backend=runtime,
        tap_id=tap_id,
        residual_scale=(0.0,) * 7,
        max_steps=config.rl.max_episode_steps,
        object_counts=config.rl.object_counts,
        layouts=config.rl.layouts,
        seed=config.rl.seed + 30_000,
        reward_mode="sparse",
        progress_reward_scale=0.0,
    ) as environment:
        cases = evaluation_cases(config)
        progress = tqdm(
            cases,
            desc=f"{backend}/{stage} closed loop",
            unit="episode",
            dynamic_ncols=True,
        )
        for case in progress:
            environment.reset(
                case_seed=int(case["environment_seed"]),
                object_count=int(case["object_count"]),
                layout=str(case["layout"]),
            )
            torch.manual_seed(int(case["policy_seed"]))
            while True:
                base, latent = environment.policy_features()
                transition = environment.step(
                    base_action=base,
                    latent=latent,
                    residual=np.zeros(7, dtype=np.float32),
                )
                if transition.done:
                    episodes.append(
                        {
                            **case,
                            "termination_reason": transition.reason,
                            "success": transition.reason == TerminationReason.SUCCESS.value,
                            "steps": transition.episode_length,
                            "action_clipping_rate": (
                                transition.episode_action_clipping_rate
                            ),
                            "mean_ik_projection_scale": (
                                transition.episode_mean_ik_projection_scale
                            ),
                        }
                    )
                    break
    report = {
        "schema_version": POLICY_EVALUATION_SCHEMA_VERSION,
        "passed": True,
        "backend": backend,
        "stage": stage,
        "stage_manifest_sha256": binding,
        "hf_revision": revision,
        "case_seed": config.rl.seed + 30_000,
        "object_counts": list(config.rl.object_counts),
        "layouts": list(config.rl.layouts),
        "max_episode_steps": config.rl.max_episode_steps,
        "case_contract": "paired environment/policy seeds across stages and interventions",
        **aggregate_episode_outcomes(episodes),
        "episode_rows": episodes,
    }
    write_json_atomic(report_path, report)
    return report
