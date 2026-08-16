from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from collections.abc import Iterable, Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from interaction_vla.device import resolve_device
from interaction_vla.env import LayoutMode, TerminationReason
from interaction_vla.lerobot_bridge.act_smoke import pilot_episode_split
from interaction_vla.lerobot_bridge.config import BridgeConfig, load_bridge_config
from interaction_vla.lerobot_bridge.rollout import (
    LoadedACTRuntime,
    _load_checkpoint_bundle,
    _make_env,
    _write_json_atomic,
    rollout_loaded_policy,
)
from interaction_vla.lerobot_bridge.validator import validate_dataset_root
from interaction_vla.physics_expert import PhysicsScriptedExpert


@dataclass(frozen=True)
class RecoveryCase:
    case_id: str
    partition: str
    seed: int
    layout: str = "normal"
    object_count: int = 2

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("recovery case_id must not be empty")
        if self.partition not in {"train_seen", "heldout"}:
            raise ValueError("recovery partition must be train_seen or heldout")
        if self.seed < 0 or self.object_count < 2:
            raise ValueError("recovery seed/object_count is invalid")
        LayoutMode(self.layout)


def train_seen_cases(
    manifest: Sequence[Mapping[str, object]],
    *,
    train_episodes: Iterable[int],
    count: int,
) -> tuple[RecoveryCase, ...]:
    if count < 1:
        raise ValueError("train-seen recovery count must be positive")
    allowed = {int(value) for value in train_episodes}
    selected = sorted(
        (
            int(record["episode_index"]),
            int(record["seed"]),
        )
        for record in manifest
        if int(record["episode_index"]) in allowed
        and int(record["object_count"]) == 2
    )
    if len(selected) < count:
        raise ValueError("not enough train-seen normal two-object episodes")
    retained = selected[:count]
    if len({seed for _, seed in retained}) != len(retained):
        raise ValueError("train-seen recovery seeds must be unique")
    return tuple(
        RecoveryCase(f"train_seen_{episode:03d}", "train_seen", seed)
        for episode, seed in retained
    )


def heldout_candidates(
    *, master_seed: int, count: int, forbidden_seeds: set[int]
) -> tuple[RecoveryCase, ...]:
    if master_seed < 0 or count < 1:
        raise ValueError("held-out master seed/count is invalid")
    forbidden = {int(value) for value in forbidden_seeds}
    selected_seeds: set[int] = set()
    cases: list[RecoveryCase] = []
    replicate = 0
    while len(cases) < count:
        seed = int(
            np.random.SeedSequence(
                (master_seed, 0x48454C44, replicate)
            ).generate_state(1, dtype=np.uint32)[0]
        )
        replicate += 1
        if seed in forbidden or seed in selected_seeds:
            continue
        selected_seeds.add(seed)
        cases.append(RecoveryCase(f"heldout_{len(cases):03d}", "heldout", seed))
    return tuple(cases)


def aggregate_recovery(
    records: Sequence[Mapping[str, object]],
    *,
    train_threshold: float,
    heldout_threshold: float,
) -> dict[str, object]:
    for name, threshold in (
        ("train_threshold", train_threshold),
        ("heldout_threshold", heldout_threshold),
    ):
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError(f"{name} must lie within [0, 1]")
    unknown = {
        str(record.get("partition")) for record in records
    } - {"train_seen", "heldout"}
    if unknown:
        raise ValueError("recovery records contain an unknown partition")
    groups: dict[str, dict[str, object]] = {}
    for partition in ("train_seen", "heldout"):
        selected = [
            record for record in records if record.get("partition") == partition
        ]
        if not selected:
            raise ValueError(f"recovery report is missing {partition} records")
        rate = float(np.mean([bool(record["success"]) for record in selected]))
        groups[partition] = {
            "cases": len(selected),
            "successes": sum(bool(record["success"]) for record in selected),
            "success_rate": rate,
            "termination_counts": dict(
                Counter(
                    str(record["termination_reason"]) for record in selected
                )
            ),
        }
    return {
        "passed": bool(
            groups["train_seen"]["success_rate"] >= train_threshold
            and groups["heldout"]["success_rate"] >= heldout_threshold
        ),
        **groups,
    }


def require_verified_training_summary(path: str | Path) -> dict[str, Any]:
    summary_path = Path(path)
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid ACT training summary: {summary_path}") from error
    if not isinstance(payload, dict):
        raise ValueError("ACT training summary must be a JSON object")
    value = payload.get("reload_max_abs_error")
    if (
        not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) > 1e-5
        or float(value) < 0.0
    ):
        raise ValueError("ACT training summary reload verification did not pass")
    return payload


def rollout_expert_case(
    config: BridgeConfig, case: RecoveryCase, *, max_steps: int
) -> dict[str, object]:
    if max_steps < 1:
        raise ValueError("expert rollout max_steps must be positive")
    env = _make_env(config, max_steps=max_steps)
    snapshot = env.reset(
        seed=case.seed,
        object_count=case.object_count,
        layout_mode=LayoutMode(case.layout),
    )
    expert = PhysicsScriptedExpert(config.source.physics)
    expert.reset(seed=case.seed)
    reason = TerminationReason.RUNNING
    steps = 0
    for step in range(max_steps):
        action = expert.act(snapshot, env.contact_diagnostics, env.grasp_state)
        transition = env.step(action)
        snapshot = transition.snapshot
        reason = transition.reason
        steps = step + 1
        if transition.done:
            break
    return {
        "case_id": case.case_id,
        "success": reason == TerminationReason.SUCCESS,
        "termination_reason": str(getattr(reason, "value", reason)),
        "steps": steps,
    }


def _selection_provenance(
    config: BridgeConfig,
    *,
    candidates: Sequence[RecoveryCase],
    inspected: Sequence[Mapping[str, object]],
    selected: Sequence[RecoveryCase],
) -> dict[str, object]:
    assert config.recovery is not None
    return {
        "heldout_master_seed": config.recovery.heldout_master_seed,
        "heldout_candidate_limit": len(candidates),
        "heldout_candidates_inspected": len(inspected),
        "heldout_selected_case_ids": [case.case_id for case in selected],
    }


def evaluate_recovery(
    config_path: str | Path,
    checkpoint: str | Path,
) -> dict[str, object]:
    config = load_bridge_config(config_path)
    if config.recovery is None:
        raise ValueError("bridge config does not define recovery")
    checkpoint_path = Path(checkpoint)
    training_summary = require_verified_training_summary(
        checkpoint_path / "training_summary.json"
    )
    validate_dataset_root(
        config.dataset.root,
        repo_id=config.dataset.repo_id,
        allow_incomplete=False,
        require_bridge_metadata=True,
        replay=True,
        bridge_config=config,
        require_collection_identity=False,
    )
    split = pilot_episode_split(
        total_episodes=config.dataset.episodes,
        seed=config.act.seed,
    )
    manifest_path = config.dataset.root / "meta" / "teacher_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid teacher manifest: {manifest_path}") from error
    if not isinstance(manifest, list) or not all(
        isinstance(record, Mapping) for record in manifest
    ):
        raise ValueError("teacher manifest must be a list of objects")
    seen = train_seen_cases(
        manifest,
        train_episodes=split["train"],
        count=config.recovery.train_seen_cases,
    )
    all_dataset_seeds = {int(record["seed"]) for record in manifest}
    candidates = heldout_candidates(
        master_seed=config.recovery.heldout_master_seed,
        count=(
            config.recovery.heldout_cases
            * config.recovery.heldout_attempt_multiplier
        ),
        forbidden_seeds=all_dataset_seeds,
    )
    expert_seen = [
        {
            **asdict(case),
            **rollout_expert_case(
                config,
                case,
                max_steps=config.recovery.max_steps,
            ),
        }
        for case in seen
    ]
    expert_candidates: list[dict[str, object]] = []
    unseen: list[RecoveryCase] = []
    for case in candidates:
        result = rollout_expert_case(
            config,
            case,
            max_steps=config.recovery.max_steps,
        )
        expert_candidates.append({**asdict(case), **result})
        if bool(result["success"]):
            unseen.append(case)
        if len(unseen) == config.recovery.heldout_cases:
            break
    selection = _selection_provenance(
        config,
        candidates=candidates,
        inspected=expert_candidates,
        selected=unseen,
    )
    destination = config.recovery.output_dir / "recovery_report.json"
    if (
        not all(bool(record["success"]) for record in expert_seen)
        or len(unseen) != config.recovery.heldout_cases
    ):
        failure = {
            "passed": False,
            "failure_stage": "expert_case_gate",
            "checkpoint": checkpoint_path.as_posix(),
            "checkpoint_reload_max_abs_error": training_summary[
                "reload_max_abs_error"
            ],
            **selection,
            "expert_train_seen": expert_seen,
            "expert_heldout_candidates": expert_candidates,
        }
        _write_json_atomic(destination, failure)
        return failure
    device = resolve_device(config.act.device)
    policy, preprocessor, postprocessor, metadata = _load_checkpoint_bundle(
        config=config,
        checkpoint=checkpoint_path,
        device=device,
    )
    runtime = LoadedACTRuntime(
        checkpoint_path,
        policy,
        preprocessor,
        postprocessor,
    )
    records = []
    for case in (*seen, *unseen):
        result = rollout_loaded_policy(
            config,
            runtime,
            seed=case.seed,
            object_count=case.object_count,
            layout=LayoutMode(case.layout),
            max_steps=config.recovery.max_steps,
        )
        records.append({**asdict(case), **result})
    report = aggregate_recovery(
        records,
        train_threshold=config.recovery.train_success_threshold,
        heldout_threshold=config.recovery.heldout_success_threshold,
    )
    report.update(
        {
            "checkpoint": checkpoint_path.as_posix(),
            "checkpoint_dataset_fingerprint": metadata.get(
                "dataset_fingerprint"
            ),
            "checkpoint_reload_max_abs_error": training_summary[
                "reload_max_abs_error"
            ],
            **selection,
            "records": records,
            "expert_train_seen": expert_seen,
            "expert_heldout_candidates": expert_candidates,
        }
    )
    _write_json_atomic(destination, report)
    return report
