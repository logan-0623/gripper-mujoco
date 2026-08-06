from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import torch
from tqdm.auto import tqdm

from .config import ExperimentConfig, load_config
from .chunked_controller import ChunkedPolicyController
from .contact_physics import InteractionSubstepEvent
from .device import resolve_device
from .env import TerminationReason
from .evaluate import shuffle_valid_edge_assignments
from .graph.builder import SceneGraphBuilder
from .models.encoders import SceneBatch
from .models.policy import ActionPolicy
from .physics_action_safety import (
    DEFAULT_IK_PROJECTION_SCALES,
    project_cartesian_action,
)
from .physics_env import FrankaContactEnv
from .physics_expert import PhysicsScriptedExpert
from .physics_data import prepare_physics_recovery_start
from .physics_recovery import PhysicsRecoveryKind, make_physics_recovery_spec
from .physics_provenance import (
    learned_rollout_source_hash,
    training_pipeline_source_hash,
)
from .train import (
    TrainingStatistics,
    build_sequence_provenance_fields,
    build_training_provenance,
    load_training_checkpoint,
    resolve_training_data,
)


@dataclass(frozen=True)
class PhysicsEvaluationCase:
    case_id: str
    seed: int
    object_count: int
    condition: str
    layout_mode: str
    randomized: bool = False
    recovery_variant_id: int | None = None
    recovery_kind: str | None = None
    source_split: str | None = None


@dataclass(frozen=True)
class PhysicsEpisodeResult:
    policy: str
    representation: str
    model_seed: int
    ablation: str
    case_id: str
    seed: int
    object_count: int
    condition: str
    layout_mode: str
    success: bool
    bilateral_contact: bool
    stable_lift: bool
    wrong_object_stable_grasp: bool
    dropped: bool
    placement: bool
    ik_failure: bool
    physics_failure: bool
    steps: int
    termination_reason: str
    physics_hash: str
    initial_state_hash: str
    transport_progress: float = 0.0
    transport_progress_rate: float = 0.0
    premature_open: bool = False
    action_saturation_rate: float = 0.0
    ik_projection_rate: float = 0.0
    zero_pose_projection_rate: float = 0.0
    mean_ik_projection_scale: float = 1.0
    post_placement_reclose: bool = False
    first_bilateral_object: str | None = None
    target_first_contact: bool = False
    target_bilateral_contact: bool = False
    stable_target_grasp: bool = False
    first_bilateral_target_substep: int | None = None
    first_stable_target_substep: int | None = None
    stable_target_substeps: int = 0
    longest_stable_target_run: int = 0
    wrong_object_interaction: bool = False
    dropped_target: bool = False
    strict_containment: bool = False
    receptacle_base_contact: bool = False
    receptacle_wall_contact: bool = False
    strict_placement: bool = False
    gripper_released: bool = False
    tcp_retreated: bool = False
    strict_task_success: bool = False
    containment_margin_x: float | None = None
    containment_margin_y: float | None = None
    rollout_device: str = "cpu"
    mean_ensemble_size: float = 1.0
    gripper_switch_count: int = 0


class InteractionRolloutTracker:
    """Measure only interaction events occurring after learned-policy handoff."""

    def __init__(
        self,
        *,
        target_name: str,
        baseline: Mapping[str, int | bool | str | None] | None = None,
    ) -> None:
        if not target_name:
            raise ValueError("target_name must not be empty")
        values = {} if baseline is None else dict(baseline)
        self.target_name = target_name
        self._baseline_substep = int(values.get("tracker_substep", 0) or 0)
        self._last_stable_total = int(
            values.get("total_stable_target_substeps", 0) or 0
        )
        self._last_target_bilateral_total = int(
            values.get("total_bilateral_target_substeps", 0) or 0
        )
        self._last_wrong_bilateral_total = int(
            values.get("total_bilateral_wrong_substeps", 0) or 0
        )
        self._baseline_dropped = bool(values.get("dropped_target", False))
        self.first_bilateral_object: str | None = None
        self.target_bilateral_contact = False
        self.stable_target_grasp = False
        self.first_bilateral_target_substep: int | None = None
        self.first_stable_target_substep: int | None = None
        self.stable_target_substeps = 0
        self.longest_stable_target_run = 0
        self._current_stable_target_run = 0
        self._last_stable_event_substep: int | None = None
        self.last_observed_substep = self._baseline_substep
        self.wrong_object_interaction = False
        self.wrong_object_stable_grasp = False
        self.dropped_target = False

    def observe(
        self,
        grasp,
        *,
        events: Iterable[InteractionSubstepEvent] | None = None,
    ) -> None:
        relative_substep = max(
            0,
            int(getattr(grasp, "tracker_substep", 0)) - self._baseline_substep,
        )
        bilateral = getattr(grasp, "bilateral_object", None)
        target_bilateral_total = int(
            getattr(grasp, "total_bilateral_target_substeps", 0)
        )
        wrong_bilateral_total = int(
            getattr(grasp, "total_bilateral_wrong_substeps", 0)
        )
        target_bilateral_delta = max(
            0,
            target_bilateral_total - self._last_target_bilateral_total,
        )
        wrong_bilateral_delta = max(
            0,
            wrong_bilateral_total - self._last_wrong_bilateral_total,
        )
        self._last_target_bilateral_total = max(
            self._last_target_bilateral_total,
            target_bilateral_total,
        )
        self._last_wrong_bilateral_total = max(
            self._last_wrong_bilateral_total,
            wrong_bilateral_total,
        )
        stable_total = int(
            getattr(grasp, "total_stable_target_substeps", 0)
        )
        stable_delta = max(0, stable_total - self._last_stable_total)
        self._last_stable_total = max(self._last_stable_total, stable_total)
        if events is not None:
            for event in events:
                if event.substep <= self.last_observed_substep:
                    continue
                event_relative = event.substep - self._baseline_substep
                bilateral_objects = tuple(event.bilateral_objects)
                if bilateral_objects and self.first_bilateral_object is None:
                    self.first_bilateral_object = (
                        self.target_name
                        if self.target_name in bilateral_objects
                        else bilateral_objects[0]
                    )
                if self.target_name in bilateral_objects:
                    self.target_bilateral_contact = True
                    if self.first_bilateral_target_substep is None:
                        self.first_bilateral_target_substep = event_relative
                if any(
                    name != self.target_name for name in bilateral_objects
                ):
                    self.wrong_object_interaction = True

                stable_objects = tuple(event.stable_objects)
                if self.target_name in stable_objects:
                    self.stable_target_grasp = True
                    self.stable_target_substeps += 1
                    if self.first_stable_target_substep is None:
                        self.first_stable_target_substep = event_relative
                    if (
                        self._last_stable_event_substep is not None
                        and event.substep == self._last_stable_event_substep + 1
                    ):
                        self._current_stable_target_run += 1
                    else:
                        self._current_stable_target_run = 1
                    self._last_stable_event_substep = event.substep
                    self.longest_stable_target_run = max(
                        self.longest_stable_target_run,
                        self._current_stable_target_run,
                    )
                if any(name != self.target_name for name in stable_objects):
                    self.wrong_object_stable_grasp = True
                self.dropped_target |= bool(event.dropped_target)
        else:
            if self.first_bilateral_object is None:
                if bilateral is not None:
                    self.first_bilateral_object = str(bilateral)
                elif target_bilateral_delta and not wrong_bilateral_delta:
                    self.first_bilateral_object = self.target_name
                elif wrong_bilateral_delta and not target_bilateral_delta:
                    self.first_bilateral_object = "wrong_object"
            if bilateral == self.target_name or target_bilateral_delta:
                self.target_bilateral_contact = True
                if self.first_bilateral_target_substep is None:
                    self.first_bilateral_target_substep = max(
                        1,
                        relative_substep - target_bilateral_delta + 1,
                    )
            if (
                (bilateral is not None and bilateral != self.target_name)
                or wrong_bilateral_delta
            ):
                self.wrong_object_interaction = True
            if stable_delta:
                self.stable_target_grasp = True
                self.stable_target_substeps += stable_delta
                if self.first_stable_target_substep is None:
                    self.first_stable_target_substep = max(
                        1,
                        relative_substep - stable_delta + 1,
                    )
            if getattr(grasp, "stable_object", None) == self.target_name:
                self._current_stable_target_run += stable_delta
                self.longest_stable_target_run = max(
                    self.longest_stable_target_run,
                    self._current_stable_target_run,
                )
            else:
                self._current_stable_target_run = 0
            stable_object = getattr(grasp, "stable_object", None)
            if stable_object is not None and stable_object != self.target_name:
                self.wrong_object_stable_grasp = True
            if (
                bool(getattr(grasp, "dropped_target", False))
                and not self._baseline_dropped
            ):
                self.dropped_target = True
        self.last_observed_substep = max(
            self.last_observed_substep,
            int(getattr(grasp, "tracker_substep", 0)),
        )

    def metrics(self) -> dict[str, object]:
        return {
            "first_bilateral_object": self.first_bilateral_object,
            "target_first_contact": (
                self.first_bilateral_object == self.target_name
            ),
            "target_bilateral_contact": self.target_bilateral_contact,
            "stable_target_grasp": self.stable_target_grasp,
            "first_bilateral_target_substep": self.first_bilateral_target_substep,
            "first_stable_target_substep": self.first_stable_target_substep,
            "stable_target_substeps": self.stable_target_substeps,
            "longest_stable_target_run": self.longest_stable_target_run,
            "wrong_object_interaction": self.wrong_object_interaction,
            "wrong_object_stable_grasp": self.wrong_object_stable_grasp,
            "dropped_target": self.dropped_target,
        }


EVALUATION_CONDITIONS = (
    "id_normal",
    "heldout_recovery",
    "count_ood",
    "crowded_ood",
    "controlled_randomization",
)


def default_evaluation_conditions(
    config: ExperimentConfig,
) -> tuple[str, ...] | None:
    return (
        ("id_normal", "heldout_recovery")
        if config.sequence.enabled
        else None
    )


def resolve_evaluation_model_seeds(
    config: ExperimentConfig, requested: Iterable[int] | None
) -> tuple[int, ...]:
    selected = tuple(
        int(value)
        for value in (
            config.train.model_seeds if requested is None else requested
        )
    )
    if not selected:
        raise ValueError("at least one evaluation model seed is required")
    if len(set(selected)) != len(selected):
        raise ValueError("evaluation model seeds must be unique")
    unknown = tuple(
        seed for seed in selected if seed not in config.train.model_seeds
    )
    if unknown:
        raise ValueError(f"evaluation model seeds are not configured: {unknown}")
    return selected


def resolve_evaluation_conditions(
    cases: Iterable[PhysicsEvaluationCase],
    requested: Iterable[str] | None,
) -> tuple[PhysicsEvaluationCase, ...]:
    values = tuple(cases)
    if requested is None:
        return values
    selected = tuple(str(value) for value in requested)
    if not selected:
        raise ValueError("at least one evaluation condition is required")
    if len(set(selected)) != len(selected):
        raise ValueError("evaluation conditions must be unique")
    unknown = tuple(
        condition
        for condition in selected
        if condition not in EVALUATION_CONDITIONS
    )
    if unknown:
        raise ValueError(f"unknown evaluation conditions: {unknown}")
    allowed = set(selected)
    filtered = tuple(case for case in values if case.condition in allowed)
    if not filtered:
        raise ValueError("selected evaluation conditions produced no cases")
    return filtered


def resolve_evaluation_output_paths(
    config: ExperimentConfig,
    output: str | Path | None,
) -> tuple[Path, Path]:
    evaluation_dir = Path(config.output_dir) / "evaluation"
    default_report = evaluation_dir / "report.json"
    default_csv = evaluation_dir / "episodes.csv"
    if output is None:
        return default_report, default_csv

    report_path = Path(output)
    if report_path.suffix.lower() != ".json":
        raise ValueError("evaluation output must be a .json path")
    csv_path = report_path.with_name(f"{report_path.stem}_episodes.csv")
    resolved_evaluation_dir = evaluation_dir.resolve()
    resolved_paths = (report_path.resolve(), csv_path.resolve())
    if any(
        not path.is_relative_to(resolved_evaluation_dir)
        for path in resolved_paths
    ):
        raise ValueError(
            "custom evaluation output must stay inside the configured evaluation directory"
        )
    protected = {default_report.resolve(), default_csv.resolve()}
    if any(path in protected for path in resolved_paths):
        raise ValueError(
            "custom evaluation output must not overwrite the default report or episode table"
        )
    return report_path, csv_path


def preload_evaluation_checkpoints(
    config: ExperimentConfig,
    *,
    model_seeds: Iterable[int],
    device: torch.device | str,
    physical_hashes: Mapping[str, str],
    expected_training_provenance: Mapping[str, object],
) -> dict[tuple[int, str], tuple[ActionPolicy, TrainingStatistics]]:
    checkpoint_paths = {
        (int(model_seed), representation): (
            Path(config.output_dir)
            / representation
            / f"seed_{model_seed}"
            / "checkpoint.pt"
        )
        for model_seed in model_seeds
        for representation in ("flat", "graph")
    }
    missing = [path for path in checkpoint_paths.values() if not path.is_file()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            f"missing selected evaluation checkpoints:\n{formatted}"
        )

    resolved_device = (
        torch.device(config.sequence.rollout_device)
        if config.sequence.enabled
        else torch.device(device)
    )
    loaded: dict[tuple[int, str], tuple[ActionPolicy, TrainingStatistics]] = {}
    for (model_seed, representation), checkpoint in checkpoint_paths.items():
        policy, statistics, payload = load_training_checkpoint(
            checkpoint,
            resolved_device,
        )
        validate_physics_checkpoint(
            payload,
            expected_provenance=physical_hashes,
            expected_representation=representation,
            expected_model_seed=model_seed,
        )
        if dict(payload.get("training_provenance", {})) != dict(
            expected_training_provenance
        ):
            raise ValueError(
                "checkpoint training provenance does not match the current configured dataset"
            )
        validate_temporal_checkpoint(payload, config)
        loaded[(model_seed, representation)] = (policy, statistics)
    return loaded


def validate_temporal_checkpoint(
    payload: Mapping[str, object],
    config: ExperimentConfig,
) -> None:
    if not config.sequence.enabled:
        return
    expected_rollout_hash = learned_rollout_source_hash()
    if payload.get("learned_rollout_source_hash") != expected_rollout_hash:
        raise ValueError(
            "checkpoint differs for learned_rollout_source_hash"
        )
    expected_temporal = {
        "contribution": "shared_infrastructure",
        "horizon": config.sequence.horizon,
        "future_loss_decay": config.sequence.future_loss_decay,
        "temporal_decay": config.sequence.temporal_decay,
        "gripper_close_threshold": config.sequence.gripper_close_threshold,
        "gripper_open_threshold": config.sequence.gripper_open_threshold,
        "recovery_loss_fraction": config.sequence.recovery_loss_fraction,
        "rollout_device": config.sequence.rollout_device,
    }
    if dict(payload.get("temporal_contract", {})) != expected_temporal:
        raise ValueError("checkpoint temporal contract does not match evaluation config")
    if int(dict(payload.get("model_kwargs", {})).get("action_horizon", 1)) != int(
        config.sequence.horizon
    ):
        raise ValueError("checkpoint action horizon does not match evaluation config")


def make_heldout_recovery_cases(
    config: ExperimentConfig,
    *,
    episodes_per_count: int,
) -> tuple[PhysicsEvaluationCase, ...]:
    if episodes_per_count < 1:
        raise ValueError("episodes_per_count must be positive")
    manifest_path = Path(config.data_dir) / "recovery_benchmark_manifest.json"
    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("held-out recovery manifest must contain accepted episodes")
    counts: Counter[int] = Counter()
    cases: list[PhysicsEvaluationCase] = []
    for raw_record in records:
        if not isinstance(raw_record, dict):
            raise ValueError("held-out recovery manifest records must be mappings")
        record = dict(raw_record)
        object_count = int(record["object_count"])
        if counts[object_count] >= episodes_per_count:
            continue
        source_seed = int(record["source_seed"])
        variant_id = int(record["variant_id"])
        kind = PhysicsRecoveryKind(str(record["kind"]))
        source_split = str(record["source_split"])
        if source_split not in {"validation", "test"}:
            raise ValueError("held-out recovery source must be validation or test")
        cases.append(
            PhysicsEvaluationCase(
                case_id=(
                    f"heldout_recovery:{object_count}:{source_seed}:"
                    f"{variant_id}:{kind.value}"
                ),
                seed=source_seed,
                object_count=object_count,
                condition="heldout_recovery",
                layout_mode="normal",
                recovery_variant_id=variant_id,
                recovery_kind=kind.value,
                source_split=source_split,
            )
        )
        counts[object_count] += 1
    if not cases:
        raise ValueError("held-out recovery filtering produced no cases")
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("held-out recovery manifest produced duplicate cases")
    return tuple(cases)


def make_physics_evaluation_cases(
    config: ExperimentConfig,
    *,
    episodes_per_count: int | None = None,
) -> tuple[PhysicsEvaluationCase, ...]:
    if config.backend != "franka_contact":
        raise ValueError("physical evaluation requires backend=franka_contact")
    resolved_episodes_per_count = (
        config.eval.episodes_per_count
        if episodes_per_count is None
        else int(episodes_per_count)
    )
    if resolved_episodes_per_count < 1:
        raise ValueError("episodes_per_count must be positive")
    groups = (
        ("id_normal", tuple(config.train.object_counts), "normal", False),
        ("count_ood", tuple(config.eval.ood_object_counts), "normal", False),
        ("crowded_ood", tuple(config.eval.crowded_object_counts), "crowded", False),
        (
            "controlled_randomization",
            tuple(config.eval.ood_object_counts),
            "normal",
            True,
        ),
    )
    cases: list[PhysicsEvaluationCase] = []
    for condition_id, (condition, object_counts, layout_mode, randomized) in enumerate(groups):
        for object_count in object_counts:
            for episode_index in range(resolved_episodes_per_count):
                sequence = np.random.SeedSequence(
                    (
                        int(config.seed),
                        0x50455641,
                        condition_id,
                        int(object_count),
                        episode_index,
                    )
                )
                seed = int(sequence.generate_state(1, dtype=np.uint32)[0])
                cases.append(
                    PhysicsEvaluationCase(
                        case_id=f"{condition}:{object_count}:{seed}",
                        seed=seed,
                        object_count=int(object_count),
                        condition=condition,
                        layout_mode=layout_mode,
                        randomized=randomized,
                    )
                )
    if len({case.case_id for case in cases}) != len(cases):
        raise RuntimeError("physical evaluation seed namespace produced a collision")
    return tuple(cases)


def shuffle_valid_physics_edges(batch: SceneBatch, *, seed: int) -> SceneBatch:
    return shuffle_valid_edge_assignments(batch, seed=seed)


def validate_physics_checkpoint(
    payload: Mapping[str, object],
    *,
    expected_provenance: Mapping[str, str] | None = None,
    expected_representation: str | None = None,
    expected_model_seed: int | None = None,
) -> None:
    model_kwargs = dict(payload.get("model_kwargs", {}))
    if payload.get("backend") != "franka_contact":
        raise ValueError("checkpoint backend must be franka_contact")
    if payload.get("feature_schema") != "physics_v2":
        raise ValueError("checkpoint feature schema must be physics_v2")
    if payload.get("action_mode") != "cartesian_7d" or payload.get("action_dim") != 7:
        raise ValueError("physical evaluation requires a 7D Cartesian checkpoint")
    if payload.get("proprioception_dim") != 23:
        raise ValueError("physical evaluation requires 23D proprioception")
    if model_kwargs.get("edge_feature_dim") != 18:
        raise ValueError("physical evaluation requires 18D edge features")
    if expected_provenance is not None and any(
        payload.get(name) != value for name, value in expected_provenance.items()
    ):
        raise ValueError(
            "physical checkpoint is stale for the current gate, controller, scene, or config"
        )
    if (
        expected_representation is not None
        and payload.get("representation") != expected_representation
    ) or (
        expected_model_seed is not None
        and payload.get("model_seed") != expected_model_seed
    ):
        raise ValueError("physical checkpoint identity does not match its evaluation label")


def initial_case_fingerprint(env: FrankaContactEnv) -> str:
    metadata = {
        "seed": env.seed,
        "object_count": env.object_count,
        "target_index": env.target_index,
        "layout_mode": env.layout_mode.value,
        "physics": env.physics_metadata(),
        "timestep": env.physics.timestep,
        "policy_hz": env.physics.policy_hz,
        "substeps": env.physics.substeps,
    }
    digest = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    )
    digest.update(np.asarray(env.data.qpos, dtype=np.float64).tobytes())
    digest.update(np.asarray(env.data.qvel, dtype=np.float64).tobytes())
    return digest.hexdigest()


def _make_env(
    config: ExperimentConfig, *, randomized: bool = False
) -> FrankaContactEnv:
    randomization = replace(config.physics.randomization, enabled=randomized)
    physics = replace(config.physics, randomization=randomization)
    return FrankaContactEnv(
        max_objects=config.max_objects,
        max_steps=config.eval.max_steps,
        min_object_distance=config.environment.min_object_distance,
        workspace_low=config.environment.workspace_low,
        workspace_high=config.environment.workspace_high,
        crowded_anchor_min_distance=config.environment.crowded_anchor_min_distance,
        crowded_anchor_max_distance=config.environment.crowded_anchor_max_distance,
        physics=physics,
    )


def _normalized_policy_inputs(
    env: FrankaContactEnv,
    builder: SceneGraphBuilder,
    statistics: TrainingStatistics,
    device: torch.device,
    *,
    shuffle_seed: int | None,
) -> tuple[SceneBatch, torch.Tensor]:
    graph = builder.build(env.snapshot())
    scene = SceneBatch(
        node_features=torch.from_numpy(graph.node_features[None]).float(),
        edge_index=torch.from_numpy(graph.edge_index).long(),
        edge_features=torch.from_numpy(graph.edge_features[None]).float(),
        node_mask=torch.from_numpy(graph.node_mask[None]).bool(),
        edge_mask=torch.from_numpy(graph.edge_mask[None]).bool(),
    )
    scene = statistics.normalize_scene(scene)
    if shuffle_seed is not None:
        scene = shuffle_valid_physics_edges(scene, seed=shuffle_seed)
    proprioception = statistics.normalize_proprioception(
        torch.from_numpy(env.proprioception()[None]).float()
    )
    return scene.to(device), proprioception.to(device)


def _grasp_counter_baseline(env: FrankaContactEnv) -> dict[str, int | bool | str | None]:
    grasp = env.grasp_state
    return {
        "tracker_substep": int(grasp.tracker_substep),
        "ever_bilateral_target_contact": bool(
            grasp.ever_bilateral_target_contact
        ),
        "ever_stable_target": bool(grasp.ever_stable_target),
        "ever_bilateral_wrong_object": bool(grasp.ever_bilateral_wrong_object),
        "total_bilateral_target_substeps": int(
            grasp.total_bilateral_target_substeps
        ),
        "total_bilateral_wrong_substeps": int(
            grasp.total_bilateral_wrong_substeps
        ),
        "ever_stable_wrong_object": bool(grasp.ever_stable_wrong_object),
        "total_stable_target_substeps": int(grasp.total_stable_target_substeps),
        "dropped_target": bool(grasp.dropped_target),
    }


def _prepare_evaluation_case(
    env: FrankaContactEnv,
    config: ExperimentConfig,
    case: PhysicsEvaluationCase,
):
    if case.condition != "heldout_recovery":
        snapshot = env.reset(
            seed=case.seed,
            object_count=case.object_count,
            layout_mode=case.layout_mode,
        )
        return snapshot, _grasp_counter_baseline(env)
    if (
        case.recovery_variant_id is None
        or case.recovery_kind is None
        or case.source_split is None
    ):
        raise ValueError("held-out recovery case is missing intervention metadata")
    kind = PhysicsRecoveryKind(case.recovery_kind)
    spec = make_physics_recovery_spec(
        case.seed,
        case.recovery_variant_id,
        kind_index=tuple(PhysicsRecoveryKind).index(kind),
    )
    prepared = prepare_physics_recovery_start(
        env,
        PhysicsScriptedExpert(config.physics),
        spec=spec,
        object_count=case.object_count,
        source_split=case.source_split,
        layout_mode=case.layout_mode,
    )
    return prepared.snapshot, dict(prepared.interaction_baseline)


@torch.no_grad()
def rollout_physics_policy(
    *,
    policy_name: str,
    representation: str,
    model_seed: int,
    policy: ActionPolicy,
    statistics: TrainingStatistics,
    config: ExperimentConfig,
    case: PhysicsEvaluationCase,
    device: torch.device | str,
    edge_shuffle: bool = False,
    ik_projection: bool = True,
    action_diagnostics: list[dict[str, object]] | None = None,
) -> PhysicsEpisodeResult:
    sequence_enabled = bool(config.sequence.enabled)
    if sequence_enabled and not ik_projection:
        raise ValueError("v3 chunked evaluation requires shared IK projection")
    resolved_device = torch.device(
        config.sequence.rollout_device if sequence_enabled else device
    )
    env = _make_env(config, randomized=case.randomized)
    snapshot, interaction_baseline = _prepare_evaluation_case(env, config, case)
    initial_hash = initial_case_fingerprint(env)
    if case.condition == "heldout_recovery":
        # Give the learned suffix its configured rollout horizon without touching
        # MuJoCo state or resetting cumulative interaction diagnostics.
        env.step_count = 0
    physics_hash = str(env.physics_metadata()["physics_hash"])
    builder = SceneGraphBuilder(max_objects=config.max_objects, feature_schema="physics_v2")
    policy.to(resolved_device).eval()
    learned_interactions = InteractionRolloutTracker(
        target_name=snapshot.target_object.name,
        baseline=interaction_baseline,
    )
    chunked_controller = None
    if sequence_enabled:
        chunked_controller = ChunkedPolicyController(
            policy=policy,
            statistics=statistics,
            builder=builder,
            horizon=config.sequence.horizon,
            temporal_decay=config.sequence.temporal_decay,
            gripper_close_threshold=config.sequence.gripper_close_threshold,
            gripper_open_threshold=config.sequence.gripper_open_threshold,
            device=resolved_device,
            edge_shuffle=edge_shuffle,
            edge_shuffle_seed=case.seed,
        )
        chunked_controller.reset(env)
    strict_containment = False
    receptacle_base_contact = False
    receptacle_wall_contact = False
    strict_placement = False
    gripper_released = False
    tcp_retreated = False
    ik_failure = False
    physics_failure = False
    reason = TerminationReason.TIMEOUT
    transport_start_distance: float | None = None
    transport_best_distance: float | None = None
    premature_open = False
    post_placement_reclose = False
    saturated_steps = 0
    projection_scales: list[float] = []
    ensemble_sizes: list[int] = []
    final_gripper_switch_count = 0

    for step_index in range(config.eval.max_steps):
        if chunked_controller is not None:
            action, controller_diagnostics = chunked_controller.act(env)
            raw_current = controller_diagnostics.raw_first_action
            aggregated = controller_diagnostics.aggregated_action
            raw_gripper_score = controller_diagnostics.raw_gripper_score
            ensemble_size = controller_diagnostics.ensemble_size
            smoothing_delta = controller_diagnostics.smoothing_delta_norm
            final_gripper_switch_count = (
                controller_diagnostics.gripper_switch_count
            )
            projection_scale = controller_diagnostics.ik_projection_scale
            saturated_steps += int(
                np.any(np.abs(controller_diagnostics.aggregated_action[:6]) >= 0.95)
            )
        else:
            scene, proprioception = _normalized_policy_inputs(
                env,
                builder,
                statistics,
                resolved_device,
                shuffle_seed=(
                    case.seed ^ step_index ^ 0x45444745
                    if edge_shuffle
                    else None
                ),
            )
            action_tensor = policy(
                scene if policy.scene_encoder is not None else None,
                proprioception,
            )
            action = action_tensor[0].detach().cpu().numpy().astype(np.float32)
            action[:6] = np.clip(action[:6], -1.0, 1.0)
            action[6] = np.clip(action[6], 0.0, 1.0)
            raw_current = action.copy()
            aggregated = action.copy()
            raw_gripper_score = float(action[6])
            ensemble_size = 1
            smoothing_delta = 0.0
            saturated_steps += int(np.any(np.abs(action[:6]) >= 0.95))
            if ik_projection:
                projection = project_cartesian_action(env.controller, action)
                action = projection.action
                projection_scale = float(projection.scale)
            else:
                projection_scale = 1.0
        projection_scales.append(projection_scale)
        ensemble_sizes.append(ensemble_size)
        if action_diagnostics is not None:
            action_diagnostics.append(
                {
                    "case_id": case.case_id,
                    "representation": representation,
                    "model_seed": int(model_seed),
                    "step": step_index,
                    "raw_current_chunk_row_0": raw_current.tolist(),
                    "aggregated_action_before_ik": aggregated.tolist(),
                    "executed_action": action.tolist(),
                    "ensemble_size": int(ensemble_size),
                    "smoothing_delta_norm": float(smoothing_delta),
                    "raw_gripper_score": float(raw_gripper_score),
                    "discrete_gripper_state": float(aggregated[6]),
                    "gripper_switch_count": int(final_gripper_switch_count),
                    "ik_projection_scale": float(projection_scale),
                }
            )
        if transport_start_distance is not None:
            before = env.snapshot()
            before_distance = float(
                np.linalg.norm(
                    before.target_object.position[:2]
                    - before.receptacle.position[:2]
                )
            )
            premature_open |= bool(
                before_distance > 0.065 and float(action[6]) >= 0.5
            )
        if strict_placement and float(action[6]) < 0.5:
            post_placement_reclose = True
        transition = env.step(action)
        reason = transition.reason
        learned_interactions.observe(
            env.grasp_state,
            events=env.grasp_tracker.interaction_events_since(
                learned_interactions.last_observed_substep
            ),
        )
        interaction_values = learned_interactions.metrics()
        placement = env.last_placement
        strict_containment |= bool(placement.fully_contained)
        receptacle_base_contact |= bool(placement.base_contact)
        receptacle_wall_contact |= bool(placement.wall_contact)
        strict_placement |= bool(
            placement.strict_stable
            or (
                not sequence_enabled
                and transition.info.get("stable_placement", False)
            )
        )
        finger_open = bool(float(np.mean(env.proprioception()[13:15])) >= 0.03)
        gripper_released |= strict_placement and finger_open
        tcp_position, _ = env.controller.tcp_pose()
        target_z = float(env.data.body(env.target_name).xpos[2])
        tcp_retreated |= gripper_released and tcp_position[2] >= target_z + 0.08
        failure = str(transition.info.get("physics_failure", ""))
        physics_failure |= transition.reason is TerminationReason.PHYSICS_FAILURE
        ik_failure |= failure == "ik_limited"
        if bool(interaction_values["stable_target_grasp"]):
            distance = float(
                np.linalg.norm(
                    transition.snapshot.target_object.position[:2]
                    - transition.snapshot.receptacle.position[:2]
                )
            )
            if transport_start_distance is None:
                transport_start_distance = distance
                transport_best_distance = distance
            else:
                assert transport_best_distance is not None
                transport_best_distance = min(transport_best_distance, distance)
        if transition.done:
            break

    transport_progress = 0.0
    transport_progress_rate = 0.0
    if transport_start_distance is not None and transport_best_distance is not None:
        transport_progress = max(
            0.0, transport_start_distance - transport_best_distance
        )
        if transport_start_distance > 1e-8:
            transport_progress_rate = float(
                np.clip(transport_progress / transport_start_distance, 0.0, 1.0)
            )
    executed_steps = len(projection_scales)
    interaction_values = learned_interactions.metrics()
    strict_task_success = reason is TerminationReason.SUCCESS
    margin_x = float(env.last_placement.containment_margin[0])
    margin_y = float(env.last_placement.containment_margin[1])

    return PhysicsEpisodeResult(
        policy=policy_name,
        representation=representation,
        model_seed=int(model_seed),
        ablation="edge_shuffle" if edge_shuffle else "none",
        case_id=case.case_id,
        seed=case.seed,
        object_count=case.object_count,
        condition=case.condition,
        layout_mode=case.layout_mode,
        success=strict_task_success,
        bilateral_contact=bool(interaction_values["target_bilateral_contact"]),
        stable_lift=bool(interaction_values["stable_target_grasp"]),
        wrong_object_stable_grasp=bool(
            interaction_values["wrong_object_stable_grasp"]
        ),
        dropped=bool(interaction_values["dropped_target"]),
        placement=strict_placement,
        ik_failure=ik_failure,
        physics_failure=physics_failure,
        steps=executed_steps,
        termination_reason=reason.value,
        physics_hash=physics_hash,
        initial_state_hash=initial_hash,
        transport_progress=transport_progress,
        transport_progress_rate=transport_progress_rate,
        premature_open=premature_open,
        action_saturation_rate=float(saturated_steps / executed_steps),
        ik_projection_rate=float(
            np.mean([scale < 1.0 for scale in projection_scales])
        ),
        zero_pose_projection_rate=float(
            np.mean([scale == 0.0 for scale in projection_scales])
        ),
        mean_ik_projection_scale=float(np.mean(projection_scales)),
        post_placement_reclose=post_placement_reclose,
        first_bilateral_object=interaction_values["first_bilateral_object"],
        target_first_contact=bool(interaction_values["target_first_contact"]),
        target_bilateral_contact=bool(
            interaction_values["target_bilateral_contact"]
        ),
        stable_target_grasp=bool(interaction_values["stable_target_grasp"]),
        first_bilateral_target_substep=interaction_values[
            "first_bilateral_target_substep"
        ],
        first_stable_target_substep=interaction_values[
            "first_stable_target_substep"
        ],
        stable_target_substeps=int(interaction_values["stable_target_substeps"]),
        longest_stable_target_run=int(
            interaction_values["longest_stable_target_run"]
        ),
        wrong_object_interaction=bool(
            interaction_values["wrong_object_interaction"]
        ),
        dropped_target=bool(interaction_values["dropped_target"]),
        strict_containment=strict_containment,
        receptacle_base_contact=receptacle_base_contact,
        receptacle_wall_contact=receptacle_wall_contact,
        strict_placement=strict_placement,
        gripper_released=gripper_released,
        tcp_retreated=tcp_retreated,
        strict_task_success=strict_task_success,
        containment_margin_x=margin_x,
        containment_margin_y=margin_y,
        rollout_device=resolved_device.type,
        mean_ensemble_size=float(np.mean(ensemble_sizes)),
        gripper_switch_count=final_gripper_switch_count,
    )


def _metrics(results: Iterable[PhysicsEpisodeResult]) -> dict[str, object]:
    values = tuple(results)
    if not values:
        return {"episodes": 0}
    contacted = [value for value in values if value.target_bilateral_contact]
    primary_interaction = {
        "target_first_contact_rate": float(
            np.mean([value.target_first_contact for value in values])
        ),
        "target_bilateral_contact_rate": float(
            np.mean([value.target_bilateral_contact for value in values])
        ),
        "stable_target_grasp_rate": float(
            np.mean([value.stable_target_grasp for value in values])
        ),
        "stable_lift_rate": float(np.mean([value.stable_lift for value in values])),
        "grasp_given_contact_rate": (
            float(np.mean([value.stable_target_grasp for value in contacted]))
            if contacted
            else None
        ),
        "mean_stable_target_substeps": float(
            np.mean([value.stable_target_substeps for value in values])
        ),
        "wrong_object_interaction_rate": float(
            np.mean([value.wrong_object_interaction for value in values])
        ),
        "wrong_object_stable_grasp_rate": float(
            np.mean([value.wrong_object_stable_grasp for value in values])
        ),
        "target_drop_rate": float(
            np.mean([value.dropped_target for value in values])
        ),
        "mean_transport_progress_rate": float(
            np.mean([value.transport_progress_rate for value in values])
        ),
    }
    secondary_task = {
        "strict_placement_rate": float(
            np.mean([value.strict_placement for value in values])
        ),
        "released_and_retreated_rate": float(
            np.mean(
                [
                    value.gripper_released and value.tcp_retreated
                    for value in values
                ]
            )
        ),
        "strict_task_success_rate": float(
            np.mean([value.strict_task_success for value in values])
        ),
        "wall_contact_without_containment_rate": float(
            np.mean(
                [
                    value.receptacle_wall_contact
                    and not value.strict_containment
                    for value in values
                ]
            )
        ),
    }
    return {
        "episodes": len(values),
        "primary_interaction": primary_interaction,
        "secondary_task": secondary_task,
        "success_rate": float(np.mean([value.success for value in values])),
        "bilateral_contact_rate": float(
            np.mean([value.bilateral_contact for value in values])
        ),
        "stable_lift_rate": float(np.mean([value.stable_lift for value in values])),
        "wrong_object_stable_grasp_rate": float(
            np.mean([value.wrong_object_stable_grasp for value in values])
        ),
        "drop_rate": float(np.mean([value.dropped for value in values])),
        "placement_rate": float(np.mean([value.placement for value in values])),
        "ik_failure_rate": float(np.mean([value.ik_failure for value in values])),
        "physics_failure_rate": float(
            np.mean([value.physics_failure for value in values])
        ),
        "mean_transport_progress": float(
            np.mean([value.transport_progress for value in values])
        ),
        "mean_transport_progress_rate": float(
            np.mean([value.transport_progress_rate for value in values])
        ),
        "premature_open_rate": float(
            np.mean([value.premature_open for value in values])
        ),
        "post_placement_reclose_rate": float(
            np.mean([value.post_placement_reclose for value in values])
        ),
        "action_saturation_rate": float(
            np.mean([value.action_saturation_rate for value in values])
        ),
        "ik_projection_rate": float(
            np.mean([value.ik_projection_rate for value in values])
        ),
        "zero_pose_projection_rate": float(
            np.mean([value.zero_pose_projection_rate for value in values])
        ),
        "mean_ik_projection_scale": float(
            np.mean([value.mean_ik_projection_scale for value in values])
        ),
        "termination_reason_counts": dict(
            sorted(Counter(value.termination_reason for value in values).items())
        ),
        "mean_steps": float(np.mean([value.steps for value in values])),
    }


def _paired_deltas(
    results: tuple[PhysicsEpisodeResult, ...],
    *,
    left_representation: str,
    left_ablation: str,
    right_representation: str,
    right_ablation: str,
) -> dict[str, object]:
    by_seed: dict[str, dict[str, float | int]] = {}
    for model_seed in sorted({value.model_seed for value in results}):
        left = {
            value.case_id: value
            for value in results
            if value.representation == left_representation
            and value.model_seed == model_seed
            and value.ablation == left_ablation
        }
        right = {
            value.case_id: value
            for value in results
            if value.representation == right_representation
            and value.model_seed == model_seed
            and value.ablation == right_ablation
        }
        keys = sorted(set(left) & set(right))
        if not keys:
            continue
        for key in keys:
            if (
                left[key].initial_state_hash != right[key].initial_state_hash
                or left[key].physics_hash != right[key].physics_hash
            ):
                raise ValueError(f"unpaired physical initial state for case {key}")

        def deltas(selected: list[str]) -> dict[str, float | int]:
            return {
                "paired_cases": len(selected),
                "target_bilateral_contact_delta": float(
                    np.mean(
                        [
                            float(right[key].target_bilateral_contact)
                            - float(left[key].target_bilateral_contact)
                            for key in selected
                        ]
                    )
                ),
                "stable_target_grasp_delta": float(
                    np.mean(
                        [
                            float(right[key].stable_target_grasp)
                            - float(left[key].stable_target_grasp)
                            for key in selected
                        ]
                    )
                ),
                "success_delta": float(
                    np.mean(
                        [
                            float(right[key].success) - float(left[key].success)
                            for key in selected
                        ]
                    )
                ),
                "stable_lift_delta": float(
                    np.mean(
                        [
                            float(right[key].stable_lift)
                            - float(left[key].stable_lift)
                            for key in selected
                        ]
                    )
                ),
                "wrong_object_delta": float(
                    np.mean(
                        [
                            float(right[key].wrong_object_stable_grasp)
                            - float(left[key].wrong_object_stable_grasp)
                            for key in selected
                        ]
                    )
                ),
                "transport_progress_rate_delta": float(
                    np.mean(
                        [
                            right[key].transport_progress_rate
                            - left[key].transport_progress_rate
                            for key in selected
                        ]
                    )
                ),
                "premature_open_delta": float(
                    np.mean(
                        [
                            float(right[key].premature_open)
                            - float(left[key].premature_open)
                            for key in selected
                        ]
                    )
                ),
                "strict_placement_delta": float(
                    np.mean(
                        [
                            float(right[key].strict_placement)
                            - float(left[key].strict_placement)
                            for key in selected
                        ]
                    )
                ),
                "strict_task_success_delta": float(
                    np.mean(
                        [
                            float(right[key].strict_task_success)
                            - float(left[key].strict_task_success)
                            for key in selected
                        ]
                    )
                ),
            }

        metrics = deltas(keys)
        metrics["by_condition"] = {
            condition: deltas(
                [key for key in keys if left[key].condition == condition]
            )
            for condition in sorted({left[key].condition for key in keys})
        }
        by_seed[str(model_seed)] = metrics
    if not by_seed:
        return {"by_model_seed": {}}
    return {
        "sign_convention": "right_minus_left",
        "left": f"{left_representation}:{left_ablation}",
        "right": f"{right_representation}:{right_ablation}",
        "by_model_seed": by_seed,
    }


def aggregate_physics_results(
    results: Iterable[PhysicsEpisodeResult],
) -> dict[str, object]:
    values = tuple(results)
    policies = sorted({value.policy for value in values})
    conditions = sorted({value.condition for value in values})
    learned_policy_sanity: dict[str, dict[str, object]] = {}
    for policy in policies:
        id_metrics = _metrics(
            value
            for value in values
            if value.policy == policy and value.condition == "id_normal"
        )
        if id_metrics["episodes"] == 0:
            continue
        physics_failure_rate = float(id_metrics["physics_failure_rate"])
        stable_lift_rate = float(id_metrics["stable_lift_rate"])
        control_passed = physics_failure_rate <= 0.10
        manipulation_passed = stable_lift_rate >= 0.10
        learned_policy_sanity[policy] = {
            "episodes": int(id_metrics["episodes"]),
            "control_passed": control_passed,
            "manipulation_passed": manipulation_passed,
            "passed": control_passed and manipulation_passed,
            "physics_failure_rate": physics_failure_rate,
            "stable_lift_rate": stable_lift_rate,
        }
    return {
        "overall": _metrics(values),
        "by_policy": {
            policy: _metrics(value for value in values if value.policy == policy)
            for policy in policies
        },
        "by_policy_and_condition": {
            policy: {
                condition: _metrics(
                    value
                    for value in values
                    if value.policy == policy and value.condition == condition
                )
                for condition in conditions
            }
            for policy in policies
        },
        "graph_vs_flat": _paired_deltas(
            values,
            left_representation="flat",
            left_ablation="none",
            right_representation="graph",
            right_ablation="none",
        ),
        "graph_vs_edge_shuffle": _paired_deltas(
            values,
            left_representation="graph",
            left_ablation="edge_shuffle",
            right_representation="graph",
            right_ablation="none",
        ),
        "learned_policy_sanity": learned_policy_sanity,
    }


def evaluate_from_config(
    config_path: str | Path,
    *,
    model_seeds: Iterable[int] | None = None,
    include_edge_shuffle: bool = False,
    episodes_per_count: int | None = None,
    conditions: Iterable[str] | None = None,
    output: str | Path | None = None,
    ik_projection: bool = True,
    show_progress: bool = False,
) -> Path:
    config = load_config(config_path)
    report_path, csv_path = resolve_evaluation_output_paths(config, output)
    selected_seeds = resolve_evaluation_model_seeds(config, model_seeds)
    resolved_episodes_per_count = (
        config.eval.episodes_per_count
        if episodes_per_count is None
        else int(episodes_per_count)
    )
    requested_conditions = (
        default_evaluation_conditions(config)
        if conditions is None
        else conditions
    )
    generated_cases = list(
        make_physics_evaluation_cases(
            config,
            episodes_per_count=resolved_episodes_per_count,
        )
    )
    if config.sequence.enabled and (
        requested_conditions is None
        or "heldout_recovery" in set(requested_conditions)
    ):
        generated_cases.extend(
            make_heldout_recovery_cases(
                config,
                episodes_per_count=resolved_episodes_per_count,
            )
        )
    cases = resolve_evaluation_conditions(
        generated_cases,
        requested_conditions,
    )
    policy_variants = ["flat", "graph"]
    if include_edge_shuffle:
        policy_variants.append("graph_edge_shuffle")
    device = (
        torch.device(config.sequence.rollout_device)
        if config.sequence.enabled
        else resolve_device(config.device)
    )
    from .physics_data import expert_gate_provenance

    physical_hashes = expert_gate_provenance(
        config_path, Path(config.output_dir) / "expert_gate.json"
    )
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
    if config.sequence.enabled:
        expected_training_provenance.update(
            build_sequence_provenance_fields(
                config,
                data_selection,
                model_seed=selected_seeds[0],
            )
        )
    loaded_checkpoints = preload_evaluation_checkpoints(
        config,
        model_seeds=selected_seeds,
        device=device,
        physical_hashes=physical_hashes,
        expected_training_provenance=expected_training_provenance,
    )
    results: list[PhysicsEpisodeResult] = []
    step_diagnostics: list[dict[str, object]] = []
    progress = (
        tqdm(
            total=len(cases) * len(selected_seeds) * len(policy_variants),
            desc="physics eval",
            unit="rollout",
            dynamic_ncols=True,
        )
        if show_progress
        else None
    )

    def record_result(
        result: PhysicsEpisodeResult, *, policy_variant: str
    ) -> None:
        results.append(result)
        if progress is not None:
            progress.set_postfix(
                seed=result.model_seed,
                policy=policy_variant,
                condition=result.condition,
                objects=result.object_count,
                success=int(result.success),
                stable_target=int(result.stable_target_grasp),
            )
            progress.update(1)

    try:
        for model_seed in selected_seeds:
            for representation in ("flat", "graph"):
                policy, statistics = loaded_checkpoints[
                    (model_seed, representation)
                ]
                for case in cases:
                    record_result(
                        rollout_physics_policy(
                            policy_name=f"{representation}_seed_{model_seed}",
                            representation=representation,
                            model_seed=model_seed,
                            policy=policy,
                            statistics=statistics,
                            config=config,
                            case=case,
                            device=device,
                            ik_projection=ik_projection,
                            action_diagnostics=step_diagnostics,
                        ),
                        policy_variant=representation,
                    )
                    if include_edge_shuffle and representation == "graph":
                        record_result(
                            rollout_physics_policy(
                                policy_name=f"graph_edge_shuffle_seed_{model_seed}",
                                representation="graph",
                                model_seed=model_seed,
                                policy=policy,
                                statistics=statistics,
                                config=config,
                                case=case,
                                device=device,
                                edge_shuffle=True,
                                ik_projection=ik_projection,
                                action_diagnostics=step_diagnostics,
                            ),
                            policy_variant="graph_edge_shuffle",
                        )
    finally:
        if progress is not None:
            progress.close()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(value) for value in results]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    diagnostics_path = (
        report_path.parent / "action_diagnostics.jsonl"
        if report_path.name == "report.json"
        else report_path.with_name(
            f"{report_path.stem}_action_diagnostics.jsonl"
        )
    )
    with diagnostics_path.open("w", encoding="utf-8") as handle:
        for row in step_diagnostics:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    diagnostics_hash = hashlib.sha256(diagnostics_path.read_bytes()).hexdigest()
    report = aggregate_physics_results(results)
    report["action_diagnostics"] = {
        "path": str(diagnostics_path.relative_to(Path(config.output_dir))),
        "sha256": diagnostics_hash,
        "steps": len(step_diagnostics),
    }
    report["code_provenance"] = {
        "learned_rollout_source_hash": learned_rollout_source_hash(),
        "training_pipeline_source_hash": training_pipeline_source_hash(),
    }
    report["evaluation_scope"] = {
        "model_seeds": list(selected_seeds),
        "episodes_per_count": resolved_episodes_per_count,
        "case_count": len(cases),
        "rollout_count": len(results),
        "policy_variants": policy_variants,
        "conditions": [
            condition
            for condition in EVALUATION_CONDITIONS
                if any(case.condition == condition for case in cases)
        ],
        "ik_projection_enabled": bool(ik_projection),
        "ik_projection_scales": list(DEFAULT_IK_PROJECTION_SCALES),
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paired Franka Graph/Flat evaluation")
    parser.add_argument("--config", default="configs/physics_smoke_macos.yaml")
    parser.add_argument("--model-seeds", nargs="+", type=int)
    parser.add_argument("--episodes-per-count", type=int)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=EVALUATION_CONDITIONS,
        help="Evaluate only the selected condition groups",
    )
    parser.add_argument(
        "--output",
        help="Write a separate JSON report (and sibling *_episodes.csv)",
    )
    parser.add_argument("--disable-ik-projection", action="store_true")
    parser.add_argument("--include-edge-shuffle", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(
        evaluate_from_config(
            args.config,
            model_seeds=args.model_seeds,
            include_edge_shuffle=args.include_edge_shuffle,
            episodes_per_count=args.episodes_per_count,
            conditions=args.conditions,
            output=args.output,
            ik_projection=not args.disable_ik_projection,
            show_progress=True,
        )
    )


if __name__ == "__main__":
    main()
