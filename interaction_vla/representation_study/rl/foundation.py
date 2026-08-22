from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import tempfile
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

from interaction_vla.lerobot_bridge.config import BridgeConfig, load_bridge_config
from interaction_vla.lerobot_bridge.provenance import (
    fingerprint_tree,
    sha256_file,
)
from interaction_vla.physics_data import PhysicsRecoveryRejected

from ..backends import make_backend
from ..state_bank.io import write_json_atomic
from ..taps.registry import registered_taps
from .actors import LatentResidualActor, OracleResidualActor, ResidualActor
from .anchoring import NominalAnchorCache
from .checkpoint import capture_rng_state, restore_rng_state
from .core import generalized_advantage_estimate, normalized_curve_auc
from .critics import OracleTwinQ, OracleValueCritic
from .distributions import (
    RecoveryCase,
    RecoveryCaseManifest,
    RecoveryCaseSampler,
    build_case_manifest,
    load_case_manifest,
    save_case_manifest,
)
from .environment import ResidualMujocoRuntime
from .evaluation_v2 import EpisodeOutcome, EvaluationReport, evaluate_case_manifest
from .gates import (
    AnchoringScreen,
    BackendScreen,
    CalibrationCandidate,
    GateDecision,
    oracle_gate,
    select_anchoring,
    select_backend,
    select_calibrated_severity,
)
from .oracle_state import (
    CompactOracleStateCodec,
    OracleNormalization,
    load_oracle_normalization,
)
from .ppo_v2 import OnPolicyBatch, PPOV2
from .protocol import GATE_SCHEMA, write_gate_atomic
from .replay import RecoveryReplay
from .sac import SAC, SACBatch
from .training import (
    collate_packed_observations,
    pack_observation,
    recompute_latents_with_policy_seeds,
)
from .v2_config import RecoveryRLV2Config


FOUNDATION_REPORT_SCHEMA = "recovery_rl_foundation_v2"
SCREEN_EVALUATION_INTERVAL = 1024
SCREEN_SEED_COUNT = 2
ANCHOR_VARIANTS = ("no_anchor", "residual_only", "full_anchoring")


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _immutable_json(path: Path, value: Mapping[str, object]) -> Path:
    encoded = json.dumps(dict(value), indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") == encoded:
            return path
        raise FileExistsError(f"completed report is immutable: {path}")
    write_json_atomic(path, dict(value))
    return path


def _artifact_hash(path: str | Path) -> str:
    source = Path(path)
    if source.is_dir():
        return fingerprint_tree(source)
    if source.is_file():
        return sha256_file(source)
    raise FileNotFoundError(f"required artifact does not exist: {source}")


def foundation_binding(config: RecoveryRLV2Config) -> str:
    return _canonical_hash(
        {
            "schema_version": config.schema_version,
            "config_sha256": sha256_file(config.config_path),
            "bridge_sha256": sha256_file(config.bridge_config),
            "sft_checkpoint": config.sft_checkpoint,
            "sft_checkpoint_sha256": _artifact_hash(config.sft_checkpoint),
            "continued_sft_checkpoint": config.continued_sft_checkpoint,
            "distribution": asdict(config.distribution),
            "reward": {
                "gamma": config.gamma,
                "progress_coefficient": config.progress_coefficient,
                "residual_coefficient": config.residual_coefficient,
                "residual_scale": list(config.residual_scale),
            },
            "ppo": asdict(config.ppo),
            "sac": asdict(config.sac),
            "seed": config.seed,
        }
    )


def _action_proximal_tap() -> str:
    matches = [
        tap.tap_id for tap in registered_taps("act") if tap.role == "action_proximal"
    ]
    if len(matches) != 1:
        raise RuntimeError("ACT must register exactly one action-proximal tap")
    return matches[0]


def _max_episode_steps(bridge: BridgeConfig) -> int:
    if bridge.recovery is None:
        raise ValueError("Recovery RL v2 bridge must define recovery.max_steps")
    return bridge.recovery.max_steps


@contextmanager
def _runtime(
    config: RecoveryRLV2Config,
    *,
    seed: int,
    residual_coefficient: float | None = None,
    checkpoint: str | None = None,
) -> Iterator[ResidualMujocoRuntime]:
    bridge = load_bridge_config(config.bridge_config)
    backend = make_backend("act", device=config.device)
    backend.load_checkpoint_for_dataset(
        config.sft_checkpoint if checkpoint is None else checkpoint,
        repo_id=bridge.dataset.repo_id,
        dataset_root=bridge.dataset.root,
    )
    normalization_path = config.output_dir / "manifests" / "oracle_normalization.json"
    normalization = (
        load_oracle_normalization(normalization_path)
        if normalization_path.is_file()
        else OracleNormalization()
    )
    environment = ResidualMujocoRuntime(
        bridge=bridge,
        backend=backend,
        tap_id=_action_proximal_tap(),
        residual_scale=config.residual_scale,
        max_steps=_max_episode_steps(bridge),
        object_counts=bridge.dataset.object_counts,
        layouts=("normal", "crowded"),
        seed=seed,
        reward_mode="recovery_v2",
        progress_reward_scale=0.0,
        oracle_codec=CompactOracleStateCodec(normalization=normalization),
        recovery_gamma=config.gamma,
        recovery_progress_coefficient=config.progress_coefficient,
        recovery_residual_coefficient=(
            config.residual_coefficient
            if residual_coefficient is None
            else float(residual_coefficient)
        ),
    )
    try:
        yield environment
    finally:
        environment.close()


class _ZeroResidualPolicy:
    def act(
        self,
        *,
        latent: torch.Tensor,
        oracle_state: np.ndarray,
        deterministic: bool,
    ) -> np.ndarray:
        del latent, oracle_state, deterministic
        return np.zeros(7, dtype=np.float32)


class _ActorEvaluationPolicy:
    def __init__(
        self,
        actor: ResidualActor,
        *,
        observation: str,
        device: torch.device,
    ) -> None:
        if observation not in {"oracle", "latent"}:
            raise ValueError("evaluation actor observation must be oracle or latent")
        self.actor = actor
        self.observation = observation
        self.device = device

    def act(
        self,
        *,
        latent: torch.Tensor,
        oracle_state: np.ndarray,
        deterministic: bool,
    ) -> np.ndarray:
        value = (
            torch.as_tensor(oracle_state, dtype=torch.float32, device=self.device)
            .reshape(1, -1)
            if self.observation == "oracle"
            else latent.to(self.device)
        )
        with torch.no_grad():
            sample = self.actor.sample(value, deterministic=deterministic)
        return sample.residual[0].detach().cpu().numpy().astype(np.float32)


def paired_evaluation_cases(
    manifest: RecoveryCaseManifest,
    *,
    partition: str,
    count: int,
) -> tuple[RecoveryCase, ...]:
    if count < 1:
        raise ValueError("paired evaluation case count must be positive")
    sources = manifest.source_seeds(partition)
    if len(sources) < count:
        raise ValueError(
            f"{partition} partition has {len(sources)} sources but {count} are required"
        )
    nominal: list[RecoveryCase] = []
    recovery: list[RecoveryCase] = []
    by_source: dict[int, tuple[RecoveryCase, ...]] = {
        source: tuple(
            case
            for case in manifest.partition(partition)
            if case.source_seed == source
        )
        for source in sources[:count]
    }
    for index, source in enumerate(sources[:count]):
        cases = by_source[source]
        nominal_rows = tuple(case for case in cases if case.family == "nominal")
        recovery_rows = tuple(
            sorted(
                (case for case in cases if case.family == "recovery"),
                key=lambda case: (case.variant_id, case.case_id),
            )
        )
        if len(nominal_rows) != 1 or len(recovery_rows) < 1:
            raise ValueError("paired source lacks nominal or recovery cases")
        nominal.append(nominal_rows[0])
        recovery.append(recovery_rows[index % len(recovery_rows)])
    return tuple((*nominal, *recovery))


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _evaluate_rows_with_rejections(
    policy: Any,
    runtime: ResidualMujocoRuntime,
    cases: Sequence[RecoveryCase],
    *,
    policy_seed: int,
    description: str | None = None,
) -> tuple[tuple[EpisodeOutcome, ...], tuple[dict[str, object], ...]]:
    rows: list[EpisodeOutcome] = []
    rejected: list[dict[str, object]] = []
    progress = tqdm(
        cases,
        desc=description,
        unit="episode",
        dynamic_ncols=True,
        disable=description is None,
    )
    for case in progress:
        try:
            report = evaluate_case_manifest(
                policy,
                runtime,
                (case,),
                policy_seed=policy_seed,
            )
        except PhysicsRecoveryRejected as error:
            rejected.append(
                {
                    "case_id": case.case_id,
                    "source_seed": case.source_seed,
                    "variant_id": case.variant_id,
                    "family": case.family,
                    "intervention_kind": case.intervention_kind,
                    "reason": str(error),
                }
            )
            continue
        rows.extend(report.rows)
    return tuple(rows), tuple(rejected)


def calibrate_distribution(config: RecoveryRLV2Config) -> dict[str, object]:
    gate_path = config.output_dir / "gates" / "distribution.json"
    if gate_path.is_file():
        return json.loads(gate_path.read_text(encoding="utf-8"))
    binding = foundation_binding(config)
    candidates: dict[float, CalibrationCandidate] = {}
    reports: dict[float, Path] = {}
    with _runtime(config, seed=config.seed + 1000) as runtime:
        for severity in config.distribution.severity_candidates:
            manifest = build_case_manifest(
                seed=config.seed,
                calibration=config.distribution.calibration_seed_count,
                training=config.distribution.training_seed_count,
                curve=config.distribution.curve_case_count,
                final=config.distribution.final_case_count,
                severity=severity,
            )
            rows, rejected = _evaluate_rows_with_rejections(
                _ZeroResidualPolicy(),
                runtime,
                manifest.partition("calibration"),
                policy_seed=config.seed + 2000,
                description=f"calibrate severity={severity:.2f}",
            )
            accepted_by_kind = {
                kind: sum(row.intervention_kind == kind for row in rows)
                for kind in sorted(
                    {
                        case.intervention_kind
                        for case in manifest.partition("calibration")
                    }
                )
            }
            nonnominal = tuple(row for row in rows if row.family != "nominal")
            recovery_success = (
                float(np.mean([row.success for row in nonnominal]))
                if nonnominal
                else 0.0
            )
            candidate = CalibrationCandidate(
                severity=severity,
                recovery_success=recovery_success,
                accepted_by_kind=accepted_by_kind,
            )
            candidates[severity] = candidate
            report = {
                "schema_version": FOUNDATION_REPORT_SCHEMA,
                "stage": "distribution_calibration",
                "binding": binding,
                "severity": severity,
                "manifest_sha256": manifest.sha256,
                "recovery_success": recovery_success,
                "accepted_by_kind": accepted_by_kind,
                "accepted_rows": [asdict(row) for row in rows],
                "rejected_rows": list(rejected),
            }
            report_path = (
                config.output_dir
                / "calibration"
                / f"severity_{severity:.2f}"
                / "report.json"
            )
            _immutable_json(report_path, report)
            reports[severity] = report_path
    selection = select_calibrated_severity(
        candidates,
        target=(0.30, 0.50),
        minimum_accepted_per_kind=(
            config.distribution.minimum_accepted_per_kind
        ),
    )
    manifest_hash: str | None = None
    manifest_path = config.output_dir / "manifests" / "cases.json"
    if selection.passed:
        assert selection.severity is not None
        selected_manifest = build_case_manifest(
            seed=config.seed,
            calibration=config.distribution.calibration_seed_count,
            training=config.distribution.training_seed_count,
            curve=config.distribution.curve_case_count,
            final=config.distribution.final_case_count,
            severity=selection.severity,
        )
        save_case_manifest(manifest_path, selected_manifest)
        manifest_hash = selected_manifest.sha256
        normalization_path = (
            config.output_dir / "manifests" / "oracle_normalization.json"
        )
        _immutable_json(normalization_path, OracleNormalization().to_json())
    gate = {
        "schema_version": GATE_SCHEMA,
        "gate": "distribution",
        "passed": selection.passed,
        "reasons": list(selection.reasons),
        "inputs": {
            "binding": binding,
            "target": list(selection.target),
            "selected_severity": selection.severity,
            "selected_recovery_success": selection.recovery_success,
            "candidates": {
                str(severity): asdict(candidate)
                for severity, candidate in sorted(candidates.items())
            },
            "rejected_candidates": {
                str(key): list(value)
                for key, value in selection.rejected_candidates.items()
            },
            "candidate_report_hashes": {
                str(severity): sha256_file(path)
                for severity, path in sorted(reports.items())
            },
            "case_manifest_sha256": manifest_hash,
            "oracle_normalization_sha256": (
                sha256_file(normalization_path) if selection.passed else None
            ),
        },
    }
    write_gate_atomic(gate_path, gate)
    return gate


def _training_seed(base_seed: int, seed_index: int) -> int:
    return int(
        np.random.SeedSequence(
            (base_seed, seed_index, 0x53435232)
        ).generate_state(1, dtype=np.uint32)[0]
    )


@dataclass
class _ACTNominalBank:
    targets: NominalAnchorCache
    case_ids: tuple[str, ...]
    agent_rgb: np.ndarray
    wrist_rgb: np.ndarray
    state: np.ndarray
    tasks: tuple[str, ...]

    def __post_init__(self) -> None:
        rows = len(self.case_ids)
        if (
            len(set(self.case_ids)) != rows
            or self.agent_rgb.shape != (rows, 256, 256, 3)
            or self.wrist_rgb.shape != (rows, 256, 256, 3)
            or self.state.shape != (rows, 10)
            or len(self.tasks) != rows
        ):
            raise ValueError("ACT nominal bank arrays are incompatible")
        self._index = {case_id: index for index, case_id in enumerate(self.case_ids)}

    def observations(
        self,
        case_ids: Sequence[str],
    ) -> tuple[dict[str, object], ...]:
        from interaction_vla.lerobot_bridge.rollout import policy_observation

        rows: list[dict[str, object]] = []
        for case_id in case_ids:
            try:
                index = self._index[str(case_id)]
            except KeyError as error:
                raise ValueError(f"nominal anchor case is unknown: {case_id}") from error
            values = policy_observation(
                agent_rgb=self.agent_rgb[index],
                wrist_rgb=self.wrist_rgb[index],
                state=self.state[index],
            )
            rows.append(
                pack_observation(
                    {
                        **{key: value.unsqueeze(0) for key, value in values.items()},
                        "task": [self.tasks[index]],
                    }
                )
            )
        return tuple(rows)


def _act_nominal_bank(
    config: RecoveryRLV2Config,
    manifest: RecoveryCaseManifest,
    runtime: ResidualMujocoRuntime,
    *,
    seed: int,
) -> _ACTNominalBank:
    root = config.output_dir / "anchors" / "act_latent"
    archive_path = root / "observations.npz"
    observation_manifest_path = root / "observations.json"
    target_root = root / "targets"
    if observation_manifest_path.is_file():
        observation_manifest = json.loads(
            observation_manifest_path.read_text(encoding="utf-8")
        )
        if observation_manifest.get("manifest_sha256") != manifest.sha256:
            raise ValueError("ACT nominal observation manifest differs")
        if observation_manifest.get("archive_sha256") != sha256_file(archive_path):
            raise ValueError("ACT nominal observation archive hash differs")
        with np.load(archive_path, allow_pickle=False) as archive:
            case_ids = tuple(str(value) for value in archive["case_ids"].tolist())
            agent_rgb = np.asarray(archive["agent_rgb"])
            wrist_rgb = np.asarray(archive["wrist_rgb"])
            state = np.asarray(archive["state"])
            tasks = tuple(str(value) for value in archive["tasks"].tolist())
        targets = NominalAnchorCache.load(
            target_root,
            dataset_fingerprint=manifest.sha256,
            sft_checkpoint=config.sft_checkpoint,
            tap_id=_action_proximal_tap(),
            seed=seed,
        )
        if case_ids != targets.case_ids:
            raise ValueError("ACT nominal observations and targets differ")
        return _ACTNominalBank(
            targets=targets,
            case_ids=case_ids,
            agent_rgb=agent_rgb,
            wrist_rgb=wrist_rgb,
            state=state,
            tasks=tasks,
        )
    case_ids_list: list[str] = []
    agent_rows: list[np.ndarray] = []
    wrist_rows: list[np.ndarray] = []
    state_rows: list[np.ndarray] = []
    latent_rows: list[np.ndarray] = []
    tasks_list: list[str] = []
    for case in manifest.partition("training"):
        if case.family != "nominal":
            continue
        runtime.reset_case(case)
        observation = runtime.current_observation
        if observation is None:
            raise RuntimeError("nominal bank reset produced no observation")
        agent, wrist, state = _rgb_state(observation)
        _, latent = runtime.policy_features()
        case_ids_list.append(case.case_id)
        agent_rows.append(agent)
        wrist_rows.append(wrist)
        state_rows.append(state)
        latent_rows.append(latent[0].cpu().numpy().astype(np.float32))
        tasks_list.append(runtime.bridge.dataset.task)
    if len(case_ids_list) < min(64, config.distribution.training_seed_count):
        raise ValueError("ACT nominal anchor bank has insufficient states")
    root.mkdir(parents=True, exist_ok=True)
    temporary = archive_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            case_ids=np.asarray(case_ids_list, dtype=np.str_),
            agent_rgb=np.stack(agent_rows).astype(np.uint8),
            wrist_rgb=np.stack(wrist_rows).astype(np.uint8),
            state=np.stack(state_rows).astype(np.float32),
            tasks=np.asarray(tasks_list, dtype=np.str_),
        )
    temporary.replace(archive_path)
    targets = NominalAnchorCache.create(
        target_root,
        latents=np.stack(latent_rows).astype(np.float32),
        case_ids=tuple(case_ids_list),
        dataset_fingerprint=manifest.sha256,
        sft_checkpoint=config.sft_checkpoint,
        tap_id=_action_proximal_tap(),
        seed=seed,
    )
    _immutable_json(
        observation_manifest_path,
        {
            "schema_version": "act_nominal_observations_v1",
            "manifest_sha256": manifest.sha256,
            "sft_checkpoint": config.sft_checkpoint,
            "tap_id": _action_proximal_tap(),
            "case_ids": case_ids_list,
            "archive_sha256": sha256_file(archive_path),
        },
    )
    return _ACTNominalBank(
        targets=targets,
        case_ids=tuple(case_ids_list),
        agent_rgb=np.stack(agent_rows).astype(np.uint8),
        wrist_rgb=np.stack(wrist_rows).astype(np.uint8),
        state=np.stack(state_rows).astype(np.float32),
        tasks=tuple(tasks_list),
    )


def _atomic_torch_save(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(dict(payload), temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _reset_training_case(
    runtime: ResidualMujocoRuntime,
    sampler: RecoveryCaseSampler,
    rejected: list[dict[str, object]],
) -> tuple[RecoveryCase, np.ndarray]:
    case = sampler.next_case()
    family = case.family
    for _ in range(1000):
        try:
            reset = runtime.reset_case(case)
            return case, reset.oracle_state.copy()
        except PhysicsRecoveryRejected as error:
            rejected.append(
                {
                    "case_id": case.case_id,
                    "source_seed": case.source_seed,
                    "variant_id": case.variant_id,
                    "family": case.family,
                    "intervention_kind": case.intervention_kind,
                    "reason": str(error),
                }
            )
            case = sampler.next_case(family=family)
    raise RuntimeError(f"could not reconstruct an accepted {family} training case")


def _oracle_anchor_cache(
    config: RecoveryRLV2Config,
    manifest: RecoveryCaseManifest,
    runtime: ResidualMujocoRuntime,
    *,
    seed: int,
) -> NominalAnchorCache:
    root = config.output_dir / "anchors" / "oracle_state"
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        return NominalAnchorCache.load(
            root,
            dataset_fingerprint=manifest.sha256,
            sft_checkpoint=config.sft_checkpoint,
            tap_id="compact_oracle_state",
            seed=seed,
        )
    case_ids: list[str] = []
    states: list[np.ndarray] = []
    for case in manifest.partition("training"):
        if case.family != "nominal":
            continue
        reset = runtime.reset_case(case)
        case_ids.append(case.case_id)
        states.append(reset.oracle_state)
    if len(states) < min(64, config.distribution.training_seed_count):
        raise ValueError("nominal Oracle anchor bank has insufficient accepted states")
    return NominalAnchorCache.create(
        root,
        latents=np.stack(states).astype(np.float32),
        case_ids=tuple(case_ids),
        dataset_fingerprint=manifest.sha256,
        sft_checkpoint=config.sft_checkpoint,
        tap_id="compact_oracle_state",
        seed=seed,
    )


def _curve_point(
    actor: ResidualActor,
    runtime: ResidualMujocoRuntime,
    cases: Sequence[RecoveryCase],
    *,
    device: torch.device,
    policy_seed: int,
    environment_steps: int,
    observation: str = "oracle",
) -> dict[str, object]:
    rng = capture_rng_state()
    training_mode = actor.training
    try:
        actor.eval()
        report = evaluate_case_manifest(
            _ActorEvaluationPolicy(
                actor,
                observation=observation,
                device=device,
            ),
            runtime,
            cases,
            policy_seed=policy_seed,
        )
    finally:
        actor.train(training_mode)
        restore_rng_state(rng)
    if report.nominal is None or report.recovery is None:
        raise ValueError("screen evaluation must contain nominal and recovery cases")
    return {
        "environment_steps": int(environment_steps),
        "case_ids": list(report.case_ids),
        "nominal_success": report.nominal.success_rate,
        "recovery_success": report.recovery.success_rate,
        "nominal": asdict(report.nominal),
        "recovery": asdict(report.recovery),
        "episode_rows": [asdict(row) for row in report.rows],
    }


def _screen_binding(
    config: RecoveryRLV2Config,
    manifest: RecoveryCaseManifest,
    *,
    backend: str,
    seed_index: int,
) -> str:
    return _canonical_hash(
        {
            "foundation": foundation_binding(config),
            "manifest": manifest.sha256,
            "backend": backend,
            "seed_index": seed_index,
            "training_seed": _training_seed(config.seed, seed_index),
            "screen_steps": config.screen_steps,
            "evaluation_interval": SCREEN_EVALUATION_INTERVAL,
        }
    )


def _save_screen_state(
    path: Path,
    *,
    binding: str,
    backend_state: Mapping[str, object],
    sampler: RecoveryCaseSampler,
    anchor: NominalAnchorCache,
    environment_steps: int,
    curve: Sequence[Mapping[str, object]],
    rejected: Sequence[Mapping[str, object]],
    replay_state: Mapping[str, object] | None = None,
    action_rng_state: Mapping[str, object] | None = None,
    policy_state: Mapping[str, torch.Tensor] | None = None,
) -> None:
    _atomic_torch_save(
        path,
        {
            "schema_version": "recovery_rl_screen_state_v2",
            "binding": binding,
            "backend": dict(backend_state),
            "sampler": sampler.state_dict(),
            "anchor": anchor.state_dict(),
            "environment_steps": int(environment_steps),
            "curve": [dict(point) for point in curve],
            "rejected": [dict(row) for row in rejected],
            "replay": None if replay_state is None else dict(replay_state),
            "action_rng_state": (
                None if action_rng_state is None else dict(action_rng_state)
            ),
            "policy_state": (
                None
                if policy_state is None
                else {
                    name: value.detach().cpu()
                    for name, value in policy_state.items()
                }
            ),
            "rng": capture_rng_state(),
        },
    )


def _trainable_policy_state(runtime: ResidualMujocoRuntime) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in runtime.backend.policy.named_parameters()
        if parameter.requires_grad
    }


def _sample_act_nominal(
    bank: _ACTNominalBank,
    runtime: ResidualMujocoRuntime,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    sampled = bank.targets.sample(batch_size)
    observations = bank.observations(sampled.case_ids)
    seeds = tuple(
        int.from_bytes(
            hashlib.sha256(case_id.encode("utf-8")).digest()[:4],
            "little",
        )
        for case_id in sampled.case_ids
    )
    current = recompute_latents_with_policy_seeds(
        runtime.backend,
        observations,
        seeds,
        tap_id=_action_proximal_tap(),
    )
    targets = torch.as_tensor(sampled.latents, dtype=torch.float32, device=device)
    return current, targets


def _load_screen_state(path: Path, *, binding: str) -> dict[str, object]:
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(loaded, dict):
        raise ValueError("screen training state must be a mapping")
    if loaded.get("schema_version") != "recovery_rl_screen_state_v2":
        raise ValueError("screen training state schema is incompatible")
    if loaded.get("binding") != binding:
        raise ValueError("screen training state binding differs")
    return loaded


def _screen_report(
    *,
    backend: str,
    seed_index: int,
    training_seed: int,
    binding: str,
    manifest: RecoveryCaseManifest,
    curve: Sequence[Mapping[str, object]],
    rejected: Sequence[Mapping[str, object]],
    last_update: Mapping[str, object],
    budget: int,
) -> dict[str, object]:
    steps = [int(point["environment_steps"]) for point in curve]
    recovery = [float(point["recovery_success"]) for point in curve]
    auc = normalized_curve_auc(steps, recovery, budget=budget)
    return {
        "schema_version": FOUNDATION_REPORT_SCHEMA,
        "stage": "algorithm_screen",
        "passed": True,
        "backend": backend,
        "seed_index": seed_index,
        "training_seed": training_seed,
        "binding": binding,
        "manifest_sha256": manifest.sha256,
        "environment_steps": budget,
        "recovery_auc": auc,
        "sft_recovery_success": float(curve[0]["recovery_success"]),
        "sft_nominal_success": float(curve[0]["nominal_success"]),
        "final_recovery_success": float(curve[-1]["recovery_success"]),
        "final_nominal_success": float(curve[-1]["nominal_success"]),
        "finite": all(
            np.isfinite(float(value)) for value in last_update.values()
        ),
        "resume_valid": True,
        "simulator_integrity_failures": 0,
        "curve": [dict(point) for point in curve],
        "last_update": dict(last_update),
        "rejected_training_cases": [dict(row) for row in rejected],
    }


def _run_ppo_screen_seed(
    config: RecoveryRLV2Config,
    manifest: RecoveryCaseManifest,
    *,
    seed_index: int,
    resume: bool,
) -> dict[str, object]:
    training_seed = _training_seed(config.seed, seed_index)
    run_dir = config.output_dir / "screen" / "ppo" / f"seed_{seed_index}"
    report_path = run_dir / "report.json"
    state_path = run_dir / "training_state.pt"
    if report_path.is_file():
        return json.loads(report_path.read_text(encoding="utf-8"))
    resuming = state_path.is_file()
    if resuming and not resume:
        raise FileExistsError(f"PPO screen is incomplete; pass --resume: {run_dir}")
    binding = _screen_binding(
        config,
        manifest,
        backend="ppo",
        seed_index=seed_index,
    )
    _seed_all(training_seed)
    with _runtime(config, seed=training_seed) as runtime:
        device = runtime.backend.device
        actor = OracleResidualActor().to(device)
        critic = OracleValueCritic().to(device)
        algorithm = PPOV2(
            actor=actor,
            critic=critic,
            config=config.ppo,
            nominal_anchor_coefficient=config.nominal_anchor_coefficient,
            latent_anchor_coefficient=0.0,
        )
        sampler = RecoveryCaseSampler(
            manifest,
            probabilities=config.distribution.probabilities,
            seed=training_seed + 1,
        )
        anchor = _oracle_anchor_cache(
            config,
            manifest,
            runtime,
            seed=training_seed + 2,
        )
        cases = paired_evaluation_cases(
            manifest,
            partition="curve",
            count=config.distribution.curve_case_count,
        )
        steps = 0
        curve: list[dict[str, object]] = []
        rejected: list[dict[str, object]] = []
        last_update: dict[str, object] = {}
        if resuming:
            loaded = _load_screen_state(state_path, binding=binding)
            algorithm.load_state_dict(loaded["backend"])
            sampler.load_state_dict(loaded["sampler"])
            anchor.load_state_dict(loaded["anchor"])
            steps = int(loaded["environment_steps"])
            curve = [dict(point) for point in loaded["curve"]]
            rejected = [dict(row) for row in loaded["rejected"]]
            restore_rng_state(loaded["rng"])
        if not curve:
            curve.append(
                _curve_point(
                    actor,
                    runtime,
                    cases,
                    device=device,
                    policy_seed=training_seed + 10,
                    environment_steps=0,
                )
            )
        progress = tqdm(
            total=config.screen_steps,
            initial=steps,
            desc=f"Oracle-PPO seed={seed_index}",
            unit="step",
            dynamic_ncols=True,
        )
        while steps < config.screen_steps:
            count = min(config.ppo.rollout_steps, config.screen_steps - steps)
            _, oracle = _reset_training_case(runtime, sampler, rejected)
            actor_rows: list[torch.Tensor] = []
            state_rows: list[torch.Tensor] = []
            residual_rows: list[torch.Tensor] = []
            log_prob_rows: list[torch.Tensor] = []
            value_rows: list[torch.Tensor] = []
            rewards: list[float] = []
            dones: list[bool] = []
            last_next_state: np.ndarray | None = None
            for index in range(count):
                base_action, latent = runtime.policy_features()
                state = torch.as_tensor(
                    oracle,
                    dtype=torch.float32,
                    device=device,
                ).reshape(1, -1)
                with torch.no_grad():
                    sample = actor.sample(state, deterministic=False)
                    value = critic(state)
                transition = runtime.step(
                    base_action=base_action,
                    latent=latent,
                    residual=sample.residual[0].cpu().numpy(),
                )
                actor_rows.append(state[0].detach())
                state_rows.append(state[0].detach())
                residual_rows.append(sample.residual[0].detach())
                log_prob_rows.append(sample.log_prob[0].detach())
                value_rows.append(value[0].detach())
                rewards.append(transition.reward)
                dones.append(transition.done)
                last_next_state = transition.next_oracle_state
                if transition.done and index + 1 < count:
                    _, oracle = _reset_training_case(runtime, sampler, rejected)
                elif not transition.done:
                    assert transition.next_oracle_state is not None
                    oracle = transition.next_oracle_state
            last_value = 0.0
            if not dones[-1]:
                assert last_next_state is not None
                with torch.no_grad():
                    last_value = float(
                        critic(
                            torch.as_tensor(
                                last_next_state,
                                dtype=torch.float32,
                                device=device,
                            ).reshape(1, -1)
                        )[0].item()
                    )
            advantages, returns = generalized_advantage_estimate(
                rewards,
                torch.stack(value_rows).cpu().numpy(),
                dones,
                last_value=last_value,
                gamma=config.gamma,
                gae_lambda=config.ppo.gae_lambda,
            )
            actor_tensor = torch.stack(actor_rows)
            state_tensor = torch.stack(state_rows)
            residual_tensor = torch.stack(residual_rows)
            log_prob_tensor = torch.stack(log_prob_rows)
            advantage_tensor = torch.as_tensor(advantages, device=device)
            return_tensor = torch.as_tensor(returns, device=device)
            for _ in range(config.ppo.update_epochs):
                permutation = torch.randperm(count, device=device)
                for start in range(0, count, config.ppo.minibatch_size):
                    indices = permutation[start : start + config.ppo.minibatch_size]
                    nominal = anchor.sample(min(64, len(indices))).latents
                    update = algorithm.update(
                        OnPolicyBatch(
                            actor_observation=actor_tensor[indices],
                            oracle_state=state_tensor[indices],
                            residual=residual_tensor[indices],
                            old_log_prob=log_prob_tensor[indices],
                            advantages=advantage_tensor[indices],
                            returns=return_tensor[indices],
                            nominal_actor_observation=torch.as_tensor(
                                nominal,
                                dtype=torch.float32,
                                device=device,
                            ),
                        )
                    )
                    last_update = asdict(update)
            steps += count
            progress.update(count)
            if steps % SCREEN_EVALUATION_INTERVAL == 0 or steps == config.screen_steps:
                curve.append(
                    _curve_point(
                        actor,
                        runtime,
                        cases,
                        device=device,
                        policy_seed=training_seed + 10,
                        environment_steps=steps,
                    )
                )
            _save_screen_state(
                state_path,
                binding=binding,
                backend_state=algorithm.state_dict(),
                sampler=sampler,
                anchor=anchor,
                environment_steps=steps,
                curve=curve,
                rejected=rejected,
            )
        progress.close()
        report = _screen_report(
            backend="ppo",
            seed_index=seed_index,
            training_seed=training_seed,
            binding=binding,
            manifest=manifest,
            curve=curve,
            rejected=rejected,
            last_update=last_update,
            budget=config.screen_steps,
        )
        _immutable_json(report_path, report)
        return report


def _rgb_state(observation: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def image(key: str) -> np.ndarray:
        value = observation.get(key)
        if not isinstance(value, torch.Tensor) or value.shape != (1, 3, 256, 256):
            raise ValueError(f"online replay observation has incompatible {key}")
        return (
            value[0]
            .detach()
            .cpu()
            .clamp(0.0, 1.0)
            .mul(255.0)
            .round()
            .to(torch.uint8)
            .permute(1, 2, 0)
            .numpy()
        )

    state = observation.get("observation.state")
    if not isinstance(state, torch.Tensor) or state.shape != (1, 10):
        raise ValueError("online replay observation has incompatible state")
    return (
        image("observation.images.agent"),
        image("observation.images.wrist"),
        state[0].detach().cpu().numpy().astype(np.float32),
    )


def _run_sac_screen_seed(
    config: RecoveryRLV2Config,
    manifest: RecoveryCaseManifest,
    *,
    seed_index: int,
    resume: bool,
) -> dict[str, object]:
    training_seed = _training_seed(config.seed, seed_index)
    run_dir = config.output_dir / "screen" / "sac" / f"seed_{seed_index}"
    report_path = run_dir / "report.json"
    state_path = run_dir / "training_state.pt"
    if report_path.is_file():
        return json.loads(report_path.read_text(encoding="utf-8"))
    resuming = state_path.is_file()
    if resuming and not resume:
        raise FileExistsError(f"SAC screen is incomplete; pass --resume: {run_dir}")
    binding = _screen_binding(
        config,
        manifest,
        backend="sac",
        seed_index=seed_index,
    )
    _seed_all(training_seed)
    with _runtime(config, seed=training_seed) as runtime:
        device = runtime.backend.device
        actor = OracleResidualActor().to(device)
        critics = OracleTwinQ().to(device)
        algorithm = SAC(
            actor=actor,
            critics=critics,
            config=config.sac,
            gamma=config.gamma,
            nominal_anchor_coefficient=config.nominal_anchor_coefficient,
            latent_anchor_coefficient=0.0,
        )
        sampler = RecoveryCaseSampler(
            manifest,
            probabilities=config.distribution.probabilities,
            seed=training_seed + 1,
        )
        anchor = _oracle_anchor_cache(
            config,
            manifest,
            runtime,
            seed=training_seed + 2,
        )
        replay = RecoveryReplay(
            root=run_dir / "replay",
            capacity=config.sac.replay_capacity,
            seed=training_seed + 3,
        )
        action_rng = np.random.default_rng(training_seed + 4)
        cases = paired_evaluation_cases(
            manifest,
            partition="curve",
            count=config.distribution.curve_case_count,
        )
        steps = 0
        curve: list[dict[str, object]] = []
        rejected: list[dict[str, object]] = []
        last_update: dict[str, object] = {}
        if resuming:
            loaded = _load_screen_state(state_path, binding=binding)
            algorithm.load_state_dict(loaded["backend"])
            sampler.load_state_dict(loaded["sampler"])
            anchor.load_state_dict(loaded["anchor"])
            replay_state = loaded.get("replay")
            if not isinstance(replay_state, Mapping):
                raise ValueError("SAC resume state has no replay state")
            replay.load_state_dict(replay_state)
            action_state = loaded.get("action_rng_state")
            if not isinstance(action_state, Mapping):
                raise ValueError("SAC resume state has no action RNG")
            action_rng.bit_generator.state = dict(action_state)
            steps = int(loaded["environment_steps"])
            curve = [dict(point) for point in loaded["curve"]]
            rejected = [dict(row) for row in loaded["rejected"]]
            restore_rng_state(loaded["rng"])
        if not curve:
            curve.append(
                _curve_point(
                    actor,
                    runtime,
                    cases,
                    device=device,
                    policy_seed=training_seed + 10,
                    environment_steps=0,
                )
            )
        progress = tqdm(
            total=config.screen_steps,
            initial=steps,
            desc=f"Oracle-SAC seed={seed_index}",
            unit="step",
            dynamic_ncols=True,
        )
        while steps < config.screen_steps:
            boundary = min(
                config.screen_steps,
                ((steps // SCREEN_EVALUATION_INTERVAL) + 1)
                * SCREEN_EVALUATION_INTERVAL,
            )
            case, oracle = _reset_training_case(runtime, sampler, rejected)
            base_action, latent = runtime.policy_features()
            while steps < boundary:
                current_observation = runtime.current_observation
                if current_observation is None:
                    raise RuntimeError("SAC runtime has no current observation")
                if steps < config.sac.warmup_steps:
                    residual = action_rng.uniform(-1.0, 1.0, size=7).astype(np.float32)
                else:
                    with torch.no_grad():
                        residual = (
                            actor.sample(
                                torch.as_tensor(
                                    oracle,
                                    dtype=torch.float32,
                                    device=device,
                                ).reshape(1, -1),
                                deterministic=False,
                            )
                            .residual[0]
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                        )
                transition = runtime.step(
                    base_action=base_action,
                    latent=latent,
                    residual=residual,
                )
                current_agent, current_wrist, current_state = _rgb_state(
                    current_observation
                )
                next_oracle = transition.next_oracle_state
                if next_oracle is None:
                    raise ValueError("SAC transition has no next Oracle-State")
                if transition.done:
                    next_observation = current_observation
                else:
                    next_observation = runtime.current_observation
                    if next_observation is None:
                        raise RuntimeError("SAC runtime lost its next observation")
                next_agent, next_wrist, next_state = _rgb_state(next_observation)
                replay.add(
                    {
                        "transition_id": f"{training_seed}:{steps:08d}",
                        "case_id": case.case_id,
                        "family": case.family,
                        "task": runtime.bridge.dataset.task,
                        "agent_rgb": current_agent,
                        "wrist_rgb": current_wrist,
                        "state": current_state,
                        "next_agent_rgb": next_agent,
                        "next_wrist_rgb": next_wrist,
                        "next_state": next_state,
                        "oracle_state": oracle,
                        "next_oracle_state": next_oracle,
                        "actor_observation": oracle,
                        "next_actor_observation": next_oracle,
                        "residual": residual,
                        "reward": transition.reward,
                        "done": transition.done,
                    }
                )
                steps += 1
                progress.update(1)
                if len(replay) >= max(config.sac.warmup_steps, config.sac.batch_size):
                    for _ in range(config.sac.updates_per_environment_step):
                        sampled = replay.sample(config.sac.batch_size)
                        nominal = anchor.sample(min(64, config.sac.batch_size)).latents
                        update = algorithm.update(
                            SACBatch(
                                actor_observation=torch.as_tensor(
                                    sampled.actor_observation,
                                    device=device,
                                ),
                                next_actor_observation=torch.as_tensor(
                                    sampled.next_actor_observation,
                                    device=device,
                                ),
                                oracle_state=torch.as_tensor(
                                    sampled.oracle_state,
                                    device=device,
                                ),
                                next_oracle_state=torch.as_tensor(
                                    sampled.next_oracle_state,
                                    device=device,
                                ),
                                residual=torch.as_tensor(
                                    sampled.residual,
                                    device=device,
                                ),
                                reward=torch.as_tensor(
                                    sampled.reward,
                                    device=device,
                                ),
                                done=torch.as_tensor(
                                    sampled.done.astype(np.float32),
                                    device=device,
                                ),
                                nominal_actor_observation=torch.as_tensor(
                                    nominal,
                                    device=device,
                                ),
                            )
                        )
                        last_update = asdict(update)
                if steps >= boundary:
                    break
                if transition.done:
                    case, oracle = _reset_training_case(runtime, sampler, rejected)
                    base_action, latent = runtime.policy_features()
                else:
                    oracle = next_oracle
                    base_action, latent = runtime.policy_features()
            curve.append(
                _curve_point(
                    actor,
                    runtime,
                    cases,
                    device=device,
                    policy_seed=training_seed + 10,
                    environment_steps=steps,
                )
            )
            replay_state = replay.state_dict()
            _save_screen_state(
                state_path,
                binding=binding,
                backend_state=algorithm.state_dict(),
                sampler=sampler,
                anchor=anchor,
                environment_steps=steps,
                curve=curve,
                rejected=rejected,
                replay_state=replay_state,
                action_rng_state=action_rng.bit_generator.state,
            )
        progress.close()
        report = _screen_report(
            backend="sac",
            seed_index=seed_index,
            training_seed=training_seed,
            binding=binding,
            manifest=manifest,
            curve=curve,
            rejected=rejected,
            last_update=last_update,
            budget=config.screen_steps,
        )
        _immutable_json(report_path, report)
        return report


def _failure_report(
    *,
    backend: str,
    seed_index: int,
    error: Exception,
) -> dict[str, object]:
    return {
        "schema_version": FOUNDATION_REPORT_SCHEMA,
        "stage": "algorithm_screen",
        "passed": False,
        "backend": backend,
        "seed_index": seed_index,
        "finite": False,
        "resume_valid": False,
        "simulator_integrity_failures": 1,
        "error": type(error).__name__,
        "message": str(error),
    }


def _backend_screen_from_reports(
    backend: str,
    reports: Sequence[Mapping[str, object]],
) -> BackendScreen:
    valid = all(report.get("passed") is True for report in reports)
    if valid:
        auc = tuple(float(report["recovery_auc"]) for report in reports)
        nominal = tuple(
            float(report["final_nominal_success"]) for report in reports
        )
        sft_nominal = float(np.median([report["sft_nominal_success"] for report in reports]))
    else:
        auc = tuple(0.0 for _ in reports)
        nominal = tuple(0.0 for _ in reports)
        sft_nominal = float(
            np.median(
                [float(report.get("sft_nominal_success", 0.0)) for report in reports]
            )
        )
    return BackendScreen(
        backend=backend,
        recovery_auc=auc,
        final_nominal_success=nominal,
        sft_nominal_success=sft_nominal,
        finite=valid and all(bool(report.get("finite")) for report in reports),
        resume_valid=valid and all(bool(report.get("resume_valid")) for report in reports),
        simulator_integrity_failures=sum(
            int(report.get("simulator_integrity_failures", 0)) for report in reports
        ),
    )


def run_algorithm_screen(
    config: RecoveryRLV2Config,
    *,
    resume: bool,
) -> dict[str, object]:
    gate_path = config.output_dir / "gates" / "backend.json"
    if gate_path.is_file():
        return json.loads(gate_path.read_text(encoding="utf-8"))
    manifest = load_case_manifest(config.output_dir / "manifests" / "cases.json")
    reports: dict[str, list[dict[str, object]]] = {"ppo": [], "sac": []}
    runners = {"ppo": _run_ppo_screen_seed, "sac": _run_sac_screen_seed}
    for backend in ("ppo", "sac"):
        for seed_index in range(SCREEN_SEED_COUNT):
            try:
                report = runners[backend](
                    config,
                    manifest,
                    seed_index=seed_index,
                    resume=resume,
                )
            except FileExistsError:
                raise
            except Exception as error:
                report = _failure_report(
                    backend=backend,
                    seed_index=seed_index,
                    error=error,
                )
                _immutable_json(
                    config.output_dir
                    / "screen"
                    / backend
                    / f"seed_{seed_index}"
                    / "failure.json",
                    report,
                )
            reports[backend].append(report)
    decision = select_backend(
        ppo=_backend_screen_from_reports("ppo", reports["ppo"]),
        sac=_backend_screen_from_reports("sac", reports["sac"]),
    )
    report_paths = {
        f"{backend}/seed_{index}": (
            config.output_dir
            / "screen"
            / backend
            / f"seed_{index}"
            / ("report.json" if report.get("passed") else "failure.json")
        )
        for backend, values in reports.items()
        for index, report in enumerate(values)
    }
    gate = decision.to_dict()
    gate["inputs"] = {
        **decision.inputs,
        "binding": foundation_binding(config),
        "case_manifest_sha256": manifest.sha256,
        "distribution_gate_sha256": sha256_file(
            config.output_dir / "gates" / "distribution.json"
        ),
        "report_hashes": {
            key: sha256_file(path) for key, path in sorted(report_paths.items())
        },
    }
    write_gate_atomic(gate_path, gate)
    return gate


def build_oracle_gate(config: RecoveryRLV2Config) -> dict[str, object]:
    destination = config.output_dir / "gates" / "oracle.json"
    if destination.is_file():
        return json.loads(destination.read_text(encoding="utf-8"))
    backend_gate = json.loads(
        (config.output_dir / "gates" / "backend.json").read_text(encoding="utf-8")
    )
    backend = str(backend_gate.get("selected_backend", ""))
    if backend not in {"ppo", "sac"}:
        raise ValueError("backend gate has no selected PPO/SAC backend")
    reports = [
        json.loads(
            (
                config.output_dir
                / "screen"
                / backend
                / f"seed_{index}"
                / "report.json"
            ).read_text(encoding="utf-8")
        )
        for index in range(SCREEN_SEED_COUNT)
    ]
    decision = oracle_gate(
        sft_recovery=tuple(
            float(report["sft_recovery_success"]) for report in reports
        ),
        rl_recovery=tuple(
            float(report["final_recovery_success"]) for report in reports
        ),
        sft_nominal=tuple(
            float(report["sft_nominal_success"]) for report in reports
        ),
        rl_nominal=tuple(
            float(report["final_nominal_success"]) for report in reports
        ),
        sft_recovery_auc=tuple(
            float(report["sft_recovery_success"]) for report in reports
        ),
        rl_recovery_auc=tuple(float(report["recovery_auc"]) for report in reports),
    )
    decision.inputs["selected_backend"] = backend
    decision.inputs["binding"] = foundation_binding(config)
    decision.inputs["backend_gate_sha256"] = sha256_file(
        config.output_dir / "gates" / "backend.json"
    )
    decision.inputs["case_manifest_sha256"] = load_case_manifest(
        config.output_dir / "manifests" / "cases.json"
    ).sha256
    decision.inputs["screen_report_hashes"] = [
        sha256_file(
            config.output_dir
            / "screen"
            / backend
            / f"seed_{index}"
            / "report.json"
        )
        for index in range(SCREEN_SEED_COUNT)
    ]
    gate = decision.to_dict()
    write_gate_atomic(destination, gate)
    return gate


def _anchor_binding(
    config: RecoveryRLV2Config,
    manifest: RecoveryCaseManifest,
    *,
    backend: str,
    variant: str,
    seed_index: int,
) -> str:
    return _canonical_hash(
        {
            "foundation": foundation_binding(config),
            "manifest": manifest.sha256,
            "backend": backend,
            "variant": variant,
            "seed_index": seed_index,
            "training_seed": _training_seed(config.seed + 50_000, seed_index),
            "screen_steps": config.screen_steps,
            "trainable_group": "act/fusion",
        }
    )


def _policy_seed(training_seed: int, environment_step: int) -> int:
    return int(
        np.random.SeedSequence(
            (training_seed, environment_step, 0x504F4C32)
        ).generate_state(1, dtype=np.uint32)[0]
    )


def _run_latent_ppo_anchor_seed(
    config: RecoveryRLV2Config,
    manifest: RecoveryCaseManifest,
    *,
    variant: str,
    seed_index: int,
    resume: bool,
) -> dict[str, object]:
    training_seed = _training_seed(config.seed + 50_000, seed_index)
    run_dir = config.output_dir / "anchor_screen" / variant / f"seed_{seed_index}"
    report_path = run_dir / "report.json"
    state_path = run_dir / "training_state.pt"
    if report_path.is_file():
        return json.loads(report_path.read_text(encoding="utf-8"))
    resuming = state_path.is_file()
    if resuming and not resume:
        raise FileExistsError(
            f"anchoring PPO run is incomplete; pass --resume: {run_dir}"
        )
    binding = _anchor_binding(
        config,
        manifest,
        backend="ppo",
        variant=variant,
        seed_index=seed_index,
    )
    _seed_all(training_seed)
    residual_penalty = 0.0 if variant == "no_anchor" else config.residual_coefficient
    with _runtime(
        config,
        seed=training_seed,
        residual_coefficient=residual_penalty,
    ) as runtime:
        device = runtime.backend.device
        runtime.backend.set_trainable_groups(("fusion",))
        representation_parameters = tuple(
            parameter
            for parameter in runtime.backend.policy.parameters()
            if parameter.requires_grad
        )
        if not representation_parameters:
            raise ValueError("ACT fusion group selected no trainable parameters")
        _, example_latent = runtime.policy_features()
        actor = LatentResidualActor(int(example_latent.shape[1])).to(device)
        critic = OracleValueCritic().to(device)
        full_anchor = variant == "full_anchoring"
        algorithm = PPOV2(
            actor=actor,
            critic=critic,
            config=config.ppo,
            representation_parameters=representation_parameters,
            representation_learning_rate=config.representation_learning_rate,
            nominal_anchor_coefficient=(
                config.nominal_anchor_coefficient if full_anchor else 0.0
            ),
            latent_anchor_coefficient=(
                config.latent_anchor_coefficient if full_anchor else 0.0
            ),
        )
        sampler = RecoveryCaseSampler(
            manifest,
            probabilities=config.distribution.probabilities,
            seed=training_seed + 1,
        )
        bank = _act_nominal_bank(
            config,
            manifest,
            runtime,
            seed=training_seed + 2,
        )
        cases = paired_evaluation_cases(
            manifest,
            partition="curve",
            count=config.distribution.curve_case_count,
        )
        steps = 0
        curve: list[dict[str, object]] = []
        rejected: list[dict[str, object]] = []
        last_update: dict[str, object] = {}
        if resuming:
            loaded = _load_screen_state(state_path, binding=binding)
            policy_state = loaded.get("policy_state")
            if not isinstance(policy_state, Mapping):
                raise ValueError("anchoring PPO state has no ACT policy state")
            runtime.backend.policy.load_state_dict(policy_state, strict=False)
            algorithm.load_state_dict(loaded["backend"])
            sampler.load_state_dict(loaded["sampler"])
            bank.targets.load_state_dict(loaded["anchor"])
            steps = int(loaded["environment_steps"])
            curve = [dict(point) for point in loaded["curve"]]
            rejected = [dict(row) for row in loaded["rejected"]]
            restore_rng_state(loaded["rng"])
        if not curve:
            curve.append(
                _curve_point(
                    actor,
                    runtime,
                    cases,
                    device=device,
                    policy_seed=training_seed + 10,
                    environment_steps=0,
                    observation="latent",
                )
            )
        progress = tqdm(
            total=config.screen_steps,
            initial=steps,
            desc=f"{variant}/PPO seed={seed_index}",
            unit="step",
            dynamic_ncols=True,
        )
        while steps < config.screen_steps:
            count = min(config.ppo.rollout_steps, config.screen_steps - steps)
            _, oracle = _reset_training_case(runtime, sampler, rejected)
            observations: list[dict[str, object]] = []
            policy_seeds: list[int] = []
            state_rows: list[torch.Tensor] = []
            residual_rows: list[torch.Tensor] = []
            log_prob_rows: list[torch.Tensor] = []
            value_rows: list[torch.Tensor] = []
            rewards: list[float] = []
            dones: list[bool] = []
            last_next_state: np.ndarray | None = None
            for index in range(count):
                observation = runtime.current_observation
                if observation is None:
                    raise RuntimeError("anchoring PPO runtime has no observation")
                policy_seed = _policy_seed(training_seed, steps + index)
                torch.manual_seed(policy_seed)
                base_action, latent_cpu = runtime.policy_features()
                latent = latent_cpu.to(device)
                state = torch.as_tensor(
                    oracle,
                    dtype=torch.float32,
                    device=device,
                ).reshape(1, -1)
                with torch.no_grad():
                    sample = actor.sample(latent, deterministic=False)
                    value = critic(state)
                transition = runtime.step(
                    base_action=base_action,
                    latent=latent_cpu,
                    residual=sample.residual[0].cpu().numpy(),
                )
                observations.append(pack_observation(observation))
                policy_seeds.append(policy_seed)
                state_rows.append(state[0].detach())
                residual_rows.append(sample.residual[0].detach())
                log_prob_rows.append(sample.log_prob[0].detach())
                value_rows.append(value[0].detach())
                rewards.append(transition.reward)
                dones.append(transition.done)
                last_next_state = transition.next_oracle_state
                if transition.done and index + 1 < count:
                    _, oracle = _reset_training_case(runtime, sampler, rejected)
                elif not transition.done:
                    assert transition.next_oracle_state is not None
                    oracle = transition.next_oracle_state
            last_value = 0.0
            if not dones[-1]:
                assert last_next_state is not None
                with torch.no_grad():
                    last_value = float(
                        critic(
                            torch.as_tensor(
                                last_next_state,
                                dtype=torch.float32,
                                device=device,
                            ).reshape(1, -1)
                        )[0].item()
                    )
            advantages, returns = generalized_advantage_estimate(
                rewards,
                torch.stack(value_rows).cpu().numpy(),
                dones,
                last_value=last_value,
                gamma=config.gamma,
                gae_lambda=config.ppo.gae_lambda,
            )
            state_tensor = torch.stack(state_rows)
            residual_tensor = torch.stack(residual_rows)
            log_prob_tensor = torch.stack(log_prob_rows)
            advantage_tensor = torch.as_tensor(advantages, device=device)
            return_tensor = torch.as_tensor(returns, device=device)
            for _ in range(config.ppo.update_epochs):
                permutation = torch.randperm(count, device=device)
                for start in range(0, count, config.ppo.minibatch_size):
                    indices = permutation[start : start + config.ppo.minibatch_size]
                    positions = [int(value) for value in indices.cpu().tolist()]
                    current_latent = recompute_latents_with_policy_seeds(
                        runtime.backend,
                        [observations[position] for position in positions],
                        [policy_seeds[position] for position in positions],
                        tap_id=_action_proximal_tap(),
                    )
                    nominal_current: torch.Tensor | None = None
                    nominal_target: torch.Tensor | None = None
                    if full_anchor:
                        nominal_current, nominal_target = _sample_act_nominal(
                            bank,
                            runtime,
                            batch_size=min(64, len(positions)),
                            device=device,
                        )
                    update = algorithm.update(
                        OnPolicyBatch(
                            actor_observation=current_latent,
                            oracle_state=state_tensor[indices],
                            residual=residual_tensor[indices],
                            old_log_prob=log_prob_tensor[indices],
                            advantages=advantage_tensor[indices],
                            returns=return_tensor[indices],
                            nominal_actor_observation=nominal_current,
                            representation_latent=nominal_current,
                            sft_target_latent=nominal_target,
                        )
                    )
                    last_update = asdict(update)
            steps += count
            progress.update(count)
            if steps % SCREEN_EVALUATION_INTERVAL == 0 or steps == config.screen_steps:
                curve.append(
                    _curve_point(
                        actor,
                        runtime,
                        cases,
                        device=device,
                        policy_seed=training_seed + 10,
                        environment_steps=steps,
                        observation="latent",
                    )
                )
            _save_screen_state(
                state_path,
                binding=binding,
                backend_state=algorithm.state_dict(),
                sampler=sampler,
                anchor=bank.targets,
                environment_steps=steps,
                curve=curve,
                rejected=rejected,
                policy_state=_trainable_policy_state(runtime),
            )
        progress.close()
        report = _screen_report(
            backend="ppo",
            seed_index=seed_index,
            training_seed=training_seed,
            binding=binding,
            manifest=manifest,
            curve=curve,
            rejected=rejected,
            last_update=last_update,
            budget=config.screen_steps,
        )
        report["stage"] = "anchoring_screen"
        report["variant"] = variant
        _immutable_json(report_path, report)
        return report


def _packed_replay_observations(
    replay_batch: Any,
    *,
    next_observation: bool = False,
) -> tuple[dict[str, object], ...]:
    from interaction_vla.lerobot_bridge.rollout import policy_observation

    agents = replay_batch.next_agent_rgb if next_observation else replay_batch.agent_rgb
    wrists = replay_batch.next_wrist_rgb if next_observation else replay_batch.wrist_rgb
    states = replay_batch.next_state if next_observation else replay_batch.state
    rows: list[dict[str, object]] = []
    for agent, wrist, state, task in zip(
        agents,
        wrists,
        states,
        replay_batch.tasks,
        strict=True,
    ):
        values = policy_observation(
            agent_rgb=agent,
            wrist_rgb=wrist,
            state=state,
        )
        rows.append(
            pack_observation(
                {
                    **{key: value.unsqueeze(0) for key, value in values.items()},
                    "task": [task],
                }
            )
        )
    return tuple(rows)


def replay_policy_seeds(
    transition_ids: Sequence[str], *, next_observation: bool
) -> tuple[int, ...]:
    """Deterministically separate stochastic policy draws for s_t and s_(t+1)."""
    suffix = ":next" if next_observation else ""
    return tuple(
        int.from_bytes(
            hashlib.sha256(f"{identifier}{suffix}".encode("utf-8")).digest()[:4],
            "little",
        )
        for identifier in transition_ids
    )


def _run_latent_sac_anchor_seed(
    config: RecoveryRLV2Config,
    manifest: RecoveryCaseManifest,
    *,
    variant: str,
    seed_index: int,
    resume: bool,
) -> dict[str, object]:
    training_seed = _training_seed(config.seed + 50_000, seed_index)
    run_dir = config.output_dir / "anchor_screen" / variant / f"seed_{seed_index}"
    report_path = run_dir / "report.json"
    state_path = run_dir / "training_state.pt"
    if report_path.is_file():
        return json.loads(report_path.read_text(encoding="utf-8"))
    resuming = state_path.is_file()
    if resuming and not resume:
        raise FileExistsError(
            f"anchoring SAC run is incomplete; pass --resume: {run_dir}"
        )
    binding = _anchor_binding(
        config,
        manifest,
        backend="sac",
        variant=variant,
        seed_index=seed_index,
    )
    _seed_all(training_seed)
    residual_penalty = 0.0 if variant == "no_anchor" else config.residual_coefficient
    with _runtime(
        config,
        seed=training_seed,
        residual_coefficient=residual_penalty,
    ) as runtime:
        device = runtime.backend.device
        runtime.backend.set_trainable_groups(("fusion",))
        representation_parameters = tuple(
            parameter
            for parameter in runtime.backend.policy.parameters()
            if parameter.requires_grad
        )
        if not representation_parameters:
            raise ValueError("ACT fusion group selected no trainable parameters")
        _, example_latent = runtime.policy_features()
        latent_dim = int(example_latent.shape[1])
        actor = LatentResidualActor(latent_dim).to(device)
        critics = OracleTwinQ().to(device)
        full_anchor = variant == "full_anchoring"
        algorithm = SAC(
            actor=actor,
            critics=critics,
            config=config.sac,
            gamma=config.gamma,
            representation_parameters=representation_parameters,
            representation_learning_rate=config.representation_learning_rate,
            nominal_anchor_coefficient=(
                config.nominal_anchor_coefficient if full_anchor else 0.0
            ),
            latent_anchor_coefficient=(
                config.latent_anchor_coefficient if full_anchor else 0.0
            ),
        )
        sampler = RecoveryCaseSampler(
            manifest,
            probabilities=config.distribution.probabilities,
            seed=training_seed + 1,
        )
        bank = _act_nominal_bank(
            config,
            manifest,
            runtime,
            seed=training_seed + 2,
        )
        replay = RecoveryReplay(
            root=run_dir / "replay",
            capacity=config.sac.replay_capacity,
            seed=training_seed + 3,
        )
        action_rng = np.random.default_rng(training_seed + 4)
        cases = paired_evaluation_cases(
            manifest,
            partition="curve",
            count=config.distribution.curve_case_count,
        )
        steps = 0
        curve: list[dict[str, object]] = []
        rejected: list[dict[str, object]] = []
        last_update: dict[str, object] = {}
        if resuming:
            loaded = _load_screen_state(state_path, binding=binding)
            policy_state = loaded.get("policy_state")
            if not isinstance(policy_state, Mapping):
                raise ValueError("anchoring SAC state has no ACT policy state")
            runtime.backend.policy.load_state_dict(policy_state, strict=False)
            algorithm.load_state_dict(loaded["backend"])
            sampler.load_state_dict(loaded["sampler"])
            bank.targets.load_state_dict(loaded["anchor"])
            replay_state = loaded.get("replay")
            if not isinstance(replay_state, Mapping):
                raise ValueError("anchoring SAC state has no replay")
            replay.load_state_dict(replay_state)
            action_state = loaded.get("action_rng_state")
            if not isinstance(action_state, Mapping):
                raise ValueError("anchoring SAC state has no action RNG")
            action_rng.bit_generator.state = dict(action_state)
            steps = int(loaded["environment_steps"])
            curve = [dict(point) for point in loaded["curve"]]
            rejected = [dict(row) for row in loaded["rejected"]]
            restore_rng_state(loaded["rng"])
        if not curve:
            curve.append(
                _curve_point(
                    actor,
                    runtime,
                    cases,
                    device=device,
                    policy_seed=training_seed + 10,
                    environment_steps=0,
                    observation="latent",
                )
            )
        progress = tqdm(
            total=config.screen_steps,
            initial=steps,
            desc=f"{variant}/SAC seed={seed_index}",
            unit="step",
            dynamic_ncols=True,
        )
        while steps < config.screen_steps:
            boundary = min(
                config.screen_steps,
                ((steps // SCREEN_EVALUATION_INTERVAL) + 1)
                * SCREEN_EVALUATION_INTERVAL,
            )
            case, oracle = _reset_training_case(runtime, sampler, rejected)
            base_action, latent = runtime.policy_features()
            while steps < boundary:
                current_observation = runtime.current_observation
                if current_observation is None:
                    raise RuntimeError("anchoring SAC runtime has no observation")
                actor_observation = latent[0].cpu().numpy().astype(np.float32)
                if steps < config.sac.warmup_steps:
                    residual = action_rng.uniform(-1.0, 1.0, size=7).astype(np.float32)
                else:
                    with torch.no_grad():
                        residual = (
                            actor.sample(
                                latent.to(device),
                                deterministic=False,
                            )
                            .residual[0]
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                        )
                transition = runtime.step(
                    base_action=base_action,
                    latent=latent,
                    residual=residual,
                )
                current_agent, current_wrist, current_state = _rgb_state(
                    current_observation
                )
                next_oracle = transition.next_oracle_state
                if next_oracle is None:
                    raise ValueError("anchoring SAC transition has no next Oracle-State")
                if transition.done:
                    next_observation = current_observation
                    next_latent = latent
                else:
                    next_observation = runtime.current_observation
                    if next_observation is None:
                        raise RuntimeError("anchoring SAC lost its next observation")
                    next_base_action, next_latent = runtime.policy_features()
                next_agent, next_wrist, next_state = _rgb_state(next_observation)
                replay.add(
                    {
                        "transition_id": f"anchor:{variant}:{training_seed}:{steps:08d}",
                        "case_id": case.case_id,
                        "family": case.family,
                        "task": runtime.bridge.dataset.task,
                        "agent_rgb": current_agent,
                        "wrist_rgb": current_wrist,
                        "state": current_state,
                        "next_agent_rgb": next_agent,
                        "next_wrist_rgb": next_wrist,
                        "next_state": next_state,
                        "oracle_state": oracle,
                        "next_oracle_state": next_oracle,
                        "actor_observation": actor_observation,
                        "next_actor_observation": (
                            next_latent[0].cpu().numpy().astype(np.float32)
                        ),
                        "residual": residual,
                        "reward": transition.reward,
                        "done": transition.done,
                    }
                )
                steps += 1
                progress.update(1)
                if len(replay) >= max(config.sac.warmup_steps, config.sac.batch_size):
                    for _ in range(config.sac.updates_per_environment_step):
                        sampled = replay.sample(config.sac.batch_size)
                        observations = _packed_replay_observations(sampled)
                        next_observations = _packed_replay_observations(
                            sampled, next_observation=True
                        )
                        seeds = replay_policy_seeds(
                            sampled.transition_ids, next_observation=False
                        )
                        next_seeds = replay_policy_seeds(
                            sampled.transition_ids, next_observation=True
                        )
                        current_latent = recompute_latents_with_policy_seeds(
                            runtime.backend,
                            observations,
                            seeds,
                            tap_id=_action_proximal_tap(),
                        )
                        next_latent = recompute_latents_with_policy_seeds(
                            runtime.backend,
                            next_observations,
                            next_seeds,
                            tap_id=_action_proximal_tap(),
                        ).detach()
                        nominal_current: torch.Tensor | None = None
                        nominal_target: torch.Tensor | None = None
                        if full_anchor:
                            nominal_current, nominal_target = _sample_act_nominal(
                                bank,
                                runtime,
                                batch_size=min(64, config.sac.batch_size),
                                device=device,
                            )
                        update = algorithm.update(
                            SACBatch(
                                actor_observation=current_latent,
                                next_actor_observation=next_latent,
                                oracle_state=torch.as_tensor(
                                    sampled.oracle_state,
                                    device=device,
                                ),
                                next_oracle_state=torch.as_tensor(
                                    sampled.next_oracle_state,
                                    device=device,
                                ),
                                residual=torch.as_tensor(
                                    sampled.residual,
                                    device=device,
                                ),
                                reward=torch.as_tensor(sampled.reward, device=device),
                                done=torch.as_tensor(
                                    sampled.done.astype(np.float32),
                                    device=device,
                                ),
                                nominal_actor_observation=nominal_current,
                                representation_latent=nominal_current,
                                sft_target_latent=nominal_target,
                            )
                        )
                        last_update = asdict(update)
                if steps >= boundary:
                    break
                if transition.done:
                    case, oracle = _reset_training_case(runtime, sampler, rejected)
                    base_action, latent = runtime.policy_features()
                else:
                    oracle = next_oracle
                    base_action, latent = next_base_action, next_latent
            curve.append(
                _curve_point(
                    actor,
                    runtime,
                    cases,
                    device=device,
                    policy_seed=training_seed + 10,
                    environment_steps=steps,
                    observation="latent",
                )
            )
            replay_state = replay.state_dict()
            _save_screen_state(
                state_path,
                binding=binding,
                backend_state=algorithm.state_dict(),
                sampler=sampler,
                anchor=bank.targets,
                environment_steps=steps,
                curve=curve,
                rejected=rejected,
                replay_state=replay_state,
                action_rng_state=action_rng.bit_generator.state,
                policy_state=_trainable_policy_state(runtime),
            )
        progress.close()
        report = _screen_report(
            backend="sac",
            seed_index=seed_index,
            training_seed=training_seed,
            binding=binding,
            manifest=manifest,
            curve=curve,
            rejected=rejected,
            last_update=last_update,
            budget=config.screen_steps,
        )
        report["stage"] = "anchoring_screen"
        report["variant"] = variant
        _immutable_json(report_path, report)
        return report


def run_anchor_screen(
    config: RecoveryRLV2Config,
    *,
    resume: bool,
) -> dict[str, object]:
    destination = config.output_dir / "gates" / "anchoring.json"
    if destination.is_file():
        return json.loads(destination.read_text(encoding="utf-8"))
    oracle = json.loads(
        (config.output_dir / "gates" / "oracle.json").read_text(encoding="utf-8")
    )
    backend = str(oracle.get("inputs", {}).get("selected_backend", ""))
    if backend not in {"ppo", "sac"}:
        raise ValueError("Oracle gate has no selected backend")
    manifest = load_case_manifest(config.output_dir / "manifests" / "cases.json")
    runner = (
        _run_latent_ppo_anchor_seed
        if backend == "ppo"
        else _run_latent_sac_anchor_seed
    )
    reports: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for variant in ANCHOR_VARIANTS:
        seed_reports: list[dict[str, object]] = []
        for seed_index in range(SCREEN_SEED_COUNT):
            try:
                seed_report = runner(
                    config,
                    manifest,
                    variant=variant,
                    seed_index=seed_index,
                    resume=resume,
                )
            except FileExistsError:
                raise
            except Exception as error:
                seed_report = {
                    **_failure_report(
                        backend=backend,
                        seed_index=seed_index,
                        error=error,
                    ),
                    "stage": "anchoring_screen",
                    "variant": variant,
                }
                failures.append(seed_report)
                _immutable_json(
                    config.output_dir
                    / "anchor_screen"
                    / variant
                    / f"seed_{seed_index}"
                    / "failure.json",
                    seed_report,
                )
            seed_reports.append(seed_report)
        aggregate = {
            "schema_version": FOUNDATION_REPORT_SCHEMA,
            "stage": "anchoring_screen",
            "passed": all(report.get("passed") is True for report in seed_reports),
            "variant": variant,
            "backend": backend,
            "manifest_sha256": manifest.sha256,
            "recovery_auc": [
                float(report.get("recovery_auc", 0.0)) for report in seed_reports
            ],
            "final_nominal_success": [
                float(report.get("final_nominal_success", 0.0))
                for report in seed_reports
            ],
            "sft_nominal_success": float(
                np.median(
                    [
                        float(report.get("sft_nominal_success", 0.0))
                        for report in seed_reports
                    ]
                )
            ),
            "seed_report_hashes": {
                str(index): sha256_file(
                    config.output_dir
                    / "anchor_screen"
                    / variant
                    / f"seed_{index}"
                    / (
                        "report.json"
                        if report.get("passed") is True
                        else "failure.json"
                    )
                )
                for index, report in enumerate(seed_reports)
            },
        }
        _immutable_json(
            config.output_dir / "anchor_screen" / variant / "report.json",
            aggregate,
        )
        reports.append(aggregate)
    if failures:
        gate = {
            "schema_version": GATE_SCHEMA,
            "gate": "anchoring",
            "passed": False,
            "reasons": [
                f"{row['variant']}/seed_{row['seed_index']} failed: {row['message']}"
                for row in failures
            ],
            "inputs": {
                "selected_backend": backend,
                "oracle_gate_sha256": sha256_file(
                    config.output_dir / "gates" / "oracle.json"
                ),
                "case_manifest_sha256": manifest.sha256,
            },
        }
        write_gate_atomic(destination, gate)
        return gate
    screens = tuple(
        AnchoringScreen(
            variant=str(report["variant"]),
            recovery_auc=tuple(float(value) for value in report["recovery_auc"]),
            final_nominal_success=tuple(
                float(value) for value in report["final_nominal_success"]
            ),
            sft_nominal_success=float(report["sft_nominal_success"]),
        )
        for report in reports
    )
    decision = select_anchoring(screens)
    decision.inputs["selected_backend"] = backend
    decision.inputs["binding"] = foundation_binding(config)
    decision.inputs["oracle_gate_sha256"] = sha256_file(
        config.output_dir / "gates" / "oracle.json"
    )
    decision.inputs["report_hashes"] = {
        report["variant"]: sha256_file(
            config.output_dir
            / "anchor_screen"
            / str(report["variant"])
            / "report.json"
        )
        for report in reports
    }
    decision.inputs["case_manifest_sha256"] = manifest.sha256
    gate = decision.to_dict()
    write_gate_atomic(destination, gate)
    return gate
