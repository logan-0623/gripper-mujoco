from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

from interaction_vla.lerobot_bridge.provenance import fingerprint_tree, sha256_file

from ..state_bank.io import write_json_atomic
from .actors import LatentResidualActor, OracleResidualActor, ResidualActor
from .checkpoint import capture_rng_state, restore_rng_state
from .core import generalized_advantage_estimate
from .critics import OracleTwinQ, OracleValueCritic
from .distributions import (
    RecoveryCaseManifest,
    RecoveryCaseSampler,
    load_case_manifest,
)
from .foundation import (
    _ACTNominalBank,
    _act_nominal_bank,
    _action_proximal_tap,
    _oracle_anchor_cache,
    _packed_replay_observations,
    _policy_seed,
    _reset_training_case,
    _rgb_state,
    _runtime,
    _sample_act_nominal,
    _seed_all,
    _trainable_policy_state,
    foundation_binding,
)
from .gates import GATE_SCHEMA
from .ppo_v2 import OnPolicyBatch, PPOV2
from .protocol import require_passing_gate
from .replay import RecoveryReplay
from .sac import SAC, SACBatch
from .snapshots import SNAPSHOT_STEPS, SnapshotStore
from .training import pack_observation, recompute_latents_with_policy_seeds
from .v2_config import RecoveryRLV2Config


FORMAL_SCHEMA = "recovery_rl_formal_act_v2"
FORMAL_CONDITIONS = (
    "sft",
    "continued_sft",
    "oracle_state",
    "rl_head",
    "rl_representation",
)
FORMAL_TRAINING_CONDITIONS = (
    "oracle_state",
    "rl_head",
    "rl_representation",
)
CONSTANT_CONTROL_CONDITIONS = ("sft", "continued_sft")
FORMAL_SEED_COUNT = 3


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _artifact_hash(path: str | Path) -> str:
    source = Path(path)
    if source.is_file():
        return sha256_file(source)
    if source.is_dir():
        return fingerprint_tree(source)
    raise FileNotFoundError(f"required formal artifact does not exist: {source}")


def formal_training_seed(base_seed: int, seed_index: int) -> int:
    if base_seed < 0 or not 0 <= seed_index < FORMAL_SEED_COUNT:
        raise ValueError("formal seed index must lie within [0, 2]")
    return int(
        np.random.SeedSequence(
            (base_seed, seed_index, 0x524C5632)
        ).generate_state(1, dtype=np.uint32)[0]
    )


@dataclass(frozen=True)
class FormalMatrix:
    conditions: tuple[str, ...]
    training_conditions: tuple[str, ...]
    training_seeds: tuple[int, ...]


def formal_matrix(*, base_seed: int) -> FormalMatrix:
    return FormalMatrix(
        conditions=FORMAL_CONDITIONS,
        training_conditions=FORMAL_TRAINING_CONDITIONS,
        training_seeds=tuple(
            formal_training_seed(base_seed, index)
            for index in range(FORMAL_SEED_COUNT)
        ),
    )


@dataclass(frozen=True)
class FormalArtifacts:
    foundation_binding: str
    backend: str
    anchoring: str
    case_manifest: Path
    oracle_normalization: Path
    state_bank: Path
    hashes: dict[str, str]


@dataclass(frozen=True)
class FormalRun:
    condition: str
    seed_index: int
    seed: int
    backend: str
    anchoring: str
    output_dir: Path
    binding: str
    parent_checkpoint: str
    trainable_groups: tuple[str, ...]
    constant_control: bool


def _load_formal_artifacts(config: RecoveryRLV2Config) -> FormalArtifacts:
    binding = foundation_binding(config)
    gates = config.output_dir / "gates"
    loaded = {
        name: require_passing_gate(
            gates / f"{name}.json",
            expected_gate=name,
            expected_binding=binding,
        )
        for name in ("distribution", "backend", "oracle", "anchoring")
    }
    backend = str(loaded["backend"].get("selected_backend", ""))
    oracle_inputs = loaded["oracle"].get("inputs")
    anchor_inputs = loaded["anchoring"].get("inputs")
    if not isinstance(oracle_inputs, Mapping) or not isinstance(anchor_inputs, Mapping):
        raise ValueError("formal foundation gate inputs are incompatible")
    if backend not in {"ppo", "sac"}:
        raise ValueError("backend gate has no selected PPO/SAC backend")
    if oracle_inputs.get("selected_backend") != backend:
        raise ValueError("oracle and backend gates select different backends")
    if anchor_inputs.get("selected_backend") != backend:
        raise ValueError("anchoring and backend gates select different backends")
    anchoring = str(anchor_inputs.get("selected_variant", ""))
    if anchoring not in {"no_anchor", "residual_only", "full_anchoring"}:
        raise ValueError("anchoring gate has no selected variant")

    case_manifest = config.output_dir / "manifests" / "cases.json"
    normalization = config.output_dir / "manifests" / "oracle_normalization.json"
    state_bank_root = config.output_dir / "state_bank_v2"
    state_bank_manifest = state_bank_root / "manifest.json"
    for label, path in (
        ("case manifest", case_manifest),
        ("Oracle normalization", normalization),
        ("State Bank v2 manifest", state_bank_manifest),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"required formal {label} not found: {path}")
    bank = json.loads(state_bank_manifest.read_text(encoding="utf-8"))
    if bank.get("schema_version") != "recovery_state_bank_v2":
        raise ValueError("State Bank v2 manifest schema is incompatible")
    artifact_hashes = bank.get("artifact_hashes")
    if not isinstance(artifact_hashes, Mapping) or not artifact_hashes:
        raise ValueError("State Bank v2 artifact hashes are missing")
    expected_case_hash = str(
        loaded["distribution"].get("inputs", {}).get(
            "case_manifest_sha256", ""
        )
    )
    if len(expected_case_hash) != 64:
        raise ValueError("distribution gate has no bound case manifest")
    for gate_name in ("backend", "anchoring"):
        inputs = loaded[gate_name].get("inputs")
        if (
            not isinstance(inputs, Mapping)
            or inputs.get("case_manifest_sha256") != expected_case_hash
        ):
            raise ValueError(
                f"{gate_name} gate case manifest binding differs"
            )
    if bank.get("source_case_manifest_sha256") != expected_case_hash:
        raise ValueError("State Bank v2 and foundation case manifests differ")
    for name, expected in artifact_hashes.items():
        artifact = state_bank_root / str(name)
        if not artifact.is_file() or sha256_file(artifact) != str(expected):
            raise ValueError(f"State Bank v2 artifact hash differs: {name}")
    hashes = {
        **{
            f"gate/{name}": sha256_file(gates / f"{name}.json")
            for name in loaded
        },
        "case_manifest": sha256_file(case_manifest),
        "oracle_normalization": sha256_file(normalization),
        "state_bank_manifest": sha256_file(state_bank_manifest),
        "sft_checkpoint": _artifact_hash(config.sft_checkpoint),
        "continued_sft_checkpoint": _artifact_hash(
            config.continued_sft_checkpoint
        ),
    }
    return FormalArtifacts(
        foundation_binding=binding,
        backend=backend,
        anchoring=anchoring,
        case_manifest=case_manifest,
        oracle_normalization=normalization,
        state_bank=state_bank_root,
        hashes=hashes,
    )


def prepare_formal_run(
    config: RecoveryRLV2Config,
    *,
    condition: str,
    seed_index: int,
) -> FormalRun:
    if condition not in FORMAL_CONDITIONS:
        raise ValueError(f"unknown formal condition: {condition}")
    if condition in CONSTANT_CONTROL_CONDITIONS and seed_index != 0:
        raise ValueError("constant control representation uses seed_index=0")
    seed = formal_training_seed(config.seed, seed_index)
    artifacts = _load_formal_artifacts(config)
    if condition == "sft":
        parent = config.sft_checkpoint
        groups: tuple[str, ...] = ()
        output = config.output_dir / "formal" / "controls" / condition
    elif condition == "continued_sft":
        parent = config.continued_sft_checkpoint
        groups = ()
        output = config.output_dir / "formal" / "controls" / condition
    else:
        parent = config.sft_checkpoint
        groups = ("fusion",) if condition == "rl_representation" else ()
        output = (
            config.output_dir
            / "formal"
            / "runs"
            / condition
            / f"seed_{seed_index}"
        )
    run_binding = _canonical_hash(
        {
            "schema_version": FORMAL_SCHEMA,
            "condition": condition,
            "seed_index": seed_index,
            "training_seed": seed,
            "backend": artifacts.backend,
            "anchoring": artifacts.anchoring,
            "parent_checkpoint": parent,
            "parent_checkpoint_sha256": artifacts.hashes[
                "continued_sft_checkpoint"
                if condition == "continued_sft"
                else "sft_checkpoint"
            ],
            "foundation_binding": artifacts.foundation_binding,
            "artifact_hashes": artifacts.hashes,
            "formal_steps": config.formal_steps,
            "snapshot_steps": list(config.snapshot_steps),
            "trainable_groups": list(groups),
        }
    )
    return FormalRun(
        condition=condition,
        seed_index=seed_index,
        seed=seed,
        backend=artifacts.backend,
        anchoring=artifacts.anchoring,
        output_dir=output,
        binding=run_binding,
        parent_checkpoint=parent,
        trainable_groups=groups,
        constant_control=condition in CONSTANT_CONTROL_CONDITIONS,
    )


def _write_immutable_json(path: Path, value: Mapping[str, object]) -> Path:
    encoded = json.dumps(dict(value), indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"formal artifact is immutable: {path}")
        return path
    write_json_atomic(path, dict(value))
    return path


def build_constant_control_timeline(
    config: RecoveryRLV2Config,
    run: FormalRun,
) -> dict[str, object]:
    if not run.constant_control:
        raise ValueError("constant timeline requires SFT or continued-SFT")
    checkpoint_hash = _artifact_hash(run.parent_checkpoint)
    report: dict[str, object] = {
        "schema_version": FORMAL_SCHEMA,
        "condition": run.condition,
        "seed_index": 0,
        "training_seed": run.seed,
        "binding": run.binding,
        "constant_control": True,
        "independent_representation_runs": 1,
        "points": [
            {
                "environment_steps": step,
                "checkpoint": run.parent_checkpoint,
                "checkpoint_sha256": checkpoint_hash,
                "constant_control": True,
            }
            for step in config.snapshot_steps
        ],
    }
    _write_immutable_json(run.output_dir / "timeline.json", report)
    return report


def _anchor_coefficients(
    config: RecoveryRLV2Config,
    variant: str,
) -> tuple[float, float, float]:
    if variant == "no_anchor":
        return 0.0, 0.0, 0.0
    if variant == "residual_only":
        return config.residual_coefficient, 0.0, 0.0
    if variant == "full_anchoring":
        return (
            config.residual_coefficient,
            config.nominal_anchor_coefficient,
            config.latent_anchor_coefficient,
        )
    raise ValueError(f"unknown formal anchoring variant: {variant}")


def _example_latent_dim(
    runtime: Any,
    manifest: RecoveryCaseManifest,
) -> int:
    errors: list[str] = []
    for case in manifest.partition("training"):
        try:
            runtime.reset_case(case)
            _, latent = runtime.policy_features()
            return int(latent.shape[1])
        except Exception as error:
            errors.append(f"{case.case_id}:{type(error).__name__}")
            if len(errors) >= 100:
                break
    raise RuntimeError("could not construct a formal ACT latent: " + ", ".join(errors))


def _make_algorithm(
    config: RecoveryRLV2Config,
    run: FormalRun,
    runtime: Any,
    manifest: RecoveryCaseManifest,
) -> tuple[ResidualActor, PPOV2 | SAC, str, Sequence[torch.nn.Parameter]]:
    device = runtime.backend.device
    if run.condition == "oracle_state":
        runtime.backend.set_trainable_groups(())
        actor: ResidualActor = OracleResidualActor(config.oracle_state_dim).to(device)
        observation = "oracle"
        representation_parameters: tuple[torch.nn.Parameter, ...] = ()
    else:
        runtime.backend.set_trainable_groups(run.trainable_groups)
        representation_parameters = tuple(
            parameter
            for parameter in runtime.backend.policy.parameters()
            if parameter.requires_grad
        )
        if run.condition == "rl_representation" and not representation_parameters:
            raise ValueError("formal ACT fusion group selected no trainable parameters")
        if run.condition == "rl_head" and representation_parameters:
            raise ValueError("formal RL-head must freeze every ACT parameter")
        actor = LatentResidualActor(_example_latent_dim(runtime, manifest)).to(device)
        observation = "latent"
    _, nominal_coefficient, latent_coefficient = _anchor_coefficients(
        config, run.anchoring
    )
    if run.condition != "rl_representation":
        latent_coefficient = 0.0
    if run.backend == "ppo":
        algorithm: PPOV2 | SAC = PPOV2(
            actor=actor,
            critic=OracleValueCritic(config.oracle_state_dim).to(device),
            config=config.ppo,
            representation_parameters=representation_parameters,
            representation_learning_rate=config.representation_learning_rate,
            nominal_anchor_coefficient=nominal_coefficient,
            latent_anchor_coefficient=latent_coefficient,
        )
    elif run.backend == "sac":
        algorithm = SAC(
            actor=actor,
            critics=OracleTwinQ(config.oracle_state_dim).to(device),
            config=config.sac,
            gamma=config.gamma,
            representation_parameters=representation_parameters,
            representation_learning_rate=config.representation_learning_rate,
            nominal_anchor_coefficient=nominal_coefficient,
            latent_anchor_coefficient=latent_coefficient,
        )
    else:
        raise ValueError(f"selected backend is incompatible: {run.backend}")
    return actor, algorithm, observation, representation_parameters


def _latest_completed_step(store: SnapshotStore) -> int | None:
    completed = [
        step
        for step in SNAPSHOT_STEPS
        if (store.root / f"step_{step:06d}" / "COMPLETED").is_file()
    ]
    return max(completed, default=None)


def _snapshot_payload(
    *,
    run: FormalRun,
    algorithm: PPOV2 | SAC,
    sampler: RecoveryCaseSampler,
    anchor: Any,
    runtime: Any,
    environment_steps: int,
    rejected: Sequence[Mapping[str, object]],
    last_update: Mapping[str, object],
    replay: RecoveryReplay | None = None,
    action_rng: np.random.Generator | None = None,
) -> dict[str, object]:
    return {
        "schema_version": FORMAL_SCHEMA,
        "condition": run.condition,
        "seed_index": run.seed_index,
        "training_seed": run.seed,
        "environment_steps": environment_steps,
        "backend_name": run.backend,
        "anchoring": run.anchoring,
        "algorithm": algorithm.state_dict(),
        "sampler": sampler.state_dict(),
        "anchor": anchor.state_dict(),
        "policy_state": _trainable_policy_state(runtime),
        "replay": None if replay is None else replay.state_dict(),
        "action_rng_state": (
            None if action_rng is None else dict(action_rng.bit_generator.state)
        ),
        "rejected_training_cases": [dict(row) for row in rejected],
        "last_update": dict(last_update),
        "rng": capture_rng_state(),
    }


def _restore_snapshot(
    payload: Mapping[str, object],
    *,
    run: FormalRun,
    algorithm: PPOV2 | SAC,
    sampler: RecoveryCaseSampler,
    anchor: Any,
    runtime: Any,
    replay: RecoveryReplay | None = None,
    action_rng: np.random.Generator | None = None,
) -> tuple[int, list[dict[str, object]], dict[str, object]]:
    if payload.get("schema_version") != FORMAL_SCHEMA:
        raise ValueError("formal snapshot payload schema is incompatible")
    if payload.get("condition") != run.condition or int(payload.get("seed_index", -1)) != run.seed_index:
        raise ValueError("formal snapshot run identity differs")
    policy_state = payload.get("policy_state")
    if policy_state is not None:
        if not isinstance(policy_state, Mapping):
            raise ValueError("formal snapshot ACT policy state is incompatible")
        runtime.backend.policy.load_state_dict(policy_state, strict=False)
    algorithm_state = payload.get("algorithm")
    sampler_state = payload.get("sampler")
    anchor_state = payload.get("anchor")
    if not all(isinstance(value, Mapping) for value in (algorithm_state, sampler_state, anchor_state)):
        raise ValueError("formal snapshot training state is incomplete")
    algorithm.load_state_dict(algorithm_state)
    sampler.load_state_dict(sampler_state)
    anchor.load_state_dict(anchor_state)
    if replay is not None:
        replay_state = payload.get("replay")
        if not isinstance(replay_state, Mapping):
            raise ValueError("formal SAC snapshot has no replay state")
        replay.load_state_dict(replay_state)
    if action_rng is not None:
        action_state = payload.get("action_rng_state")
        if not isinstance(action_state, Mapping):
            raise ValueError("formal SAC snapshot has no action RNG state")
        action_rng.bit_generator.state = dict(action_state)
    rng = payload.get("rng")
    if not isinstance(rng, Mapping):
        raise ValueError("formal snapshot RNG state is missing")
    restore_rng_state(rng)
    rejected = [dict(row) for row in payload.get("rejected_training_cases", [])]
    last_update = dict(payload.get("last_update", {}))
    return int(payload["environment_steps"]), rejected, last_update


def _formal_anchor(
    config: RecoveryRLV2Config,
    run: FormalRun,
    manifest: RecoveryCaseManifest,
    runtime: Any,
) -> Any:
    if run.condition == "oracle_state":
        return _oracle_anchor_cache(
            config, manifest, runtime, seed=run.seed + 2
        )
    return _act_nominal_bank(config, manifest, runtime, seed=run.seed + 2)


def _sample_nominal(
    config: RecoveryRLV2Config,
    run: FormalRun,
    anchor: Any,
    runtime: Any,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if run.anchoring != "full_anchoring":
        return None, None
    if run.condition == "oracle_state":
        values = anchor.sample(batch_size).latents
        return torch.as_tensor(values, dtype=torch.float32, device=device), None
    assert isinstance(anchor, _ACTNominalBank)
    current, target = _sample_act_nominal(
        anchor,
        runtime,
        batch_size=batch_size,
        device=device,
    )
    if run.condition == "rl_representation":
        return current, target
    return current.detach(), None


def _save_registered_snapshot(
    store: SnapshotStore,
    *,
    step: int,
    run: FormalRun,
    algorithm: PPOV2 | SAC,
    sampler: RecoveryCaseSampler,
    anchor: Any,
    runtime: Any,
    rejected: Sequence[Mapping[str, object]],
    last_update: Mapping[str, object],
    replay: RecoveryReplay | None = None,
    action_rng: np.random.Generator | None = None,
) -> Path:
    target = store.root / f"step_{step:06d}" / "COMPLETED"
    if target.is_file():
        return target.parent
    return store.save(
        step=step,
        binding=run.binding,
        payload=_snapshot_payload(
            run=run,
            algorithm=algorithm,
            sampler=sampler,
            anchor=anchor.targets if isinstance(anchor, _ACTNominalBank) else anchor,
            runtime=runtime,
            environment_steps=step,
            rejected=rejected,
            last_update=last_update,
            replay=replay,
            action_rng=action_rng,
        ),
    )


def _run_formal_ppo(
    config: RecoveryRLV2Config,
    run: FormalRun,
    manifest: RecoveryCaseManifest,
    runtime: Any,
    actor: ResidualActor,
    algorithm: PPOV2,
    observation_kind: str,
    anchor: Any,
    store: SnapshotStore,
    *,
    resume: bool,
) -> dict[str, object]:
    device = runtime.backend.device
    sampler = RecoveryCaseSampler(
        manifest,
        probabilities=config.distribution.probabilities,
        seed=run.seed + 1,
    )
    latest = _latest_completed_step(store)
    steps = 0
    rejected: list[dict[str, object]] = []
    last_update: dict[str, object] = {}
    if latest is not None:
        if latest < config.formal_steps and not resume:
            raise FileExistsError(
                f"formal run is incomplete; pass --resume: {run.output_dir}"
            )
        payload = store.load(step=latest, expected_binding=run.binding, map_location=device)
        anchor_state = anchor.targets if isinstance(anchor, _ACTNominalBank) else anchor
        steps, rejected, last_update = _restore_snapshot(
            payload,
            run=run,
            algorithm=algorithm,
            sampler=sampler,
            anchor=anchor_state,
            runtime=runtime,
        )
    else:
        _save_registered_snapshot(
            store,
            step=0,
            run=run,
            algorithm=algorithm,
            sampler=sampler,
            anchor=anchor,
            runtime=runtime,
            rejected=rejected,
            last_update=last_update,
        )
    progress = tqdm(
        total=config.formal_steps,
        initial=steps,
        desc=f"formal/{run.condition}/PPO seed={run.seed_index}",
        unit="step",
        dynamic_ncols=True,
    )
    while steps < config.formal_steps:
        boundary = next(step for step in config.snapshot_steps if step > steps)
        count = min(config.ppo.rollout_steps, boundary - steps)
        _, oracle = _reset_training_case(runtime, sampler, rejected)
        observations: list[dict[str, object]] = []
        policy_seeds: list[int] = []
        actor_rows: list[torch.Tensor] = []
        state_rows: list[torch.Tensor] = []
        residual_rows: list[torch.Tensor] = []
        log_prob_rows: list[torch.Tensor] = []
        value_rows: list[torch.Tensor] = []
        rewards: list[float] = []
        dones: list[bool] = []
        last_next_state: np.ndarray | None = None
        for index in range(count):
            policy_seed = _policy_seed(run.seed, steps + index)
            torch.manual_seed(policy_seed)
            current_observation = runtime.current_observation
            if current_observation is None:
                raise RuntimeError("formal PPO runtime has no observation")
            base_action, latent_cpu = runtime.policy_features()
            latent = latent_cpu.to(device)
            state = torch.as_tensor(
                oracle, dtype=torch.float32, device=device
            ).reshape(1, -1)
            actor_observation = state if observation_kind == "oracle" else latent
            with torch.no_grad():
                sample = actor.sample(actor_observation, deterministic=False)
                value = algorithm.critic(state)
            transition = runtime.step(
                base_action=base_action,
                latent=latent_cpu,
                residual=sample.residual[0].cpu().numpy(),
            )
            if observation_kind == "latent":
                observations.append(pack_observation(current_observation))
                policy_seeds.append(policy_seed)
            actor_rows.append(actor_observation[0].detach())
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
                    algorithm.critic(
                        torch.as_tensor(
                            last_next_state, dtype=torch.float32, device=device
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
                positions = [int(value) for value in indices.cpu().tolist()]
                if run.condition == "rl_representation":
                    current_actor = recompute_latents_with_policy_seeds(
                        runtime.backend,
                        [observations[position] for position in positions],
                        [policy_seeds[position] for position in positions],
                        tap_id=_action_proximal_tap(),
                    )
                else:
                    current_actor = actor_tensor[indices]
                nominal_current, nominal_target = _sample_nominal(
                    config,
                    run,
                    anchor,
                    runtime,
                    batch_size=min(64, len(positions)),
                    device=device,
                )
                update = algorithm.update(
                    OnPolicyBatch(
                        actor_observation=current_actor,
                        oracle_state=state_tensor[indices],
                        residual=residual_tensor[indices],
                        old_log_prob=log_prob_tensor[indices],
                        advantages=advantage_tensor[indices],
                        returns=return_tensor[indices],
                        nominal_actor_observation=nominal_current,
                        representation_latent=(
                            nominal_current if nominal_target is not None else None
                        ),
                        sft_target_latent=nominal_target,
                    )
                )
                last_update = asdict(update)
        steps += count
        progress.update(count)
        if steps in config.snapshot_steps:
            _save_registered_snapshot(
                store,
                step=steps,
                run=run,
                algorithm=algorithm,
                sampler=sampler,
                anchor=anchor,
                runtime=runtime,
                rejected=rejected,
                last_update=last_update,
            )
    progress.close()
    return {
        "environment_steps": steps,
        "last_update": last_update,
        "rejected_training_cases": rejected,
    }


def _run_formal_sac(
    config: RecoveryRLV2Config,
    run: FormalRun,
    manifest: RecoveryCaseManifest,
    runtime: Any,
    actor: ResidualActor,
    algorithm: SAC,
    observation_kind: str,
    anchor: Any,
    store: SnapshotStore,
    *,
    resume: bool,
) -> dict[str, object]:
    device = runtime.backend.device
    sampler = RecoveryCaseSampler(
        manifest,
        probabilities=config.distribution.probabilities,
        seed=run.seed + 1,
    )
    replay = RecoveryReplay(
        root=run.output_dir / "replay",
        capacity=config.sac.replay_capacity,
        seed=run.seed + 3,
    )
    action_rng = np.random.default_rng(run.seed + 4)
    latest = _latest_completed_step(store)
    steps = 0
    rejected: list[dict[str, object]] = []
    last_update: dict[str, object] = {}
    if latest is not None:
        if latest < config.formal_steps and not resume:
            raise FileExistsError(
                f"formal run is incomplete; pass --resume: {run.output_dir}"
            )
        payload = store.load(step=latest, expected_binding=run.binding, map_location=device)
        anchor_state = anchor.targets if isinstance(anchor, _ACTNominalBank) else anchor
        steps, rejected, last_update = _restore_snapshot(
            payload,
            run=run,
            algorithm=algorithm,
            sampler=sampler,
            anchor=anchor_state,
            runtime=runtime,
            replay=replay,
            action_rng=action_rng,
        )
    else:
        _save_registered_snapshot(
            store,
            step=0,
            run=run,
            algorithm=algorithm,
            sampler=sampler,
            anchor=anchor,
            runtime=runtime,
            rejected=rejected,
            last_update=last_update,
            replay=replay,
            action_rng=action_rng,
        )
    progress = tqdm(
        total=config.formal_steps,
        initial=steps,
        desc=f"formal/{run.condition}/SAC seed={run.seed_index}",
        unit="step",
        dynamic_ncols=True,
    )
    while steps < config.formal_steps:
        boundary = next(step for step in config.snapshot_steps if step > steps)
        case, oracle = _reset_training_case(runtime, sampler, rejected)
        base_action, latent = runtime.policy_features()
        while steps < boundary:
            current_observation = runtime.current_observation
            if current_observation is None:
                raise RuntimeError("formal SAC runtime has no observation")
            actor_observation = (
                oracle.astype(np.float32)
                if observation_kind == "oracle"
                else latent[0].cpu().numpy().astype(np.float32)
            )
            if steps < config.sac.warmup_steps:
                residual = action_rng.uniform(-1.0, 1.0, size=7).astype(np.float32)
            else:
                actor_tensor = torch.as_tensor(
                    actor_observation, dtype=torch.float32, device=device
                ).reshape(1, -1)
                with torch.no_grad():
                    residual = (
                        actor.sample(actor_tensor, deterministic=False)
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
            current_agent, current_wrist, current_state = _rgb_state(current_observation)
            next_oracle = transition.next_oracle_state
            if next_oracle is None:
                raise ValueError("formal SAC transition has no next Oracle-State")
            if transition.done:
                next_observation = current_observation
                next_latent = latent
            else:
                next_observation = runtime.current_observation
                if next_observation is None:
                    raise RuntimeError("formal SAC lost its next observation")
                next_base_action, next_latent = runtime.policy_features()
            next_actor_observation = (
                next_oracle.astype(np.float32)
                if observation_kind == "oracle"
                else next_latent[0].cpu().numpy().astype(np.float32)
            )
            next_agent, next_wrist, next_state = _rgb_state(next_observation)
            replay.add(
                {
                    "transition_id": f"formal:{run.condition}:{run.seed}:{steps:08d}",
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
                    "next_actor_observation": next_actor_observation,
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
                    current_actor = torch.as_tensor(
                        sampled.actor_observation, device=device
                    )
                    if run.condition == "rl_representation":
                        observations = _packed_replay_observations(sampled)
                        seeds = tuple(
                            int.from_bytes(
                                hashlib.sha256(identifier.encode("utf-8")).digest()[:4],
                                "little",
                            )
                            for identifier in sampled.transition_ids
                        )
                        current_actor = recompute_latents_with_policy_seeds(
                            runtime.backend,
                            observations,
                            seeds,
                            tap_id=_action_proximal_tap(),
                        )
                    nominal_current, nominal_target = _sample_nominal(
                        config,
                        run,
                        anchor,
                        runtime,
                        batch_size=min(64, config.sac.batch_size),
                        device=device,
                    )
                    update = algorithm.update(
                        SACBatch(
                            actor_observation=current_actor,
                            next_actor_observation=torch.as_tensor(
                                sampled.next_actor_observation, device=device
                            ),
                            oracle_state=torch.as_tensor(sampled.oracle_state, device=device),
                            next_oracle_state=torch.as_tensor(
                                sampled.next_oracle_state, device=device
                            ),
                            residual=torch.as_tensor(sampled.residual, device=device),
                            reward=torch.as_tensor(sampled.reward, device=device),
                            done=torch.as_tensor(
                                sampled.done.astype(np.float32), device=device
                            ),
                            nominal_actor_observation=nominal_current,
                            representation_latent=(
                                nominal_current if nominal_target is not None else None
                            ),
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
        _save_registered_snapshot(
            store,
            step=steps,
            run=run,
            algorithm=algorithm,
            sampler=sampler,
            anchor=anchor,
            runtime=runtime,
            rejected=rejected,
            last_update=last_update,
            replay=replay,
            action_rng=action_rng,
        )
    progress.close()
    return {
        "environment_steps": steps,
        "last_update": last_update,
        "rejected_training_cases": rejected,
    }


def run_formal_training(
    config: RecoveryRLV2Config,
    *,
    condition: str,
    seed_index: int,
    resume: bool,
) -> dict[str, object]:
    run = prepare_formal_run(
        config, condition=condition, seed_index=seed_index
    )
    if run.constant_control:
        return build_constant_control_timeline(config, run)
    report_path = run.output_dir / "training_report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("binding") != run.binding:
            raise ValueError("formal training report binding differs")
        return report
    manifest = load_case_manifest(
        config.output_dir / "manifests" / "cases.json"
    )
    _seed_all(run.seed)
    residual_coefficient, _, _ = _anchor_coefficients(config, run.anchoring)
    with _runtime(
        config,
        seed=run.seed,
        residual_coefficient=residual_coefficient,
    ) as runtime:
        actor, algorithm, observation, _ = _make_algorithm(
            config, run, runtime, manifest
        )
        anchor = _formal_anchor(config, run, manifest, runtime)
        store = SnapshotStore(run.output_dir / "snapshots")
        if run.backend == "ppo":
            assert isinstance(algorithm, PPOV2)
            result = _run_formal_ppo(
                config,
                run,
                manifest,
                runtime,
                actor,
                algorithm,
                observation,
                anchor,
                store,
                resume=resume,
            )
        else:
            assert isinstance(algorithm, SAC)
            result = _run_formal_sac(
                config,
                run,
                manifest,
                runtime,
                actor,
                algorithm,
                observation,
                anchor,
                store,
                resume=resume,
            )
    completed = [
        step
        for step in config.snapshot_steps
        if (run.output_dir / "snapshots" / f"step_{step:06d}" / "COMPLETED").is_file()
    ]
    report = {
        "schema_version": FORMAL_SCHEMA,
        "passed": completed == list(config.snapshot_steps),
        "condition": run.condition,
        "seed_index": run.seed_index,
        "training_seed": run.seed,
        "backend": run.backend,
        "anchoring": run.anchoring,
        "binding": run.binding,
        "parent_checkpoint": run.parent_checkpoint,
        "trainable_groups": list(run.trainable_groups),
        "environment_steps": int(result["environment_steps"]),
        "snapshot_steps": completed,
        "last_update": result["last_update"],
        "rejected_training_cases": result["rejected_training_cases"],
    }
    if report["passed"] is not True:
        raise RuntimeError("formal training ended without every registered snapshot")
    _write_immutable_json(report_path, report)
    return report
