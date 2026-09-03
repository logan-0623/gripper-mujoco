from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from ..state_bank.io import write_json_atomic
from ..statistics import (
    benjamini_hochberg,
    clustered_bootstrap_mean,
    paired_sign_flip_pvalue,
)
from .config import LiberoStudyConfig
from .feature_binding import LIBERO_SMOLVLA_RENAME_MAP
from .latents import (
    _file_sha256,
    collate_state_bank_observations,
    load_latent_cache,
)
from .positive_control import load_positive_control_plan, positive_control_root
from .probes import _binary_auprc
from .recruitment import (
    _action_effect,
    _atomic_npz,
    _context_batch_plan,
    _predict,
    same_norm_random_delta,
)
from .schema import StateRecord
from .sparse_autoencoder import (
    TopKSparseAutoencoder,
    load_sparse_autoencoder,
    match_decoder_features,
    save_sparse_autoencoder,
    train_sparse_autoencoder,
)
from .state_bank import load_state_bank


SPARSE_FEATURE_SCHEMA = "libero_smolvla_sparse_features_v1"
PRIMARY_TAP = "action_expert_input"
FEATURE_MULTIPLIER = 2
TOP_K = 32
TRAINING_STEPS = 5000
TRAINING_BATCH_SIZE = 512
CANDIDATE_LIMIT = 8
MINIMUM_CANDIDATES = 4
MINIMUM_TASKS = 10
MINIMUM_EPISODES = 30
MINIMUM_DECODER_COSINE = 0.70
MINIMUM_ACTIVATION_CORRELATION = 0.50
MINIMUM_BREADTH_ENTROPY = 0.50


def sparse_feature_root(output_dir: str | Path) -> Path:
    return Path(output_dir) / "protocol_v5" / "sparse_features"


def _episode_key(record: StateRecord) -> str:
    return f"{record.suite}:{record.task_id}:{record.source_episode_id}"


def _task_key(record: StateRecord) -> str:
    return f"{record.suite}:{record.task_id}"


def _normalized_entropy(mass: np.ndarray) -> float:
    values = np.asarray(mass, dtype=np.float64)
    total = float(values.sum())
    if len(values) <= 1 or total <= 0:
        return 0.0
    probabilities = values[values > 0] / total
    return float(-(probabilities * np.log(probabilities)).sum() / np.log(len(values)))


def summarize_activations(
    activations: np.ndarray,
    tasks: Sequence[str],
    episodes: Sequence[str],
    frames: Sequence[int],
    *,
    seed: int,
) -> list[dict[str, object]]:
    values = np.asarray(activations, dtype=np.float64)
    tasks = np.asarray(tasks)
    episodes = np.asarray(episodes)
    frames = np.asarray(frames, dtype=np.int64)
    if (
        values.ndim != 2
        or len(values) == 0
        or tasks.shape != (len(values),)
        or episodes.shape != tasks.shape
        or frames.shape != tasks.shape
        or not np.isfinite(values).all()
        or np.any(values < 0)
    ):
        raise ValueError("feature activations and metadata must be aligned and finite")
    task_names = np.unique(tasks)
    episode_names = np.unique(episodes)
    task_mass = np.stack([values[tasks == key].sum(axis=0) for key in task_names])
    episode_mass = np.stack([values[episodes == key].sum(axis=0) for key in episode_names])
    active = values > 1e-8
    adjacent: list[np.ndarray] = []
    shuffled: list[np.ndarray] = []
    rng = np.random.default_rng(seed)
    for episode in episode_names:
        rows = np.flatnonzero(episodes == episode)
        rows = rows[np.argsort(frames[rows], kind="stable")]
        if len(rows) < 2:
            continue
        adjacent.append(np.abs(np.diff(values[rows], axis=0)).mean(axis=0))
        order = rng.permutation(len(rows))
        shuffled.append(np.abs(np.diff(values[rows[order]], axis=0)).mean(axis=0))
    adjacent_mean = np.mean(adjacent, axis=0) if adjacent else np.zeros(values.shape[1])
    shuffled_mean = np.mean(shuffled, axis=0) if shuffled else np.zeros(values.shape[1])
    ratio = adjacent_mean / np.maximum(shuffled_mean, 1e-12)
    return [
        {
            "feature": feature,
            "active_fraction": float(active[:, feature].mean()),
            "mean_activation": float(values[:, feature].mean()),
            "task_count": int(np.sum(task_mass[:, feature] > 0)),
            "episode_count": int(np.sum(episode_mass[:, feature] > 0)),
            "task_entropy": _normalized_entropy(task_mass[:, feature]),
            "episode_entropy": _normalized_entropy(episode_mass[:, feature]),
            "temporal_difference_ratio": float(ratio[feature]),
        }
        for feature in range(values.shape[1])
    ]


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    if np.std(first) <= 1e-12 or np.std(second) <= 1e-12:
        return -1.0
    return float(np.corrcoef(first, second)[0, 1])


def select_stable_features(
    activations_by_seed: Sequence[np.ndarray],
    decoders_by_seed: Sequence[np.ndarray],
    profiles: Sequence[Mapping[str, object]],
    *,
    limit: int = CANDIDATE_LIMIT,
    min_tasks: int = MINIMUM_TASKS,
    min_episodes: int = MINIMUM_EPISODES,
    min_decoder_cosine: float = MINIMUM_DECODER_COSINE,
    min_activation_correlation: float = MINIMUM_ACTIVATION_CORRELATION,
    min_breadth_entropy: float = MINIMUM_BREADTH_ENTROPY,
) -> list[dict[str, object]]:
    if len(activations_by_seed) < 2 or len(activations_by_seed) != len(decoders_by_seed):
        raise ValueError("stable feature selection requires at least two aligned seeds")
    reference = np.asarray(activations_by_seed[0])
    if len(profiles) != reference.shape[1]:
        raise ValueError("feature profiles do not match the reference dictionary")
    mappings: list[dict[int, tuple[int, float]]] = []
    for decoder in decoders_by_seed[1:]:
        mappings.append(
            {
                left: (right, cosine)
                for left, right, cosine in match_decoder_features(
                    np.asarray(decoders_by_seed[0]), np.asarray(decoder)
                )
            }
        )
    candidates: list[dict[str, object]] = []
    for feature, profile in enumerate(profiles):
        if (
            int(profile["task_count"]) < min_tasks
            or int(profile["episode_count"]) < min_episodes
            or float(profile["task_entropy"]) < min_breadth_entropy
            or float(profile["episode_entropy"]) < min_breadth_entropy
        ):
            continue
        matched = [mapping.get(feature) for mapping in mappings]
        if any(row is None for row in matched):
            continue
        rows = [row for row in matched if row is not None]
        cosines = [float(row[1]) for row in rows]
        correlations = [
            _correlation(reference[:, feature], np.asarray(values)[:, row[0]])
            for values, row in zip(activations_by_seed[1:], rows, strict=True)
        ]
        if min(cosines) < min_decoder_cosine or min(correlations) < min_activation_correlation:
            continue
        breadth = float(
            np.sqrt(float(profile["task_entropy"]) * float(profile["episode_entropy"]))
        )
        candidates.append(
            {
                "feature": feature,
                "matched_features": [feature, *[int(row[0]) for row in rows]],
                "minimum_decoder_cosine": float(min(cosines)),
                "minimum_activation_correlation": float(min(correlations)),
                "stability_score": float(min((*cosines, *correlations))),
                "breadth_score": breadth,
                "selection_score": float(min((*cosines, *correlations))) * breadth,
                "profile": dict(profile),
            }
        )
    return sorted(candidates, key=lambda row: (-float(row["selection_score"]), int(row["feature"])))[:limit]


def feature_removal_delta(
    activation: np.ndarray, decoder_feature: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    strength = np.asarray(activation, dtype=np.float64)
    direction = np.asarray(decoder_feature, dtype=np.float64)
    normalization = np.asarray(scale, dtype=np.float64)
    if strength.ndim != 1 or direction.ndim != 1 or normalization.shape != direction.shape:
        raise ValueError("feature contribution dimensions are incompatible")
    if not all(np.isfinite(value).all() for value in (strength, direction, normalization)):
        raise ValueError("feature contribution must be finite")
    return -strength[:, None] * direction[None, :] * normalization[None, :]


def matched_random_feature_delta(
    target_delta: np.ndarray, *, direction: np.ndarray, seed: int
) -> np.ndarray:
    return same_norm_random_delta(
        target_delta, target_direction=np.asarray(direction, dtype=np.float64), seed=seed
    )


def summarize_action_usage(
    target_effect: np.ndarray,
    random_effect: np.ndarray,
    episodes: Sequence[str],
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> dict[str, object]:
    target = np.asarray(target_effect, dtype=np.float64)
    random = np.asarray(random_effect, dtype=np.float64)
    groups = np.asarray(episodes)
    if target.ndim != 1 or random.shape != target.shape or groups.shape != target.shape:
        raise ValueError("action effects and episode clusters must be aligned")
    difference = target - random
    unique = np.unique(groups)
    cluster_means = np.asarray([difference[groups == group].mean() for group in unique])
    return {
        "target_effect": float(target.mean()),
        "matched_random_effect": float(random.mean()),
        "target_minus_random": clustered_bootstrap_mean(
            difference, groups, samples=samples, confidence=confidence, seed=seed
        ),
        "p_value": paired_sign_flip_pvalue(cluster_means, samples=samples, seed=seed + 1),
    }


def sparse_feature_decision(
    *, candidate_count: int, causal_rows: Sequence[Mapping[str, object]]
) -> str:
    if candidate_count < MINIMUM_CANDIDATES:
        return "stop_no_stable_features"
    if any(float(row["ci_low"]) > 0 and float(row["q_value"]) <= 0.05 for row in causal_rows):
        return "authorize_separate_longitudinal_design"
    return "stop_no_causal_features"


def _encode(
    model: TopKSparseAutoencoder,
    mean: np.ndarray,
    scale: np.ndarray,
    values: np.ndarray,
    *,
    batch_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    activations: list[np.ndarray] = []
    reconstructions: list[np.ndarray] = []
    normalized = (np.asarray(values, dtype=np.float32) - mean) / scale
    with torch.no_grad():
        for start in range(0, len(normalized), batch_size):
            batch = torch.from_numpy(normalized[start : start + batch_size]).to(device)
            reconstructed, encoded = model(batch)
            activations.append(encoded.cpu().numpy())
            reconstructions.append(reconstructed.cpu().numpy())
    model.cpu()
    return np.concatenate(activations), np.concatenate(reconstructions)


def _reconstruction_metrics(values: np.ndarray, reconstructed: np.ndarray) -> dict[str, float]:
    residual = float(np.sum((values - reconstructed) ** 2))
    centered = float(np.sum((values - values.mean(axis=0, keepdims=True)) ** 2))
    return {
        "mse": float(np.mean((values - reconstructed) ** 2)),
        "explained_variance": 1.0 - residual / centered if centered > 0 else 0.0,
    }


def _binary_association(activation: np.ndarray, target: np.ndarray) -> dict[str, object]:
    positive = _binary_auprc(target.astype(np.int64), activation)
    negative = _binary_auprc(target.astype(np.int64), -activation)
    sign = "positive" if positive >= negative else "negative"
    return {"auprc": float(max(positive, negative)), "direction": sign}


def _candidate_associations(
    candidates: Sequence[Mapping[str, object]], activations: np.ndarray, records: Sequence[StateRecord]
) -> list[dict[str, object]]:
    contact = np.asarray([bool(record.labels.contact and record.labels.contact.gripper_target) for record in records])
    grasp = np.asarray([bool(record.labels.stable_grasp) for record in records])
    phases = np.asarray([str(record.labels.phase) for record in records])
    result: list[dict[str, object]] = []
    for candidate in candidates:
        feature = int(candidate["feature"])
        values = activations[:, feature]
        phase_means = {phase: float(values[phases == phase].mean()) for phase in sorted(set(phases))}
        result.append(
            {
                "feature": feature,
                "contact": _binary_association(values, contact),
                "stable_grasp": _binary_association(values, grasp),
                "phase_mean_activation": phase_means,
            }
        )
    return result


def _code_sha256() -> str:
    digest = hashlib.sha256()
    for path in (Path(__file__), Path(__file__).with_name("sparse_autoencoder.py")):
        digest.update(path.name.encode())
        digest.update(_file_sha256(path).encode())
    return digest.hexdigest()


def _pca_reference(train: np.ndarray, validation: np.ndarray, rank: int = TOP_K) -> dict[str, float]:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = torch.from_numpy(train.astype(np.float32)).to(device)
    _, _, components = torch.pca_lowrank(source, q=rank, center=False)
    held_out = torch.from_numpy(validation.astype(np.float32)).to(device)
    reconstructed = (held_out @ components) @ components.T
    return _reconstruction_metrics(validation, reconstructed.cpu().numpy())


def discover_sparse_features(config: LiberoStudyConfig) -> dict[str, object]:
    plan = load_positive_control_plan(config)
    source_root = positive_control_root(config.output_dir)
    latent_root = source_root / "latents" / PRIMARY_TAP
    state_ids, features, latent_manifest = load_latent_cache(latent_root)
    records, state_manifest, _, episode_split = load_state_bank(config.output_dir / "state_bank")
    if state_ids != tuple(record.state_id for record in records):
        raise ValueError("sparse discovery latent rows differ from the State Bank")
    root = sparse_feature_root(config.output_dir)
    report_path = root / "discovery.json"
    binding = {
        "schema_version": SPARSE_FEATURE_SCHEMA,
        "plan_sha256": _file_sha256(source_root / "plan.json"),
        "checkpoint_sha256": plan["checkpoint_sha256"],
        "latent_values_sha256": latent_manifest["values_sha256"],
        "state_bank_records_sha256": state_manifest["records_sha256"],
        "code_sha256": _code_sha256(),
        "tap": PRIMARY_TAP,
    }
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("binding") != binding:
            raise FileExistsError(f"sparse discovery binding changed: {report_path}")
        if _file_sha256(Path(str(report["candidates"]))) != report.get(
            "candidates_sha256"
        ):
            raise ValueError("sparse candidate artifact changed")
        if _file_sha256(Path(str(report["feature_profiles"]))) != report.get(
            "feature_profiles_sha256"
        ):
            raise ValueError("sparse feature-profile artifact changed")
        for row in report.get("seed_reports", ()):
            if _file_sha256(Path(str(row["model"]))) != row.get("model_sha256"):
                raise ValueError("sparse model artifact changed")
        return report
    partitions = np.asarray([episode_split.assignments[state_id] for state_id in state_ids])
    train_rows = np.flatnonzero(partitions == "train")
    validation_rows = np.flatnonzero(partitions == "validation")
    test_rows = np.flatnonzero(partitions == "test")
    if min(map(len, (train_rows, validation_rows, test_rows))) == 0:
        raise ValueError("episode-group discovery partitions must be non-empty")
    seeds = tuple(config.seed + offset for offset in (101, 211, 307))
    activations: list[np.ndarray] = []
    decoders: list[np.ndarray] = []
    seed_reports: list[dict[str, object]] = []
    feature_dim = FEATURE_MULTIPLIER * features.shape[1]
    for seed in seeds:
        model_path = root / "models" / f"seed_{seed}.npz"
        seed_path = root / "models" / f"seed_{seed}.json"
        seed_binding = {
            **binding,
            "seed": seed,
            "feature_dim": feature_dim,
            "top_k": TOP_K,
            "steps": TRAINING_STEPS,
            "batch_size": TRAINING_BATCH_SIZE,
        }
        if model_path.is_file() or seed_path.is_file():
            if not model_path.is_file() or not seed_path.is_file():
                raise FileExistsError(f"incomplete sparse seed artifact: {model_path}")
            seed_report = json.loads(seed_path.read_text(encoding="utf-8"))
            if seed_report.get("binding") != seed_binding or seed_report.get(
                "model_sha256"
            ) != _file_sha256(model_path):
                raise FileExistsError(f"sparse seed binding changed: {model_path}")
            model, mean, scale = load_sparse_autoencoder(model_path)
            history = seed_report["history"]
        else:
            model, mean, scale, history = train_sparse_autoencoder(
                features[train_rows],
                feature_dim=feature_dim,
                top_k=TOP_K,
                steps=TRAINING_STEPS,
                batch_size=TRAINING_BATCH_SIZE,
                seed=seed,
            )
            save_sparse_autoencoder(model_path, model, mean=mean, scale=scale)
            write_json_atomic(
                seed_path,
                {
                    "binding": seed_binding,
                    "history": history,
                    "model_sha256": _file_sha256(model_path),
                },
            )
            model, mean, scale = load_sparse_autoencoder(model_path)
        encoded, reconstructed = _encode(model, mean, scale, features)
        normalized = (features - mean) / scale
        seed_reports.append(
            {
                "seed": seed,
                "model": str(model_path),
                "model_sha256": _file_sha256(model_path),
                "history": history,
                "mean_l0": float(np.mean(np.sum(encoded > 0, axis=1))),
                "alive_features_validation": int(np.sum(np.any(encoded[validation_rows] > 0, axis=0))),
                "validation": _reconstruction_metrics(normalized[validation_rows], reconstructed[validation_rows]),
                "test": _reconstruction_metrics(normalized[test_rows], reconstructed[test_rows]),
            }
        )
        activations.append(encoded)
        decoders.append(model.decoder.weight.detach().numpy().T)
    profile_rows = np.concatenate((train_rows, validation_rows))
    profiles = summarize_activations(
        activations[0][profile_rows],
        [_task_key(records[index]) for index in profile_rows],
        [_episode_key(records[index]) for index in profile_rows],
        [records[index].frame_index for index in profile_rows],
        seed=config.seed,
    )
    for profile in profiles:
        if float(profile["active_fraction"]) == 0:
            scope = "inactive"
        elif (
            int(profile["task_count"]) >= MINIMUM_TASKS
            and int(profile["episode_count"]) >= MINIMUM_EPISODES
            and float(profile["task_entropy"]) >= MINIMUM_BREADTH_ENTROPY
            and float(profile["episode_entropy"]) >= MINIMUM_BREADTH_ENTROPY
        ):
            scope = "broad"
        elif float(profile["task_entropy"]) < MINIMUM_BREADTH_ENTROPY:
            scope = "task_concentrated"
        elif float(profile["episode_entropy"]) < MINIMUM_BREADTH_ENTROPY:
            scope = "episode_concentrated"
        else:
            scope = "intermediate"
        profile["scope"] = scope
    profiles_path = root / "feature_profiles.json"
    write_json_atomic(
        profiles_path,
        {
            "schema_version": SPARSE_FEATURE_SCHEMA,
            "selection_uses_interaction_labels": False,
            "profiles": profiles,
        },
    )
    candidates = select_stable_features(
        [values[validation_rows] for values in activations], decoders, profiles
    )
    frozen = {
        "schema_version": SPARSE_FEATURE_SCHEMA,
        "passed": len(candidates) >= MINIMUM_CANDIDATES,
        "selection_uses_interaction_labels": False,
        "selection_partition": "episode_group_train_plus_validation_metadata",
        "stability_partition": "episode_group_validation",
        "thresholds": {
            "minimum_candidates": MINIMUM_CANDIDATES,
            "minimum_tasks": MINIMUM_TASKS,
            "minimum_episodes": MINIMUM_EPISODES,
            "minimum_decoder_cosine": MINIMUM_DECODER_COSINE,
            "minimum_activation_correlation": MINIMUM_ACTIVATION_CORRELATION,
            "minimum_breadth_entropy": MINIMUM_BREADTH_ENTROPY,
        },
        "candidates": candidates,
    }
    candidates_path = root / "candidates.json"
    write_json_atomic(candidates_path, frozen)
    test_records = [records[index] for index in test_rows]
    report = {
        "schema_version": SPARSE_FEATURE_SCHEMA,
        "passed": bool(frozen["passed"]),
        "status": "complete",
        "binding": binding,
        "settings": {
            "seeds": seeds,
            "feature_dim": feature_dim,
            "top_k": TOP_K,
            "steps": TRAINING_STEPS,
            "batch_size": TRAINING_BATCH_SIZE,
        },
        "states_by_partition": {
            "train": len(train_rows), "validation": len(validation_rows), "test": len(test_rows)
        },
        "pca_rank_32_validation": _pca_reference(
            ((features[train_rows] - features[train_rows].mean(axis=0)) / np.where(features[train_rows].std(axis=0) > 1e-6, features[train_rows].std(axis=0), 1.0)).astype(np.float32),
            ((features[validation_rows] - features[train_rows].mean(axis=0)) / np.where(features[train_rows].std(axis=0) > 1e-6, features[train_rows].std(axis=0), 1.0)).astype(np.float32),
        ),
        "seed_reports": seed_reports,
        "candidate_count": len(candidates),
        "feature_scopes": {
            scope: sum(profile["scope"] == scope for profile in profiles)
            for scope in ("broad", "task_concentrated", "episode_concentrated", "intermediate", "inactive")
        },
        "feature_profiles": str(profiles_path),
        "feature_profiles_sha256": _file_sha256(profiles_path),
        "candidates": str(candidates_path),
        "candidates_sha256": _file_sha256(candidates_path),
        "test_associations_after_freeze": _candidate_associations(
            candidates, activations[0][test_rows], test_records
        ),
        "next_gate": "feature_intervention_vs_matched_random" if frozen["passed"] else "stop",
    }
    write_json_atomic(report_path, report)
    return report


def _select_feature_states(
    activation: np.ndarray,
    source_rows: np.ndarray,
    records: Sequence[StateRecord],
    *,
    max_states: int,
) -> tuple[int, ...]:
    buckets: dict[str, list[int]] = {}
    for index in source_rows:
        if activation[index] <= 1e-8:
            continue
        buckets.setdefault(_task_key(records[index]), []).append(int(index))
    for rows in buckets.values():
        rows.sort(key=lambda index: (-float(activation[index]), records[index].state_id))
    selected: list[int] = []
    for cursor in range(max(map(len, buckets.values()), default=0)):
        for task in sorted(buckets):
            if cursor < len(buckets[task]):
                selected.append(buckets[task][cursor])
                if len(selected) == max_states:
                    return tuple(selected)
    return tuple(selected)


def intervene_sparse_features(
    config: LiberoStudyConfig, *, max_states: int = 512, batch_size: int = 32
) -> dict[str, object]:
    if max_states <= 0 or batch_size <= 0:
        raise ValueError("sparse feature intervention sizes must be positive")
    root = sparse_feature_root(config.output_dir)
    discovery_path = root / "discovery.json"
    discovery = json.loads(discovery_path.read_text(encoding="utf-8"))
    if not discovery.get("passed"):
        raise ValueError("sparse feature discovery gate did not pass")
    candidates_path = root / "candidates.json"
    frozen = json.loads(candidates_path.read_text(encoding="utf-8"))
    if _file_sha256(candidates_path) != discovery.get("candidates_sha256"):
        raise ValueError("sparse candidate artifact changed")
    candidates = frozen["candidates"]
    source_root = positive_control_root(config.output_dir)
    latent_report = json.loads((source_root / "latents" / "report.json").read_text())
    required_batch_size = int(latent_report["runtime"]["batch_size"])
    if batch_size != required_batch_size:
        raise ValueError(
            f"action batch size must match official latent extraction ({required_batch_size})"
        )
    report_path = root / f"action_sensitivity_n_{max_states:04d}.json"
    model_path = Path(str(discovery["seed_reports"][0]["model"]))
    binding = {
        "schema_version": SPARSE_FEATURE_SCHEMA,
        "discovery_sha256": _file_sha256(discovery_path),
        "candidates_sha256": _file_sha256(candidates_path),
        "model_sha256": _file_sha256(model_path),
        "max_states": max_states,
        "batch_size": batch_size,
        "code_sha256": _code_sha256(),
    }
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("binding") != binding:
            raise FileExistsError(f"sparse intervention binding changed: {report_path}")
        if _file_sha256(Path(str(report["effects"]))) != report.get("effects_sha256"):
            raise ValueError("sparse action-effect artifact changed")
        return report
    plan = load_positive_control_plan(config)
    records, _, _, episode_split = load_state_bank(config.output_dir / "state_bank")
    state_ids, features, _ = load_latent_cache(source_root / "latents" / PRIMARY_TAP)
    if state_ids != tuple(record.state_id for record in records):
        raise ValueError("sparse intervention rows differ from the State Bank")
    model, mean, scale = load_sparse_autoencoder(model_path)
    activations, _ = _encode(model, mean, scale, features)
    test_rows = np.asarray(
        [index for index, state_id in enumerate(state_ids) if episode_split.assignments[state_id] == "test"]
    )
    selected = {
        int(row["feature"]): _select_feature_states(
            activations[:, int(row["feature"])], test_rows, records, max_states=max_states
        )
        for row in candidates
    }
    estimable = {feature: rows for feature, rows in selected.items() if len(rows) >= 32}
    if not estimable:
        raise ValueError("no frozen sparse feature has 32 active held-out states")
    union_rows = tuple(sorted({index for rows in estimable.values() for index in rows}))
    try:
        from lerobot.datasets import LeRobotDataset
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("LeRobotDataset is required for sparse feature actions") from error
    dataset = LeRobotDataset(
        config.sources.lerobot_repo_id,
        root=config.sources.lerobot_root,
        revision=config.sources.lerobot_revision,
        download_videos=True,
    )
    from ..backends.lerobot import SmolVLABackend

    backend = SmolVLABackend(device="auto")
    backend.load_checkpoint_for_dataset(
        str(plan["checkpoint"]),
        repo_id=config.sources.lerobot_repo_id,
        dataset_root=dataset.root,
        rename_map=LIBERO_SMOLVLA_RENAME_MAP,
    )
    policy, preprocessor, postprocessor = backend._loaded()
    checkpoint_id = f"official:{str(plan['checkpoint_sha256'])[:16]}"

    def predict_rows(rows: tuple[int, ...], delta_values: np.ndarray) -> np.ndarray:
        ids = tuple(state_ids[index] for index in rows)
        output: np.ndarray | None = None
        for start, stop, context_rows, output_rows in _context_batch_plan(
            state_ids, ids, batch_size=batch_size
        ):
            expected = features[start:stop]
            delta = np.zeros_like(expected)
            delta[np.asarray(context_rows)] = delta_values[np.asarray(output_rows)]
            predicted = _predict(
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                backend=backend,
                batch=collate_state_bank_observations(records[start:stop], dataset),
                state_ids=state_ids[start:stop],
                checkpoint_id=checkpoint_id,
                delta=delta,
                expected=expected,
            )
            if output is None:
                output = np.empty((len(rows), *predicted.shape[1:]), dtype=predicted.dtype)
            output[np.asarray(output_rows)] = predicted[np.asarray(context_rows)]
        if output is None:
            raise ValueError("sparse action batching produced no outputs")
        return output

    zero = np.zeros((len(union_rows), features.shape[1]), dtype=np.float32)
    original_union = predict_rows(union_rows, zero)
    union_lookup = {index: row for row, index in enumerate(union_rows)}
    effect_arrays: dict[str, np.ndarray] = {}
    rows_report: list[dict[str, object]] = []
    decoder = model.decoder.weight.detach().numpy().T
    for feature, rows in sorted(estimable.items()):
        row_array = np.asarray(rows)
        strength = activations[row_array, feature]
        target_delta = feature_removal_delta(strength, decoder[feature], scale).astype(np.float32)
        raw_direction = decoder[feature] * scale
        random_delta = matched_random_feature_delta(
            target_delta, direction=raw_direction, seed=config.seed + feature
        ).astype(np.float32)
        target_actions = predict_rows(rows, target_delta)
        random_actions = predict_rows(rows, random_delta)
        original = original_union[np.asarray([union_lookup[index] for index in rows])]
        target_effects = _action_effect(original, target_actions)
        random_effects = _action_effect(original, random_actions)
        episodes = [_episode_key(records[index]) for index in rows]
        metrics = {
            metric: summarize_action_usage(
                target_effects[metric],
                random_effects[metric],
                episodes,
                samples=max(100, config.probes.bootstrap_samples),
                confidence=config.probes.confidence_level,
                seed=config.seed + feature + metric_index,
            )
            for metric_index, metric in enumerate(target_effects)
        }
        for metric in target_effects:
            effect_arrays[f"f{feature}_{metric}_target"] = target_effects[metric].astype(np.float32)
            effect_arrays[f"f{feature}_{metric}_random"] = random_effects[metric].astype(np.float32)
        effect_arrays[f"f{feature}_state_ids"] = np.asarray(
            [state_ids[index] for index in rows]
        )
        rows_report.append(
            {
                "feature": feature,
                "states": len(rows),
                "episodes": len(set(episodes)),
                "tasks": len({_task_key(records[index]) for index in rows}),
                "same_norm_max_abs_error": float(
                    np.max(np.abs(np.linalg.norm(target_delta, axis=1) - np.linalg.norm(random_delta, axis=1)))
                ),
                "activation_norm_ratio": float(
                    np.mean(np.linalg.norm(features[row_array] + target_delta, axis=1))
                    / max(np.mean(np.linalg.norm(features[row_array], axis=1)), 1e-12)
                ),
                "metrics": metrics,
                "primary_p_value": metrics["first_action_l2"]["p_value"],
            }
        )
    q_values = benjamini_hochberg([float(row["primary_p_value"]) for row in rows_report])
    causal_rows: list[dict[str, object]] = []
    for row, q_value in zip(rows_report, q_values, strict=True):
        interval = row["metrics"]["first_action_l2"]["target_minus_random"]
        support_passed = (
            0.8 <= float(row["activation_norm_ratio"]) <= 1.2
            and float(row["same_norm_max_abs_error"]) <= 1e-5
        )
        row["primary_q_value"] = q_value
        row["support_passed"] = support_passed
        row["causal_action_feature"] = (
            support_passed and float(interval["ci_low"]) > 0 and q_value <= 0.05
        )
        causal_rows.append(
            {
                "feature": row["feature"],
                "ci_low": interval["ci_low"] if support_passed else -1.0,
                "q_value": q_value,
            }
        )
    effects_path = root / f"action_effects_n_{max_states:04d}.npz"
    _atomic_npz(effects_path, **effect_arrays)
    decision = sparse_feature_decision(candidate_count=len(candidates), causal_rows=causal_rows)
    report = {
        "schema_version": SPARSE_FEATURE_SCHEMA,
        "passed": True,
        "status": "complete",
        "binding": binding,
        "primary_metric": "first_action_l2_target_minus_matched_random",
        "primary_cluster": "episode",
        "multiple_comparison": "Benjamini-Hochberg across frozen candidates",
        "candidate_rows": rows_report,
        "not_estimable_features": sorted(set(selected) - set(estimable)),
        "decision": decision,
        "authorize_longitudinal_design": decision == "authorize_separate_longitudinal_design",
        "closed_loop_useful": "not_measured",
        "effects": str(effects_path),
        "effects_sha256": _file_sha256(effects_path),
    }
    write_json_atomic(report_path, report)
    return report


def report_sparse_features(
    config: LiberoStudyConfig, *, max_states: int = 512
) -> dict[str, object]:
    root = sparse_feature_root(config.output_dir)
    discovery = json.loads((root / "discovery.json").read_text(encoding="utf-8"))
    if not discovery.get("passed"):
        return {
            "schema_version": SPARSE_FEATURE_SCHEMA,
            "passed": False,
            "status": "complete",
            "decision": "stop_no_stable_features",
            "candidate_count": discovery.get("candidate_count", 0),
        }
    actions_path = root / f"action_sensitivity_n_{max_states:04d}.json"
    actions = json.loads(actions_path.read_text(encoding="utf-8"))
    if _file_sha256(Path(str(actions["effects"]))) != actions.get("effects_sha256"):
        raise ValueError("sparse action-effect artifact changed")
    decision = str(actions["decision"])
    report = {
        "schema_version": SPARSE_FEATURE_SCHEMA,
        "passed": decision == "authorize_separate_longitudinal_design",
        "status": "complete",
        "candidate_count": discovery["candidate_count"],
        "causal_feature_count": sum(
            bool(row["causal_action_feature"]) for row in actions["candidate_rows"]
        ),
        "decision": decision,
        "discovery_sha256": _file_sha256(root / "discovery.json"),
        "action_sensitivity_sha256": _file_sha256(actions_path),
        "interpretation_boundary": {
            "feature_labels": "post-freeze associations, not selection criteria",
            "closed_loop_useful": "not measured",
            "new_training": "authorized only by a positive causal feature gate",
        },
    }
    write_json_atomic(root / "report.json", report)
    return report
