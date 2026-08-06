from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np

from .config import load_config
from .env import EnvStep, KinematicTabletopEnv, TerminationReason
from .expert import ExpertPhase, ScriptedExpert
from .graph.builder import SceneGraphBuilder
from .graph.schema import SceneGraph, SceneSnapshot
from .recovery import RecoverySpec, apply_recovery_spec
from .recovery import make_recovery_spec


class TabletopEnvironment(Protocol):
    def reset(self, seed: int, object_count: int, target_index: int | None = None) -> SceneSnapshot: ...
    def step(self, action: np.ndarray) -> EnvStep: ...
    def proprioception(self) -> np.ndarray: ...
    def perturb_gripper_state(
        self, delta: np.ndarray, gripper_open: float | None = None
    ) -> SceneSnapshot: ...


@dataclass(frozen=True)
class EpisodeStep:
    snapshot: SceneSnapshot
    graph: SceneGraph
    proprioception: np.ndarray
    action: np.ndarray
    phase: ExpertPhase


@dataclass(frozen=True)
class Episode:
    seed: int
    object_count: int
    target_name: str
    steps: tuple[EpisodeStep, ...]
    reason: TerminationReason
    trajectory_kind: str = "base"
    source_seed: int | None = None
    variant_id: int | None = None
    perturbation_kind: str | None = None
    injection_phase: str | None = None


@dataclass(frozen=True)
class EpisodeArrays:
    seed: int
    object_count: int
    target_name: str
    reason: str
    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    node_mask: np.ndarray
    edge_mask: np.ndarray
    proprioception: np.ndarray
    actions: np.ndarray
    phases: np.ndarray
    trajectory_kind: str = "base"
    source_seed: int | None = None
    variant_id: int | None = None
    perturbation_kind: str | None = None
    injection_phase: str | None = None


@dataclass(frozen=True)
class EpisodeSplits:
    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]


def episode_paths_from_manifest(data_dir: str | Path) -> tuple[Path, ...]:
    return _paths_from_manifest(data_dir, "manifest.json", label="episode")


def recovery_paths_from_manifest(data_dir: str | Path) -> tuple[Path, ...]:
    return _paths_from_manifest(
        data_dir, "recovery_manifest.json", label="recovery episode"
    )


def _paths_from_manifest(
    data_dir: str | Path,
    manifest_name: str,
    *,
    label: str,
) -> tuple[Path, ...]:
    directory = Path(data_dir)
    manifest = directory / manifest_name
    if not manifest.exists():
        raise FileNotFoundError(f"{label} manifest not found: {manifest}")
    records = json.loads(manifest.read_text(encoding="utf-8"))
    paths: list[Path] = []
    for record in records:
        filename = str(record["path"])
        if Path(filename).name != filename:
            raise ValueError(f"manifest {label} path must be a filename: {filename}")
        path = directory / filename
        if not path.exists():
            raise FileNotFoundError(f"manifest {label} does not exist: {path}")
        paths.append(path)
    if not paths:
        raise ValueError(f"{label} manifest is empty: {manifest}")
    if len(set(paths)) != len(paths):
        raise ValueError(f"{label} manifest contains duplicate paths")
    return tuple(paths)


def collect_episode(
    env: TabletopEnvironment,
    expert: ScriptedExpert,
    *,
    seed: int,
    object_count: int,
    builder: SceneGraphBuilder | None = None,
) -> Episode:
    builder = builder or SceneGraphBuilder(max_objects=5)
    expert.reset()
    snapshot = env.reset(seed=seed, object_count=object_count)
    target_name = snapshot.target_object.name
    steps: list[EpisodeStep] = []

    while True:
        action = expert.act(snapshot)
        graph = builder.build(snapshot)
        step = EpisodeStep(
            snapshot=snapshot,
            graph=graph,
            proprioception=env.proprioception().copy(),
            action=np.asarray(action, dtype=np.float32).copy(),
            phase=expert.phase,
        )
        result = env.step(action)
        steps.append(step)
        snapshot = result.snapshot
        if result.done:
            return Episode(
                seed=seed,
                object_count=object_count,
                target_name=target_name,
                steps=tuple(steps),
                reason=result.reason,
            )


def collect_recovery_episode(
    env: TabletopEnvironment,
    expert: ScriptedExpert,
    *,
    source_seed: int,
    object_count: int,
    spec: RecoverySpec,
    builder: SceneGraphBuilder | None = None,
) -> Episode:
    if spec.source_seed != source_seed:
        raise ValueError("recovery spec source_seed does not match the episode source")
    builder = builder or SceneGraphBuilder(max_objects=5)
    expert.reset()
    snapshot = env.reset(seed=source_seed, object_count=object_count)
    target_name = snapshot.target_object.name
    steps: list[EpisodeStep] = []
    injected = False

    while True:
        action = expert.act(snapshot)
        if not injected and expert.phase is spec.injection_phase:
            snapshot = apply_recovery_spec(env, spec)  # type: ignore[arg-type]
            injected = True
            action = expert.act(snapshot)
            frame_phase = spec.injection_phase
        elif injected:
            frame_phase = expert.phase
        else:
            result = env.step(action)
            snapshot = result.snapshot
            if result.done:
                raise RuntimeError(
                    f"base prefix ended with {result.reason.value} before "
                    f"{spec.injection_phase.value} injection"
                )
            continue

        steps.append(
            EpisodeStep(
                snapshot=snapshot,
                graph=builder.build(snapshot),
                proprioception=env.proprioception().copy(),
                action=np.asarray(action, dtype=np.float32).copy(),
                phase=frame_phase,
            )
        )
        result = env.step(action)
        snapshot = result.snapshot
        if result.done:
            if result.reason is not TerminationReason.SUCCESS:
                raise RuntimeError(
                    f"recovery trajectory ended with {result.reason.value} after "
                    f"{spec.kind.value} injection"
                )
            return Episode(
                seed=source_seed,
                object_count=object_count,
                target_name=target_name,
                steps=tuple(steps),
                reason=result.reason,
                trajectory_kind="recovery",
                source_seed=source_seed,
                variant_id=spec.variant_id,
                perturbation_kind=spec.kind.value,
                injection_phase=spec.injection_phase.value,
            )


def save_episode(episode: Episode, path: str | Path) -> Path:
    if not episode.steps:
        raise ValueError("cannot save an episode with no steps")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.dumps(
        {
            "seed": episode.seed,
            "object_count": episode.object_count,
            "target_name": episode.target_name,
            "reason": episode.reason.value,
            "trajectory_kind": episode.trajectory_kind,
            "source_seed": episode.source_seed,
            "variant_id": episode.variant_id,
            "perturbation_kind": episode.perturbation_kind,
            "injection_phase": episode.injection_phase,
        },
        sort_keys=True,
    )
    np.savez_compressed(
        destination,
        metadata=np.asarray(metadata),
        node_features=np.stack([step.graph.node_features for step in episode.steps]),
        edge_index=episode.steps[0].graph.edge_index,
        edge_features=np.stack([step.graph.edge_features for step in episode.steps]),
        node_mask=np.stack([step.graph.node_mask for step in episode.steps]),
        edge_mask=np.stack([step.graph.edge_mask for step in episode.steps]),
        proprioception=np.stack([step.proprioception for step in episode.steps]),
        actions=np.stack([step.action for step in episode.steps]),
        phases=np.asarray([step.phase.value for step in episode.steps]),
    )
    return destination


def load_episode_arrays(path: str | Path) -> EpisodeArrays:
    with np.load(Path(path), allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        return EpisodeArrays(
            seed=int(metadata["seed"]),
            object_count=int(metadata["object_count"]),
            target_name=str(metadata["target_name"]),
            reason=str(metadata["reason"]),
            node_features=archive["node_features"].copy(),
            edge_index=archive["edge_index"].copy(),
            edge_features=archive["edge_features"].copy(),
            node_mask=archive["node_mask"].copy(),
            edge_mask=archive["edge_mask"].copy(),
            proprioception=archive["proprioception"].copy(),
            actions=archive["actions"].copy(),
            phases=archive["phases"].copy(),
            trajectory_kind=str(metadata.get("trajectory_kind", "base")),
            source_seed=(
                None if metadata.get("source_seed") is None else int(metadata["source_seed"])
            ),
            variant_id=(
                None if metadata.get("variant_id") is None else int(metadata["variant_id"])
            ),
            perturbation_kind=(
                None
                if metadata.get("perturbation_kind") is None
                else str(metadata["perturbation_kind"])
            ),
            injection_phase=(
                None
                if metadata.get("injection_phase") is None
                else str(metadata["injection_phase"])
            ),
        )


def split_episode_seeds(
    seeds: Iterable[int],
    *,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> EpisodeSplits:
    if validation_fraction < 0 or test_fraction < 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("validation/test fractions must be non-negative and sum to less than one")
    values = np.asarray(tuple(int(value) for value in seeds), dtype=np.int64)
    if len(np.unique(values)) != len(values):
        raise ValueError("episode seeds must be unique")
    rng = np.random.default_rng(seed)
    shuffled = values[rng.permutation(len(values))]
    validation_size = int(round(len(values) * validation_fraction))
    test_size = int(round(len(values) * test_fraction))
    validation = shuffled[:validation_size]
    test = shuffled[validation_size : validation_size + test_size]
    train = shuffled[validation_size + test_size :]
    return EpisodeSplits(
        train=tuple(int(value) for value in train),
        validation=tuple(int(value) for value in validation),
        test=tuple(int(value) for value in test),
    )


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def collect_from_config(config_path: str | Path) -> Path:
    cfg = load_config(config_path)
    output_dir = Path(cfg.data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    builder = SceneGraphBuilder(max_objects=cfg.max_objects)
    records: list[dict] = []
    rejections: list[dict] = []
    manifest = output_dir / "manifest.json"
    rejection_log = output_dir / "rejections.json"
    _write_json_atomic(manifest, records)
    _write_json_atomic(rejection_log, rejections)
    for episode_index in range(cfg.train.episodes):
        object_count = cfg.train.object_counts[episode_index % len(cfg.train.object_counts)]
        episode_seed = cfg.seed + episode_index
        env = KinematicTabletopEnv(
            max_objects=cfg.max_objects,
            max_steps=cfg.environment.max_steps,
            min_object_distance=cfg.environment.min_object_distance,
            workspace_low=cfg.environment.workspace_low,
            workspace_high=cfg.environment.workspace_high,
            crowded_anchor_min_distance=cfg.environment.crowded_anchor_min_distance,
            crowded_anchor_max_distance=cfg.environment.crowded_anchor_max_distance,
        )
        try:
            episode = collect_episode(
                env,
                ScriptedExpert(),
                seed=episode_seed,
                object_count=object_count,
                builder=builder,
            )
        except RuntimeError as error:
            rejections.append(
                {
                    "episode_index": episode_index,
                    "seed": episode_seed,
                    "object_count": object_count,
                    "error": str(error),
                }
            )
            _write_json_atomic(rejection_log, rejections)
            continue
        path = save_episode(episode, output_dir / f"episode_{episode_index:06d}.npz")
        records.append(
            {
                "episode_index": episode_index,
                "seed": episode_seed,
                "object_count": object_count,
                "frames": len(episode.steps),
                "reason": episode.reason.value,
                "path": path.name,
            }
        )
        _write_json_atomic(manifest, records)
    if not records:
        raise RuntimeError(f"all episode collection attempts failed; see {rejection_log}")
    return manifest


def augment_recovery_from_config(config_path: str | Path) -> Path:
    cfg = load_config(config_path)
    if not cfg.recovery.enabled:
        raise ValueError("recovery augmentation is not enabled in the configuration")

    output_dir = Path(cfg.data_dir)
    base_paths = episode_paths_from_manifest(output_dir)
    arrays_by_path = {path: load_episode_arrays(path) for path in base_paths}
    splits = split_episode_seeds(
        (arrays.seed for arrays in arrays_by_path.values()),
        validation_fraction=0.1,
        test_fraction=0.1,
        seed=cfg.seed,
    )
    train_seeds = set(splits.train)
    manifest = output_dir / "recovery_manifest.json"
    rejection_log = output_dir / "recovery_rejections.json"
    split_path = output_dir / "recovery_source_split.json"
    records: list[dict] = []
    rejections: list[dict] = []
    _write_json_atomic(manifest, records)
    _write_json_atomic(rejection_log, rejections)
    _write_json_atomic(
        split_path,
        {
            "train": list(splits.train),
            "validation": list(splits.validation),
            "test": list(splits.test),
        },
    )
    builder = SceneGraphBuilder(max_objects=cfg.max_objects)

    for base_path in base_paths:
        base = arrays_by_path[base_path]
        if base.seed not in train_seeds:
            continue
        for variant_id in range(cfg.recovery.variants_per_episode):
            spec = make_recovery_spec(base.seed, variant_id)
            env = KinematicTabletopEnv(
                max_objects=cfg.max_objects,
                max_steps=cfg.environment.max_steps,
                min_object_distance=cfg.environment.min_object_distance,
                workspace_low=cfg.environment.workspace_low,
                workspace_high=cfg.environment.workspace_high,
                crowded_anchor_min_distance=cfg.environment.crowded_anchor_min_distance,
                crowded_anchor_max_distance=cfg.environment.crowded_anchor_max_distance,
            )
            try:
                episode = collect_recovery_episode(
                    env,
                    ScriptedExpert(),
                    source_seed=base.seed,
                    object_count=base.object_count,
                    spec=spec,
                    builder=builder,
                )
            except RuntimeError as error:
                rejections.append(
                    {
                        "source_seed": base.seed,
                        "object_count": base.object_count,
                        "variant_id": variant_id,
                        "perturbation_kind": spec.kind.value,
                        "error": str(error),
                    }
                )
                _write_json_atomic(rejection_log, rejections)
                continue

            filename = f"recovery_seed_{base.seed:010d}_variant_{variant_id:03d}.npz"
            path = save_episode(episode, output_dir / filename)
            records.append(
                {
                    "source_seed": base.seed,
                    "object_count": base.object_count,
                    "variant_id": variant_id,
                    "perturbation_kind": spec.kind.value,
                    "injection_phase": spec.injection_phase.value,
                    "frames": len(episode.steps),
                    "reason": episode.reason.value,
                    "path": path.name,
                }
            )
            _write_json_atomic(manifest, records)

    if not records:
        raise RuntimeError(
            f"all recovery generation attempts failed; see {rejection_log}"
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect interaction-graph expert episodes")
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--config", default="configs/pilot_macos.yaml")
    recovery_parser = subparsers.add_parser("augment-recovery")
    recovery_parser.add_argument("--config", default="configs/recovery_macos.yaml")
    args = parser.parse_args()
    if args.command == "collect":
        print(collect_from_config(args.config))
    elif args.command == "augment-recovery":
        print(augment_recovery_from_config(args.config))


if __name__ == "__main__":
    main()
