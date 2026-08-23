from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from tqdm.auto import tqdm

from .alignment import EpisodeDescriptor
from .config import LiberoStudyConfig


@dataclass(frozen=True)
class RawTaskSource:
    suite: str
    task_id: int
    task_name: str
    language: str
    bddl_path: Path
    demonstration_path: Path


@dataclass(frozen=True)
class LeRobotEpisodeSource:
    descriptor: EpisodeDescriptor
    episode_index: int
    dataset_from_index: int
    dataset_to_index: int
    language: str


@dataclass(frozen=True)
class SourceCatalog:
    tasks: tuple[RawTaskSource, ...]
    raw_episodes: tuple[EpisodeDescriptor, ...]
    lerobot_episodes: tuple[LeRobotEpisodeSource, ...]
    dataset_revision: str
    dataset_fps: float


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_optional_sources() -> tuple[object, object, object]:
    try:
        import h5py
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from libero.libero import benchmark, get_libero_path
    except ImportError as error:  # pragma: no cover - Linux optional dependency
        raise RuntimeError(
            "LIBERO source discovery requires Linux and "
            "lerobot[dataset,training,smolvla,libero] plus h5py"
        ) from error
    return h5py, LeRobotDataset, (benchmark, get_libero_path)


def discover_raw_tasks(config: LiberoStudyConfig) -> tuple[RawTaskSource, ...]:
    _, _, libero = _require_optional_sources()
    benchmark, get_libero_path = libero
    tasks: list[RawTaskSource] = []
    for suite_name in config.coverage.suites:
        suite = benchmark.get_benchmark_dict()[suite_name]()
        task_count = suite.get_num_tasks()
        if config.coverage.tasks_per_suite is not None:
            task_count = min(task_count, config.coverage.tasks_per_suite)
        for task_id in range(task_count):
            task = suite.get_task(task_id)
            relative_demo = Path(suite.get_task_demonstration(task_id))
            demo = config.sources.raw_hdf5_root / relative_demo
            bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            if not demo.is_file():
                raise FileNotFoundError(
                    f"original privileged LIBERO demo is missing: {demo}"
                )
            if not bddl.is_file():
                raise FileNotFoundError(f"LIBERO BDDL file is missing: {bddl}")
            tasks.append(
                RawTaskSource(
                    suite=suite_name,
                    task_id=task_id,
                    task_name=str(task.name),
                    language=str(task.language),
                    bddl_path=bddl,
                    demonstration_path=demo,
                )
            )
    return tuple(tasks)


def load_raw_episode_descriptors(
    tasks: Sequence[RawTaskSource],
    *,
    raw_hdf5_root: Path,
) -> tuple[EpisodeDescriptor, ...]:
    h5py, _, _ = _require_optional_sources()
    result: list[EpisodeDescriptor] = []
    for task in tqdm(tasks, desc="LIBERO raw HDF5", unit="task"):
        with h5py.File(task.demonstration_path, "r") as handle:
            if "data" not in handle:
                raise ValueError(f"raw LIBERO file has no data group: {task.demonstration_path}")
            for demo_key in sorted(handle["data"], key=lambda value: int(value.split("_")[-1])):
                group = handle["data"][demo_key]
                if not {"states", "actions"}.issubset(group):
                    raise ValueError(f"raw demo lacks states/actions: {demo_key}")
                model_xml = group.attrs.get("model_file")
                if isinstance(model_xml, bytes):
                    model_xml = model_xml.decode("utf-8")
                if not model_xml:
                    raise ValueError(f"raw demo lacks model_file: {demo_key}")
                actions = np.asarray(group["actions"], dtype=np.float64)
                result.append(
                    EpisodeDescriptor(
                        source_kind="raw",
                        suite=task.suite,
                        task_id=task.task_id,
                        episode_id=str(demo_key),
                        actions=actions,
                        relative_path=str(
                            task.demonstration_path.relative_to(raw_hdf5_root)
                        ),
                        demo_key=str(demo_key),
                        model_xml_sha256=sha256_bytes(str(model_xml).encode("utf-8")),
                    )
                )
    return tuple(result)


def _normalize_language(value: str) -> str:
    return " ".join(value.lower().replace("_", " ").split())


def _resolve_dataset_revision(repo_id: str, requested: str) -> str:
    if len(requested) == 40 and all(char in "0123456789abcdef" for char in requested):
        return requested
    try:
        from huggingface_hub import HfApi

        revision = str(HfApi().dataset_info(repo_id, revision=requested).sha)
    except Exception as error:  # pragma: no cover - network dependent
        raise RuntimeError(
            "could not resolve the mutable LeRobot dataset revision; use a 40-character Hub commit"
        ) from error
    if len(revision) != 40:
        raise ValueError("Hugging Face did not return an immutable dataset commit")
    return revision


def load_lerobot_episode_sources(
    config: LiberoStudyConfig, tasks: Sequence[RawTaskSource]
) -> tuple[tuple[LeRobotEpisodeSource, ...], str, float]:
    _, LeRobotDataset, _ = _require_optional_sources()
    revision = _resolve_dataset_revision(
        config.sources.lerobot_repo_id, config.sources.lerobot_revision
    )
    dataset = LeRobotDataset(
        config.sources.lerobot_repo_id,
        root=config.sources.lerobot_root,
        revision=revision,
        download_videos=False,
    )
    required_features = {
        "observation.images.image",
        "observation.images.image2",
        "observation.state",
        "action",
        "episode_index",
        "task_index",
    }
    missing_features = sorted(required_features.difference(dataset.meta.features))
    if missing_features:
        raise ValueError(
            f"LeRobot LIBERO schema is missing required policy fields: {missing_features}"
        )
    language_to_task: dict[str, RawTaskSource] = {}
    for task in tasks:
        key = _normalize_language(task.language)
        if key in language_to_task:
            raise ValueError(f"task language is ambiguous across configured suites: {task.language}")
        language_to_task[key] = task
    columns = dataset.select_columns(["action", "observation.state", "episode_index", "task_index"])
    result: list[LeRobotEpisodeSource] = []
    for episode_index in tqdm(
        range(dataset.meta.total_episodes), desc="LeRobot episode catalog", unit="episode"
    ):
        metadata = dataset.meta.episodes[episode_index]
        start = int(metadata["dataset_from_index"])
        end = int(metadata["dataset_to_index"])
        task_names = metadata.get("tasks")
        if task_names:
            language = str(task_names[0])
        else:
            first = columns[start]
            task_index = int(first["task_index"])
            language = str(dataset.meta.tasks.index[task_index])
        task = language_to_task.get(_normalize_language(language))
        if task is None:
            continue
        selected = columns.select(range(start, end))
        actions = np.asarray(selected["action"], dtype=np.float64)
        robot_states = np.asarray(selected["observation.state"], dtype=np.float64)
        descriptor = EpisodeDescriptor(
            source_kind="lerobot",
            suite=task.suite,
            task_id=task.task_id,
            episode_id=str(episode_index),
            actions=actions,
            robot_states=robot_states,
        )
        result.append(
            LeRobotEpisodeSource(
                descriptor=descriptor,
                episode_index=episode_index,
                dataset_from_index=start,
                dataset_to_index=end,
                language=language,
            )
        )
    if not result:
        raise ValueError("no configured LIBERO tasks were found in the LeRobotDataset")
    fps = float(dataset.meta.fps)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("LeRobotDataset FPS must be finite and positive")
    return tuple(result), revision, fps


def load_source_catalog(config: LiberoStudyConfig) -> SourceCatalog:
    tasks = discover_raw_tasks(config)
    raw = load_raw_episode_descriptors(
        tasks, raw_hdf5_root=config.sources.raw_hdf5_root
    )
    lerobot, revision, fps = load_lerobot_episode_sources(config, tasks)
    return SourceCatalog(tasks, raw, lerobot, revision, fps)


def source_catalog_binding(catalog: SourceCatalog) -> str:
    payload = {
        "dataset_revision": catalog.dataset_revision,
        "dataset_fps": catalog.dataset_fps,
        "tasks": [
            {
                "suite": task.suite,
                "task_id": task.task_id,
                "task_name": task.task_name,
                "language": task.language,
                "bddl_sha256": sha256_file(task.bddl_path),
                "demo_sha256": sha256_file(task.demonstration_path),
            }
            for task in catalog.tasks
        ],
    }
    return sha256_bytes(json.dumps(payload, sort_keys=True).encode("utf-8"))
