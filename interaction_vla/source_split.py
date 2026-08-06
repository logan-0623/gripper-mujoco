from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class SourceSplit:
    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]

    def __post_init__(self) -> None:
        groups = tuple(
            set(group) for group in (self.train, self.validation, self.test)
        )
        if any(len(group) == 0 for group in groups):
            raise ValueError("source split groups must not be empty")
        if any(
            len(group) != len(values)
            for group, values in zip(
                groups,
                (self.train, self.validation, self.test),
                strict=True,
            )
        ):
            raise ValueError("source split groups must contain unique seeds")
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise ValueError("source split groups must be disjoint")

    def payload(self) -> dict[str, list[int]]:
        return {
            "train": list(self.train),
            "validation": list(self.validation),
            "test": list(self.test),
        }


@dataclass(frozen=True)
class LoadedSourceSplit:
    split: SourceSplit
    training_recovery_sources: tuple[int, ...]
    benchmark_sources: tuple[int, ...]
    content_hash: str


@dataclass(frozen=True)
class SourceDataLayout:
    split: SourceSplit
    base_by_split: Mapping[str, tuple[Path, ...]]
    training_recovery_paths: tuple[Path, ...]
    benchmark_recovery_paths: tuple[Path, ...]
    training_recovery_sources: tuple[int, ...]
    benchmark_sources: tuple[int, ...]
    source_split_hash: str
    manifest_hash: str
    training_recovery_manifest_hash: str
    benchmark_recovery_manifest_hash: str


def deterministic_source_split(
    seeds: Iterable[int],
    *,
    seed: int,
) -> SourceSplit:
    values = tuple(sorted(int(value) for value in seeds))
    if len(set(values)) != len(values):
        raise ValueError("source seeds must be unique")
    if len(values) < 10:
        raise ValueError("source split requires at least ten successful sources")
    rng = np.random.default_rng(np.random.SeedSequence((int(seed), 0x53504C54)))
    shuffled = np.asarray(values, dtype=np.int64)[rng.permutation(len(values))]
    validation_count = int(round(len(values) * 0.10))
    test_count = int(round(len(values) * 0.10))
    validation = tuple(sorted(int(value) for value in shuffled[:validation_count]))
    test = tuple(
        sorted(
            int(value)
            for value in shuffled[
                validation_count : validation_count + test_count
            ]
        )
    )
    train = tuple(
        sorted(int(value) for value in shuffled[validation_count + test_count :])
    )
    return SourceSplit(train=train, validation=validation, test=test)


def select_training_recovery_sources(
    train_sources: Iterable[int],
    *,
    fraction: float,
    seed: int,
) -> tuple[int, ...]:
    values = tuple(sorted(int(value) for value in train_sources))
    if len(set(values)) != len(values):
        raise ValueError("training source seeds must be unique")
    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError("training recovery fraction must be within (0, 1]")
    count = int(round(len(values) * fraction))
    if count < 1:
        raise ValueError("training recovery source selection is empty")
    rng = np.random.default_rng(np.random.SeedSequence((int(seed), 0x52454353)))
    selected = np.asarray(values, dtype=np.int64)[rng.permutation(len(values))[:count]]
    return tuple(sorted(int(value) for value in selected))


def validate_derived_sources(
    split: SourceSplit,
    *,
    training_recovery_sources: Iterable[int],
    benchmark_sources: Iterable[int],
) -> None:
    training = tuple(int(value) for value in training_recovery_sources)
    benchmark = tuple(int(value) for value in benchmark_sources)
    if len(set(training)) != len(training):
        raise ValueError("training recovery sources must be unique")
    if len(set(benchmark)) != len(benchmark):
        raise ValueError("benchmark sources must be unique")
    if not set(training).issubset(split.train):
        raise ValueError("training recovery sources must belong to the train split")
    expected_benchmark = set(split.validation) | set(split.test)
    if set(benchmark) != expected_benchmark:
        raise ValueError(
            "benchmark sources must exactly cover validation and test splits"
        )


def source_split_hash(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def save_source_split(
    path: str | Path,
    split: SourceSplit,
    *,
    training_recovery_sources: Iterable[int],
    benchmark_sources: Iterable[int],
) -> Path:
    training = tuple(sorted(int(value) for value in training_recovery_sources))
    benchmark = tuple(sorted(int(value) for value in benchmark_sources))
    validate_derived_sources(
        split,
        training_recovery_sources=training,
        benchmark_sources=benchmark,
    )
    payload: dict[str, object] = {
        **split.payload(),
        "training_recovery_sources": list(training),
        "benchmark_sources": list(benchmark),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def load_source_split(path: str | Path) -> LoadedSourceSplit:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    split = SourceSplit(
        train=tuple(sorted(int(value) for value in payload["train"])),
        validation=tuple(sorted(int(value) for value in payload["validation"])),
        test=tuple(sorted(int(value) for value in payload["test"])),
    )
    training = tuple(
        sorted(int(value) for value in payload["training_recovery_sources"])
    )
    benchmark = tuple(sorted(int(value) for value in payload["benchmark_sources"]))
    validate_derived_sources(
        split,
        training_recovery_sources=training,
        benchmark_sources=benchmark,
    )
    return LoadedSourceSplit(
        split=split,
        training_recovery_sources=training,
        benchmark_sources=benchmark,
        content_hash=source_split_hash(payload),
    )


def _load_json(path: Path) -> object:
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _episode_metadata(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as archive:
        return dict(json.loads(str(archive["metadata"].item())))


def _record_path(data_dir: Path, record: Mapping[str, object]) -> Path:
    filename = str(record["path"])
    if Path(filename).name != filename:
        raise ValueError(f"manifest episode path must be a filename: {filename}")
    path = data_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"manifest episode does not exist: {path}")
    return path


def _load_recovery_records(
    *,
    data_dir: Path,
    records: object,
    allowed_sources: set[int],
    expected_split_by_seed: Mapping[int, str],
    label: str,
) -> tuple[Path, ...]:
    if not isinstance(records, list) or not records:
        raise ValueError(f"{label} manifest must contain accepted episodes")
    paths: list[Path] = []
    for raw_record in records:
        if not isinstance(raw_record, dict):
            raise ValueError(f"{label} manifest records must be mappings")
        record = dict(raw_record)
        path = _record_path(data_dir, record)
        metadata = _episode_metadata(path)
        source_seed = int(record["source_seed"])
        source_split = str(record["source_split"])
        if source_seed not in allowed_sources:
            raise ValueError(f"{label} source seed {source_seed} is outside its split")
        if source_split != expected_split_by_seed[source_seed]:
            raise ValueError(f"{label} source split mismatch for seed {source_seed}")
        if (
            metadata.get("trajectory_kind") != "recovery"
            or int(metadata.get("source_seed", -1)) != source_seed
            or metadata.get("source_split") != source_split
            or int(metadata.get("variant_id", -1)) != int(record["variant_id"])
            or metadata.get("perturbation_kind") != record.get("kind")
        ):
            raise ValueError(f"{label} metadata mismatch: {path}")
        paths.append(path)
    if len(set(paths)) != len(paths):
        raise ValueError(f"{label} manifest contains duplicate paths")
    return tuple(paths)


def load_source_data_layout(data_dir: str | Path) -> SourceDataLayout:
    directory = Path(data_dir)
    loaded_split = load_source_split(directory / "source_split.json")
    split = loaded_split.split
    expected_split_by_seed = {
        source_seed: split_name
        for split_name, source_seeds in (
            ("train", split.train),
            ("validation", split.validation),
            ("test", split.test),
        )
        for source_seed in source_seeds
    }
    manifest_payload = _load_json(directory / "manifest.json")
    if not isinstance(manifest_payload, list) or not manifest_payload:
        raise ValueError("base manifest must contain successful source episodes")
    base_by_split: dict[str, list[Path]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    observed_sources: set[int] = set()
    for raw_record in manifest_payload:
        if not isinstance(raw_record, dict):
            raise ValueError("base manifest records must be mappings")
        record = dict(raw_record)
        path = _record_path(directory, record)
        metadata = _episode_metadata(path)
        source_seed = int(record["source_seed"])
        if source_seed in observed_sources:
            raise ValueError(f"duplicate base source seed: {source_seed}")
        observed_sources.add(source_seed)
        source_split = str(record["source_split"])
        if (
            source_seed not in expected_split_by_seed
            or source_split != expected_split_by_seed[source_seed]
        ):
            raise ValueError(f"source split mismatch for seed {source_seed}")
        if int(metadata["seed"]) != source_seed:
            raise ValueError(f"base source seed metadata mismatch: {path}")
        base_by_split[source_split].append(path)
    if observed_sources != set(expected_split_by_seed):
        raise ValueError("base manifest does not exactly cover the saved source split")

    training_payload = _load_json(directory / "recovery_manifest.json")
    benchmark_payload = _load_json(
        directory / "recovery_benchmark_manifest.json"
    )
    training_paths = _load_recovery_records(
        data_dir=directory,
        records=training_payload,
        allowed_sources=set(loaded_split.training_recovery_sources),
        expected_split_by_seed=expected_split_by_seed,
        label="training recovery",
    )
    benchmark_paths = _load_recovery_records(
        data_dir=directory,
        records=benchmark_payload,
        allowed_sources=set(loaded_split.benchmark_sources),
        expected_split_by_seed=expected_split_by_seed,
        label="benchmark recovery",
    )
    if set(training_paths) & set(benchmark_paths):
        raise ValueError("training and benchmark recovery paths must be disjoint")
    return SourceDataLayout(
        split=split,
        base_by_split={
            name: tuple(paths) for name, paths in base_by_split.items()
        },
        training_recovery_paths=training_paths,
        benchmark_recovery_paths=benchmark_paths,
        training_recovery_sources=loaded_split.training_recovery_sources,
        benchmark_sources=loaded_split.benchmark_sources,
        source_split_hash=loaded_split.content_hash,
        manifest_hash=_canonical_hash(manifest_payload),
        training_recovery_manifest_hash=_canonical_hash(training_payload),
        benchmark_recovery_manifest_hash=_canonical_hash(benchmark_payload),
    )
