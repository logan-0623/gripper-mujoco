from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

import numpy as np
from tqdm.auto import tqdm

from interaction_vla.lerobot_bridge.provenance import sha256_file

from .io import write_json_atomic


STATE_BANK_V2_SCHEMA = "recovery_state_bank_v2"
PRIMARY_STRATA = ("nominal", "perturbation", "recovery")
PRIMARY_FACTORS = ("geometry", "phase", "recovery_state", "next_relation")
SECONDARY_FACTORS = ("contact", "stable_grasp")
V2_PARTITIONS = ("train", "validation", "test")
V2_COUNTS = {"train": 280, "validation": 60, "test": 60}
REQUIRED_LABELS = {
    "geometry",
    "phase",
    "recovery_state",
    "recovery_type",
    "next_relation",
    "contact",
    "stable_grasp",
}


def _vector(value: object, width: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (width,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite vector of width {width}")
    return result.copy()


def _image(value: object, name: str) -> np.ndarray:
    result = np.asarray(value)
    if (
        result.ndim != 3
        or result.shape[-1] != 3
        or result.dtype != np.uint8
        or min(result.shape[:2]) < 1
    ):
        raise ValueError(f"{name} must be a non-empty HWC uint8 RGB image")
    return result.copy()


@dataclass(frozen=True)
class StateBankV2Candidate:
    candidate_id: str
    case_id: str
    source_seed: int
    partition: str
    family: str
    intervention_kind: str
    step: int
    phase: int
    robot_state: np.ndarray
    oracle_state: np.ndarray
    agent_rgb: np.ndarray
    wrist_rgb: np.ndarray
    labels: dict[str, object]

    def __post_init__(self) -> None:
        if not self.candidate_id.strip() or not self.case_id.strip():
            raise ValueError("State Bank v2 candidate ids must be non-empty")
        if self.source_seed < 0 or self.step < 0:
            raise ValueError("candidate source seed and step must be non-negative")
        if self.partition not in V2_PARTITIONS:
            raise ValueError(f"unknown State Bank v2 partition: {self.partition}")
        if self.family not in PRIMARY_STRATA:
            raise ValueError(f"unknown State Bank v2 family: {self.family}")
        if not 0 <= self.phase < 6:
            raise ValueError("candidate phase must lie within [0, 5]")
        if set(self.labels) != REQUIRED_LABELS:
            raise ValueError("candidate labels do not match the v2 ontology")
        geometry = np.asarray(self.labels["geometry"], dtype=np.float32)
        if geometry.shape != (16,) or not np.isfinite(geometry).all():
            raise ValueError("candidate geometry label must be finite 16D")
        object.__setattr__(self, "robot_state", _vector(self.robot_state, 10, "robot_state"))
        object.__setattr__(self, "oracle_state", _vector(self.oracle_state, 36, "oracle_state"))
        agent = _image(self.agent_rgb, "agent_rgb")
        wrist = _image(self.wrist_rgb, "wrist_rgb")
        if agent.shape != wrist.shape:
            raise ValueError("candidate RGB views must share shape")
        object.__setattr__(self, "agent_rgb", agent)
        object.__setattr__(self, "wrist_rgb", wrist)


@dataclass(frozen=True)
class StateBankV2Split:
    partitions: dict[str, tuple[str, ...]]
    source_seed_by_id: dict[str, int]
    family_by_id: dict[str, str]

    def __post_init__(self) -> None:
        if set(self.partitions) != set(V2_PARTITIONS):
            raise ValueError("State Bank v2 split partitions are incompatible")
        ids = tuple(
            state_id
            for partition in V2_PARTITIONS
            for state_id in self.partitions[partition]
        )
        if len(ids) != len(set(ids)) or not ids:
            raise ValueError("State Bank v2 split ids must be non-empty and disjoint")
        if set(ids) != set(self.source_seed_by_id) or set(ids) != set(self.family_by_id):
            raise ValueError("State Bank v2 split metadata must align with ids")
        sources = [self.source_seeds(partition) for partition in V2_PARTITIONS]
        if any(
            left & right
            for index, left in enumerate(sources)
            for right in sources[index + 1 :]
        ):
            raise ValueError("State Bank v2 source seed crosses partitions")

    def source_seeds(
        self,
        partition: str,
        *,
        family: str | None = None,
    ) -> set[int]:
        if partition not in V2_PARTITIONS:
            raise ValueError(f"unknown State Bank v2 partition: {partition}")
        return {
            self.source_seed_by_id[state_id]
            for state_id in self.partitions[partition]
            if family is None or self.family_by_id[state_id] == family
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "recovery_state_bank_split_v2",
            "partitions": {
                name: list(self.partitions[name]) for name in V2_PARTITIONS
            },
            "source_seed_by_id": dict(sorted(self.source_seed_by_id.items())),
            "family_by_id": dict(sorted(self.family_by_id.items())),
        }


@dataclass(frozen=True)
class StateBankV2Report:
    passed: bool
    record_count: int
    stratum_counts: dict[str, int]
    partition_counts: dict[str, int]
    manifest_sha256: str
    output_dir: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_v2_split(path: str | Path) -> StateBankV2Split:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("schema_version") != "recovery_state_bank_split_v2":
        raise ValueError("State Bank v2 split schema is incompatible")
    partitions = raw.get("partitions")
    sources = raw.get("source_seed_by_id")
    families = raw.get("family_by_id")
    if not all(isinstance(value, Mapping) for value in (partitions, sources, families)):
        raise ValueError("State Bank v2 split structure is incompatible")
    return StateBankV2Split(
        partitions={
            name: tuple(str(value) for value in partitions[name])
            for name in V2_PARTITIONS
        },
        source_seed_by_id={str(key): int(value) for key, value in sources.items()},
        family_by_id={str(key): str(value) for key, value in families.items()},
    )


def _round_robin_select(
    candidates: Sequence[StateBankV2Candidate],
    *,
    count: int,
    rng: np.random.Generator,
) -> tuple[StateBankV2Candidate, ...]:
    by_source: dict[int, list[StateBankV2Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_source[candidate.source_seed].append(candidate)
    if len(by_source) < 3:
        raise ValueError("State Bank v2 stratum needs at least three source groups")
    sources = np.asarray(sorted(by_source), dtype=np.int64)
    rng.shuffle(sources)
    for values in by_source.values():
        values.sort(key=lambda item: (item.step, item.candidate_id))
        rng.shuffle(values)
    selected: list[StateBankV2Candidate] = []
    cursor = {int(source): 0 for source in sources}
    while len(selected) < count:
        changed = False
        for source_value in sources:
            source = int(source_value)
            index = cursor[source]
            values = by_source[source]
            if index < len(values):
                selected.append(values[index])
                cursor[source] += 1
                changed = True
                if len(selected) == count:
                    break
        if not changed:
            raise ValueError(
                f"State Bank v2 has only {len(selected)} candidates for required {count}"
            )
    return tuple(selected)


def _records_bytes(
    selected: Sequence[StateBankV2Candidate],
    *,
    observation_uri: str,
) -> bytes:
    rows = []
    for index, candidate in enumerate(selected):
        rows.append(
            {
                "schema_version": STATE_BANK_V2_SCHEMA,
                "state_id": candidate.candidate_id,
                "case_id": candidate.case_id,
                "source_seed": candidate.source_seed,
                "partition": candidate.partition,
                "family": candidate.family,
                "intervention_kind": candidate.intervention_kind,
                "step": candidate.step,
                "phase": candidate.phase,
                "robot_state": candidate.robot_state.tolist(),
                "oracle_state": candidate.oracle_state.tolist(),
                "labels": candidate.labels,
                "observation": {
                    "uri": observation_uri,
                    "index": index,
                    "agent_rgb_key": "agent_rgb",
                    "wrist_rgb_key": "wrist_rgb",
                },
            }
        )
    return b"".join(
        (
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        for row in rows
    )


def _inspect_existing(output_dir: Path) -> StateBankV2Report:
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != STATE_BANK_V2_SCHEMA:
        raise ValueError("State Bank v2 manifest schema is incompatible")
    for name in ("records.jsonl", "observations.npz", "split.json", "ontology.json"):
        if manifest["artifact_hashes"].get(name) != sha256_file(output_dir / name):
            raise ValueError(f"State Bank v2 artifact hash differs: {name}")
    return StateBankV2Report(**report)


def build_state_bank_v2(
    candidates: Sequence[StateBankV2Candidate],
    *,
    output_dir: str | Path,
    manifest_hash: str,
    seed: int,
) -> StateBankV2Report:
    destination = Path(output_dir)
    if destination.exists():
        return _inspect_existing(destination)
    if len(manifest_hash) != 64 or seed < 0:
        raise ValueError("State Bank v2 requires a manifest SHA-256 and non-negative seed")
    if not candidates or len({value.candidate_id for value in candidates}) != len(candidates):
        raise ValueError("State Bank v2 candidates must be non-empty and uniquely identified")
    partition_by_source: dict[int, str] = {}
    for candidate in candidates:
        observed = partition_by_source.setdefault(candidate.source_seed, candidate.partition)
        if observed != candidate.partition:
            raise ValueError("State Bank v2 candidate source crosses partitions")
    rng = np.random.default_rng(seed)
    selected: list[StateBankV2Candidate] = []
    for partition in V2_PARTITIONS:
        for family in PRIMARY_STRATA:
            pool = tuple(
                candidate
                for candidate in candidates
                if candidate.partition == partition and candidate.family == family
            )
            selected.extend(
                _round_robin_select(pool, count=V2_COUNTS[partition], rng=rng)
            )
    selected.sort(
        key=lambda item: (
            V2_PARTITIONS.index(item.partition),
            PRIMARY_STRATA.index(item.family),
            item.candidate_id,
        )
    )
    split = StateBankV2Split(
        partitions={
            partition: tuple(
                candidate.candidate_id
                for candidate in selected
                if candidate.partition == partition
            )
            for partition in V2_PARTITIONS
        },
        source_seed_by_id={
            candidate.candidate_id: candidate.source_seed for candidate in selected
        },
        family_by_id={
            candidate.candidate_id: candidate.family for candidate in selected
        },
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
    )
    try:
        observation_uri = (destination / "observations.npz").as_posix()
        with (staging / "observations.npz").open("wb") as handle:
            np.savez_compressed(
                handle,
                state_id=np.asarray(
                    [candidate.candidate_id for candidate in selected], dtype=np.str_
                ),
                agent_rgb=np.stack([candidate.agent_rgb for candidate in selected]),
                wrist_rgb=np.stack([candidate.wrist_rgb for candidate in selected]),
            )
        records = _records_bytes(selected, observation_uri=observation_uri)
        (staging / "records.jsonl").write_bytes(records)
        write_json_atomic(staging / "split.json", split.to_dict())
        write_json_atomic(
            staging / "ontology.json",
            {
                "schema_version": "recovery_measurement_ontology_v2",
                "role": "measurement_language",
                "policy_input": False,
                "primary_factors": list(PRIMARY_FACTORS),
                "secondary_factors": list(SECONDARY_FACTORS),
                "descriptive_factors": ["entity"],
                "geometry_width": 16,
            },
        )
        artifact_hashes = {
            name: sha256_file(staging / name)
            for name in ("records.jsonl", "observations.npz", "split.json", "ontology.json")
        }
        manifest = {
            "schema_version": STATE_BANK_V2_SCHEMA,
            "source_case_manifest_sha256": manifest_hash,
            "selection_seed": seed,
            "record_count": len(selected),
            "stratum_counts": dict(Counter(value.family for value in selected)),
            "partition_counts": dict(Counter(value.partition for value in selected)),
            "artifact_hashes": artifact_hashes,
        }
        write_json_atomic(staging / "manifest.json", manifest)
        report = StateBankV2Report(
            passed=True,
            record_count=len(selected),
            stratum_counts={
                family: sum(value.family == family for value in selected)
                for family in PRIMARY_STRATA
            },
            partition_counts={
                partition: sum(value.partition == partition for value in selected)
                for partition in V2_PARTITIONS
            },
            manifest_sha256=sha256_file(staging / "manifest.json"),
            output_dir=destination.as_posix(),
        )
        write_json_atomic(staging / "report.json", report.to_dict())
        os.replace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return report


def _phase_from_oracle(oracle_state: np.ndarray) -> int:
    aperture = float(oracle_state[16])
    contact = bool(oracle_state[17] >= 0.5)
    stable_grasp = bool(oracle_state[18] >= 0.5)
    supported = bool(oracle_state[19] >= 0.5)
    goal_distance = float(oracle_state[13])
    if supported and goal_distance <= 0.20 and aperture >= 0.5 and not stable_grasp:
        return 5
    if stable_grasp and goal_distance <= 0.20:
        return 4
    if stable_grasp and supported:
        return 2
    if stable_grasp:
        return 3
    if contact:
        return 1
    return 0


def _labels_from_oracle(
    oracle_state: np.ndarray,
    *,
    family: str,
    intervention_kind: str,
) -> dict[str, object]:
    from interaction_vla.representation_study.rl.oracle_state import INTERVENTIONS

    phase = _phase_from_oracle(oracle_state)
    return {
        "geometry": oracle_state[:16].astype(np.float32).tolist(),
        "phase": phase,
        "recovery_state": PRIMARY_STRATA.index(family),
        "recovery_type": INTERVENTIONS.index(intervention_kind),
        "next_relation": min(phase + 1, 5),
        "contact": int(oracle_state[17] >= 0.5),
        "stable_grasp": int(oracle_state[18] >= 0.5),
    }


def _case_by_family(
    cases: Sequence[object],
    *,
    family: str,
    offset: int,
) -> object:
    selected = tuple(
        sorted(
            (case for case in cases if getattr(case, "family") == family),
            key=lambda case: (getattr(case, "variant_id"), getattr(case, "case_id")),
        )
    )
    if not selected:
        raise ValueError(f"source group has no {family} case")
    return selected[offset % len(selected)]


def collect_state_bank_v2(config: object) -> dict[str, object]:
    from interaction_vla.physics_data import PhysicsRecoveryRejected
    from interaction_vla.representation_study.rl.distributions import load_case_manifest
    from interaction_vla.representation_study.rl.foundation import (
        _rgb_state,
        _runtime,
        foundation_binding,
    )
    from interaction_vla.representation_study.rl.oracle_state import INTERVENTIONS
    from interaction_vla.representation_study.rl.protocol import require_passing_gate

    output_dir = Path(getattr(config, "output_dir")) / "state_bank_v2"
    if output_dir.exists():
        return _inspect_existing(output_dir).to_dict()
    binding = foundation_binding(config)
    gates = Path(getattr(config, "output_dir")) / "gates"
    for name in ("distribution", "backend", "oracle", "anchoring"):
        require_passing_gate(
            gates / f"{name}.json",
            expected_gate=name,
            expected_binding=binding,
        )
    manifest = load_case_manifest(
        Path(getattr(config, "output_dir")) / "manifests" / "cases.json"
    )
    partition_map = {
        "training": "train",
        "curve": "validation",
        "final": "test",
    }
    cases_by_partition_source: dict[tuple[str, int], tuple[object, ...]] = {}
    for source_partition, output_partition in partition_map.items():
        for source_seed in manifest.source_seeds(source_partition):
            cases_by_partition_source[(output_partition, source_seed)] = tuple(
                case
                for case in manifest.partition(source_partition)
                if case.source_seed == source_seed
            )
    candidates: list[StateBankV2Candidate] = []
    rejected: list[dict[str, object]] = []
    progress = tqdm(
        total=sum(V2_COUNTS.values()) * len(PRIMARY_STRATA),
        desc="State Bank v2",
        unit="state",
        dynamic_ncols=True,
    )
    with _runtime(config, seed=int(getattr(config, "seed")) + 70_000) as runtime:
        max_steps = int(runtime.max_steps)
        for partition in V2_PARTITIONS:
            sources = sorted(
                source
                for (candidate_partition, source) in cases_by_partition_source
                if candidate_partition == partition
            )
            for family in PRIMARY_STRATA:
                required = V2_COUNTS[partition]
                group_target = 20 if partition == "train" else 6
                per_case = max(1, int(np.ceil(required / group_target)))
                stride = max(1, max_steps // per_case)
                family_rows: list[StateBankV2Candidate] = []
                for source_index, source_seed in enumerate(sources):
                    case = _case_by_family(
                        cases_by_partition_source[(partition, source_seed)],
                        family=family,
                        offset=source_index,
                    )
                    try:
                        runtime.reset_case(case)
                    except PhysicsRecoveryRejected as error:
                        rejected.append(
                            {
                                "case_id": case.case_id,
                                "source_seed": source_seed,
                                "family": family,
                                "reason": str(error),
                            }
                        )
                        continue
                    captured = 0
                    step = 0
                    while captured < per_case:
                        observation = runtime.current_observation
                        oracle = runtime.current_oracle_state
                        if observation is None or oracle is None:
                            break
                        if step % stride == 0:
                            agent, wrist, robot_state = _rgb_state(observation)
                            phase = _phase_from_oracle(oracle)
                            identity = hashlib.sha256(
                                f"{case.case_id}:{step}".encode("utf-8")
                            ).hexdigest()[:16]
                            family_rows.append(
                                StateBankV2Candidate(
                                    candidate_id=f"v2-{identity}-{step:04d}",
                                    case_id=case.case_id,
                                    source_seed=source_seed,
                                    partition=partition,
                                    family=family,
                                    intervention_kind=case.intervention_kind,
                                    step=step,
                                    phase=phase,
                                    robot_state=robot_state,
                                    oracle_state=oracle,
                                    agent_rgb=agent,
                                    wrist_rgb=wrist,
                                    labels=_labels_from_oracle(
                                        oracle,
                                        family=family,
                                        intervention_kind=case.intervention_kind,
                                    ),
                                )
                            )
                            captured += 1
                            progress.update(1)
                        base_action, latent = runtime.policy_features()
                        transition = runtime.step(
                            base_action=base_action,
                            latent=latent,
                            residual=np.zeros(7, dtype=np.float32),
                        )
                        step += 1
                        if transition.done:
                            break
                    if len(family_rows) >= required:
                        break
                if len(family_rows) < required:
                    raise ValueError(
                        f"State Bank v2 collected {len(family_rows)} {partition}/{family} "
                        f"states but requires {required}"
                    )
                candidates.extend(family_rows[:required])
    progress.close()
    report = build_state_bank_v2(
        candidates,
        output_dir=output_dir,
        manifest_hash=manifest.sha256,
        seed=int(getattr(config, "seed")) + 70_000,
    )
    write_json_atomic(
        output_dir / "collection_report.json",
        {
            "schema_version": "recovery_state_bank_collection_v2",
            "passed": True,
            "binding": binding,
            "source_case_manifest_sha256": manifest.sha256,
            "rejected_cases": rejected,
            "registered_interventions": list(INTERVENTIONS),
        },
    )
    return report.to_dict()
