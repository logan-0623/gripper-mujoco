from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Literal, Mapping

import numpy as np


CASE_MANIFEST_SCHEMA = "recovery_rl_case_manifest_v2"
DISTRIBUTION_VERSION = "recovery_rl_distribution_v2"
PARTITION_NAMES = ("calibration", "training", "curve", "final")
FAMILY_NAMES = ("recovery", "perturbation", "nominal")
PERTURBATIONS = (
    ("approach", "approach_offset"),
    ("grasp", "grasp_offset"),
    ("lift", "lift_offset"),
)
RECOVERIES = (
    ("transport", "wrong_way_transport"),
    ("transport", "premature_open"),
    ("transport", "receptacle_misalignment"),
)


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class RecoveryCase:
    case_id: str
    partition: str
    family: Literal["recovery", "perturbation", "nominal"]
    source_seed: int
    variant_id: int
    object_count: int
    layout: str
    phase: str
    intervention_kind: str
    severity: float

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("case_id must be non-empty")
        if self.partition not in PARTITION_NAMES:
            raise ValueError(f"unknown recovery partition: {self.partition}")
        if self.family not in FAMILY_NAMES:
            raise ValueError(f"unknown recovery family: {self.family}")
        if self.source_seed < 0 or self.variant_id < 0:
            raise ValueError("source_seed and variant_id must be non-negative")
        if self.object_count not in {2, 3}:
            raise ValueError("Recovery RL v2 object_count must be two or three")
        if self.layout not in {"normal", "crowded"}:
            raise ValueError("Recovery RL v2 layout must be normal or crowded")
        if not np.isfinite(self.severity) or not 0.0 <= self.severity <= 1.0:
            raise ValueError("case severity must be finite and within [0, 1]")
        if self.family == "nominal":
            if self.intervention_kind != "nominal" or self.severity != 0.0:
                raise ValueError("nominal cases must have kind=nominal and severity=0")
        elif self.family == "perturbation":
            expected = dict(PERTURBATIONS).get(self.phase)
            if expected != self.intervention_kind or self.severity <= 0.0:
                raise ValueError("perturbation case phase/kind/severity is incompatible")
        else:
            kinds = {kind for _, kind in RECOVERIES}
            if self.phase != "transport" or self.intervention_kind not in kinds or self.severity <= 0.0:
                raise ValueError("recovery case phase/kind/severity is incompatible")


@dataclass(frozen=True)
class RecoveryCaseManifest:
    schema_version: str
    distribution_version: str
    cases: tuple[RecoveryCase, ...]
    source_hash: str

    def __post_init__(self) -> None:
        if self.schema_version != CASE_MANIFEST_SCHEMA:
            raise ValueError("recovery case manifest schema is incompatible")
        if self.distribution_version != DISTRIBUTION_VERSION:
            raise ValueError("recovery distribution version is incompatible")
        if not self.cases:
            raise ValueError("recovery case manifest must not be empty")
        if len({case.case_id for case in self.cases}) != len(self.cases):
            raise ValueError("recovery case ids must be unique")
        if tuple(sorted(self.cases, key=lambda item: item.case_id)) != self.cases:
            raise ValueError("recovery cases must be sorted by case id")
        if len(self.source_hash) != 64:
            raise ValueError("source_hash must be a SHA-256 digest")
        partition_by_seed: dict[int, str] = {}
        for case in self.cases:
            observed = partition_by_seed.setdefault(case.source_seed, case.partition)
            if observed != case.partition:
                raise ValueError("source seed crosses recovery partitions")

    @property
    def partition_names(self) -> tuple[str, ...]:
        return PARTITION_NAMES

    def partition(self, name: str) -> tuple[RecoveryCase, ...]:
        if name not in PARTITION_NAMES:
            raise ValueError(f"unknown recovery partition: {name}")
        return tuple(case for case in self.cases if case.partition == name)

    def source_seeds(self, name: str) -> tuple[int, ...]:
        return tuple(sorted({case.source_seed for case in self.partition(name)}))

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "distribution_version": self.distribution_version,
            "source_hash": self.source_hash,
            "cases": [asdict(case) for case in self.cases],
        }

    @property
    def sha256(self) -> str:
        return _canonical_hash(self.payload())


def _source_seeds(count: int, *, seed: int) -> tuple[int, ...]:
    if count < 1:
        raise ValueError("source seed count must be positive")
    rng = np.random.default_rng(np.random.SeedSequence((seed, 0x52525632)))
    values: list[int] = []
    observed: set[int] = set()
    while len(values) < count:
        candidate = int(rng.integers(0, 2**32, dtype=np.uint64))
        if candidate not in observed:
            observed.add(candidate)
            values.append(candidate)
    return tuple(values)


def _source_scene(source_seed: int) -> tuple[int, str]:
    rng = np.random.default_rng(
        np.random.SeedSequence((source_seed, 0x53434E32))
    )
    object_count = 2 + int(rng.integers(0, 2))
    layout = ("normal", "crowded")[int(rng.integers(0, 2))]
    return object_count, layout


def _source_cases(
    *,
    partition: str,
    source_seed: int,
    severity: float,
) -> tuple[RecoveryCase, ...]:
    object_count, layout = _source_scene(source_seed)
    records = [
        RecoveryCase(
            case_id=f"{partition}:{source_seed}:nominal:0:nominal",
            partition=partition,
            family="nominal",
            source_seed=source_seed,
            variant_id=0,
            object_count=object_count,
            layout=layout,
            phase="approach",
            intervention_kind="nominal",
            severity=0.0,
        )
    ]
    for variant_id, (phase, kind) in enumerate(PERTURBATIONS):
        records.append(
            RecoveryCase(
                case_id=f"{partition}:{source_seed}:perturbation:{variant_id}:{kind}",
                partition=partition,
                family="perturbation",
                source_seed=source_seed,
                variant_id=variant_id,
                object_count=object_count,
                layout=layout,
                phase=phase,
                intervention_kind=kind,
                severity=severity,
            )
        )
    for variant_id, (phase, kind) in enumerate(RECOVERIES):
        records.append(
            RecoveryCase(
                case_id=f"{partition}:{source_seed}:recovery:{variant_id}:{kind}",
                partition=partition,
                family="recovery",
                source_seed=source_seed,
                variant_id=variant_id,
                object_count=object_count,
                layout=layout,
                phase=phase,
                intervention_kind=kind,
                severity=severity,
            )
        )
    return tuple(records)


def build_case_manifest(
    *,
    seed: int,
    calibration: int,
    training: int,
    curve: int,
    final: int,
    severity: float = 1.0,
) -> RecoveryCaseManifest:
    counts = {
        "calibration": int(calibration),
        "training": int(training),
        "curve": int(curve),
        "final": int(final),
    }
    if seed < 0 or min(counts.values()) < 1:
        raise ValueError("manifest seed must be non-negative and counts positive")
    if not np.isfinite(severity) or not 0.0 < severity <= 1.0:
        raise ValueError("manifest severity must be finite and within (0, 1]")
    all_sources = _source_seeds(sum(counts.values()), seed=seed)
    offset = 0
    partition_sources: dict[str, tuple[int, ...]] = {}
    for name in PARTITION_NAMES:
        count = counts[name]
        partition_sources[name] = all_sources[offset : offset + count]
        offset += count
    source_payload = {
        "seed": int(seed),
        "severity": float(severity),
        "partition_sources": {
            name: list(partition_sources[name]) for name in PARTITION_NAMES
        },
    }
    cases = tuple(
        sorted(
            (
                case
                for partition in PARTITION_NAMES
                for source_seed in partition_sources[partition]
                for case in _source_cases(
                    partition=partition,
                    source_seed=source_seed,
                    severity=float(severity),
                )
            ),
            key=lambda item: item.case_id,
        )
    )
    return RecoveryCaseManifest(
        schema_version=CASE_MANIFEST_SCHEMA,
        distribution_version=DISTRIBUTION_VERSION,
        cases=cases,
        source_hash=_canonical_hash(source_payload),
    )


def save_case_manifest(
    path: str | Path,
    manifest: RecoveryCaseManifest,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        manifest.payload(), indent=2, sort_keys=True
    ) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"case manifest is immutable: {destination}")
        return destination
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(destination)
    return destination


def load_case_manifest(path: str | Path) -> RecoveryCaseManifest:
    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("case manifest must be a mapping")
    expected = {"schema_version", "distribution_version", "source_hash", "cases"}
    unknown = set(raw) - expected
    if unknown:
        raise ValueError("unknown case manifest fields: " + ", ".join(sorted(unknown)))
    records = raw.get("cases")
    if not isinstance(records, list):
        raise ValueError("case manifest cases must be a list")
    cases = tuple(RecoveryCase(**dict(record)) for record in records)
    return RecoveryCaseManifest(
        schema_version=str(raw.get("schema_version", "")),
        distribution_version=str(raw.get("distribution_version", "")),
        source_hash=str(raw.get("source_hash", "")),
        cases=cases,
    )


class RecoveryCaseSampler:
    def __init__(
        self,
        manifest: RecoveryCaseManifest,
        *,
        probabilities: tuple[float, float, float],
        seed: int,
    ) -> None:
        values = np.asarray(probabilities, dtype=np.float64)
        if (
            values.shape != (3,)
            or not np.isfinite(values).all()
            or np.any(values < 0.0)
            or not np.isclose(values.sum(), 1.0)
        ):
            raise ValueError("sampler probabilities must be finite, non-negative, and sum to one")
        if seed < 0:
            raise ValueError("sampler seed must be non-negative")
        training = manifest.partition("training")
        self._by_family = {
            family: tuple(case for case in training if case.family == family)
            for family in FAMILY_NAMES
        }
        if any(not cases for cases in self._by_family.values()):
            raise ValueError("training partition must contain every recovery family")
        self.manifest_hash = manifest.sha256
        self.probabilities = tuple(float(value) for value in values)
        self.rng = np.random.default_rng(seed)

    def next_case(self) -> RecoveryCase:
        family_index = int(self.rng.choice(len(FAMILY_NAMES), p=self.probabilities))
        family = FAMILY_NAMES[family_index]
        cases = self._by_family[family]
        return cases[int(self.rng.integers(0, len(cases)))]

    def state_dict(self) -> dict[str, object]:
        return {
            "schema_version": "recovery_case_sampler_v2",
            "manifest_hash": self.manifest_hash,
            "probabilities": list(self.probabilities),
            "rng_state": dict(self.rng.bit_generator.state),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if state.get("schema_version") != "recovery_case_sampler_v2":
            raise ValueError("sampler state schema is incompatible")
        if state.get("manifest_hash") != self.manifest_hash:
            raise ValueError("sampler state manifest hash is incompatible")
        probabilities = tuple(float(value) for value in state["probabilities"])
        if probabilities != self.probabilities:
            raise ValueError("sampler state probabilities are incompatible")
        rng_state = state.get("rng_state")
        if not isinstance(rng_state, Mapping):
            raise ValueError("sampler state RNG must be a mapping")
        self.rng.bit_generator.state = dict(rng_state)
