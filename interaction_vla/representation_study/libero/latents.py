from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

from ..state_bank.io import write_json_atomic
from .config import LiberoStudyConfig
from .feature_binding import LIBERO_SMOLVLA_RENAME_MAP
from .schema import StateRecord
from .state_bank import load_state_bank
from .taps import SEMANTIC_TAPS, SmolVLASemanticTapCapture


LATENT_SCHEMA = "libero_semantic_latent_cache_v1"


def validate_requested_taps(taps: Sequence[str] | None) -> tuple[str, ...]:
    selected = tuple(SEMANTIC_TAPS if taps is None else taps)
    if not selected:
        raise ValueError("at least one semantic tap is required")
    if len(set(selected)) != len(selected):
        raise ValueError("duplicate semantic tap")
    unknown = sorted(set(selected) - set(SEMANTIC_TAPS))
    if unknown:
        raise ValueError(f"unknown semantic tap: {unknown}")
    return selected


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    files = (
        sorted(
            item
            for item in path.rglob("*")
            if item.is_file() and ".cache" not in item.relative_to(path).parts
        )
        if path.is_dir()
        else [path]
    )
    if not files:
        raise ValueError(f"checkpoint contains no scientific files: {path}")
    digest = hashlib.sha256()
    for item in files:
        relative = str(item.relative_to(path)) if path.is_dir() else item.name
        digest.update(relative.encode("utf-8"))
        digest.update(_file_sha256(item).encode("ascii"))
    return digest.hexdigest()


def _latent_implementation_source_paths() -> tuple[Path, ...]:
    return (
        Path(__file__),
        Path(__file__).with_name("taps.py"),
        Path(__file__).with_name("feature_binding.py"),
        Path(__file__).parents[1] / "backends" / "lerobot.py",
    )


def _latent_implementation_sha256() -> str:
    source_root = Path(__file__).parents[1]
    digest = hashlib.sha256()
    for source in _latent_implementation_source_paths():
        digest.update(source.relative_to(source_root).as_posix().encode("utf-8"))
        digest.update(_file_sha256(source).encode("ascii"))
    return digest.hexdigest()


def deterministic_inference_noise(
    state_ids: Sequence[str],
    *,
    checkpoint_id: str,
    row_shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not state_ids or not row_shape or any(dimension <= 0 for dimension in row_shape):
        raise ValueError("deterministic noise requires state IDs and a positive row shape")
    rows: list[torch.Tensor] = []
    for state_id in state_ids:
        digest = hashlib.sha256(f"{checkpoint_id}:{state_id}".encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "little") % (2**63 - 1)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        rows.append(
            torch.randn(row_shape, generator=generator, device=device, dtype=dtype)
        )
    return torch.stack(rows, dim=0)


def collate_state_bank_observations(
    records: Sequence[StateRecord], dataset: object
) -> dict[str, object]:
    if not records:
        raise ValueError("cannot collate an empty LIBERO State Bank batch")
    samples = [dataset[record.observation.dataset_index] for record in records]  # type: ignore[index]
    global_values: list[torch.Tensor] = []
    wrist_values: list[torch.Tensor] = []
    for record, sample in zip(records, samples, strict=True):
        global_value = sample[record.observation.global_rgb_key]
        global_values.append(torch.as_tensor(global_value))
        if record.observation.wrist_rgb_key is not None:
            wrist_values.append(
                torch.as_tensor(sample[record.observation.wrist_rgb_key])
            )
    batch: dict[str, object] = {
        records[0].observation.global_rgb_key: torch.stack(global_values),
        "observation.state": torch.tensor(
            [record.observation.robot_state for record in records], dtype=torch.float32
        ),
        "task": [record.language for record in records],
    }
    wrist_key = records[0].observation.wrist_rgb_key
    if wrist_key is not None:
        if len(wrist_values) != len(records):
            raise ValueError("wrist RGB applicability differs within one latent batch")
        batch[wrist_key] = torch.stack(wrist_values)
    return batch


def _resolve_checkpoint(
    config: LiberoStudyConfig, stage: str
) -> tuple[str, str, str]:
    manifest_path = config.output_dir / "stages" / stage / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"stage manifest does not exist; run libero stages plan first: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint = str(manifest["checkpoint"])
    path = Path(checkpoint)
    if not path.is_dir():
        raise FileNotFoundError(
            f"SmolVLA checkpoint for {stage} does not exist: {path}"
        )
    checkpoint_hash = _tree_sha256(path)
    expected_hash = manifest.get("checkpoint_sha256")
    if manifest.get("status") != "complete" or expected_hash != checkpoint_hash:
        raise ValueError(f"SmolVLA checkpoint manifest is stale or incomplete: {stage}")
    return str(path), f"{stage}:{checkpoint_hash[:16]}", checkpoint_hash


def extract_smolvla_latents(
    config: LiberoStudyConfig,
    *,
    stage: str,
    batch_size: int = 8,
) -> dict[str, object]:
    if stage not in {"pretrained", "sft_25", "sft_50", "sft_100"}:
        raise ValueError(f"unknown SmolVLA stage: {stage}")
    if batch_size <= 0:
        raise ValueError("latent batch size must be positive")
    checkpoint, checkpoint_id, checkpoint_hash = _resolve_checkpoint(config, stage)
    return extract_smolvla_latents_from_checkpoint(
        config,
        checkpoint=checkpoint,
        checkpoint_id=checkpoint_id,
        checkpoint_hash=checkpoint_hash,
        output_dir=config.output_dir / "latents" / stage,
        label=stage,
        batch_size=batch_size,
        report_fields={"stage": stage},
    )


def _runtime_provenance(policy: object, *, batch_size: int) -> dict[str, object]:
    parameter = next(policy.parameters())  # type: ignore[attr-defined]
    versions = {}
    for package in ("lerobot", "transformers", "torch"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    device = parameter.device
    cuda = device.type == "cuda"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "device_type": device.type,
        "device_name": torch.cuda.get_device_name(device) if cuda else device.type,
        "device_capability": list(torch.cuda.get_device_capability(device)) if cuda else None,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "model_dtype": str(parameter.dtype),
        "batch_size": batch_size,
    }


def _canonical_sha256(value: Mapping[str, object]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def extract_smolvla_latents_from_checkpoint(
    config: LiberoStudyConfig,
    *,
    checkpoint: str,
    checkpoint_id: str,
    checkpoint_hash: str,
    output_dir: str | Path,
    label: str,
    batch_size: int = 8,
    runtime_binding: bool = False,
    report_schema: str = "libero_smolvla_latent_extraction_v1",
    report_fields: Mapping[str, object] | None = None,
    taps: Sequence[str] | None = None,
) -> dict[str, object]:
    if batch_size <= 0:
        raise ValueError("latent batch size must be positive")
    selected_taps = validate_requested_taps(taps)
    output_dir = Path(output_dir)
    bank_root = config.output_dir / "state_bank"
    records, bank_manifest, _, _ = load_state_bank(bank_root)
    if not bank_manifest.get("audit_passed"):
        raise ValueError("LIBERO State Bank audit has not passed")
    source_revisions = {record.source_revision for record in records}
    if len(source_revisions) != 1:
        raise ValueError("State Bank records do not share one LeRobot dataset revision")
    dataset_revision = next(iter(source_revisions))
    try:
        from lerobot.datasets import LeRobotDataset
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("LeRobotDataset is required for latent extraction") from error
    dataset = LeRobotDataset(
        config.sources.lerobot_repo_id,
        root=config.sources.lerobot_root,
        revision=dataset_revision,
        download_videos=True,
    )
    from ..backends.lerobot import SmolVLABackend

    backend = SmolVLABackend(device="auto")
    backend.load_checkpoint_for_dataset(
        checkpoint,
        repo_id=config.sources.lerobot_repo_id,
        dataset_root=dataset.root,
        rename_map=LIBERO_SMOLVLA_RENAME_MAP,
    )
    policy, preprocessor, _ = backend._loaded()
    runtime = _runtime_provenance(policy, batch_size=batch_size)
    runtime_fingerprint = _canonical_sha256(runtime) if runtime_binding else None
    state_bank_hash = _file_sha256(bank_root / "manifest.json")
    implementation_hash = _latent_implementation_sha256()
    state_ids = tuple(record.state_id for record in records)
    writers = {
        tap: LatentCacheWriter(
            output_dir / tap,
            checkpoint_id=checkpoint_id,
            checkpoint_sha256=checkpoint_hash,
            state_bank_sha256=state_bank_hash,
            tap=tap,
            pooling="valid_token_mean",
            implementation_sha256=implementation_hash,
            runtime_fingerprint_sha256=runtime_fingerprint,
            expected_state_ids=state_ids,
        )
        for tap in selected_taps
    }
    tap_metadata: dict[str, Mapping[str, object]] | None = None
    for start in tqdm(
        range(0, len(records), batch_size),
        desc=f"SmolVLA latents {label}",
        unit="batch",
    ):
        selected = records[start : start + batch_size]
        if tap_metadata is not None and all(
            writer.has(record.state_id)
            for record in selected
            for writer in writers.values()
        ):
            continue
        batch = collate_state_bank_observations(selected, dataset)
        processed = preprocessor(backend._raw_batch(batch))
        flow = policy.model
        original_sample_noise = flow.sample_noise

        def sample_noise(shape, device, selected=selected):
            return deterministic_inference_noise(
                tuple(record.state_id for record in selected),
                checkpoint_id=checkpoint_id,
                row_shape=tuple(int(item) for item in shape[1:]),
                device=torch.device(device),
                dtype=processed["observation.state"].dtype,
            )

        flow.sample_noise = sample_noise
        policy.reset()
        capture = SmolVLASemanticTapCapture(policy)
        try:
            with torch.no_grad():
                _, captured_values, captured_metadata = capture.capture(
                    lambda: policy.predict_action_chunk(processed)
                )
        finally:
            flow.sample_noise = original_sample_noise
        values = {tap: captured_values[tap] for tap in selected_taps}
        metadata = {tap: captured_metadata[tap] for tap in selected_taps}
        if tap_metadata is None:
            tap_metadata = metadata
        elif tap_metadata != metadata:
            raise ValueError("SmolVLA semantic tap metadata changed across batches")
        for tap, matrix in values.items():
            for record, row in zip(selected, matrix.numpy(), strict=True):
                writers[tap].add(record.state_id, row)
    manifests = {tap: writer.finalize() for tap, writer in writers.items()}
    report = {
        "schema_version": report_schema,
        "passed": True,
        "checkpoint_id": checkpoint_id,
        "checkpoint_sha256": checkpoint_hash,
        "state_bank_sha256": state_bank_hash,
        "implementation_sha256": implementation_hash,
        "runtime": runtime,
        "runtime_fingerprint_sha256": runtime_fingerprint,
        "states": len(records),
        "tap_metadata": tap_metadata,
        "caches": manifests,
        **dict(report_fields or {}),
    }
    write_json_atomic(output_dir / "report.json", report)
    return report


def inspect_stage_latents(config: LiberoStudyConfig, *, stage: str) -> dict[str, object]:
    report_path = config.output_dir / "latents" / stage / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = {}
    for tap in SEMANTIC_TAPS:
        state_ids, values, manifest = load_latent_cache(
            config.output_dir / "latents" / stage / tap
        )
        rows[tap] = {
            "states": len(state_ids),
            "feature_dim": int(values.shape[1]),
            "manifest": manifest,
        }
    if len({row["states"] for row in rows.values()}) != 1:
        raise ValueError("semantic tap caches do not cover the same states")
    return {**report, "validated_caches": rows}


class LatentCacheWriter:
    def __init__(
        self,
        output_dir: str | Path,
        *,
        checkpoint_id: str,
        checkpoint_sha256: str,
        state_bank_sha256: str,
        tap: str,
        pooling: str,
        expected_state_ids: Sequence[str],
        implementation_sha256: str | None = None,
        runtime_fingerprint_sha256: str | None = None,
    ) -> None:
        self.root = Path(output_dir)
        self.rows = self.root / ".rows"
        self.expected_state_ids = tuple(str(item) for item in expected_state_ids)
        if len(set(self.expected_state_ids)) != len(self.expected_state_ids):
            raise ValueError("latent cache expected state IDs must be unique")
        self.expected_state_id_set = frozenset(self.expected_state_ids)
        self.binding = {
            "schema_version": LATENT_SCHEMA,
            "checkpoint_id": checkpoint_id,
            "checkpoint_sha256": checkpoint_sha256,
            "state_bank_sha256": state_bank_sha256,
            "tap": tap,
            "pooling": pooling,
            "implementation_sha256": implementation_sha256,
            "expected_state_ids": list(self.expected_state_ids),
        }
        if runtime_fingerprint_sha256 is not None:
            self.binding["runtime_fingerprint_sha256"] = runtime_fingerprint_sha256
        binding_path = self.root / "binding.json"
        if binding_path.exists():
            existing = json.loads(binding_path.read_text(encoding="utf-8"))
            if existing != self.binding:
                raise ValueError("latent cache has a different scientific binding")
        else:
            write_json_atomic(binding_path, self.binding)

    def _row_path(self, state_id: str) -> Path:
        digest = hashlib.sha256(state_id.encode("utf-8")).hexdigest()
        return self.rows / f"{digest}.npy"

    def _row_hash_path(self, state_id: str) -> Path:
        return self._row_path(state_id).with_suffix(".sha256.json")

    def has(self, state_id: str) -> bool:
        if state_id not in self.expected_state_id_set:
            raise ValueError(f"latent state ID is not expected: {state_id}")
        path = self._row_path(state_id)
        hash_path = self._row_hash_path(state_id)
        if not path.is_file() or not hash_path.is_file():
            return False
        try:
            expected = json.loads(hash_path.read_text(encoding="utf-8"))["sha256"]
            array = np.load(path, allow_pickle=False)
        except (OSError, KeyError, TypeError, ValueError):
            return False
        return (
            expected == _file_sha256(path)
            and array.ndim == 1
            and np.isfinite(array).all()
        )

    def add(self, state_id: str, value: np.ndarray) -> Path:
        if state_id not in self.expected_state_id_set:
            raise ValueError(f"latent state ID is not expected: {state_id}")
        array = np.asarray(value, dtype=np.float32)
        if array.ndim != 1 or not np.isfinite(array).all():
            raise ValueError("latent row must be a finite one-dimensional array")
        path = self._row_path(state_id)
        if path.exists():
            existing = np.load(path, allow_pickle=False)
            if not np.array_equal(existing, array):
                raise ValueError(f"cached latent differs for state: {state_id}")
            write_json_atomic(
                self._row_hash_path(state_id), {"sha256": _file_sha256(path)}
            )
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".npy.tmp")
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        write_json_atomic(
            self._row_hash_path(state_id), {"sha256": _file_sha256(path)}
        )
        return path

    def finalize(self) -> dict[str, object]:
        missing = [state_id for state_id in self.expected_state_ids if not self.has(state_id)]
        if missing:
            raise ValueError(f"latent cache is missing {len(missing)} expected states")
        arrays = [np.load(self._row_path(state_id), allow_pickle=False) for state_id in self.expected_state_ids]
        widths = {array.shape for array in arrays}
        if len(widths) != 1:
            raise ValueError("latent cache rows have inconsistent feature widths")
        matrix = np.stack(arrays).astype(np.float32, copy=False)
        value_path = self.root / "values.npy"
        temporary = value_path.with_suffix(".npy.tmp")
        with temporary.open("wb") as handle:
            np.save(handle, matrix, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, value_path)
        manifest = {
            **self.binding,
            "states": len(self.expected_state_ids),
            "feature_dim": int(matrix.shape[1]),
            "values_sha256": _file_sha256(value_path),
            "complete": True,
        }
        write_json_atomic(self.root / "manifest.json", manifest)
        return manifest


def load_latent_cache(
    output_dir: str | Path,
) -> tuple[tuple[str, ...], np.ndarray, dict[str, object]]:
    root = Path(output_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != LATENT_SCHEMA or not manifest.get("complete"):
        raise ValueError("latent cache manifest is incomplete or incompatible")
    values_path = root / "values.npy"
    if _file_sha256(values_path) != manifest.get("values_sha256"):
        raise ValueError("latent cache values hash does not match manifest")
    values = np.load(values_path, allow_pickle=False)
    state_ids = tuple(str(item) for item in manifest["expected_state_ids"])
    if len(set(state_ids)) != len(state_ids):
        raise ValueError("latent cache manifest contains duplicate state IDs")
    if values.shape != (len(state_ids), int(manifest["feature_dim"])):
        raise ValueError("latent cache matrix shape does not match manifest")
    if values.dtype != np.float32 or not np.isfinite(values).all():
        raise ValueError("latent cache matrix must be finite float32")
    return state_ids, values, manifest
