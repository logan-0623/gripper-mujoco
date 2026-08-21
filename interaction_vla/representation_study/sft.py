from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterator, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
from tqdm.auto import tqdm

from interaction_vla.lerobot_bridge.provenance import sha256_file

from .backends import make_backend
from .config import RepresentationStudyConfig
from .state_bank.io import write_json_atomic


SFT_REPORT_SCHEMA_VERSION = "interaction_stage_sft_report_v1"


class DeterministicStepBatchSampler(Sampler[list[int]]):
    """Maps every optimization step to a reproducible dataset batch."""

    def __init__(
        self,
        *,
        dataset_size: int,
        batch_size: int,
        seed: int,
        start_step: int,
        total_steps: int,
    ) -> None:
        if dataset_size < 1 or batch_size < 1 or not 0 <= start_step <= total_steps:
            raise ValueError("deterministic sampler arguments are invalid")
        self.dataset_size = int(dataset_size)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.start_step = int(start_step)
        self.total_steps = int(total_steps)

    def __iter__(self) -> Iterator[list[int]]:
        for step in range(self.start_step, self.total_steps):
            rng = np.random.default_rng(np.random.SeedSequence((self.seed, step)))
            yield rng.integers(
                0, self.dataset_size, size=self.batch_size, endpoint=False
            ).astype(int).tolist()

    def __len__(self) -> int:
        return self.total_steps - self.start_step


def _parent_stage(stage: str) -> str:
    parents = {"sft": "pretrained", "continued_sft": "sft"}
    if stage not in parents:
        raise ValueError("SFT stage must be sft or continued_sft")
    return parents[stage]


def _destination(config: RepresentationStudyConfig, backend: str, stage: str) -> Path:
    destination = config.sft.output_dir / backend / stage
    expected = destination / "checkpoint"
    actual = Path(config.stage_config(backend, stage).checkpoint)
    if actual != expected:
        raise ValueError(f"configured {backend}/{stage} checkpoint must be {expected}")
    return destination


def _binding(config: RepresentationStudyConfig, backend: str, stage: str) -> str:
    payload = {
        "config_sha256": sha256_file(config.config_path),
        "backend": backend,
        "stage": stage,
        "parent": config.stage_config(backend, _parent_stage(stage)).checkpoint,
        "sft": asdict(config.sft),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _train_episodes(path: Path) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        values = payload["episode_indices"]["train"]
    except (KeyError, TypeError) as error:
        raise ValueError("dataset split manifest has no train episode indices") from error
    result = [int(value) for value in values]
    if not result or len(result) != len(set(result)):
        raise ValueError("SFT train episodes must be non-empty and unique")
    return result


def _dataset(config: RepresentationStudyConfig, policy_config: Any) -> Any:
    from lerobot.datasets import LeRobotDataset
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    metadata = LeRobotDatasetMetadata(config.dataset.repo_id, root=config.dataset.root)
    delta_timestamps: dict[str, list[float]] = {}
    if policy_config.action_delta_indices is not None:
        delta_timestamps["action"] = [
            int(index) / metadata.fps for index in policy_config.action_delta_indices
        ]
    for key in getattr(policy_config, "image_features", {}):
        indices = policy_config.observation_delta_indices
        if indices is not None and len(indices) > 1:
            delta_timestamps[key] = [int(index) / metadata.fps for index in indices]
    return LeRobotDataset(
        config.dataset.repo_id,
        root=config.dataset.root,
        episodes=_train_episodes(config.dataset.split_manifest),
        delta_timestamps=delta_timestamps or None,
    )


def _load_policy_for_dataset(
    config: RepresentationStudyConfig,
    *,
    backend_name: str,
    parent_checkpoint: str,
) -> tuple[Any, Any, Any, Any]:
    if backend_name == "act":
        runtime = make_backend("act", device=config.extraction.device)
        runtime.load_checkpoint(parent_checkpoint)
        dataset = _dataset(config, runtime.policy.config)
        return runtime, dataset, runtime.preprocessor, runtime.postprocessor
    runtime = make_backend(backend_name, device=config.extraction.device)
    runtime.load_checkpoint_for_dataset(
        parent_checkpoint,
        repo_id=config.dataset.repo_id,
        dataset_root=config.dataset.root,
    )
    dataset = _dataset(config, runtime.policy.config)
    return runtime, dataset, runtime.preprocessor, runtime.postprocessor


def _save(
    destination: Path,
    *,
    runtime: Any,
    optimizer: torch.optim.Optimizer,
    step: int,
    losses: list[float],
    binding: str,
) -> None:
    checkpoint = destination / "checkpoint"
    runtime.policy.save_pretrained(checkpoint)
    runtime.preprocessor.save_pretrained(checkpoint)
    runtime.postprocessor.save_pretrained(checkpoint)
    torch.save(
        {
            "schema_version": "interaction_sft_training_state_v1",
            "step": int(step),
            "optimizer": optimizer.state_dict(),
            "losses": list(losses),
            "binding": binding,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.random.get_rng_state(),
        },
        destination / "training_state.pt",
    )


def train_sft(
    config: RepresentationStudyConfig,
    *,
    backend: str,
    stage: str,
    resume: bool,
) -> dict[str, object]:
    parent = _parent_stage(stage)
    parent_checkpoint = config.stage_config(backend, parent).checkpoint
    destination = _destination(config, backend, stage)
    training_state = destination / "training_state.pt"
    if destination.exists() and not resume:
        raise FileExistsError(f"SFT output already exists; pass --resume: {destination}")
    if resume and not training_state.is_file():
        raise FileNotFoundError(f"SFT resume state not found: {training_state}")
    destination.mkdir(parents=True, exist_ok=True)
    binding = _binding(config, backend, stage)
    start = 0
    losses: list[float] = []
    load_checkpoint = parent_checkpoint
    loaded: Mapping[str, object] | None = None
    if resume:
        loaded = torch.load(training_state, map_location="cpu", weights_only=False)
        if loaded.get("schema_version") != "interaction_sft_training_state_v1":
            raise ValueError("SFT training state schema is incompatible")
        if loaded.get("binding") != binding:
            raise ValueError("SFT resume state differs from current config")
        start = int(loaded["step"])
        losses = [float(value) for value in loaded["losses"]]  # type: ignore[arg-type]
        load_checkpoint = (destination / "checkpoint").as_posix()
    runtime, dataset, _, _ = _load_policy_for_dataset(
        config, backend_name=backend, parent_checkpoint=load_checkpoint
    )
    runtime.set_trainable_groups(config.stage_config(backend, stage).trainable_groups)
    trainable = [parameter for parameter in runtime.policy.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("SFT selected no trainable policy parameters")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.sft.learning_rate,
        weight_decay=config.sft.weight_decay,
    )
    if loaded is not None:
        optimizer.load_state_dict(loaded["optimizer"])  # type: ignore[arg-type]
        random.setstate(loaded["python_rng"])  # type: ignore[arg-type]
        np.random.set_state(loaded["numpy_rng"])  # type: ignore[arg-type]
        torch.random.set_rng_state(loaded["torch_rng"])  # type: ignore[arg-type]
    else:
        random.seed(config.sft.seed)
        np.random.seed(config.sft.seed)
        torch.manual_seed(config.sft.seed)
        _save(
            destination,
            runtime=runtime,
            optimizer=optimizer,
            step=0,
            losses=losses,
            binding=binding,
        )
    sampler = DeterministicStepBatchSampler(
        dataset_size=len(dataset),
        batch_size=config.sft.batch_size,
        seed=config.sft.seed,
        start_step=start,
        total_steps=config.sft.steps,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=config.sft.num_workers,
        pin_memory=runtime.device.type == "cuda",
    )
    progress = tqdm(
        loader,
        total=config.sft.steps,
        initial=start,
        desc=f"{backend}/{stage} SFT",
        unit="step",
        dynamic_ncols=True,
    )
    runtime.policy.train()
    for step, raw_batch in enumerate(progress, start=start + 1):
        batch = runtime.preprocessor(runtime._raw_batch(raw_batch))
        loss, _ = runtime.policy.forward(batch)
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not torch.isfinite(loss):
            raise ValueError("SFT policy loss must be a finite scalar tensor")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable, config.sft.grad_clip_norm
        )
        if not torch.isfinite(gradient_norm):
            raise ValueError("SFT gradient norm is non-finite")
        optimizer.step()
        losses.append(float(loss.detach().item()))
        progress.set_postfix(loss=losses[-1])
        if step % config.sft.save_every == 0 or step == config.sft.steps:
            _save(
                destination,
                runtime=runtime,
                optimizer=optimizer,
                step=step,
                losses=losses,
                binding=binding,
            )
    report = {
        "schema_version": SFT_REPORT_SCHEMA_VERSION,
        "passed": True,
        "backend": backend,
        "stage": stage,
        "parent_stage": parent,
        "parent_checkpoint": parent_checkpoint,
        "checkpoint": (destination / "checkpoint").as_posix(),
        "steps": config.sft.steps,
        "train_episodes": len(_train_episodes(config.dataset.split_manifest)),
        "dataset_frames": len(dataset),
        "mean_loss_last_100": float(np.mean(losses[-100:])),
    }
    write_json_atomic(destination / "report.json", report)
    return report
