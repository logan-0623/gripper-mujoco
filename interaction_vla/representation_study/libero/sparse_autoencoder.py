from __future__ import annotations

import os
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm


class TopKSparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, feature_dim: int, top_k: int) -> None:
        super().__init__()
        if not 0 < top_k <= feature_dim or min(input_dim, feature_dim) <= 0:
            raise ValueError("invalid Top-K sparse autoencoder dimensions")
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.top_k = top_k
        self.encoder = nn.Linear(input_dim, feature_dim)
        self.decoder = nn.Linear(feature_dim, input_dim, bias=False)
        nn.init.kaiming_uniform_(self.encoder.weight, a=5**0.5)
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.T)
            self.normalize_decoder_()

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        dense = torch.relu(self.encoder(values))
        if self.top_k == self.feature_dim:
            return dense
        indices = torch.topk(dense, self.top_k, dim=-1, sorted=False).indices
        mask = torch.zeros_like(dense).scatter_(-1, indices, 1.0)
        return dense * mask

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        activations = self.encode(values)
        return self.decoder(activations), activations

    @torch.no_grad()
    def normalize_decoder_(self) -> None:
        self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-8))


def train_sparse_autoencoder(
    values: np.ndarray,
    *,
    feature_dim: int,
    top_k: int,
    steps: int,
    batch_size: int,
    seed: int,
    device: str = "auto",
    progress: bool = True,
) -> tuple[TopKSparseAutoencoder, np.ndarray, np.ndarray, dict[str, float]]:
    data = np.asarray(values, dtype=np.float32)
    if data.ndim != 2 or not len(data) or not np.isfinite(data).all():
        raise ValueError("SAE training values must be a finite non-empty matrix")
    if min(steps, batch_size) <= 0:
        raise ValueError("SAE training steps and batch size must be positive")
    mean = data.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = data.std(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.where(scale > 1e-6, scale, 1.0).astype(np.float32)
    normalized = (data - mean) / scale
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch_device = torch.device(
        "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    )
    model = TopKSparseAutoencoder(data.shape[1], feature_dim, top_k).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    tensor = torch.from_numpy(normalized)
    first_loss = last_loss = 0.0
    for step in tqdm(range(steps), desc=f"Top-K SAE seed {seed}", disable=not progress):
        indices = torch.randint(len(tensor), (batch_size,), generator=generator)
        batch = tensor[indices].to(torch_device)
        reconstructed, _ = model(batch)
        loss = nn.functional.mse_loss(reconstructed, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.normalize_decoder_()
        last_loss = float(loss.detach().cpu())
        if step == 0:
            first_loss = last_loss
    model = model.cpu().eval()
    return model, mean, scale, {"first_loss": first_loss, "last_loss": last_loss}


def save_sparse_autoencoder(
    path: str | Path,
    model: TopKSparseAutoencoder,
    *,
    mean: np.ndarray,
    scale: np.ndarray,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "input_dim": np.asarray(model.input_dim, dtype=np.int64),
        "feature_dim": np.asarray(model.feature_dim, dtype=np.int64),
        "top_k": np.asarray(model.top_k, dtype=np.int64),
        "encoder_weight": model.encoder.weight.detach().cpu().numpy(),
        "encoder_bias": model.encoder.bias.detach().cpu().numpy(),
        "decoder_weight": model.decoder.weight.detach().cpu().numpy(),
        "mean": np.asarray(mean, dtype=np.float32),
        "scale": np.asarray(scale, dtype=np.float32),
    }
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def load_sparse_autoencoder(
    path: str | Path,
) -> tuple[TopKSparseAutoencoder, np.ndarray, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as archive:
        model = TopKSparseAutoencoder(
            int(archive["input_dim"]), int(archive["feature_dim"]), int(archive["top_k"])
        )
        with torch.no_grad():
            model.encoder.weight.copy_(torch.from_numpy(archive["encoder_weight"]))
            model.encoder.bias.copy_(torch.from_numpy(archive["encoder_bias"]))
            model.decoder.weight.copy_(torch.from_numpy(archive["decoder_weight"]))
        mean = archive["mean"].astype(np.float32, copy=True)
        scale = archive["scale"].astype(np.float32, copy=True)
    if mean.shape != (model.input_dim,) or scale.shape != mean.shape or np.any(scale <= 0):
        raise ValueError("SAE normalization metadata is incompatible")
    return model.eval(), mean, scale


def match_decoder_features(
    reference: np.ndarray, candidate: np.ndarray
) -> tuple[tuple[int, int, float], ...]:
    first = np.asarray(reference, dtype=np.float64)
    second = np.asarray(candidate, dtype=np.float64)
    if first.ndim != 2 or second.ndim != 2 or first.shape[1] != second.shape[1]:
        raise ValueError("decoder features must be two aligned matrices")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("decoder features must be finite")
    first = first / np.linalg.norm(first, axis=1, keepdims=True).clip(1e-12)
    second = second / np.linalg.norm(second, axis=1, keepdims=True).clip(1e-12)
    cosine = first @ second.T
    used_first: set[int] = set()
    used_second: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for flat_index in np.argsort(cosine, axis=None, kind="stable")[::-1]:
        left, right = np.unravel_index(flat_index, cosine.shape)
        if left in used_first or right in used_second:
            continue
        matches.append((int(left), int(right), float(cosine[left, right])))
        used_first.add(int(left))
        used_second.add(int(right))
        if len(matches) == min(len(first), len(second)):
            break
    return tuple(sorted(matches))
