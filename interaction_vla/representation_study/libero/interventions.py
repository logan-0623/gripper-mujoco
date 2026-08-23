from __future__ import annotations

from typing import Mapping

import numpy as np


INTERVENTION_CONTROLS = (
    "original",
    "factor_rowspace_matched_donor",
    "matched_random",
    "matched_mean",
    "zero_ood",
    "instruction_shuffle",
)


def _row_basis(weight: np.ndarray) -> np.ndarray:
    matrix = np.asarray(weight, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("probe weight must be a finite matrix")
    _, singular, right = np.linalg.svd(matrix, full_matrices=False)
    rank = int(np.sum(singular > max(matrix.shape) * np.finfo(float).eps * singular.max(initial=0.0)))
    if rank == 0:
        raise ValueError("probe weight has an empty row space")
    return right[:rank]


def factor_rowspace_intervention(
    source: np.ndarray, donor: np.ndarray, probe_weight: np.ndarray
) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64)
    donor = np.asarray(donor, dtype=np.float64)
    if source.shape != donor.shape or source.ndim != 2:
        raise ValueError("source and donor latents must share shape [states, features]")
    basis = _row_basis(probe_weight)
    if basis.shape[1] != source.shape[1]:
        raise ValueError("probe row space and latent feature dimensions differ")
    difference = donor - source
    return source + (difference @ basis.T) @ basis


def matched_mean_intervention(
    source: np.ndarray, train_mean: np.ndarray, probe_weight: np.ndarray
) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64)
    mean = np.asarray(train_mean, dtype=np.float64)
    if source.ndim != 2 or mean.shape != (source.shape[1],) or not np.isfinite(mean).all():
        raise ValueError("matched mean must be a finite training-partition feature mean")
    donor = np.broadcast_to(mean, source.shape)
    return factor_rowspace_intervention(source, donor, probe_weight)


def matched_random_subspace_intervention(
    source: np.ndarray,
    donor: np.ndarray,
    *,
    rank: int,
    seed: int,
) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64)
    donor = np.asarray(donor, dtype=np.float64)
    if source.shape != donor.shape or source.ndim != 2:
        raise ValueError("matched-random source and donor must share [states, features]")
    if rank <= 0 or rank > source.shape[1]:
        raise ValueError("matched-random rank must be within latent feature dimension")
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.normal(size=(source.shape[1], rank)))
    return source + ((donor - source) @ basis) @ basis.T


def zero_ood_intervention(source: np.ndarray) -> np.ndarray:
    source = np.asarray(source)
    if source.ndim != 2 or not np.isfinite(source).all():
        raise ValueError("zero OOD control requires finite [states, features]")
    return np.zeros_like(source)


def matched_donor_indices(
    *,
    factor_labels: np.ndarray,
    nuisance_groups: np.ndarray,
    seed: int,
) -> np.ndarray:
    labels = np.asarray(factor_labels)
    nuisance = np.asarray(nuisance_groups)
    if labels.ndim != 1 or nuisance.shape != labels.shape:
        raise ValueError("factor labels and nuisance groups must be aligned vectors")
    rng = np.random.default_rng(seed)
    result = np.empty(len(labels), dtype=np.int64)
    for index in range(len(labels)):
        candidates = np.flatnonzero(
            (nuisance == nuisance[index]) & (labels != labels[index])
        )
        if len(candidates) == 0:
            raise ValueError(f"state {index} has no matched donor with a different factor")
        result[index] = int(rng.choice(candidates))
    return result


def instruction_shuffle_indices(instructions: np.ndarray, *, seed: int) -> np.ndarray:
    values = np.asarray(instructions)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("instruction shuffle requires an aligned instruction vector")
    rng = np.random.default_rng(seed)
    result = np.empty(len(values), dtype=np.int64)
    for index, value in enumerate(values):
        candidates = np.flatnonzero(values != value)
        if not len(candidates):
            raise ValueError("instruction shuffle requires at least two distinct instructions")
        result[index] = int(rng.choice(candidates))
    return result


def intervention_diagnostics(
    source: np.ndarray,
    intervened: np.ndarray,
    *,
    target_weight: np.ndarray,
    non_target_weights: Mapping[str, np.ndarray],
) -> dict[str, object]:
    source = np.asarray(source, dtype=np.float64)
    changed = np.asarray(intervened, dtype=np.float64)
    if source.shape != changed.shape or source.ndim != 2:
        raise ValueError("intervention diagnostics require aligned latent matrices")

    def probe_change(weight: np.ndarray) -> float:
        matrix = np.asarray(weight, dtype=np.float64)
        return float(np.mean(np.abs((changed - source) @ matrix.T)))

    source_norm = np.linalg.norm(source, axis=1).mean()
    changed_norm = np.linalg.norm(changed, axis=1).mean()
    return {
        "target_probe_change": probe_change(target_weight),
        "non_target_probe_change": {
            name: probe_change(weight) for name, weight in sorted(non_target_weights.items())
        },
        "activation_norm_ratio": float(changed_norm / source_norm) if source_norm > 0 else None,
        "mean_activation_displacement": float(np.linalg.norm(changed - source, axis=1).mean()),
    }
