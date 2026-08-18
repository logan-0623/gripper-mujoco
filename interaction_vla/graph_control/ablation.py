from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from interaction_vla.graph_control.schema import (
    ABLATION_CONDITIONS,
    TOKEN_DIM,
    TOKEN_SLICES,
    validate_token,
)


_ENTITY_GEOMETRY_GROUPS = frozenset(
    {
        "entity_presence",
        "entity_visibility",
        "gripper_target_geometry",
        "target_receptacle_geometry",
        "distractor_geometry",
    }
)
_INTERACTION_STATE_GROUPS = _ENTITY_GEOMETRY_GROUPS | {
    "relation_presence",
    "phase",
}


def _validate_token_rows(value: object) -> np.ndarray:
    tokens = np.asarray(value, dtype=np.float32)
    if tokens.ndim != 2 or tokens.shape[1] != TOKEN_DIM:
        raise ValueError(f"Graph tokens must have shape [rows, {TOKEN_DIM}]")
    if not np.isfinite(tokens).all():
        raise ValueError("Graph tokens must be finite")
    return tokens


def representation_transform(tokens: object, condition: str) -> np.ndarray:
    """Apply a fixed-width progressive representation mask.

    ``shuffled_graph`` retains the full token here; temporal/entity correspondence is
    broken separately at the episode level so that its marginal distribution stays
    unchanged.
    """

    source = _validate_token_rows(tokens)
    if condition not in ABLATION_CONDITIONS:
        raise ValueError(f"unsupported ablation condition: {condition}")

    if condition == "flat":
        active_groups: frozenset[str] = frozenset()
    elif condition == "entity_geometry":
        active_groups = _ENTITY_GEOMETRY_GROUPS
    elif condition == "interaction_state":
        active_groups = frozenset(_INTERACTION_STATE_GROUPS)
    else:
        active_groups = frozenset(TOKEN_SLICES)

    transformed = np.zeros_like(source)
    for group in active_groups:
        transformed[:, TOKEN_SLICES[group]] = source[:, TOKEN_SLICES[group]]
    return transformed


def stratified_episode_permutation(
    lengths: Mapping[int, int],
    *,
    seed: int,
) -> tuple[dict[int, int], dict[int, int]]:
    """Return a deterministic, within-length-stratum derangement.

    At most four rank-based strata are used.  The number of strata is capped at
    ``floor(n / 2)`` so every stratum has at least two episodes and can be
    deranged without silently falling back to self-pairing.
    """

    normalized = {int(episode): int(length) for episode, length in lengths.items()}
    if len(normalized) < 2:
        raise ValueError("at least two episodes are required for shuffling")
    if len(normalized) != len(lengths) or any(length <= 0 for length in normalized.values()):
        raise ValueError("episode ids must be unique and lengths must be positive")

    ordered = sorted(normalized, key=lambda episode: (normalized[episode], episode))
    stratum_count = min(4, len(ordered) // 2)
    groups = [group.tolist() for group in np.array_split(ordered, stratum_count)]
    if any(len(group) < 2 for group in groups):
        raise RuntimeError("internal error: shuffle stratum cannot be deranged")

    rng = np.random.default_rng(int(seed))
    permutation: dict[int, int] = {}
    strata: dict[int, int] = {}
    for stratum, group in enumerate(groups):
        shuffled = list(group)
        rng.shuffle(shuffled)
        sources = shuffled[1:] + shuffled[:1]
        permutation.update(zip(shuffled, sources, strict=True))
        strata.update({episode: stratum for episode in group})

    return permutation, strata


def resample_sequence_nearest(tokens: object, *, destination_length: int) -> np.ndarray:
    """Match episode lengths by nearest normalized progress, without interpolation."""

    source = np.asarray(tokens, dtype=np.float32)
    if source.ndim != 2 or source.shape[0] == 0:
        raise ValueError("source sequence must be a non-empty 2D array")
    if not np.isfinite(source).all():
        raise ValueError("source sequence must be finite")
    if destination_length <= 0:
        raise ValueError("destination_length must be positive")

    indices = np.rint(
        np.linspace(0, source.shape[0] - 1, int(destination_length))
    ).astype(np.int64)
    return source[indices].copy()


class MaskedPredictedTokenProvider:
    """Apply the exact training-time ablation mask to a live Graph estimate."""

    def __init__(self, provider: object, *, condition: str) -> None:
        if condition not in {"entity_geometry", "interaction_state", "full_graph"}:
            raise ValueError("masked provider condition is incompatible")
        self.provider = provider
        self.condition = condition

    def reset(self) -> None:
        self.provider.reset()  # type: ignore[attr-defined]

    def bind_model(self, model: object) -> None:
        if hasattr(self.provider, "bind_model"):
            self.provider.bind_model(model)  # type: ignore[attr-defined]

    def token(self, **kwargs: object) -> np.ndarray:
        value = self.provider.token(**kwargs)  # type: ignore[attr-defined]
        transformed = representation_transform(
            validate_token(value)[None, :], self.condition
        )
        return transformed[0]


class ScheduledShuffledTokenProvider:
    """Serve a predeclared observation-independent test-reservoir sequence."""

    def __init__(
        self,
        *,
        sequences: Mapping[int, object],
        case_schedule: Mapping[str, int],
        max_steps: int,
    ) -> None:
        if max_steps < 1:
            raise ValueError("shuffled provider max_steps must be positive")
        self.sequences = {
            int(episode): _validate_token_rows(tokens).copy()
            for episode, tokens in sequences.items()
        }
        self.case_schedule = {
            str(case): int(episode) for case, episode in case_schedule.items()
        }
        if not self.sequences or not self.case_schedule:
            raise ValueError("shuffled provider requires sequences and a case schedule")
        missing = set(self.case_schedule.values()) - set(self.sequences)
        if missing:
            raise ValueError("shuffled provider schedule references missing sequences")
        self.max_steps = int(max_steps)
        self._selected_episode: int | None = None
        self._step = 0

    def select_case(self, case_id: str) -> None:
        try:
            self._selected_episode = self.case_schedule[str(case_id)]
        except KeyError as error:
            raise ValueError(f"shuffled provider case is not scheduled: {case_id}") from error

    def reset(self) -> None:
        if self._selected_episode is None:
            raise ValueError("shuffled provider case must be selected before reset")
        self._step = 0

    def token(self, **kwargs: object) -> np.ndarray:
        del kwargs
        if self._selected_episode is None:
            raise ValueError("shuffled provider case must be selected before inference")
        if self._step >= self.max_steps:
            raise ValueError("shuffled provider exceeded configured max_steps")
        sequence = self.sequences[self._selected_episode]
        if self.max_steps == 1:
            index = 0
        else:
            index = int(
                np.rint(
                    self._step * (sequence.shape[0] - 1) / (self.max_steps - 1)
                )
            )
        self._step += 1
        return validate_token(sequence[index])
