from __future__ import annotations

from typing import Iterable, Mapping, Protocol, Sequence

import numpy as np

from .schema import InteractionLabels, StateRecord


class ReplayFrame(Protocol):
    frame_index: int
    action: np.ndarray
    simulator_state: np.ndarray


class ReplayAdapter(Protocol):
    def replay_episode(self, source: object) -> Iterable[ReplayFrame]: ...


class InteractionAnnotator(Protocol):
    def annotate_episode(self, frames: Sequence[ReplayFrame]) -> Sequence[InteractionLabels]: ...


class StateBank(Protocol):
    def records(self) -> Iterable[StateRecord]: ...


class PolicyAdapter(Protocol):
    def act(self, observation: Mapping[str, object]) -> np.ndarray: ...


class LatentTapRegistry(Protocol):
    def capture(self, policy: PolicyAdapter, observation: Mapping[str, object]) -> Mapping[str, object]: ...


class ProbeRunner(Protocol):
    def fit_and_evaluate(self, bank: StateBank, latents: object) -> Mapping[str, object]: ...


class InterventionRunner(Protocol):
    def intervene(self, factor: str, latent: object, control: str) -> object: ...


class ClosedLoopEvaluator(Protocol):
    def paired_rollout(self, case: object, intervention: object) -> Mapping[str, object]: ...
