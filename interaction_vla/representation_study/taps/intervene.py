from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import nn

from .capture import ModuleTap, _tensors, resolve_module


def replace_selected_tensor(
    value: object, *, selector: str, transform: Callable[[torch.Tensor], torch.Tensor]
) -> object:
    tensors = _tensors(value)
    if not tensors:
        raise ValueError("intervention target contains no tensor")
    target = tensors[0] if selector == "first" else tensors[-1]

    def replace(item: object) -> object:
        if item is target:
            return transform(target)
        if isinstance(item, Mapping):
            return {key: replace(child) for key, child in item.items()}
        if isinstance(item, tuple):
            return tuple(replace(child) for child in item)
        if isinstance(item, list):
            return [replace(child) for child in item]
        return item

    return replace(value)


def intervention_tensor(
    tensor: torch.Tensor, *, batch_size: int, mode: str
) -> torch.Tensor:
    if tensor.ndim < 1:
        raise ValueError("intervention tensor must have a batch axis")
    if tensor.shape[0] == batch_size:
        axis = 0
    elif tensor.ndim >= 2 and tensor.shape[1] == batch_size:
        axis = 1
    else:
        raise ValueError("cannot identify intervention tensor batch axis")
    if mode == "zero":
        return torch.zeros_like(tensor)
    if mode == "mean":
        return tensor.mean(dim=axis, keepdim=True).expand_as(tensor)
    if mode == "matched_random":
        if batch_size < 2:
            raise ValueError("matched_random intervention requires at least two states")
        return torch.roll(tensor, shifts=1, dims=axis)
    raise ValueError("intervention mode must be zero, mean, or matched_random")


class ForwardTapIntervention:
    def __init__(self, policy: nn.Module, tap: ModuleTap) -> None:
        self.policy = policy
        self.tap = tap
        self.last_tensor: torch.Tensor | None = None

    def run(
        self,
        invoke: Callable[[], object],
        *,
        batch_size: int,
        mode: str,
    ) -> object:
        module = resolve_module(self.policy, self.tap.module_path)
        self.last_tensor = None

        def transform(value: object) -> object:
            def changed(tensor: torch.Tensor) -> torch.Tensor:
                result = intervention_tensor(
                    tensor, batch_size=batch_size, mode=mode
                )
                self.last_tensor = result
                return result

            return replace_selected_tensor(
                value,
                selector=self.tap.tensor_selector,
                transform=changed,
            )

        if self.tap.capture_input:
            handle = module.register_forward_pre_hook(
                lambda _module, inputs: transform(inputs)
            )
        else:
            handle = module.register_forward_hook(
                lambda _module, _inputs, output: transform(output)
            )
        try:
            return invoke()
        finally:
            handle.remove()
