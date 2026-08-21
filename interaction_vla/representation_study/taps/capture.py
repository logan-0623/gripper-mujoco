from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class ModuleTap:
    tap_id: str
    module_path: str
    capture_input: bool = False
    tensor_selector: str = "last"
    call_reducer: str = "last"

    def __post_init__(self) -> None:
        if not self.tap_id or not self.module_path:
            raise ValueError("module taps require non-empty identifiers and paths")
        if self.tensor_selector not in {"first", "last"}:
            raise ValueError("tensor_selector must be first or last")
        if self.call_reducer not in {"first", "last", "mean"}:
            raise ValueError("call_reducer must be first, last, or mean")


def resolve_module(root: nn.Module, path: str) -> nn.Module:
    current: Any = root
    for component in path.split("."):
        if not component:
            raise ValueError("module path components must be non-empty")
        if component.isdigit():
            current = current[int(component)]
        else:
            current = getattr(current, component, None)
        if current is None:
            raise ValueError(f"policy module path does not exist: {path}")
    if not isinstance(current, nn.Module):
        raise ValueError(f"policy module path is not a torch module: {path}")
    return current


def _tensors(value: object) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, Mapping):
        result: list[torch.Tensor] = []
        for key in sorted(value, key=str):
            result.extend(_tensors(value[key]))
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(_tensors(item))
        return result
    return []


def pool_latent(value: object, *, batch_size: int, selector: str) -> torch.Tensor:
    candidates = [tensor for tensor in _tensors(value) if tensor.ndim >= 1]
    if not candidates:
        raise ValueError("tap output contains no tensor")
    tensor = candidates[0] if selector == "first" else candidates[-1]
    if tensor.ndim == 1:
        if batch_size != 1:
            raise ValueError("one-dimensional tap output requires batch size one")
        tensor = tensor.unsqueeze(0)
    elif tensor.shape[0] == batch_size:
        pass
    elif tensor.ndim >= 2 and tensor.shape[1] == batch_size:
        tensor = tensor.transpose(0, 1)
    else:
        raise ValueError(
            f"cannot identify batch axis for tap tensor {tuple(tensor.shape)} and batch {batch_size}"
        )
    if tensor.ndim == 2:
        pooled = tensor
    elif tensor.ndim == 3:
        pooled = tensor.mean(dim=1)
    else:
        pooled = tensor.flatten(start_dim=2).mean(dim=2)
    result = pooled.detach().to(dtype=torch.float32, device="cpu")
    if result.shape[0] != batch_size or result.ndim != 2 or not torch.isfinite(result).all():
        raise ValueError("pooled tap latent must be finite with shape [batch, features]")
    return result


def pool_latent_with_grad(
    value: object, *, batch_size: int, selector: str
) -> torch.Tensor:
    candidates = [tensor for tensor in _tensors(value) if tensor.ndim >= 1]
    if not candidates:
        raise ValueError("tap output contains no tensor")
    tensor = candidates[0] if selector == "first" else candidates[-1]
    if tensor.ndim == 1:
        if batch_size != 1:
            raise ValueError("one-dimensional tap output requires batch size one")
        tensor = tensor.unsqueeze(0)
    elif tensor.shape[0] == batch_size:
        pass
    elif tensor.ndim >= 2 and tensor.shape[1] == batch_size:
        tensor = tensor.transpose(0, 1)
    else:
        raise ValueError("cannot identify differentiable tap batch axis")
    if tensor.ndim == 2:
        result = tensor
    elif tensor.ndim == 3:
        result = tensor.mean(dim=1)
    else:
        result = tensor.flatten(start_dim=2).mean(dim=2)
    if result.ndim != 2 or result.shape[0] != batch_size:
        raise ValueError("differentiable tap must pool to [batch, features]")
    return result.to(torch.float32)


class ForwardTapCapture:
    def __init__(self, policy: nn.Module, taps: Sequence[ModuleTap]) -> None:
        if not taps or len({tap.tap_id for tap in taps}) != len(taps):
            raise ValueError("forward capture requires unique taps")
        self.policy = policy
        self.taps = tuple(taps)

    def capture(
        self,
        invoke: Callable[[], object],
        *,
        batch_size: int,
    ) -> tuple[object, dict[str, torch.Tensor]]:
        captured: dict[str, list[object]] = {tap.tap_id: [] for tap in self.taps}
        handles: list[Any] = []
        for tap in self.taps:
            module = resolve_module(self.policy, tap.module_path)
            if tap.capture_input:
                handles.append(
                    module.register_forward_pre_hook(
                        lambda _module, inputs, tap_id=tap.tap_id: captured[tap_id].append(inputs)
                    )
                )
            else:
                handles.append(
                    module.register_forward_hook(
                        lambda _module, _inputs, output, tap_id=tap.tap_id: captured[tap_id].append(output)
                    )
                )
        try:
            output = invoke()
        finally:
            for handle in handles:
                handle.remove()
        result: dict[str, torch.Tensor] = {}
        for tap in self.taps:
            values = captured[tap.tap_id]
            if not values:
                raise ValueError(f"policy execution did not reach tap: {tap.tap_id}")
            pooled = [
                pool_latent(value, batch_size=batch_size, selector=tap.tensor_selector)
                for value in values
            ]
            feature_dims = {value.shape[1] for value in pooled}
            if len(feature_dims) != 1:
                raise ValueError(f"tap {tap.tap_id} changed feature width across calls")
            if tap.call_reducer == "first":
                result[tap.tap_id] = pooled[0]
            elif tap.call_reducer == "last":
                result[tap.tap_id] = pooled[-1]
            else:
                result[tap.tap_id] = torch.stack(pooled, dim=0).mean(dim=0)
        return output, result
