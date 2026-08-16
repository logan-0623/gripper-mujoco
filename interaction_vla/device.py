from __future__ import annotations

import torch


DEVICE_TYPES = frozenset({"cpu", "mps", "cuda"})
DEVICE_REQUESTS = frozenset({"auto", *DEVICE_TYPES})


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available in this PyTorch runtime"
            )
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available in this PyTorch runtime")
        return torch.device("mps")
    if requested != "auto":
        raise ValueError("requested device must be auto, cpu, mps, or cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
