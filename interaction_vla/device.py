from __future__ import annotations

import torch


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available in this PyTorch runtime")
        return torch.device("mps")
    if requested != "auto":
        raise ValueError("requested device must be auto, cpu, or mps")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
