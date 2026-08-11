from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from interaction_vla.lerobot_bridge.act_smoke import FORBIDDEN_BATCH_FRAGMENTS

from .cache import TokenCache
from .schema import TOKEN_DIM, TOKEN_FEATURE_NAMES


ENVIRONMENT_STATE_KEY = "observation.environment_state"


class GraphDatasetMetadata:
    """Read-only metadata view adding ACT's native environment-state feature."""

    def __init__(self, base: Any) -> None:
        self._base = base
        features = {name: dict(value) for name, value in base.features.items()}
        if ENVIRONMENT_STATE_KEY in features:
            raise ValueError("base dataset already declares observation.environment_state")
        features[ENVIRONMENT_STATE_KEY] = {
            "dtype": "float32",
            "shape": [TOKEN_DIM],
            "names": list(TOKEN_FEATURE_NAMES),
        }
        self._features = features

    @property
    def features(self) -> dict[str, dict[str, object]]:
        return self._features

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)


class GraphConditionedDataset(Dataset[dict[str, Any]]):
    def __init__(self, base: Any, cache: TokenCache) -> None:
        if not hasattr(base, "meta") or not hasattr(base, "features"):
            raise ValueError("base dataset must expose LeRobot metadata and features")
        self.base = base
        self.cache = cache
        self.meta = GraphDatasetMetadata(base.meta)
        self._tokens = cache.by_row

    @property
    def features(self) -> dict[str, dict[str, object]]:
        return self.meta.features

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, item: int) -> dict[str, Any]:
        source = self.base[item]
        forbidden = sorted(
            name
            for name in source
            if any(fragment in name.lower() for fragment in FORBIDDEN_BATCH_FRAGMENTS)
        )
        if forbidden:
            raise ValueError(f"policy sample contains forbidden teacher keys: {forbidden}")
        row = int(torch.as_tensor(source.get("index", -1)).item())
        try:
            token = self._tokens[row]
        except KeyError as error:
            raise ValueError(f"graph token cache does not contain global row {row}") from error
        result = dict(source)
        result[ENVIRONMENT_STATE_KEY] = torch.from_numpy(token.copy())
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

