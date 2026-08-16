from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .pipeline import (
    cache_from_config,
    compare_from_config,
    evaluate_from_config,
    inspect_from_config,
    smoke_from_config,
)


class CLIUsageError(ValueError):
    pass


class JSONArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CLIUsageError(message)


def build_parser() -> argparse.ArgumentParser:
    parser = JSONArgumentParser(
        prog="python -m interaction_vla.graph_control",
        description="Paired Graph-conditioned ACT continuous-control experiment.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("inspect", "validate dataset, split, and Graph checkpoints"),
        ("cache", "freeze and cache all Graph control tokens"),
        ("smoke", "run one ACT update and reload per condition"),
        ("compare", "train the fixed-epoch paired ACT matrix"),
        ("evaluate", "run paired MuJoCo closed-loop evaluation"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--config", required=True, type=Path)
    return parser


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _dispatch(args: argparse.Namespace) -> Any:
    functions = {
        "inspect": inspect_from_config,
        "cache": cache_from_config,
        "smoke": smoke_from_config,
        "compare": compare_from_config,
        "evaluate": evaluate_from_config,
    }
    return functions[args.command](args.config)


def main(argv: Sequence[str] | None = None) -> None:
    try:
        args = build_parser().parse_args(argv)
        result = _dispatch(args)
    except Exception as error:
        print(
            json.dumps(
                {
                    "passed": False,
                    "error": type(error).__name__,
                    "message": str(error),
                },
                sort_keys=True,
            )
        )
        raise SystemExit(2 if isinstance(error, CLIUsageError) else 1) from error
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
    if isinstance(result, Mapping) and result.get("passed") is False:
        raise SystemExit(1)
