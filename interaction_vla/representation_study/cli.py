from __future__ import annotations

import argparse
import json
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m interaction_vla.representation_study",
        description="SmolVLA LIBERO interaction-representation study",
    )
    families = parser.add_subparsers(dest="family", required=True)
    from .libero.cli import add_libero_parser

    add_libero_parser(families)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        from .libero.cli import dispatch

        result = dispatch(args)
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
        raise SystemExit(1) from error
    print(json.dumps(result, indent=2, sort_keys=True))
    if result.get("passed") is False:
        raise SystemExit(1)
