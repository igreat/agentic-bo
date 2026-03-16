"""Top-level CLI entrypoint — composes subcommands from each module."""

import argparse
from pathlib import Path
import sys

from .engine import BOEngine
from . import engine_cli
from .evaluation import cli as evaluation_cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BO workflow CLI")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("bo_runs"),
        help="Directory where run state and artifacts are stored",
    )
    parser.add_argument(
        "--backends-root",
        type=Path,
        default=Path("evaluation_backends"),
        help="Directory where evaluation backend artifacts are stored",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    engine_cli.register_commands(sub)
    evaluation_cli.register_commands(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    engine = BOEngine(runs_root=args.runs_root)

    for handler in (engine_cli.handle, evaluation_cli.handle):
        result = handler(args, engine)
        if result is not None:
            return result

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
