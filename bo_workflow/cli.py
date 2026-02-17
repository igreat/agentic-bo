"""Top-level CLI entrypoint — composes subcommands from each module."""

import argparse
from pathlib import Path
import sys

from .engine import BOEngine
from . import engine_cli, oracle_cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BO workflow CLI")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs"),
        help="Directory where run state and artifacts are stored",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    engine_cli.register_commands(sub)
    oracle_cli.register_commands(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    engine = BOEngine(runs_root=args.runs_root)

    for handler in (engine_cli.handle, oracle_cli.handle):
        result = handler(args, engine)
        if result is not None:
            return result

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
