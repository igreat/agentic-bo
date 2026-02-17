"""CLI subcommands for the oracle layer: build-oracle, run-proxy."""

import argparse
import json
from typing import Any

from .engine import BOEngine


def _json_print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def register_commands(sub: argparse._SubParsersAction) -> None:
    """Register oracle subcommands on an existing subparsers group."""
    oracle_cmd = sub.add_parser("build-oracle", help="Train and persist proxy oracle")
    oracle_cmd.add_argument("--run-id", type=str, required=True)
    oracle_cmd.add_argument("--cv-folds", type=int, default=5)
    oracle_cmd.add_argument("--max-features", type=int, default=None)
    oracle_cmd.add_argument("--verbose", action="store_true")

    run_cmd = sub.add_parser("run-proxy", help="Run iterative proxy optimization loop")
    run_cmd.add_argument("--run-id", type=str, required=True)
    run_cmd.add_argument("--iterations", type=int, required=True)
    run_cmd.add_argument("--batch-size", type=int, default=1)
    run_cmd.add_argument("--verbose", action="store_true")


def handle(args: argparse.Namespace, engine: BOEngine) -> int | None:
    """Handle an oracle subcommand. Returns exit code, or None if not ours."""
    if args.command == "build-oracle":
        from .oracle import build_proxy_oracle

        run_dir = engine._paths(args.run_id).run_dir
        payload = build_proxy_oracle(
            run_dir,
            cv_folds=args.cv_folds,
            max_features=args.max_features,
            verbose=args.verbose,
        )
        _json_print(payload)
        return 0

    if args.command == "run-proxy":
        from .observers import ProxyObserver

        run_dir = engine._paths(args.run_id).run_dir
        observer = ProxyObserver(run_dir)
        payload = engine.run_optimization(
            args.run_id,
            observer=observer,
            num_iterations=args.iterations,
            batch_size=args.batch_size,
            verbose=args.verbose,
        )
        _json_print(payload)
        return 0

    return None
