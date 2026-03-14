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
    run_cmd.add_argument(
        "--seed-pool",
        type=str,
        default=None,
        help=(
            "Path to pool CSV. Injects all rows as initial observations so "
            "HEBO starts with real data instead of random sampling."
        ),
    )
    run_cmd.add_argument("--verbose", action="store_true")


def _validate_run_proxy_preconditions(
    engine: BOEngine, run_id: str, batch_size: int
) -> None:
    """Validate run-proxy preconditions before any state mutation happens."""
    state = engine._load_state(run_id)
    if state["status"] not in {"initialized", "oracle_ready", "running"}:
        raise ValueError(
            f"Run '{run_id}' is not ready for suggestions. "
            f"Current status: {state['status']}"
        )

    engine_name = str(state.get("default_engine", "hebo"))
    if engine_name == "bo_lcb" and int(batch_size) != 1:
        raise ValueError("bo_lcb currently supports batch-size=1 only.")


def _seed_pool_observations(
    engine: BOEngine, run_id: str, pool_path: str, verbose: bool
) -> int:
    """Inject all pool rows as initial observations so HEBO starts informed.

    Reads the pool CSV and feeds every (x, y) pair to the engine as
    source="pool-seed".  Returns the number of observations injected.
    """
    import sys

    import pandas as pd

    from .utils import read_json, to_python_scalar

    state = read_json(engine.get_run_dir(run_id) / "state.json")
    pool_df = pd.read_csv(pool_path)
    target_col = state["target_column"]
    active = list(state["active_features"])

    obs_list: list[dict] = []
    for _, row in pool_df.iterrows():
        y_val = row.get(target_col)
        if pd.isna(y_val):
            continue
        x = {}
        for f in active:
            if f in row.index and not pd.isna(row[f]):
                # Preserve original feature types (e.g., categorical labels)
                # and let engine.observe validate required schema.
                x[f] = to_python_scalar(row[f])
        obs_list.append({"x": x, "y": float(y_val)})

    if not obs_list:
        return 0

    engine.observe(run_id, obs_list, source="pool-seed", verbose=verbose)
    if verbose:
        print(
            f"[seed-pool] injected {len(obs_list)} pool observations",
            file=sys.stderr,
        )
    return len(obs_list)


def handle(args: argparse.Namespace, engine: BOEngine) -> int | None:
    """Handle an oracle subcommand. Returns exit code, or None if not ours."""
    if args.command == "build-oracle":
        from .oracle import build_proxy_oracle

        run_dir = engine.get_run_dir(args.run_id)
        payload = build_proxy_oracle(
            run_dir,
            cv_folds=args.cv_folds,
            max_features=args.max_features,
            verbose=args.verbose,
        )
        _json_print(payload)
        return 0

    if args.command == "run-proxy":
        from .observers.proxy import ProxyObserver

        run_dir = engine.get_run_dir(args.run_id)

        # Validate failure-prone preconditions before any seed writes happen.
        observer = ProxyObserver(run_dir)
        _validate_run_proxy_preconditions(engine, args.run_id, args.batch_size)

        # Optionally seed HEBO with real pool observations
        seed_pool = getattr(args, "seed_pool", None)
        if seed_pool:
            _seed_pool_observations(engine, args.run_id, seed_pool, args.verbose)

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
