"""CLI subcommands for the evaluation layer."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from ..engine import BOEngine
from ..utils import RunPaths, read_json, read_jsonl, to_python_scalar, utc_now_iso
from .oracle import predict_original_scale
from .proxy import ProxyObserver


def _json_print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def register_commands(sub: argparse._SubParsersAction) -> None:
    """Register evaluation subcommands on an existing subparsers group."""
    oracle_cmd = sub.add_parser("build-oracle", help="Train and persist proxy oracle")
    oracle_cmd.add_argument("--run-id", type=str, required=True)
    oracle_cmd.add_argument("--cv-folds", type=int, default=5)
    oracle_cmd.add_argument("--max-features", type=int, default=None)
    oracle_cmd.add_argument("--verbose", action="store_true")

    run_proxy_cmd = sub.add_parser(
        "run-proxy", help="Run iterative proxy optimization loop"
    )
    run_proxy_cmd.add_argument("--run-id", type=str, required=True)
    run_proxy_cmd.add_argument("--iterations", type=int, required=True)
    run_proxy_cmd.add_argument("--batch-size", type=int, default=1)
    run_proxy_cmd.add_argument(
        "--seed-pool",
        type=str,
        default=None,
        help=(
            "Path to pool CSV. Injects all rows as initial observations so "
            "HEBO starts with real data instead of random sampling."
        ),
    )
    run_proxy_cmd.add_argument("--verbose", action="store_true")

    run_cmd = sub.add_parser(
        "run-evaluator",
        help="Run suggest/observe loop against an external oracle/backend",
    )
    run_cmd.add_argument("--run-id", type=str, required=True)
    run_cmd.add_argument(
        "--oracle-dir",
        type=Path,
        required=True,
        help="Directory containing oracle.pkl, oracle_meta.json, and matching state.json",
    )
    run_cmd.add_argument("--iterations", type=int, required=True)
    run_cmd.add_argument("--batch-size", type=int, default=1)
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
    """Inject all pool rows as initial observations so HEBO starts informed."""
    state = read_json(engine.get_run_dir(run_id) / "state.json")
    pool_df = pd.read_csv(pool_path)
    target_col = state["target_column"]
    active = list(state["active_features"])

    obs_list: list[dict[str, Any]] = []
    for _, row in pool_df.iterrows():
        y_val = row.get(target_col)
        if pd.isna(y_val):
            continue
        x: dict[str, Any] = {}
        for feature in active:
            if feature in row.index and not pd.isna(row[feature]):
                x[feature] = to_python_scalar(row[feature])
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


def _validate_evaluator_preconditions(
    engine: BOEngine,
    run_id: str,
    oracle_dir: Path,
    batch_size: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    state = engine._load_state(run_id)
    if state["status"] not in {"initialized", "oracle_ready", "running"}:
        raise ValueError(
            f"Run '{run_id}' is not ready for suggestions. Current status: {state['status']}"
        )

    engine_name = str(state.get("default_engine", "hebo"))
    if engine_name == "bo_lcb" and int(batch_size) != 1:
        raise ValueError("bo_lcb currently supports batch-size=1 only.")

    paths = RunPaths(run_dir=oracle_dir)
    if not paths.state.exists():
        raise FileNotFoundError(f"Oracle state not found at {paths.state}")
    if not paths.oracle_model.exists():
        raise FileNotFoundError(f"Oracle model not found at {paths.oracle_model}")
    if not paths.oracle_meta.exists():
        raise FileNotFoundError(f"Oracle metadata not found at {paths.oracle_meta}")

    oracle_state = read_json(paths.state)
    oracle_meta = read_json(paths.oracle_meta)

    if oracle_state.get("objective") != state.get("objective"):
        raise ValueError(
            "Evaluator oracle objective does not match the current run objective."
        )

    oracle_features = list(oracle_meta.get("active_features", []))
    run_features = set(state.get("active_features", []))
    missing = [name for name in oracle_features if name not in run_features]
    if missing:
        raise ValueError(
            f"Current run is missing oracle-required features: {missing}"
        )

    return state, oracle_state, oracle_meta


def _pending_suggestions(engine: BOEngine, run_id: str) -> list[dict[str, Any]]:
    """Return suggestions that were logged but not yet observed."""
    paths = engine._paths(run_id)
    suggestions = read_jsonl(paths.suggestions)
    observations = read_jsonl(paths.observations)
    observed_ids = {
        str(row["suggestion_id"])
        for row in observations
        if row.get("suggestion_id") is not None
    }
    return [
        row
        for row in suggestions
        if row.get("suggestion_id") is not None
        and str(row["suggestion_id"]) not in observed_ids
    ]


def _evaluate_suggestions(
    *,
    oracle_dir: Path,
    oracle_state: dict[str, Any],
    oracle_features: list[str],
    suggestions: list[dict[str, Any]],
    default_engine: str,
) -> list[dict[str, Any]]:
    x_df = pd.DataFrame([row["x"] for row in suggestions])[oracle_features]
    y_pred = predict_original_scale(oracle_dir, oracle_state, x_df)

    observations = []
    for suggestion, y_value in zip(suggestions, y_pred, strict=True):
        observations.append(
            {
                "x": suggestion["x"],
                "y": float(y_value),
                "engine": suggestion.get("engine", default_engine),
                "suggestion_id": suggestion.get("suggestion_id"),
            }
        )
    return observations


def run_hidden_oracle_evaluator(
    engine: BOEngine,
    *,
    run_id: str,
    oracle_dir: str | Path,
    num_iterations: int,
    batch_size: int = 1,
    verbose: bool = False,
) -> dict[str, Any]:
    oracle_dir = Path(oracle_dir)
    state, oracle_state, oracle_meta = _validate_evaluator_preconditions(
        engine, run_id, oracle_dir, batch_size
    )
    oracle_features = list(oracle_meta.get("active_features", []))
    default_engine = str(state.get("default_engine", "hebo"))

    recorded = 0
    pending = _pending_suggestions(engine, run_id)
    if pending:
        observations = _evaluate_suggestions(
            oracle_dir=oracle_dir,
            oracle_state=oracle_state,
            oracle_features=oracle_features,
            suggestions=pending,
            default_engine=default_engine,
        )
        engine.observe(
            run_id,
            observations,
            source="benchmark-evaluator",
            verbose=verbose,
        )
        recorded += len(observations)

    for _ in range(int(num_iterations)):
        suggestions_payload = engine.suggest(
            run_id, batch_size=int(batch_size), verbose=verbose
        )
        suggestions = suggestions_payload["suggestions"]
        observations = _evaluate_suggestions(
            oracle_dir=oracle_dir,
            oracle_state=oracle_state,
            oracle_features=oracle_features,
            suggestions=suggestions,
            default_engine=default_engine,
        )

        engine.observe(
            run_id,
            observations,
            source="benchmark-evaluator",
            verbose=verbose,
        )
        recorded += len(observations)

    if int(num_iterations) > 0:
        updated = engine._load_state(run_id)
        updated["status"] = "completed"
        updated["updated_at"] = utc_now_iso()
        engine._save_state(run_id, updated)

    report = engine.report(run_id, verbose=verbose)
    return {
        "run_id": run_id,
        "oracle_dir": str(oracle_dir),
        "iterations": int(num_iterations),
        "batch_size": int(batch_size),
        "recorded": recorded,
        "resolved_pending": len(pending),
        "best_value": report.get("best_value"),
        "best_iteration": report.get("best_iteration"),
        "report_path": str(engine._paths(run_id).report),
        "convergence_plot_path": str(engine._paths(run_id).convergence_plot),
    }


def handle(args: argparse.Namespace, engine: BOEngine) -> int | None:
    """Handle an evaluation subcommand. Returns exit code, or None if not ours."""
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
        run_dir = engine.get_run_dir(args.run_id)

        observer = ProxyObserver(run_dir)
        _validate_run_proxy_preconditions(engine, args.run_id, args.batch_size)

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

    if args.command == "run-evaluator":
        payload = run_hidden_oracle_evaluator(
            engine,
            run_id=args.run_id,
            oracle_dir=args.oracle_dir,
            num_iterations=args.iterations,
            batch_size=args.batch_size,
            verbose=args.verbose,
        )
        _json_print(payload)
        return 0

    return None


def main(argv: list[str] | None = None) -> int:
    """Standalone entrypoint for evaluation-only commands."""
    parser = argparse.ArgumentParser(prog="python -m bo_workflow.evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    register_commands(sub)
    args = parser.parse_args(argv)
    exit_code = handle(args, BOEngine())
    return int(exit_code or 0)
