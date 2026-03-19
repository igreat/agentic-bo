"""CLI subcommands for the evaluation layer."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from ..engine import BOEngine
from ..utils import (
    EvaluationBackendPaths,
    read_jsonl,
    to_python_scalar,
    utc_now_iso,
)
from .oracle import build_proxy_oracle, predict_original_scale, read_backend_meta
from .python_module import run_python_module_evaluator
from .proxy import ProxyObserver


def _json_print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def register_commands(sub: argparse._SubParsersAction) -> None:
    """Register evaluation subcommands on an existing subparsers group."""
    oracle_cmd = sub.add_parser("build-oracle", help="Train and persist proxy oracle")
    oracle_cmd.add_argument("--dataset", type=Path, required=True)
    oracle_cmd.add_argument("--target", type=str, required=True)
    oracle_cmd.add_argument("--objective", choices=["min", "max"], required=True)
    oracle_cmd.add_argument("--backend-id", type=str, required=True)
    oracle_cmd.add_argument(
        "--drop-cols",
        nargs="*",
        default=None,
        help="Columns to exclude from backend training before inferring features.",
    )
    oracle_cmd.add_argument("--seed", type=int, default=7)
    oracle_cmd.add_argument(
        "--engine",
        choices=["hebo", "bo_lcb", "random", "botorch"],
        default="hebo",
        help="Default engine metadata attached to the backend.",
    )
    oracle_cmd.add_argument("--cv-folds", type=int, default=5)
    oracle_cmd.add_argument(
        "--max-features",
        type=int,
        default=None,
        help=(
            "Limit backend features for high-dimensional datasets. Avoid this "
            "for simplex-constrained composition runs unless you are sure the "
            "reduced backend still covers every constrained column."
        ),
    )
    oracle_cmd.add_argument("--verbose", action="store_true")

    run_proxy_cmd = sub.add_parser(
        "run-proxy", help="Run iterative proxy optimization loop"
    )
    run_proxy_cmd.add_argument("--run-id", type=str, required=True)
    run_proxy_cmd.add_argument(
        "--backend-id",
        type=str,
        default=None,
        help="Backend id under <backends-root>; defaults to the run id.",
    )
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
        "--backend-id",
        type=str,
        required=True,
        help="Backend id under <backends-root> to use for hidden evaluation.",
    )
    run_cmd.add_argument("--iterations", type=int, required=True)
    run_cmd.add_argument("--batch-size", type=int, default=1)
    run_cmd.add_argument("--verbose", action="store_true")

    python_cmd = sub.add_parser(
        "run-python-evaluator",
        help="Run suggest/observe loop against a local Python evaluator module",
    )
    python_cmd.add_argument("--run-id", type=str, required=True)
    python_cmd.add_argument(
        "--module-path",
        type=Path,
        required=True,
        help="Path to a Python module exposing an evaluator function.",
    )
    python_cmd.add_argument(
        "--function",
        type=str,
        default="evaluate",
        help="Name of the evaluator function inside the module.",
    )
    python_cmd.add_argument("--iterations", type=int, required=True)
    python_cmd.add_argument("--batch-size", type=int, default=1)
    python_cmd.add_argument("--verbose", action="store_true")


def _resolve_backend_dir(
    backends_root: Path, *, run_id: str, backend_id: str | None
) -> Path:
    resolved_id = backend_id or run_id
    return backends_root / resolved_id


def _validate_backend_compatibility(
    state: dict[str, Any], backend_meta: dict[str, Any]
) -> None:
    oracle_features = list(backend_meta.get("active_features", []))
    run_features = set(state.get("active_features", []))
    run_features.update(str(name) for name in state.get("fixed_features", {}).keys())
    missing = [name for name in oracle_features if name not in run_features]
    if missing:
        raise ValueError(
            f"Current run is missing oracle-required features: {missing}"
        )

    constrained_missing: list[str] = []
    for constraint in state.get("constraints", []):
        if constraint.get("type") != "simplex":
            continue
        for col in constraint.get("cols", []):
            if col not in oracle_features and col not in constrained_missing:
                constrained_missing.append(str(col))
    if constrained_missing:
        raise ValueError(
            "Backend is incompatible with the current run's simplex constraints. "
            f"Backend is missing constrained columns: {constrained_missing}. "
            "Rebuild the backend without dropping those columns."
        )


def _validate_run_proxy_preconditions(
    engine: BOEngine,
    run_id: str,
    batch_size: int,
    backend_meta: dict[str, Any],
) -> None:
    """Validate run-proxy preconditions before any state mutation happens."""
    state = engine._load_state(run_id)
    if state["status"] not in {"initialized", "running"}:
        raise ValueError(
            f"Run '{run_id}' is not ready for suggestions. "
            f"Current status: {state['status']}"
        )

    engine_name = str(state.get("default_engine", "hebo"))
    if engine_name == "bo_lcb" and int(batch_size) != 1:
        raise ValueError("bo_lcb currently supports batch-size=1 only.")

    _validate_backend_compatibility(state, backend_meta)


def _seed_pool_observations(
    engine: BOEngine, run_id: str, pool_path: str, verbose: bool
) -> int:
    """Inject all pool rows as initial observations so HEBO starts informed."""
    state = engine._load_state(run_id)
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


def _attach_backend_summary(
    engine: BOEngine,
    run_id: str,
    backend_meta: dict[str, Any],
) -> None:
    """Copy non-pointer backend summary into run state for reporting."""
    state = engine._load_state(run_id)
    state["oracle"] = {
        "source": "evaluation_backend_metadata",
        "selected_model": backend_meta["selected_model"],
        "selected_rmse": backend_meta["selected_rmse"],
        "cv_rmse": backend_meta.get("cv_rmse", {}),
        "active_features": backend_meta.get("active_features", []),
    }
    state["updated_at"] = utc_now_iso()
    engine._save_state(run_id, state)


def _validate_evaluator_preconditions(
    engine: BOEngine,
    run_id: str,
    backend_dir: Path,
    batch_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    state = engine._load_state(run_id)
    if state["status"] not in {"initialized", "running"}:
        raise ValueError(
            f"Run '{run_id}' is not ready for suggestions. Current status: {state['status']}"
        )

    engine_name = str(state.get("default_engine", "hebo"))
    if engine_name == "bo_lcb" and int(batch_size) != 1:
        raise ValueError("bo_lcb currently supports batch-size=1 only.")

    backend_paths = EvaluationBackendPaths(backend_dir=backend_dir)
    if not backend_paths.oracle_model.exists():
        raise FileNotFoundError(f"Oracle model not found at {backend_paths.oracle_model}")
    if not backend_paths.oracle_meta.exists():
        raise FileNotFoundError(
            f"Oracle metadata not found at {backend_paths.oracle_meta}"
        )

    backend_meta = read_backend_meta(backend_dir)

    if backend_meta.get("objective") != state.get("objective"):
        raise ValueError(
            "Evaluator oracle objective does not match the current run objective."
        )

    _validate_backend_compatibility(state, backend_meta)

    return state, backend_meta


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
    backend_dir: Path,
    backend_objective: str,
    backend_features: list[str],
    suggestions: list[dict[str, Any]],
    default_engine: str,
) -> list[dict[str, Any]]:
    x_df = pd.DataFrame([row["x"] for row in suggestions])[backend_features]
    y_pred = predict_original_scale(backend_dir, backend_objective, x_df)

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
    backend_dir: str | Path,
    num_iterations: int,
    batch_size: int = 1,
    verbose: bool = False,
) -> dict[str, Any]:
    backend_dir = Path(backend_dir)
    state, backend_meta = _validate_evaluator_preconditions(
        engine, run_id, backend_dir, batch_size
    )
    _attach_backend_summary(engine, run_id, backend_meta)
    backend_features = list(backend_meta.get("active_features", []))
    backend_objective = str(backend_meta["objective"])
    default_engine = str(state.get("default_engine", "hebo"))

    recorded = 0
    pending = _pending_suggestions(engine, run_id)
    if pending:
        observations = _evaluate_suggestions(
            backend_dir=backend_dir,
            backend_objective=backend_objective,
            backend_features=backend_features,
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
            backend_dir=backend_dir,
            backend_objective=backend_objective,
            backend_features=backend_features,
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
        "backend_id": backend_dir.name,
        "backend_dir": str(backend_dir),
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
        backend_dir = args.backends_root / args.backend_id
        payload = build_proxy_oracle(
            dataset_path=args.dataset,
            target_column=args.target,
            objective=args.objective,
            backend_dir=backend_dir,
            drop_cols=getattr(args, "drop_cols", None),
            seed=int(getattr(args, "seed", 7)),
            default_engine=str(getattr(args, "engine", "hebo")),
            cv_folds=args.cv_folds,
            max_features=args.max_features,
            verbose=args.verbose,
        )
        _json_print(payload)
        return 0

    if args.command == "run-proxy":
        backend_dir = _resolve_backend_dir(
            args.backends_root,
            run_id=args.run_id,
            backend_id=getattr(args, "backend_id", None),
        )
        backend_meta = read_backend_meta(backend_dir)

        observer = ProxyObserver(backend_dir)
        _validate_run_proxy_preconditions(
            engine,
            args.run_id,
            args.batch_size,
            backend_meta,
        )
        _attach_backend_summary(engine, args.run_id, backend_meta)

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
        backend_dir = _resolve_backend_dir(
            args.backends_root,
            run_id=args.run_id,
            backend_id=args.backend_id,
        )
        payload = run_hidden_oracle_evaluator(
            engine,
            run_id=args.run_id,
            backend_dir=backend_dir,
            num_iterations=args.iterations,
            batch_size=args.batch_size,
            verbose=args.verbose,
        )
        _json_print(payload)
        return 0

    if args.command == "run-python-evaluator":
        payload = run_python_module_evaluator(
            engine,
            run_id=args.run_id,
            module_path=args.module_path,
            function_name=args.function,
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
    parser.add_argument(
        "--backends-root",
        type=Path,
        default=Path("evaluation_backends"),
        help="Directory where evaluation backend artifacts are stored",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    register_commands(sub)
    args = parser.parse_args(argv)
    exit_code = handle(args, BOEngine())
    return int(exit_code or 0)
