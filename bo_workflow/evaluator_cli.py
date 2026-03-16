"""CLI subcommand for an operator-owned hidden evaluator loop."""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .engine import BOEngine
from .oracle import predict_original_scale
from .utils import RunPaths, read_json, read_jsonl, utc_now_iso


def _json_print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def register_commands(sub: argparse._SubParsersAction) -> None:
    """Register hidden-evaluator subcommands on an existing subparsers group."""
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
            "Harness oracle objective does not match the current run objective."
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
    """Handle an evaluator subcommand. Returns exit code, or None if not ours."""
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
