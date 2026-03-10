"""CLI subcommands for the BO engine: init, suggest, observe, status, report."""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .engine import BOEngine


def _json_print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _parse_json_object(value: str) -> dict[str, Any]:
    path = Path(value)
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object.")
    return dict(payload)


def _parse_observation_records(value: str) -> list[dict[str, Any]]:
    path = Path(value)
    if path.exists():
        if path.suffix.lower() == ".json":
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, list):
                raise ValueError("JSON observation payload must be a list of objects.")
            return [dict(x) for x in payload]

        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
            if "y" not in frame.columns:
                raise ValueError("CSV observations must include a 'y' column.")
            rows: list[dict[str, Any]] = []
            for _, row in frame.iterrows():
                x = row.drop(labels=["y"]).to_dict()
                rows.append({"x": x, "y": float(row["y"])})
            return rows

    payload = json.loads(value)
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise ValueError("Inline observation payload must be JSON object or list.")
    return [dict(x) for x in payload]


def _parse_simplex_groups(raw: list[str] | None) -> list[dict[str, Any]]:
    """Parse --simplex-groups 'col1,col2,col3:100' into constraint dicts."""
    if not raw:
        return []
    result = []
    for entry in raw:
        if ":" not in entry:
            raise ValueError(
                f"--simplex-groups entry '{entry}' must be 'col1,col2,...:total' "
                "(columns separated by commas, then colon, then numeric total)."
            )
        cols_part, total_part = entry.rsplit(":", 1)
        cols = [c.strip() for c in cols_part.split(",") if c.strip()]
        if len(cols) < 2:
            raise ValueError(
                f"--simplex-groups entry '{entry}' must list at least 2 columns."
            )
        try:
            total = float(total_part.strip())
        except ValueError as exc:
            raise ValueError(
                f"--simplex-groups total '{total_part}' is not a valid number."
            ) from exc
        result.append({"type": "simplex", "cols": cols, "total": total})
    return result


def register_commands(sub: argparse._SubParsersAction) -> None:
    """Register engine subcommands on an existing subparsers group."""
    init_cmd = sub.add_parser("init", help="Initialize run from a dataset")
    init_cmd.add_argument("--dataset", type=Path, required=True)
    init_cmd.add_argument("--target", type=str, required=True)
    init_cmd.add_argument(
        "--objective", type=str, choices=["min", "max"], required=True
    )
    init_cmd.add_argument("--run-id", type=str, default=None)
    init_cmd.add_argument("--seed", type=int, default=7)
    init_cmd.add_argument(
        "--engine",
        type=str,
        choices=["hebo", "bo_lcb", "random", "botorch"],
        default="hebo",
        help="Default optimizer engine for this run",
    )
    init_cmd.add_argument("--init-random", type=int, default=10)
    init_cmd.add_argument("--batch-size", type=int, default=1)
    init_cmd.add_argument("--max-categories", type=int, default=64)
    init_cmd.add_argument(
        "--drop-cols",
        type=str,
        default=None,
        help="Comma-separated column names to exclude from the feature space (e.g. rxn_smiles)",
    )
    init_cmd.add_argument(
        "--simplex-groups",
        type=str,
        action="append",
        default=None,
        metavar="COLS:TOTAL",
        help=(
            "Declare a simplex constraint: comma-separated column names followed by "
            "a colon and the required sum (e.g. 'Metal_1,Metal_2,Metal_3:100'). "
            "Repeat the flag for multiple groups."
        ),
    )
    init_cmd.add_argument(
        "--intent-json",
        type=str,
        default=None,
        help="Optional JSON object or path to JSON object for original user intent",
    )
    init_cmd.add_argument("--verbose", action="store_true")

    suggest_cmd = sub.add_parser("suggest", help="Suggest next experimental candidates")
    suggest_cmd.add_argument("--run-id", type=str, required=True)
    suggest_cmd.add_argument("--batch-size", type=int, default=None)
    suggest_cmd.add_argument("--verbose", action="store_true")

    observe_cmd = sub.add_parser("observe", help="Record observation(s)")
    observe_cmd.add_argument("--run-id", type=str, required=True)
    observe_cmd.add_argument(
        "--data",
        type=str,
        required=True,
        help="Observations as JSON string/object/list, or path to CSV/JSON file",
    )
    observe_cmd.add_argument("--verbose", action="store_true")

    status_cmd = sub.add_parser("status", help="Show run status")
    status_cmd.add_argument("--run-id", type=str, required=True)

    report_cmd = sub.add_parser("report", help="Generate report and plot")
    report_cmd.add_argument("--run-id", type=str, required=True)
    report_cmd.add_argument("--verbose", action="store_true")


def handle(args: argparse.Namespace, engine: BOEngine) -> int | None:
    """Handle an engine subcommand. Returns exit code, or None if not ours."""
    if args.command == "init":
        intent_payload = None
        if args.intent_json is not None:
            intent_payload = _parse_json_object(args.intent_json)
        drop_cols = [c.strip() for c in args.drop_cols.split(",")] if args.drop_cols else []
        constraints = _parse_simplex_groups(args.simplex_groups)
        payload = engine.init_run(
            dataset_path=args.dataset,
            target_column=args.target,
            objective=args.objective,
            default_engine=args.engine,
            run_id=args.run_id,
            seed=args.seed,
            num_initial_random_samples=args.init_random,
            default_batch_size=args.batch_size,
            max_categories=args.max_categories,
            drop_cols=drop_cols,
            constraints=constraints or None,
            intent=intent_payload,
            verbose=args.verbose,
        )
        _json_print(payload)
        return 0

    if args.command == "suggest":
        payload = engine.suggest(
            args.run_id,
            batch_size=args.batch_size,
            verbose=args.verbose,
        )
        _json_print(payload)
        return 0

    if args.command == "observe":
        observations = _parse_observation_records(args.data)
        payload = engine.observe(args.run_id, observations, verbose=args.verbose)
        _json_print(payload)
        return 0

    if args.command == "status":
        payload = engine.status(args.run_id)
        _json_print(payload)
        return 0

    if args.command == "report":
        payload = engine.report(args.run_id, verbose=args.verbose)
        _json_print(payload)
        return 0

    return None
