import argparse
import json
from pathlib import Path
import sys
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BO workflow core engine CLI")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs"),
        help="Directory where run state and artifacts are stored",
    )

    sub = parser.add_subparsers(dest="command", required=True)

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
        choices=["hebo", "bo_lcb", "random"],
        default="hebo",
        help="Default optimizer engine for this run",
    )
    init_cmd.add_argument("--init-random", type=int, default=10)
    init_cmd.add_argument("--batch-size", type=int, default=1)
    init_cmd.add_argument("--max-categories", type=int, default=64)
    init_cmd.add_argument(
        "--intent-json",
        type=str,
        default=None,
        help="Optional JSON object or path to JSON object for original user intent",
    )
    init_cmd.add_argument("--verbose", action="store_true")

    oracle_cmd = sub.add_parser("build-oracle", help="Train and persist proxy oracle")
    oracle_cmd.add_argument("--run-id", type=str, required=True)
    oracle_cmd.add_argument("--cv-folds", type=int, default=5)
    oracle_cmd.add_argument("--max-features", type=int, default=None)
    oracle_cmd.add_argument("--verbose", action="store_true")

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

    run_cmd = sub.add_parser("run-proxy", help="Run iterative proxy optimization loop")
    run_cmd.add_argument("--run-id", type=str, required=True)
    run_cmd.add_argument("--iterations", type=int, required=True)
    run_cmd.add_argument("--batch-size", type=int, default=1)
    run_cmd.add_argument("--verbose", action="store_true")

    status_cmd = sub.add_parser("status", help="Show run status")
    status_cmd.add_argument("--run-id", type=str, required=True)

    report_cmd = sub.add_parser("report", help="Generate report and plot")
    report_cmd.add_argument("--run-id", type=str, required=True)
    report_cmd.add_argument("--verbose", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    engine = BOEngine(runs_root=args.runs_root)

    if args.command == "init":
        intent_payload = None
        if args.intent_json is not None:
            intent_payload = _parse_json_object(args.intent_json)
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
            intent=intent_payload,
            verbose=args.verbose,
        )
        _json_print(payload)
        return 0

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

    if args.command == "status":
        payload = engine.status(args.run_id)
        _json_print(payload)
        return 0

    if args.command == "report":
        payload = engine.report(args.run_id, verbose=args.verbose)
        _json_print(payload)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
