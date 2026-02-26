#!/usr/bin/env python3
"""EGFR real-experiment workflow (human-in-the-loop, no lookup mapping).

This script is designed for real laboratory usage:
- Initializes a run from labeled seed data (+ optional unlabeled candidate pool)
- Suggests next molecules to test
- Records real observations from experiments
- Retrains oracle after each observation batch

Unlike `egfr_ic50_global_experiment.py`, this script does NOT use lookup-table
simulation and does NOT map predictions back to a fixed reference library.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bo_workflow.engine import BOEngine
from bo_workflow.workflows.smiles_workflow_common import (
    build_runtime_dataset,
    load_candidate_smiles,
    load_seed_rows,
    parse_observation_rows,
    to_pic50,
)


def cmd_init(args: argparse.Namespace) -> int:
    engine = BOEngine()

    target_col = "pIC50"
    if args.seed_target == "pIC50":
        seed_rows = load_seed_rows(args.seed_data, args.smiles_column, "pIC50")
    else:
        raw_rows = load_seed_rows(args.seed_data, args.smiles_column, "ic50_nM")
        seed_rows = [(smi, to_pic50(y)) for smi, y in raw_rows]

    candidate_smiles: list[str] = []
    if args.candidate_data is not None:
        candidate_smiles = load_candidate_smiles(args.candidate_data, args.smiles_column)

    temp_dataset = Path("data") / f"egfr_real_runtime_{args.run_id or 'auto'}.csv"
    counts = build_runtime_dataset(temp_dataset, seed_rows, candidate_smiles)

    init_state = engine.init_smiles_run(
        dataset_path=temp_dataset,
        target_column=target_col,
        objective="max",
        smiles_column="smiles",
        default_engine=args.engine,
        run_id=args.run_id,
        num_initial_random_samples=args.init_random,
        default_batch_size=args.batch_size,
        seed=args.seed,
        fingerprint_bits=args.fp_bits,
        energy_backend=args.energy_backend,
        discovery=False,
        verbose=args.verbose,
    )
    run_id = init_state["run_id"]

    oracle = engine.build_oracle(
        run_id,
        cv_folds=args.cv_folds,
        max_features=args.max_features,
        verbose=args.verbose,
    )

    payload = {
        "mode": "egfr_real_experiment",
        "note": "No lookup mapping. Suggestions are for real lab evaluation.",
        "run_id": run_id,
        "runtime_dataset": str(temp_dataset.resolve()),
        "counts": counts,
        "oracle": {
            "selected_model": oracle["selected_model"],
            "selected_rmse": oracle["selected_rmse"],
            "cv_rmse": oracle["cv_rmse"],
        },
    }
    print(json.dumps(payload, indent=2))
    return 0


def cmd_suggest(args: argparse.Namespace) -> int:
    engine = BOEngine()
    result = engine.suggest(args.run_id, batch_size=args.batch_size, verbose=args.verbose)

    rows = []
    for s in result.get("suggestions", []):
        x = s.get("x", {})
        rows.append(
            {
                "suggestion_id": s.get("suggestion_id"),
                "smiles": x.get("smiles"),
                "engine": s.get("engine"),
                "iteration": s.get("iteration"),
            }
        )

    out_csv = Path("runs") / args.run_id / "lab_suggestions.csv"
    if rows:
        pd.DataFrame(rows).to_csv(out_csv, index=False)

    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "num_suggestions": len(rows),
                "suggestions_csv": str(out_csv.resolve()),
                "suggestions": rows,
            },
            indent=2,
        )
    )
    return 0


def cmd_observe(args: argparse.Namespace) -> int:
    engine = BOEngine()

    if args.target_type == "pIC50":
        obs_rows = parse_observation_rows(args.data, "pIC50")
    else:
        raw = parse_observation_rows(args.data, "ic50_nM")
        obs_rows = []
        for row in raw:
            obs_rows.append({**row, "y": to_pic50(float(row["y"]))})

    result = engine.observe(
        args.run_id,
        obs_rows,
        source="real_experiment",
        verbose=args.verbose,
    )

    oracle = engine.build_oracle(
        args.run_id,
        cv_folds=args.cv_folds,
        max_features=args.max_features,
        verbose=args.verbose,
    )

    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "recorded": result.get("recorded"),
                "oracle": {
                    "selected_model": oracle["selected_model"],
                    "selected_rmse": oracle["selected_rmse"],
                },
            },
            indent=2,
        )
    )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    engine = BOEngine()
    print(json.dumps(engine.status(args.run_id), indent=2))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    engine = BOEngine()
    print(json.dumps(engine.report(args.run_id, verbose=args.verbose), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="EGFR real-experiment workflow (no lookup mapping)")
    sub = p.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="Initialize real-experiment run from seed labels")
    init.add_argument("--seed-data", type=Path, required=True, help="CSV with labeled seed molecules")
    init.add_argument("--candidate-data", type=Path, default=None, help="Optional CSV with candidate molecules (smiles column)")
    init.add_argument("--smiles-column", type=str, default="smiles")
    init.add_argument("--seed-target", type=str, choices=["pIC50", "ic50_nM"], default="pIC50")
    init.add_argument("--run-id", type=str, default=None)
    init.add_argument("--engine", type=str, choices=["hebo", "bo_lcb", "random"], default="hebo")
    init.add_argument("--seed", type=int, default=42)
    init.add_argument("--init-random", type=int, default=10)
    init.add_argument("--batch-size", type=int, default=1)
    init.add_argument("--fp-bits", type=int, default=256)
    init.add_argument("--energy-backend", type=str, choices=["none", "xtb", "auto", "ml", "dft"], default="none")
    init.add_argument("--cv-folds", type=int, default=5)
    init.add_argument("--max-features", type=int, default=None)
    init.add_argument("--verbose", action="store_true")

    suggest = sub.add_parser("suggest", help="Suggest next molecules for real experiments")
    suggest.add_argument("--run-id", type=str, required=True)
    suggest.add_argument("--batch-size", type=int, default=4)
    suggest.add_argument("--verbose", action="store_true")

    observe = sub.add_parser("observe", help="Record real experimental observations and retrain oracle")
    observe.add_argument("--run-id", type=str, required=True)
    observe.add_argument("--data", type=Path, required=True, help="CSV with columns smiles + pIC50|ic50_nM (+ optional suggestion_id)")
    observe.add_argument("--target-type", type=str, choices=["pIC50", "ic50_nM"], default="pIC50")
    observe.add_argument("--cv-folds", type=int, default=5)
    observe.add_argument("--max-features", type=int, default=None)
    observe.add_argument("--verbose", action="store_true")

    status = sub.add_parser("status", help="Show current run status")
    status.add_argument("--run-id", type=str, required=True)

    report = sub.add_parser("report", help="Generate report")
    report.add_argument("--run-id", type=str, required=True)
    report.add_argument("--verbose", action="store_true")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "init":
        return cmd_init(args)
    if args.command == "suggest":
        return cmd_suggest(args)
    if args.command == "observe":
        return cmd_observe(args)
    if args.command == "status":
        return cmd_status(args)
    if args.command == "report":
        return cmd_report(args)
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
