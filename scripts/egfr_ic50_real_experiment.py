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
import csv
import json
from pathlib import Path
from typing import Any
import sys

import pandas as pd
from rdkit import Chem

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bo_workflow.engine import BOEngine


def _canonical(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _to_pic50(ic50_nm: float) -> float:
    import math

    if ic50_nm <= 0:
        raise ValueError("ic50_nM must be > 0")
    return 9.0 - math.log10(float(ic50_nm))


def _load_seed_rows(
    seed_csv: Path,
    smiles_column: str,
    target_column: str,
) -> list[tuple[str, float]]:
    frame = pd.read_csv(seed_csv)
    if smiles_column not in frame.columns:
        raise ValueError(f"Missing smiles column '{smiles_column}' in {seed_csv}")
    if target_column not in frame.columns:
        raise ValueError(f"Missing target column '{target_column}' in {seed_csv}")

    rows: list[tuple[str, float]] = []
    for _, row in frame.iterrows():
        smi = str(row[smiles_column]).strip()
        if not smi:
            continue
        can = _canonical(smi)
        if can is None:
            continue

        y_raw = row[target_column]
        if pd.isna(y_raw):
            continue
        rows.append((can, float(y_raw)))

    if len(rows) < 5:
        raise ValueError("Need at least 5 valid labeled seed rows.")
    return rows


def _load_candidate_smiles(candidate_csv: Path, smiles_column: str) -> list[str]:
    frame = pd.read_csv(candidate_csv)
    if smiles_column not in frame.columns:
        raise ValueError(f"Missing smiles column '{smiles_column}' in {candidate_csv}")

    smiles: list[str] = []
    seen: set[str] = set()
    for value in frame[smiles_column].dropna().tolist():
        smi = str(value).strip()
        if not smi:
            continue
        can = _canonical(smi)
        if can is None or can in seen:
            continue
        seen.add(can)
        smiles.append(can)
    return smiles


def _build_runtime_dataset(
    out_csv: Path,
    seed_rows: list[tuple[str, float]],
    candidate_smiles: list[str],
) -> dict[str, int]:
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    seed_map = {smi: y for smi, y in seed_rows}
    all_smiles = list(seed_map.keys())
    for smi in candidate_smiles:
        if smi not in seed_map:
            all_smiles.append(smi)

    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["smiles", "pIC50"])
        for smi in all_smiles:
            y = seed_map.get(smi)
            writer.writerow([smi, "" if y is None else y])

    return {
        "seed_count": len(seed_map),
        "candidate_count": len(candidate_smiles),
        "total_pool": len(all_smiles),
    }


def cmd_init(args: argparse.Namespace) -> int:
    engine = BOEngine()

    target_col = "pIC50"
    if args.seed_target == "pIC50":
        seed_rows = _load_seed_rows(args.seed_data, args.smiles_column, "pIC50")
    else:
        raw_rows = _load_seed_rows(args.seed_data, args.smiles_column, "ic50_nM")
        seed_rows = [(smi, _to_pic50(y)) for smi, y in raw_rows]

    candidate_smiles: list[str] = []
    if args.candidate_data is not None:
        candidate_smiles = _load_candidate_smiles(args.candidate_data, args.smiles_column)

    temp_dataset = Path("data") / f"egfr_real_runtime_{args.run_id or 'auto'}.csv"
    counts = _build_runtime_dataset(temp_dataset, seed_rows, candidate_smiles)

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


def _parse_observation_rows(
    obs_path: Path,
    target_column: str,
) -> list[dict[str, Any]]:
    frame = pd.read_csv(obs_path)

    required = {"smiles", target_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Observation CSV missing columns: {sorted(missing)}")

    has_suggestion_id = "suggestion_id" in frame.columns

    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        smi = str(row["smiles"]).strip()
        can = _canonical(smi)
        if can is None:
            continue

        y_raw = row[target_column]
        if pd.isna(y_raw):
            continue

        rec: dict[str, Any] = {
            "x": {"smiles": can},
            "y": float(y_raw),
        }
        if has_suggestion_id and pd.notna(row["suggestion_id"]):
            rec["suggestion_id"] = str(row["suggestion_id"])
        rows.append(rec)

    if not rows:
        raise ValueError("No valid observations parsed from CSV.")
    return rows


def cmd_observe(args: argparse.Namespace) -> int:
    engine = BOEngine()

    if args.target_type == "pIC50":
        obs_rows = _parse_observation_rows(args.data, "pIC50")
    else:
        raw = _parse_observation_rows(args.data, "ic50_nM")
        obs_rows = []
        for row in raw:
            obs_rows.append({**row, "y": _to_pic50(float(row["y"]))})

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
