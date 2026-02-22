#!/usr/bin/env python3
"""Global EGFR IC50 experiment with representative train/lookup split.

Workflow:
1) Read full EGFR dataset (ic50_nM or pIC50), convert to pIC50 if needed.
2) Build initial train set either:
    - auto representative split (default), or
    - from user-provided seed SMILES CSV.
    Train size defaults to 50.
3) Run the same experiment loop style as lookup_experiment (HEBO + A/B/C/D prescreen).
4) Evaluate selected candidates against full dataset as simulation lookup.

All results are simulated proxy-oracle experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.bo_workflow.engine import BOEngine
from scripts.egfr_lookup_experiment import (
    _canonical,
    _tanimoto_fp,
    prescreen_candidates,
    select_diverse_train,
)


def _to_pic50(ic50_nm: float) -> float:
    if ic50_nm <= 0:
        raise ValueError("ic50_nM must be > 0")
    return 9.0 - math.log10(float(ic50_nm))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="EGFR global ic50 experiment with representative split")
    p.add_argument("--dataset", type=Path, default=Path("data/egfr_ic50.csv"))
    p.add_argument("--target-column", default="ic50_nM", choices=["ic50_nM", "pIC50"])
    p.add_argument("--split-mode", default="auto", choices=["auto", "from-seed-csv"])
    p.add_argument("--seed-smiles-csv", type=Path, help="CSV with user seed smiles (required when --split-mode from-seed-csv)")
    p.add_argument("--seed-smiles-column", default="smiles")
    p.add_argument("--seed-count", type=int, default=50)

    p.add_argument("--iterations", type=int, default=30)
    p.add_argument("--rounds", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--experiments-per-round", type=int, default=4)
    p.add_argument("--fp-bits", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ucb-beta", type=float, default=2.0)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def _load_full_dataset(dataset_path: Path, target_column: str):
    with open(dataset_path) as f:
        rows = list(csv.DictReader(f))

    data = []
    for r in rows:
        smi = r.get("smiles", "")
        can = _canonical(smi)
        if not can:
            continue

        if target_column == "pIC50":
            y = float(r["pIC50"])
        else:
            y = _to_pic50(float(r["ic50_nM"]))

        data.append((smi, can, y))

    if not data:
        raise ValueError("No valid molecules found in dataset")
    return data


def _load_seed_smiles(path: Path, column: str, seed_count: int):
    with open(path) as f:
        rows = list(csv.DictReader(f))

    seeds = []
    seen = set()
    for r in rows:
        smi = r.get(column, "")
        can = _canonical(smi)
        if not can or can in seen:
            continue
        seen.add(can)
        seeds.append((smi, can))
        if len(seeds) >= seed_count:
            break

    if not seeds:
        raise ValueError("No valid seed smiles found")
    return seeds


def _select_representative_split(full_data, seed_count: int, seed: int):
    all_smiles = [s for s, _, _ in full_data]
    all_targets = [y for _, _, y in full_data]

    train_idx, lookup_idx = select_diverse_train(all_smiles, all_targets, seed_count, seed)

    train_pairs = [full_data[i] for i in train_idx]
    lookup_pairs = [full_data[i] for i in lookup_idx]
    return train_pairs, lookup_pairs


def main(argv=None) -> int:
    args = parse_args(argv)

    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    if args.split_mode == "from-seed-csv":
        if args.seed_smiles_csv is None:
            raise ValueError("--seed-smiles-csv is required when --split-mode from-seed-csv")
        if not args.seed_smiles_csv.exists():
            raise FileNotFoundError(f"Seed CSV not found: {args.seed_smiles_csv}")

    full_data = _load_full_dataset(args.dataset, args.target_column)

    if args.split_mode == "from-seed-csv":
        seed_smiles = _load_seed_smiles(args.seed_smiles_csv, args.seed_smiles_column, args.seed_count)
    else:
        seed_smiles = []

    full_by_can = {}
    full_smi_by_can = {}
    for smi, can, y in full_data:
        full_by_can[can] = y
        full_smi_by_can[can] = smi

    if args.split_mode == "from-seed-csv":
        train_pairs = []
        for _, can in seed_smiles:
            if can in full_by_can:
                train_pairs.append((full_smi_by_can[can], can, full_by_can[can]))

        if len(train_pairs) < 5:
            raise ValueError(f"Only {len(train_pairs)} seed molecules overlap with dataset; need at least 5")

        train_pairs = train_pairs[: args.seed_count]
        train_can = {can for _, can, _ in train_pairs}
        lookup_pairs = [row for row in full_data if row[1] not in train_can]
    else:
        train_pairs, lookup_pairs = _select_representative_split(full_data, args.seed_count, args.seed)

    print(f"Loaded full dataset: {len(full_by_can)} molecules")
    print(f"Split mode: {args.split_mode}")
    print(f"Initial train set: {len(train_pairs)} molecules")
    print(f"Lookup set: {len(lookup_pairs)} molecules")

    # Build full lookup table and exploration pool
    lookup_table = {}
    for smi, can, y in lookup_pairs:
        lookup_table[can] = y
    for smi, can, y in train_pairs:
        lookup_table[can] = y
    all_pool_smiles = list({smi for smi, _, _ in full_data})

    # Write pIC50 normalized full dataset for BOEngine init
    normalized_dataset_csv = Path("data/egfr_ic50_normalized_pic50.csv")
    with open(normalized_dataset_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["smiles", "pIC50"])
        for smi, _, y in full_data:
            w.writerow([smi, y])

    split_suffix = "auto" if args.split_mode == "auto" else "seeded"
    train_csv = Path(f"data/egfr_ic50_train_{split_suffix}.csv")
    with open(train_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["smiles", "pIC50"])
        for smi, _, y in train_pairs:
            w.writerow([smi, y])

    lookup_csv = Path(f"data/egfr_ic50_lookup_{split_suffix}.csv")
    with open(lookup_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["smiles", "pIC50"])
        for smi, _, y in lookup_pairs:
            w.writerow([smi, y])

    engine = BOEngine()
    init_result = engine.init_smiles_run(
        dataset_path=str(train_csv),
        target_column="pIC50",
        objective="max",
        discovery=True,
        seed=args.seed,
        fingerprint_bits=args.fp_bits,
        verbose=args.verbose,
    )
    run_id = init_result["run_id"]

    state_path = Path(f"runs/{run_id}/state.json")
    state = json.loads(state_path.read_text())
    state["_full_dataset"] = state.get("dataset_path", state.get("dataset"))
    state["dataset_path"] = str(train_csv.resolve())
    state_path.write_text(json.dumps(state, indent=2))

    train_set = {can for _, can, _ in train_pairs}
    observed_smiles = set(train_set)
    observed_results = {can: y for _, can, y in train_pairs}
    observed_fps = {}
    for can in observed_smiles:
        fp = _tanimoto_fp(can)
        if fp is not None:
            observed_fps[can] = fp

    round_results = []
    discoveries_order = []
    best_found_so_far = -999.0
    rounds_since_improvement = 0

    for round_num in range(1, args.rounds + 1):
        print()
        print("=" * 70)
        print(f"  ROUND {round_num} / {args.rounds}")
        print("=" * 70)

        if round_num > 1:
            prev = round_results[-1]
            with open(train_csv, "a", newline="") as f:
                w = csv.writer(f)
                for m in prev["matches"]:
                    w.writerow([m["smiles"], m["real_pIC50"]])
                for m in prev["misses"]:
                    w.writerow([m["smiles"], m["penalty_y"]])

            state = json.loads(state_path.read_text())
            state["status"] = "running"
            state_path.write_text(json.dumps(state, indent=2))

        oracle = engine.build_oracle(run_id, verbose=args.verbose)
        print(f"  Oracle model={oracle['selected_model']} RMSE={oracle['selected_rmse']:.4f}")

        engine.run_proxy_optimization(
            run_id,
            num_iterations=args.iterations,
            batch_size=args.batch_size,
            verbose=args.verbose,
        )

        obs_path = Path(f"runs/{run_id}/observations.jsonl")
        all_obs = [json.loads(line) for line in obs_path.read_text().strip().split("\n") if line.strip()]

        proposal_map = {}
        for obs in all_obs:
            smi = obs.get("matched_smiles") or obs.get("x", {}).get("smiles")
            if not smi:
                continue
            can = _canonical(smi)
            if not can:
                continue
            pred = obs.get("y")
            if can not in proposal_map or (pred is not None and pred > (proposal_map[can].get("predicted") or -999)):
                proposal_map[can] = {"smiles": smi, "canonical": can, "predicted": pred}

        proposals = list(proposal_map.values())

        desc_lookup_path = Path(f"runs/{run_id}/descriptor_lookup.json")
        descriptor_lookup = json.loads(desc_lookup_path.read_text())
        state = json.loads(state_path.read_text())
        active_features = state.get("active_features", [])

        selected = prescreen_candidates(
            proposals,
            descriptor_lookup,
            active_features,
            observed_smiles,
            observed_results,
            observed_fps,
            budget=args.experiments_per_round,
            all_pool_smiles=all_pool_smiles,
            discoveries_order=discoveries_order,
            round_num=round_num,
            best_found_so_far=best_found_so_far,
            rounds_since_improvement=rounds_since_improvement,
            oracle_rmse=oracle.get("selected_rmse"),
            ucb_beta=args.ucb_beta,
            verbose=args.verbose,
        )

        matches = []
        misses = []

        for s in selected:
            can = s["canonical"]
            smi = s["smiles"]
            pred = s.get("predicted")
            knn = s.get("knn_pred")
            source = s.get("source", "hebo_exploit")

            if can in lookup_table:
                real = lookup_table[can]
                matches.append(
                    {
                        "smiles": smi,
                        "predicted_target": pred,
                        "knn_pred": knn,
                        "real_pIC50": real,
                        "prediction_error": abs((pred or 0) - real) if pred is not None else None,
                        "source": source,
                    }
                )
                observed_smiles.add(can)
                observed_results[can] = real
                fp = _tanimoto_fp(can)
                if fp is not None:
                    observed_fps[can] = fp
                if can not in train_set and can not in set(discoveries_order):
                    discoveries_order.append(can)
            else:
                penalty_y = min(lookup_table.values()) - 1.0
                misses.append(
                    {
                        "smiles": smi,
                        "predicted_target": pred,
                        "knn_pred": knn,
                        "penalty_y": penalty_y,
                        "source": source,
                        "reason": "not_in_lookup",
                    }
                )
                observed_smiles.add(can)

        round_best = -999.0
        if matches:
            round_best = max(m["real_pIC50"] for m in matches)

        if round_best > best_found_so_far and round_best > -999.0:
            best_found_so_far = round_best
            rounds_since_improvement = 0
        else:
            rounds_since_improvement += 1

        print(
            f"  Round {round_num}: hits={len(matches)} misses={len(misses)} "
            f"best={best_found_so_far:.3f}"
        )

        round_results.append(
            {
                "round": round_num,
                "matches": matches,
                "misses": misses,
                "oracle_rmse": oracle["selected_rmse"],
                "best_found_so_far": best_found_so_far,
                "rounds_since_improvement": rounds_since_improvement,
            }
        )

    best_lookup = max(lookup_table.values())
    best_lookup_smi = next(s for s, y in lookup_table.items() if y == best_lookup)
    all_found = [m["real_pIC50"] for r in round_results for m in r["matches"]]

    output = {
        "run_id": run_id,
        "version": "ic50-global-seeded-v1",
        "dataset": str(args.dataset),
        "split_mode": args.split_mode,
        "target_column": args.target_column,
        "seed_smiles_csv": str(args.seed_smiles_csv) if args.seed_smiles_csv else None,
        "seed_count_used": len(train_pairs),
        "train_csv": str(train_csv),
        "lookup_csv": str(lookup_csv),
        "oracle_note": "All optimisation results are simulations via proxy oracle + lookup.",
        "best_in_dataset": {"smiles": best_lookup_smi, "pIC50": best_lookup},
        "oracle_cv_rmse_last_round": round_results[-1]["oracle_rmse"] if round_results else None,
        "overall_best_found": max(all_found) if all_found else None,
        "gap_to_best": (best_lookup - max(all_found)) if all_found else None,
        "rounds": round_results,
    }

    out_path = Path(f"runs/{run_id}/ic50_global_experiment_results.json")
    out_path.write_text(json.dumps(output, indent=2))

    print("\n" + "=" * 70)
    print("  GLOBAL IC50 EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"  Run ID: {run_id}")
    print(f"  Best in full dataset (sim lookup): {best_lookup:.3f}")
    if all_found:
        print(f"  Best found by experiment: {max(all_found):.3f}")
        print(f"  Gap: {best_lookup - max(all_found):.3f}")
    else:
        print("  No lookup hits found.")
    print(f"  Last-round oracle CV RMSE: {round_results[-1]['oracle_rmse']:.4f}")
    print(f"  Saved: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
