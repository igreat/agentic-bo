#!/usr/bin/env python3
"""EGFR IC50 global simulation experiment.

Simulates a multi-round molecular optimization campaign against the full EGFR
IC50 dataset.  Each "round" asks HEBO for suggestions in descriptor space,
maps them back to the nearest real molecule, and looks up the true pIC50 as
a proxy for a real experiment.

Workflow:
  1. Load full EGFR IC50 dataset, convert to pIC50.
  2. Select seed molecules as initial labeled training set.
  3. Encode all molecules to RDKit descriptor features.
  4. Init BO run (seeds labeled, remaining molecules unlabeled).
  5. Build proxy oracle on labeled molecules.
  6. Each round:
       a. Suggest next batch via HEBO in descriptor space.
       b. Decode suggestions to nearest real molecules.
       c. Look up real pIC50 from full dataset.
       d. Record observations; rebuild oracle.
  7. Report best found vs best in dataset.

Usage:
    uv run python -m bo_workflow.scripts.egfr_ic50_global_experiment \\
        --dataset data/egfr_ic50.csv \\
        --seed-count 50 --rounds 20 --batch-size 4
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bo_workflow.engine import BOEngine
from bo_workflow.oracle import build_proxy_oracle
from bo_workflow.converters.molecule_descriptors import (
    canonicalize_smiles,
    decode_nearest,
    encode_molecules,
    is_descriptor_col,
)
from bo_workflow.scripts.egfr_utils import (
    is_reasonable_seed_smiles,
    load_full_dataset,
    reselect_active_features,
)


def select_seed_molecules(
    data: list[tuple[str, str, float]],
    n: int,
    seed: int = 42,
    filter_fn=None,
) -> list[tuple[str, str, float]]:
    """Stratified seed selection across pIC50 quantiles.

    Samples evenly from n quantile bins so the seed set covers the full
    activity range rather than clustering at one end.

    # TODO: port Kevin's select_representative_split / build_lookup_table from
    # bo_workflow.tools.data.splits (data-conversion branch). That approach
    # separates the data into a labeled train set and a held-out lookup pool,
    # and supports quality modes (top/decent/mixed) + diversity fill.
    """
    pool = data
    if filter_fn is not None:
        pool = [row for row in pool if filter_fn(row[0])]
    if not pool:
        raise ValueError("No molecules passed the seed filter")

    rng = np.random.default_rng(seed)
    arr = sorted(pool, key=lambda r: r[2])
    n = min(n, len(arr))
    bins = np.array_split(np.arange(len(arr)), n)

    selected_idx: set[int] = set()
    for b in bins:
        if len(b) > 0:
            selected_idx.add(int(rng.choice(b)))

    # fill if rounding left us short
    while len(selected_idx) < n:
        selected_idx.add(int(rng.integers(0, len(arr))))

    return [arr[i] for i in sorted(selected_idx)]


def build_features_csv(
    output_path: Path,
    seed_rows: list[tuple[str, str, float]],
    all_data: list[tuple[str, str, float]],
    morgan_bits: int,
) -> tuple[pd.DataFrame, list[str], dict[str, dict[str, float]], dict[str, float]]:
    """Encode all molecules to descriptor features and write a features CSV.

    Seeds have their real pIC50; all other molecules have NaN (unlabeled).
    Returns (catalog_df, descriptor_cols, descriptor_lookup, pic50_lookup).
    """
    seed_pic50 = {can: y for _, can, y in seed_rows}

    # Write raw CSV: smiles + pIC50 (NaN for non-seeds)
    raw_csv = output_path.with_suffix(".raw.csv")
    raw_csv.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {"smiles": smi, "pIC50": seed_pic50.get(can, float("nan"))}
        for smi, can, _ in all_data
    ]
    pd.DataFrame(records).to_csv(raw_csv, index=False)

    # Encode SMILES columns to descriptor features
    _, catalog_df = encode_molecules(raw_csv, smiles_cols=["smiles"], morgan_bits=morgan_bits)

    descriptor_cols = sorted([c for c in catalog_df.columns if is_descriptor_col(c)])

    catalog_df = catalog_df.copy()
    catalog_df["canonical_smiles"] = catalog_df["smiles"].map(canonicalize_smiles)
    catalog_df["pIC50"] = pd.to_numeric(catalog_df["pIC50"], errors="coerce")
    catalog_df = catalog_df.dropna(subset=["canonical_smiles"]).copy()
    catalog_df = (
        catalog_df.sort_values("pIC50", ascending=False, na_position="last")
        .drop_duplicates(subset=["canonical_smiles"], keep="first")
        .reset_index(drop=True)
    )

    # Build lookups
    # pic50_lookup uses all_data (full ground-truth) so BO-suggested molecules
    # that aren't seeds can still be looked up during simulation.
    all_pic50: dict[str, float] = {can: y for _, can, y in all_data}
    descriptor_lookup: dict[str, dict[str, float]] = {}
    pic50_lookup: dict[str, float] = {}
    for _, row in catalog_df.iterrows():
        can = str(row["canonical_smiles"])
        descriptor_lookup[can] = {
            col: float(row[col]) if pd.notna(row[col]) else 0.0
            for col in descriptor_cols
        }
        if can in all_pic50:
            pic50_lookup[can] = all_pic50[can]

    # Write features CSV: descriptor columns + pIC50 (the oracle target)
    feat_rows = [
        {**descriptor_lookup[can], "pIC50": pic50_lookup.get(can, float("nan"))}
        for can in catalog_df["canonical_smiles"].tolist()
        if can in descriptor_lookup
    ]
    pd.DataFrame(feat_rows).to_csv(output_path, index=False)

    return catalog_df, descriptor_cols, descriptor_lookup, pic50_lookup


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="EGFR IC50 global simulation experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", type=Path, default=Path("data/egfr_ic50.csv"))
    p.add_argument("--target-column", default="ic50_nM", choices=["ic50_nM", "pIC50"])
    p.add_argument("--seed-count", type=int, default=50)
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--morgan-bits", type=int, default=64)
    p.add_argument("--no-seed-filter", action="store_true", help="Disable med-chem seed filter")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    if not args.dataset.exists():
        print(f"Dataset not found: {args.dataset}", file=sys.stderr)
        return 1

    # 1. Load dataset
    print(f"Loading {args.dataset}")
    all_data = load_full_dataset(args.dataset, args.target_column)
    print(f"  {len(all_data)} unique molecules")

    best_in_dataset = max(y for _, _, y in all_data)
    best_dataset_smi = next(smi for smi, _, y in all_data if y == best_in_dataset)
    print(f"  Best pIC50 in dataset: {best_in_dataset:.3f}")

    # 2. Seed selection
    filter_fn = None if args.no_seed_filter else is_reasonable_seed_smiles
    seed_rows = select_seed_molecules(all_data, args.seed_count, seed=args.seed, filter_fn=filter_fn)
    seed_pic50_values = [y for _, _, y in seed_rows]
    print(f"  {len(seed_rows)} seed molecules (pIC50 range: {min(seed_pic50_values):.2f}–{max(seed_pic50_values):.2f})")

    # 3. Encode to descriptors
    features_csv = Path("data/egfr_global_features.csv")
    print(f"Encoding {len(all_data)} molecules to descriptors (morgan_bits={args.morgan_bits})...")
    catalog_df, descriptor_cols, descriptor_lookup, pic50_lookup = build_features_csv(
        features_csv, seed_rows, all_data, morgan_bits=args.morgan_bits
    )
    print(f"  {len(descriptor_cols)} descriptor features, {len(pic50_lookup)} labeled molecules")

    # 4. Init BO run
    engine = BOEngine()
    init = engine.init_run(
        dataset_path=features_csv,
        target_column="pIC50",
        objective="max",
        seed=args.seed,
        default_batch_size=args.batch_size,
        verbose=args.verbose,
    )
    run_id = init["run_id"]
    run_dir = engine.get_run_dir(run_id)
    print(f"Run: {run_id}")

    # 5. Optimization rounds
    observed_canonical: set[str] = {can for _, can, _ in seed_rows}
    best_found = max(y for _, _, y in seed_rows)
    round_results = []

    print(f"\nRunning {args.rounds} rounds (batch_size={args.batch_size})...")

    for round_num in range(1, args.rounds + 1):
        # Reselect top features via RF importance before suggesting
        reselect_active_features(
            state_path=run_dir / "state.json",
            descriptor_lookup=descriptor_lookup,
            descriptor_cols=descriptor_cols,
            observed_canonical=observed_canonical,
            pic50_lookup=pic50_lookup,
            max_dims=15,
            verbose=args.verbose,
        )

        suggestions_result = engine.suggest(run_id, batch_size=args.batch_size, verbose=False)
        suggestions = suggestions_result.get("suggestions", [])

        # Read back the active features used for this suggest (may be reduced)
        state = json.loads((run_dir / "state.json").read_text())
        active_features = state.get("active_features", descriptor_cols)

        obs_to_record = []
        round_hits = []

        for s in suggestions:
            x = s.get("x", {})
            query_vec = np.array([float(x.get(c, 0.0)) for c in active_features])

            # Decode to nearest non-observed molecule using active feature subspace
            neighbors = decode_nearest(query_vec, catalog_df, active_features, k=40)
            chosen_can = None
            chosen_smi = None
            for neighbor in neighbors:
                can = str(neighbor.get("canonical_smiles", ""))
                smi = str(neighbor.get("smiles", ""))
                if not can:
                    can = canonicalize_smiles(smi) or ""
                if can and can not in observed_canonical:
                    chosen_can = can
                    chosen_smi = smi
                    break

            if chosen_can is None:
                continue

            # Look up real pIC50 (penalty if not in dataset)
            real_y = pic50_lookup.get(chosen_can, min(pic50_lookup.values()) - 1.0)
            in_dataset = chosen_can in pic50_lookup

            observed_canonical.add(chosen_can)
            obs_to_record.append({
                "x": descriptor_lookup.get(chosen_can, x),
                "y": real_y,
            })
            round_hits.append({
                "smiles": chosen_smi,
                "canonical": chosen_can,
                "real_pIC50": real_y,
                "in_dataset": in_dataset,
            })

        if obs_to_record:
            engine.observe(run_id, obs_to_record, verbose=False)

        # TODO: port Kevin's prescreen_candidates from bo_workflow.domain.strategies
        # (data-conversion branch). It adds exploit/explore/novelty slots with
        # adaptive reallocation, UCB scoring, kNN Tanimoto predictions, and
        # structural diversity filters. Currently we just take the first
        # non-observed nearest neighbor from every HEBO suggestion.

        round_best = max((h["real_pIC50"] for h in round_hits if h["in_dataset"]), default=float("-inf"))
        if round_best > best_found:
            best_found = round_best

        round_results.append({
            "round": round_num,
            "hits": round_hits,
            "best_found_so_far": best_found,
        })

        print(
            f"  Round {round_num:2d}: suggestions={len(round_hits)}"
            f"  round_best={round_best:.3f}"
            f"  overall_best={best_found:.3f}"
        )

    # 7. Final oracle build for surrogate quality report
    final_oracle = build_proxy_oracle(run_dir, cv_folds=args.cv_folds, verbose=False)

    output = {
        "run_id": run_id,
        "dataset": str(args.dataset),
        "seed_count": len(seed_rows),
        "total_molecules": len(all_data),
        "oracle_note": "All results are simulations via proxy oracle + real dataset lookup.",
        "best_in_dataset": {"smiles": best_dataset_smi, "pIC50": best_in_dataset},
        "best_found": best_found,
        "gap_to_best": best_in_dataset - best_found,
        "final_oracle_rmse": final_oracle["selected_rmse"],
        "rounds": round_results,
    }

    out_path = Path(f"runs/{run_id}/egfr_global_results.json")
    out_path.write_text(json.dumps(output, indent=2))

    print(f"\nBest in full dataset:  {best_in_dataset:.3f}")
    print(f"Best found by BO:      {best_found:.3f}")
    print(f"Gap to best:           {best_in_dataset - best_found:.3f}")
    print(f"Final oracle CV RMSE:  {final_oracle['selected_rmse']:.4f}")
    print(f"Results saved:         {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
