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

import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.ensemble import RandomForestRegressor

from bo_workflow.engine import BOEngine
from bo_workflow.experiment_strategies import (
    _canonical,
    _tanimoto_fp,
    prescreen_candidates,
    select_diverse_train,
)
from bo_workflow.workflows.data_utils import (
    append_labeled_rows,
    build_lookup_table,
    load_seed_smiles,
    select_representative_split,
    split_from_seed_smiles,
    write_smiles_target_csv,
)


def _to_pic50(ic50_nm: float) -> float:
    if ic50_nm <= 0:
        raise ValueError("ic50_nM must be > 0")
    return 9.0 - math.log10(float(ic50_nm))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="EGFR global ic50 experiment with representative split")
    p.add_argument("--dataset", type=Path, default=Path("data/egfr_ic50.csv"))
    p.add_argument("--target-column", default="ic50_nM", choices=["ic50_nM", "pIC50"])
    p.add_argument(
        "--max-pic50",
        type=float,
        default=12.0,
        help="Drop rows whose normalized pIC50 exceeds this threshold (default: 12.0)",
    )
    p.add_argument(
        "--fix-tiny-ic50-as-molar",
        action="store_true",
        help="If ic50_nM is < 1e-6, treat it as molar and convert to nM before pIC50 conversion",
    )
    p.add_argument("--split-mode", default="auto", choices=["auto", "from-seed-csv"])
    p.add_argument("--seed-smiles-csv", type=Path, help="CSV with user seed smiles (required when --split-mode from-seed-csv)")
    p.add_argument("--seed-smiles-column", default="smiles")
    p.add_argument("--seed-count", type=int, default=50)
    p.add_argument("--seed-top-fraction", type=float, default=0.1,
                   help="In auto mode, fraction of quality-prior seeds kept before diversity fill (default: 0.1)")
    p.add_argument(
        "--seed-quality-mode",
        default="decent",
        choices=["top", "decent", "mixed"],
        help="Auto seed quality policy: top=strong exploit, decent=mid-high band, mixed=half top half decent",
    )
    p.add_argument(
        "--seed-decent-low-q",
        type=float,
        default=0.60,
        help="Lower quantile for decent seed band (default: 0.60)",
    )
    p.add_argument(
        "--seed-decent-high-q",
        type=float,
        default=0.90,
        help="Upper quantile for decent seed band (default: 0.90)",
    )
    p.add_argument("--no-seed-filter", action="store_true",
                   help="Disable basic seed quality filter (salts/multi-fragment/inorganic)")

    p.add_argument("--iterations", type=int, default=30)
    p.add_argument("--rounds", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--experiments-per-round", type=int, default=4)
    p.add_argument("--fp-bits", type=int, default=256)
    p.add_argument(
        "--energy-backend",
        type=str,
        default="none",
        choices=["none", "xtb", "auto", "ml", "dft"],
        help="Energy feature backend for descriptors (xTB/auto/ML/DFT)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ucb-beta", type=float, default=1.0)
    p.add_argument(
        "--hebo-exploit-slots",
        type=int,
        default=1,
        help="Number of HEBO exploit picks per round (high predicted activity)",
    )
    p.add_argument(
        "--hebo-explore-slots",
        type=int,
        default=0,
        help="Number of HEBO explore picks per round (high uncertainty/UCB)",
    )
    p.add_argument(
        "--hebo-explore-beta",
        type=float,
        default=2.0,
        help="Exploration weight for HEBO explore slot ranking",
    )
    p.add_argument(
        "--all-hebo-exploit",
        action="store_true",
        help="Use HEBO UCB only for all per-round selections (disable greedy/novelty slots)",
    )
    p.add_argument(
        "--adaptive-novelty",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable adaptive reallocation of novelty slot to exploit when stalled (default: on)",
    )
    p.add_argument(
        "--novelty-grace-rounds",
        type=int,
        default=10,
        help="Keep novelty slot for at least this many initial rounds (default: 10)",
    )
    p.add_argument(
        "--stall-rounds-threshold",
        type=int,
        default=3,
        help="After this many no-improvement rounds, reallocate novelty slot if adaptive mode is on (default: 3)",
    )
    p.add_argument("--verbose", action="store_true")
    return p


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def _load_full_dataset(
    dataset_path: Path,
    target_column: str,
    *,
    max_pic50: float | None,
    fix_tiny_ic50_as_molar: bool,
):
    with open(dataset_path) as f:
        rows = list(csv.DictReader(f))

    data = []
    dropped_extreme = 0
    fixed_tiny = 0
    for r in rows:
        smi = r.get("smiles", "")
        can = _canonical(smi)
        if not can:
            continue

        if target_column == "pIC50":
            y = float(r["pIC50"])
        else:
            ic50_nm = float(r["ic50_nM"])
            if fix_tiny_ic50_as_molar and ic50_nm < 1e-6:
                ic50_nm = ic50_nm * 1e9
                fixed_tiny += 1
            y = _to_pic50(ic50_nm)

        if max_pic50 is not None and y > max_pic50:
            dropped_extreme += 1
            continue

        data.append((smi, can, y))

    if not data:
        raise ValueError("No valid molecules found in dataset")

    if fixed_tiny > 0:
        print(
            f"Data quality: converted {fixed_tiny} tiny ic50_nM values (<1e-6) from M to nM before pIC50"
        )
    if dropped_extreme > 0:
        print(
            f"Data quality: dropped {dropped_extreme} rows with pIC50 > {max_pic50}"
        )

    return data


def _is_reasonable_seed_smiles(smiles: str) -> bool:
    """Basic med-chem seed filter: exclude salts/multi-fragments/inorganics."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False

    if "." in Chem.MolToSmiles(mol):
        return False

    allowed = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53}
    atoms = [a.GetAtomicNum() for a in mol.GetAtoms()]
    if not atoms:
        return False
    if any(z not in allowed for z in atoms):
        return False

    heavy = mol.GetNumHeavyAtoms()
    if heavy < 8 or heavy > 90:
        return False

    if 6 not in atoms:
        return False

    return True


def _reselect_active_features(
    *,
    state_path: Path,
    train_csv: Path,
    desc_lookup_path: Path,
    seed: int,
    max_dims: int = 15,
    verbose: bool = False,
) -> int:
    """Reselect top descriptor features from currently observed labeled data.

    This keeps the model aligned with newly added descriptors (e.g., dft_* from
    xTB) while limiting HEBO dimensionality for GP stability.
    """
    state = json.loads(state_path.read_text())
    desc_lookup = json.loads(desc_lookup_path.read_text())
    train_df = pd.read_csv(train_csv)

    target_col = str(state.get("target_column", "pIC50"))
    smiles_col = str(state.get("smiles_direct", {}).get("smiles_column", "smiles"))
    if target_col not in train_df.columns or smiles_col not in train_df.columns:
        return 0

    y_series = pd.to_numeric(train_df[target_col], errors="coerce")
    valid_rows: list[dict[str, float]] = []
    y_vals: list[float] = []
    for smi, y in zip(train_df[smiles_col].astype(str), y_series):
        if pd.isna(y):
            continue
        row_desc = desc_lookup.get(smi)
        if row_desc is None:
            continue
        valid_rows.append(row_desc)
        y_vals.append(float(y))

    if len(valid_rows) < 5:
        return 0

    x_imp = pd.DataFrame(valid_rows)
    variable_cols: list[str] = []
    for col in x_imp.columns:
        vals = pd.to_numeric(x_imp[col], errors="coerce").dropna()
        if len(vals) == 0:
            continue
        if np.isclose(float(vals.min()), float(vals.max())):
            continue
        variable_cols.append(str(col))

    if not variable_cols:
        return 0

    x_var = x_imp[variable_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y_arr = np.asarray(y_vals, dtype=float)

    keep_n = min(max_dims, len(variable_cols))
    if len(variable_cols) > keep_n:
        rf = RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=1)
        rf.fit(x_var, y_arr)
        top_idx = np.argsort(rf.feature_importances_)[::-1][:keep_n]
        selected = [variable_cols[i] for i in top_idx]
    else:
        selected = list(variable_cols)

    design_params: list[dict[str, float | str]] = []
    selected_final: list[str] = []
    for col in selected:
        vals = pd.to_numeric(x_var[col], errors="coerce").dropna()
        if len(vals) == 0:
            continue
        lb = float(vals.min())
        ub = float(vals.max())
        if np.isclose(lb, ub):
            continue
        selected_final.append(col)
        design_params.append({"name": col, "type": "num", "lb": lb, "ub": ub})

    if not selected_final:
        return 0

    state["active_features"] = selected_final
    state["design_parameters"] = design_params
    state["updated_at"] = pd.Timestamp.utcnow().isoformat()
    state_path.write_text(json.dumps(state, indent=2))

    if verbose:
        n_energy = sum(1 for f in selected_final if str(f).startswith("dft_"))
        print(
            f"  Reselected {len(selected_final)} active features (energy={n_energy})"
        )

    return len(selected_final)


def run(args: argparse.Namespace) -> int:
    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    if args.split_mode == "from-seed-csv":
        if args.seed_smiles_csv is None:
            raise ValueError("--seed-smiles-csv is required when --split-mode from-seed-csv")
        if not args.seed_smiles_csv.exists():
            raise FileNotFoundError(f"Seed CSV not found: {args.seed_smiles_csv}")

    full_data = _load_full_dataset(
        args.dataset,
        args.target_column,
        max_pic50=args.max_pic50,
        fix_tiny_ic50_as_molar=args.fix_tiny_ic50_as_molar,
    )

    apply_filter = not args.no_seed_filter

    if args.split_mode == "from-seed-csv":
        seed_smiles = load_seed_smiles(
            args.seed_smiles_csv,
            args.seed_smiles_column,
            args.seed_count,
            canonicalize=_canonical,
            filter_fn=_is_reasonable_seed_smiles if apply_filter else None,
        )
    else:
        seed_smiles = []

    full_by_can = {}
    full_smi_by_can = {}
    for smi, can, y in full_data:
        full_by_can[can] = y
        full_smi_by_can[can] = smi

    if args.split_mode == "from-seed-csv":
        train_pairs, lookup_pairs = split_from_seed_smiles(
            full_data,
            seed_smiles,
            seed_count=args.seed_count,
            min_overlap=5,
        )
    else:
        train_pairs, lookup_pairs = select_representative_split(
            full_data,
            args.seed_count,
            args.seed,
            select_diverse_train_fn=select_diverse_train,
            top_fraction=args.seed_top_fraction,
            quality_mode=args.seed_quality_mode,
            decent_low_q=args.seed_decent_low_q,
            decent_high_q=args.seed_decent_high_q,
            filter_fn=_is_reasonable_seed_smiles if apply_filter else None,
        )

    print(f"Loaded full dataset: {len(full_by_can)} molecules")
    print(f"Split mode: {args.split_mode}")
    if args.split_mode == "auto":
        print(
            "Seed quality mode: "
            f"{args.seed_quality_mode} "
            f"(top_fraction={args.seed_top_fraction}, decent_q=[{args.seed_decent_low_q}, {args.seed_decent_high_q}])"
        )
    print(
        "Adaptive novelty: "
        f"{'on' if args.adaptive_novelty else 'off'} "
        f"(grace_rounds={args.novelty_grace_rounds}, stall_threshold={args.stall_rounds_threshold})"
    )
    print(f"Selection mode: {'all_hebo_exploit' if args.all_hebo_exploit else 'hybrid'}")
    print(
        "HEBO slots: "
        f"exploit={args.hebo_exploit_slots}, explore={args.hebo_explore_slots}, "
        f"explore_beta={args.hebo_explore_beta}"
    )
    print(f"Seed filter: {'on' if apply_filter else 'off'}")
    print(f"Initial train set: {len(train_pairs)} molecules")
    print(f"Lookup set: {len(lookup_pairs)} molecules")

    lookup_table = build_lookup_table(lookup_pairs, train_pairs)
    all_pool_smiles = list({smi for smi, _, _ in full_data})

    split_suffix = "auto" if args.split_mode == "auto" else "seeded"
    train_csv = Path(f"data/egfr_ic50_train_{split_suffix}.csv")
    write_smiles_target_csv(train_csv, train_pairs, target_name="pIC50")

    lookup_csv = Path(f"data/egfr_ic50_lookup_{split_suffix}.csv")
    write_smiles_target_csv(lookup_csv, lookup_pairs, target_name="pIC50")

    engine = BOEngine()
    init_result = engine.init_smiles_run(
        dataset_path=str(train_csv),
        target_column="pIC50",
        objective="max",
        energy_backend=args.energy_backend,
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

    from bo_workflow.molecular.features import compute_descriptors
    from bo_workflow.molecular.energy import EnergyCache, get_energy_features
    from bo_workflow.molecular.types import DescriptorConfig

    dc = state["smiles_direct"]["descriptor_config"]
    fp_cfg = dc.get("fingerprint", {})
    desc_config = DescriptorConfig(
        basic=dc.get("basic", True),
        fingerprint_enabled=fp_cfg.get("enabled", True),
        fingerprint_n_bits=fp_cfg.get("n_bits", 256),
        fingerprint_radius=fp_cfg.get("radius", 2),
        electronic=dc.get("electronic", True),
        steric=dc.get("steric", False),
    )

    energy_backend = dc.get("energy_backend", "none")
    energy_cache = None
    if energy_backend != "none":
        method_map = {
            "xtb": "GFN2-xTB",
            "auto": "auto",
            "ml": "ANI-2x",
            "dft": "B3LYP/6-31G*",
        }
        energy_cache = EnergyCache(
            path=Path(f"runs/{run_id}/energy_cache.json"),
            method=method_map.get(energy_backend, energy_backend),
        )
        print(f"  Energy backend enabled: {energy_backend}")

    initial_train_smiles = {smi for smi, _, _ in train_pairs}

    full_descriptor_lookup = {}
    xtb_initial_count = 0
    for smi, _, _ in full_data:
        if smi in full_descriptor_lookup:
            continue
        try:
            desc = compute_descriptors(smi, config=desc_config)
            if energy_backend != "none" and smi in initial_train_smiles:
                energy_feats = get_energy_features(
                    smi,
                    cache=energy_cache,
                    backend=energy_backend,
                )
                for k, v in energy_feats.items():
                    desc[f"dft_{k}"] = float(v)
                xtb_initial_count += 1
            full_descriptor_lookup[smi] = desc
        except Exception:
            pass

    desc_lookup_path = Path(f"runs/{run_id}/descriptor_lookup.json")
    desc_lookup_path.write_text(json.dumps(full_descriptor_lookup))
    print(f"  Descriptor lookup expanded: {len(full_descriptor_lookup)} molecules (base descriptors)")
    if energy_backend != "none":
        print(
            f"  Incremental energy mode: computed {xtb_initial_count} initial energy profiles "
            f"(train set only); new molecules will be computed on demand with cache reuse"
        )

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
            append_labeled_rows(
                train_csv,
                [(m["smiles"], m["real_pIC50"]) for m in prev["matches"]]
                + [(m["smiles"], m["penalty_y"]) for m in prev["misses"]],
            )

            dl_path = Path(f"runs/{run_id}/descriptor_lookup.json")
            dl = json.loads(dl_path.read_text())
            for m in prev["matches"] + prev["misses"]:
                smi = m["smiles"]
                if smi not in dl:
                    try:
                        desc = compute_descriptors(smi, config=desc_config)
                        if energy_backend != "none":
                            energy_feats = get_energy_features(
                                smi,
                                cache=energy_cache,
                                backend=energy_backend,
                            )
                            for k, v in energy_feats.items():
                                desc[f"dft_{k}"] = float(v)
                        dl[smi] = desc
                    except Exception:
                        pass
            dl_path.write_text(json.dumps(dl))

            state = json.loads(state_path.read_text())
            state["status"] = "running"
            state_path.write_text(json.dumps(state, indent=2))

        # Ensure descriptors/energy are available for any newly observed molecules
        # before prescreen scoring in this round.
        dl_path = Path(f"runs/{run_id}/descriptor_lookup.json")
        dl = json.loads(dl_path.read_text())
        obs_path = Path(f"runs/{run_id}/observations.jsonl")
        obs_rows = [
            json.loads(line)
            for line in obs_path.read_text().strip().split("\n")
            if line.strip()
        ] if obs_path.exists() and obs_path.read_text().strip() else []
        newly_enriched = 0
        for obs in obs_rows:
            smi = obs.get("matched_smiles") or obs.get("x", {}).get("smiles")
            if not smi:
                continue
            if smi in dl and (energy_backend == "none" or any(k.startswith("dft_") for k in dl[smi].keys())):
                continue
            try:
                desc = compute_descriptors(smi, config=desc_config)
                if energy_backend != "none":
                    energy_feats = get_energy_features(
                        smi,
                        cache=energy_cache,
                        backend=energy_backend,
                    )
                    for k, v in energy_feats.items():
                        desc[f"dft_{k}"] = float(v)
                dl[smi] = desc
                newly_enriched += 1
            except Exception:
                pass
        if newly_enriched > 0:
            dl_path.write_text(json.dumps(dl))
            if args.verbose:
                print(f"  Enriched {newly_enriched} newly observed molecules with descriptors/energy")

        _reselect_active_features(
            state_path=state_path,
            train_csv=train_csv,
            desc_lookup_path=dl_path,
            seed=args.seed,
            max_dims=15,
            verbose=args.verbose,
        )

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
            hebo_exploit_slots=args.hebo_exploit_slots,
            hebo_explore_slots=args.hebo_explore_slots,
            hebo_explore_beta=args.hebo_explore_beta,
            exploit_only=args.all_hebo_exploit,
            adaptive_novelty=args.adaptive_novelty,
            novelty_grace_rounds=args.novelty_grace_rounds,
            stall_rounds_threshold=args.stall_rounds_threshold,
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


def main(argv=None) -> int:
    args = parse_args(argv)
    return run(args)
