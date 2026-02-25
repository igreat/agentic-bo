"""Compare feature representations: DRFP-only vs descriptors-only vs combined.

Encodes the same dataset three ways, runs BO with a proxy oracle for each,
and produces a single convergence plot + JSON summary.
"""

import argparse
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bo_workflow.converters.reaction_drfp import encode_reactions  # noqa: E402
from bo_workflow.converters.molecule_descriptors import encode_molecules  # noqa: E402
from bo_workflow.converters.combined import encode_combined  # noqa: E402
from bo_workflow.engine import BOEngine  # noqa: E402
from bo_workflow.observers.proxy import ProxyObserver  # noqa: E402
from bo_workflow.oracle import build_proxy_oracle  # noqa: E402
from bo_workflow.plotting import plot_optimization_convergence  # noqa: E402


REP_LABELS = {
    "drfp": "DRFP-only",
    "descriptors": "Descriptors-only",
    "combined": "Combined (DRFP + Desc)",
}


def _read_observation_values(path: Path) -> list[float]:
    values: list[float] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            values.append(float(json.loads(line)["y"]))
    return values


def _stack_traces(traces: list[list[float]]) -> np.ndarray:
    min_len = min(len(t) for t in traces)
    return np.array([t[:min_len] for t in traces], dtype=float)


def _encode_dataset(
    rep: str,
    input_path: Path,
    output_dir: Path,
    rxn_col: str,
    smiles_cols: list[str],
    n_bits: int,
    morgan_bits: int,
) -> Path:
    """Encode dataset for a given representation. Returns path to features.csv."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if rep == "drfp":
        features_df, catalog_df = encode_reactions(input_path, n_bits=n_bits, rxn_col=rxn_col)
    elif rep == "descriptors":
        features_df, catalog_df = encode_molecules(input_path, smiles_cols=smiles_cols, morgan_bits=morgan_bits)
        # Drop rxn_col if present — it's a string column that the engine can't handle
        for col_to_drop in [rxn_col]:
            if col_to_drop in features_df.columns:
                features_df = features_df.drop(columns=[col_to_drop])
    elif rep == "combined":
        features_df, catalog_df = encode_combined(input_path, rxn_col=rxn_col, smiles_cols=smiles_cols, n_bits=n_bits)
    else:
        raise ValueError(f"Unknown representation: {rep}")

    features_path = output_dir / "features.csv"
    catalog_path = output_dir / "catalog.csv"
    features_df.to_csv(features_path, index=False)
    catalog_df.to_csv(catalog_path, index=False)
    return features_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare DRFP-only vs descriptors-only vs combined representations",
    )
    parser.add_argument("--input", type=Path, required=True, help="Input CSV")
    parser.add_argument("--rxn-col", default="rxn_smiles", help="Reaction SMILES column")
    parser.add_argument("--smiles-cols", nargs="+", required=True, help="Component SMILES columns")
    parser.add_argument("--target", type=str, required=True, help="Target column name")
    parser.add_argument("--objective", choices=["min", "max"], required=True)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--init-random", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bits", type=int, default=256, help="DRFP fingerprint length")
    parser.add_argument("--morgan-bits", type=int, default=64, help="Morgan FP bits per SMILES col")
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument(
        "--reps", nargs="+",
        choices=list(REP_LABELS),
        default=list(REP_LABELS),
        help="Representations to compare",
    )
    parser.add_argument("--plot-out", type=Path, default=Path("results/compare/representations.pdf"))
    parser.add_argument("--summary-out", type=Path, default=Path("results/compare/representations_summary.json"))
    parser.add_argument("--keep-data", action="store_true", help="Keep encoded data directories")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    data_root = Path("data/cmp")
    engine = BOEngine(runs_root=args.runs_root)

    methods_data: dict[str, np.ndarray] = {}
    summary_runs: list[dict] = []
    rep_final_bests: dict[str, list[float]] = {}

    for rep in args.reps:
        label = REP_LABELS[rep]
        if args.verbose:
            print(f"\n=== {label} ===")

        # Step 1: Encode
        data_dir = data_root / rep
        if args.verbose:
            print(f"  Encoding to {data_dir} ...")
        features_path = _encode_dataset(
            rep, args.input, data_dir,
            args.rxn_col, args.smiles_cols, args.n_bits, args.morgan_bits,
        )

        # Count features
        import pandas as pd
        feat_df = pd.read_csv(features_path)
        n_features = len([c for c in feat_df.columns if c != args.target])
        if args.verbose:
            print(f"  Features: {n_features} columns, {len(feat_df)} rows")

        traces: list[list[float]] = []

        for r in range(args.repeats):
            run_seed = args.seed + r
            if args.verbose:
                print(f"  Repeat {r + 1}/{args.repeats} (seed={run_seed})")

            # Step 2: Init + build oracle + run
            state = engine.init_run(
                dataset_path=features_path,
                target_column=args.target,
                objective=args.objective,
                default_engine="hebo",
                seed=run_seed,
                num_initial_random_samples=args.init_random,
                default_batch_size=args.batch_size,
            )
            run_id = str(state["run_id"])
            run_dir = engine.get_run_dir(run_id)

            oracle_info = build_proxy_oracle(run_dir)
            if args.verbose:
                print(
                    f"    Oracle: {oracle_info.get('selected_model')} "
                    f"(cv_rmse={oracle_info.get('selected_rmse', 0):.4f})"
                )

            observer = ProxyObserver(run_dir)
            engine.run_optimization(
                run_id,
                observer=observer,
                num_iterations=args.iterations,
                batch_size=args.batch_size,
            )

            obs_path = args.runs_root / run_id / "observations.jsonl"
            values = _read_observation_values(obs_path)
            traces.append(values)

            if args.objective == "max":
                final_best = float(max(values))
            else:
                final_best = float(min(values))

            rep_final_bests.setdefault(rep, []).append(final_best)

            if args.verbose:
                print(f"    Result: n_obs={len(values)} final_best={final_best:.4f}")

            summary_runs.append({
                "run_id": run_id,
                "representation": rep,
                "label": label,
                "seed": run_seed,
                "n_features": n_features,
                "num_observations": len(values),
                "best_value": final_best,
                "oracle_model": oracle_info.get("selected_model"),
                "oracle_rmse": oracle_info.get("selected_rmse"),
            })

        methods_data[label] = _stack_traces(traces)

    # Plot
    plot_optimization_convergence(
        methods_data,
        title="Representation Comparison",
        ylabel=f"{args.target} (best-so-far)",
        objective=args.objective,
        fig_path=str(args.plot_out),
        show=False,
    )

    # Summary
    summary = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "input": str(args.input),
        "target": args.target,
        "objective": args.objective,
        "iterations": args.iterations,
        "batch_size": args.batch_size,
        "repeats": args.repeats,
        "seed": args.seed,
        "representations": args.reps,
        "plot_path": str(args.plot_out),
        "runs": summary_runs,
        "rep_stats": {
            REP_LABELS[rep]: {
                "n_runs": len(bests),
                "mean_final_best": float(np.mean(bests)),
                "std_final_best": float(np.std(bests)),
                "best_final_best": float(max(bests) if args.objective == "max" else min(bests)),
            }
            for rep, bests in rep_final_bests.items()
        },
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    # Cleanup encoded data
    if not args.keep_data and data_root.exists():
        shutil.rmtree(data_root)

    if args.verbose:
        print("\n=== Results ===")
        for label, stats in summary["rep_stats"].items():
            print(f"  {label}: best={stats['best_final_best']:.4f}, mean={stats['mean_final_best']:.4f}")
        print(f"\n  Plot: {args.plot_out}")
        print(f"  Summary: {args.summary_out}")

    print(json.dumps({
        "plot": str(args.plot_out),
        "summary": str(args.summary_out),
        "num_runs": len(summary_runs),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
