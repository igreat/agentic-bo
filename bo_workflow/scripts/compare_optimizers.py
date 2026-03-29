"""Benchmark helper to compare hebo/bo_lcb/random on one dataset.

This script is intentionally separate from the core engine so benchmark/demo
logic does not complicate the production run workflow.
"""

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from bo_workflow.engine import BOEngine
from bo_workflow.evaluation.proxy import ProxyObserver
from bo_workflow.evaluation.oracle import build_proxy_oracle

ENGINE_CHOICES = ("hebo", "bo_lcb", "random", "botorch")
ENGINE_LABELS = {
    "hebo": "HEBO",
    "bo_lcb": "BO (LCB)",
    "random": "Random Search",
    "botorch": "BoTorch",
}


def _read_observation_values(path: Path) -> list[float]:
    if not path.exists():
        raise FileNotFoundError(f"Observation file not found: {path}")
    values: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            values.append(float(row["y"]))
    return values


def _stack_traces(traces: list[list[float]]) -> np.ndarray:
    if not traces:
        raise ValueError("No traces to stack")
    min_len = min(len(trace) for trace in traces)
    if min_len == 0:
        raise ValueError("At least one trace is empty")
    return np.array([trace[:min_len] for trace in traces], dtype=float)


def _final_best(values: list[float], objective: str) -> float:
    if objective == "min":
        return float(min(values))
    return float(max(values))


def _engine_summary(final_bests: list[float], objective: str) -> dict[str, float | int]:
    arr = np.asarray(final_bests, dtype=float)
    if objective == "min":
        best_value = float(arr.min())
        worst_value = float(arr.max())
    else:
        best_value = float(arr.max())
        worst_value = float(arr.min())
    return {
        "n_runs": int(arr.size),
        "mean_final_best": float(arr.mean()),
        "std_final_best": float(arr.std()),
        "median_final_best": float(np.median(arr)),
        "best_final_best": best_value,
        "worst_final_best": worst_value,
    }


def _cumulative_best(values: np.ndarray, objective: str) -> np.ndarray:
    if objective == "min":
        return np.minimum.accumulate(values, axis=1)
    return np.maximum.accumulate(values, axis=1)


def _plot_convergence(
    methods_data: dict[str, np.ndarray],
    *,
    title: str,
    ylabel: str,
    objective: str,
    fig_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not colors:
        colors = ["#2E86AB", "#A23B72", "#F18F01", "#454645"]

    for idx, (label, series) in enumerate(methods_data.items()):
        best_so_far = _cumulative_best(series, objective)
        mean = np.mean(best_so_far, axis=0)
        stderr = np.std(best_so_far, axis=0) / np.sqrt(best_so_far.shape[0])
        iters = np.arange(best_so_far.shape[1], dtype=int)
        color = colors[idx % len(colors)]
        ax.plot(iters, mean, lw=2.2, color=color, label=label)
        ax.fill_between(iters, mean - stderr, mean + stderr, color=color, alpha=0.18)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run and compare multiple BO engines on one dataset",
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--objective", type=str, choices=["min", "max"], required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--init-random", type=int, default=10)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--max-features", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--runs-root", type=Path, default=Path("bo_runs"))
    parser.add_argument(
        "--drop-cols",
        type=str,
        default=None,
        help="Comma-separated column names to exclude from the feature space",
    )
    parser.add_argument(
        "--engines",
        nargs="+",
        choices=list(ENGINE_CHOICES),
        default=list(ENGINE_CHOICES),
    )
    parser.add_argument(
        "--plot-out", type=Path, default=Path("results/compare/optimizers.pdf")
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("results/compare/optimizers_summary.json"),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show progress bars and per-run logs",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if "bo_lcb" in args.engines and args.batch_size != 1:
        raise ValueError("bo_lcb currently supports batch-size=1 only")

    drop_cols = [c.strip() for c in args.drop_cols.split(",") if c.strip()] if args.drop_cols else []

    engine = BOEngine(runs_root=args.runs_root)
    methods_data: dict[str, np.ndarray] = {}
    summary_runs: list[dict[str, object]] = []
    engine_final_bests: dict[str, list[float]] = {}

    total_runs = len(args.engines) * args.repeats
    run_progress = tqdm(
        total=total_runs,
        desc="Optimizer runs",
        unit="run",
    )

    for engine_name in args.engines:
        if args.verbose:
            print(f"\n=== Engine: {ENGINE_LABELS[engine_name]} ===")

        traces: list[list[float]] = []

        for rep in range(args.repeats):
            run_seed = args.seed + rep
            if args.verbose:
                print(f"- repeat {rep + 1}/{args.repeats} (seed={run_seed})")

            state = engine.init_run(
                dataset_path=args.dataset,
                target_column=args.target,
                objective=args.objective,
                default_engine=engine_name,
                seed=run_seed,
                num_initial_random_samples=args.init_random,
                default_batch_size=args.batch_size,
                drop_cols=drop_cols,
            )
            run_id = str(state["run_id"])
            if args.verbose:
                print(f"  init: run_id={run_id}")

            backend_dir = args.runs_root.parent / "evaluation_backends" / run_id
            oracle_info = build_proxy_oracle(
                dataset_path=args.dataset,
                target_column=args.target,
                objective=args.objective,
                backend_dir=backend_dir,
                drop_cols=drop_cols,
                seed=run_seed,
                default_engine=engine_name,
                cv_folds=args.cv_folds,
                max_features=args.max_features,
            )
            if args.verbose:
                print(
                    "  oracle: "
                    f"{oracle_info.get('selected_model')} "
                    f"(cv_rmse={oracle_info.get('selected_rmse'):.4f})"
                )

            observer = ProxyObserver(backend_dir)
            engine.run_optimization(
                run_id,
                observer=observer,
                num_iterations=args.iterations,
                batch_size=args.batch_size,
            )

            obs_path = args.runs_root / run_id / "observations.jsonl"
            values = _read_observation_values(obs_path)
            traces.append(values)
            final_best = _final_best(values, args.objective)
            engine_final_bests.setdefault(engine_name, []).append(final_best)
            if args.verbose:
                print(f"  result: n_obs={len(values)} final_best={final_best:.4f}")

            summary_runs.append(
                {
                    "run_id": run_id,
                    "engine": engine_name,
                    "seed": run_seed,
                    "num_observations": len(values),
                    "best_value": final_best,
                    "oracle_selected_model": oracle_info.get("selected_model"),
                    "oracle_selected_rmse": oracle_info.get("selected_rmse"),
                }
            )
            run_progress.update(1)

        label = ENGINE_LABELS[engine_name]
        methods_data[label] = _stack_traces(traces)

    run_progress.close()

    _plot_convergence(
        methods_data,
        title="Optimizer Comparison",
        ylabel=f"{args.target} (best-so-far)",
        objective=args.objective,
        fig_path=args.plot_out,
    )

    summary: dict[str, object] = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "dataset": str(args.dataset),
        "target": args.target,
        "objective": args.objective,
        "iterations": int(args.iterations),
        "batch_size": int(args.batch_size),
        "repeats": int(args.repeats),
        "engines": list(args.engines),
        "plot_path": str(args.plot_out),
        "runs": summary_runs,
        "engine_stats": {
            ENGINE_LABELS[name]: _engine_summary(bests, args.objective)
            for name, bests in engine_final_bests.items()
        },
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_out.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    if args.verbose:
        print("\n=== Aggregate final-best stats ===")
        for label, stats in summary["engine_stats"].items():
            print(
                f"- {label}: mean={stats['mean_final_best']:.4f}, "
                f"std={stats['std_final_best']:.4f}, "
                f"median={stats['median_final_best']:.4f}, "
                f"n={stats['n_runs']}"
            )

    print(
        json.dumps(
            {
                "plot": str(args.plot_out),
                "summary": str(args.summary_out),
                "num_runs": len(summary_runs),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
