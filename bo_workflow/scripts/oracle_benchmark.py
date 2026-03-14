#!/usr/bin/env python3
"""Oracle performance benchmark.

Measures how well the oracle guides BO toward the true optimum, producing
before/after metrics for comparing oracle versions.

Metrics reported:
  - CV RMSE              : oracle prediction accuracy (cross-validated)
  - Spearman (all)       : rank correlation across full dataset (cross-validated)
  - Spearman (top-K%)    : rank correlation within top-K% rows (cross-validated)
  - Cumulative best curve: how quickly BO finds good true values
  - Top-K% recovery rate : fraction of suggestions landing in true top-K%
  - Iters to first top-K% hit
  - Random baseline      : same metrics for pure random sampling (reference)

Design: true-seeded, oracle-guided benchmark
  - Seed observations use true target values (mirrors real workflow: a chemist
    has genuine initial measurements before BO begins).
  - All subsequent BO iterations are fed oracle predictions, not true values.
  - True values of suggestions are tracked secretly via nearest-neighbour
    lookup and used only for metric computation, never fed back to the engine.
  - A better oracle guides BO toward better true values faster; differences
    in the metrics reflect oracle quality given this warm-started setup.

Spearman correlations are computed from cross-validated held-out predictions,
so they reflect genuine generalisation rather than in-sample memorisation.
The top-K% Spearman is the most informative: it tells you whether the oracle
correctly orders the best candidates relative to each other.

Works with any all-numeric tabular dataset; primary development dataset is
HER_virtual_data.csv.

Usage:
    # Benchmark on HER dataset (primary development dataset)
    uv run python -m bo_workflow.scripts.oracle_benchmark run \\
        --dataset data/HER_virtual_data.csv \\
        --target Target --objective max \\
        --seed-count 20 --iterations 50 --repeats 10 \\
        --output results/her_baseline.json \\
        --plot-out results/her_baseline.pdf

    # Compare baseline vs improved oracle
    uv run python -m bo_workflow.scripts.oracle_benchmark compare \\
        results/her_baseline.json results/her_improved.json \\
        --labels "Baseline" "Improved" \\
        --plot-out results/her_comparison.pdf
"""

import argparse
import json
import pickle
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict

from bo_workflow.engine import BOEngine
from bo_workflow.oracle import build_proxy_oracle, predict_original_scale
from bo_workflow.utils import read_json


# ---------------------------------------------------------------------------
# Stratified seeding
# ---------------------------------------------------------------------------


def _stratified_seed_indices(
    y_series: pd.Series,
    valid_indices: list[int],
    n_seeds: int,
    n_strata: int,
    rng: np.random.Generator,
) -> list[int]:
    """Sample seeds proportionally across quantile strata of the target.

    Ensures the seed set covers the full target distribution rather than
    clustering in the mode (useful for skewed target distributions).
    """
    strata = pd.qcut(
        y_series[valid_indices], q=n_strata, labels=False, duplicates="drop"
    )
    unique_strata = sorted(strata.dropna().unique())
    per_stratum = max(1, n_seeds // len(unique_strata))

    selected: list[int] = []
    for s in unique_strata:
        s_idx = [i for i in valid_indices if strata.get(i) == s]
        n_pick = min(per_stratum, len(s_idx))
        selected.extend(rng.choice(s_idx, size=n_pick, replace=False).tolist())

    # Top up to n_seeds if rounding left us short
    used = set(selected)
    remaining = [i for i in valid_indices if i not in used]
    shortfall = n_seeds - len(selected)
    if shortfall > 0 and remaining:
        selected.extend(
            rng.choice(remaining, size=min(shortfall, len(remaining)), replace=False).tolist()
        )

    return [int(i) for i in selected[:n_seeds]]


# ---------------------------------------------------------------------------
# Nearest-neighbour lookup
# ---------------------------------------------------------------------------


def _build_norm_matrix(
    df: pd.DataFrame, feature_cols: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (norm_matrix, mean, std) for fast normalised Euclidean search."""
    matrix = df[feature_cols].values.astype(float)
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std[std == 0] = 1.0
    return (matrix - mean) / std, mean, std


def _find_nearest(
    suggestion_x: dict[str, Any],
    feature_cols: list[str],
    norm_matrix: np.ndarray,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    excluded: set[int],
) -> int:
    """Return nearest non-excluded row index by normalised Euclidean distance."""
    query = np.array(
        [float(suggestion_x.get(c, 0.0)) for c in feature_cols], dtype=float
    )
    query_norm = (query - norm_mean) / norm_std
    distances = np.sqrt(((norm_matrix - query_norm) ** 2).sum(axis=1))
    for idx in np.argsort(distances):
        if int(idx) not in excluded:
            return int(idx)
    return int(np.argmin(distances))  # fallback: all rows excluded


# ---------------------------------------------------------------------------
# Single repeat
# ---------------------------------------------------------------------------


def _run_single_repeat(
    full_df: pd.DataFrame,
    target_col: str,
    objective: str,
    seed_indices: list[int],
    valid_indices: list[int],
    n_iterations: int,
    top_k_set: set[int],
    runs_root: Path,
    repeat_seed: int,
    constraints: list[dict[str, Any]] | None,
    cv_folds: int,
    max_features: int | None,
    drop_cols: list[str],
    verbose: bool,
) -> dict[str, Any]:
    """Run one benchmark repeat. Returns raw metrics dict."""
    with tempfile.NamedTemporaryFile(
        suffix=".csv", delete=False, mode="w", newline=""
    ) as f:
        tmp_path = Path(f.name)
    full_df.to_csv(tmp_path, index=False)

    try:
        engine = BOEngine(runs_root=runs_root)
        init_result = engine.init_run(
            dataset_path=tmp_path,
            target_column=target_col,
            objective=objective,
            seed=repeat_seed,
            drop_cols=drop_cols,
            constraints=constraints or [],
            verbose=verbose,
        )
        run_id = init_result["run_id"]
        run_dir = engine.get_run_dir(run_id)

        oracle_result = build_proxy_oracle(
            run_dir,
            cv_folds=cv_folds,
            max_features=max_features,
            verbose=verbose,
        )
        cv_rmse = float(oracle_result["selected_rmse"])

        # Re-read state after oracle build — active_features may have been pruned
        state = read_json(run_dir / "state.json")
        active_features: list[str] = list(state["active_features"])

        # ------------------------------------------------------------------
        # Spearman rank correlation (out-of-sample via cross-validation)
        #
        # The oracle's build step uses cross_val_score (scalars only).
        # We clone the fitted pipeline and run cross_val_predict to get
        # genuine held-out predictions for every row, then compute Spearman
        # on those. Internal-scale predictions are converted back to
        # original scale before computing correlation.
        # ------------------------------------------------------------------
        with open(run_dir / "oracle.pkl", "rb") as f:
            fitted_pipeline = pickle.load(f)
        unfitted_pipeline = clone(fitted_pipeline)

        x_all = full_df.reindex(columns=active_features).fillna(0.0)
        y_true_all = full_df[target_col].values.astype(float)

        # Oracle trains on internally-normalized targets (always minimise).
        # For max objectives: y_internal = target_max - y_true
        obj_transform = state.get("objective_transform", {})
        target_max = obj_transform.get("target_max_for_restore")
        y_internal = (target_max - y_true_all) if target_max is not None else y_true_all

        cv_preds_internal = cross_val_predict(
            unfitted_pipeline, x_all, y_internal, cv=cv_folds
        )
        # Convert back to original scale
        cv_preds = (target_max - cv_preds_internal) if target_max is not None else cv_preds_internal

        def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float | None:
            if len(a) < 2:
                return None
            val = spearmanr(a, b).statistic
            return None if pd.isna(val) else float(val)

        sr_all = _safe_spearman(cv_preds, y_true_all)

        top_k_mask = np.array([i in top_k_set for i in full_df.index])
        sr_top_k = _safe_spearman(cv_preds[top_k_mask], y_true_all[top_k_mask])

        # ------------------------------------------------------------------
        # Random baseline: shuffle non-seed rows, pick first n_iterations
        # Uses a separate rng to avoid coupling with seed selection.
        # ------------------------------------------------------------------
        rand_rng = np.random.default_rng(repeat_seed + 7919)
        rand_available = [i for i in valid_indices if i not in set(seed_indices)]
        n_rand = min(n_iterations, len(rand_available))
        rand_indices = rand_rng.choice(rand_available, size=n_rand, replace=False).tolist()

        rand_true_values: list[float] = [float(full_df.at[i, target_col]) for i in rand_indices]
        rand_in_top_k: list[bool] = [i in top_k_set for i in rand_indices]

        rand_cum_best: list[float] = []
        running = float("inf") if objective == "min" else float("-inf")
        for v in rand_true_values:
            running = min(running, v) if objective == "min" else max(running, v)
            rand_cum_best.append(running)

        rand_top_k_cumulative = [
            sum(rand_in_top_k[: i + 1]) / (i + 1) for i in range(len(rand_in_top_k))
        ]
        rand_first_hit = next(
            (i + 1 for i, hit in enumerate(rand_in_top_k) if hit), None
        )

        # ------------------------------------------------------------------
        # Oracle-guided BO loop
        # ------------------------------------------------------------------
        norm_matrix, norm_mean, norm_std = _build_norm_matrix(full_df, active_features)

        seed_obs = [
            {
                "x": {
                    c: float(full_df.at[i, c])
                    for c in active_features
                    if c in full_df.columns
                },
                "y": float(full_df.at[i, target_col]),
            }
            for i in seed_indices
            if pd.notna(full_df.at[i, target_col])
        ]
        engine.observe(run_id, seed_obs, verbose=False)

        excluded: set[int] = set(seed_indices)
        true_values: list[float] = []
        in_top_k: list[bool] = []

        for _ in range(n_iterations):
            suggest_result = engine.suggest(run_id, batch_size=1, verbose=False)
            suggestions = suggest_result.get("suggestions", [])
            if not suggestions:
                continue

            obs_for_engine: list[dict[str, Any]] = []
            for s in suggestions:
                x = s["x"]

                nn_idx = _find_nearest(
                    x, active_features, norm_matrix, norm_mean, norm_std, excluded
                )
                excluded.add(nn_idx)
                true_y = float(full_df.at[nn_idx, target_col])
                true_values.append(true_y)
                in_top_k.append(nn_idx in top_k_set)

                x_df = pd.DataFrame([{c: x.get(c, 0.0) for c in active_features}])
                y_pred = float(predict_original_scale(run_dir, state, x_df)[0])
                obs_for_engine.append({"x": x, "y": y_pred})

            engine.observe(run_id, obs_for_engine, source="proxy-oracle", verbose=False)

        # Cumulative best (true values)
        cum_best: list[float] = []
        running = float("inf") if objective == "min" else float("-inf")
        for v in true_values:
            running = min(running, v) if objective == "min" else max(running, v)
            cum_best.append(running)

        top_k_cumulative = [
            sum(in_top_k[: i + 1]) / (i + 1) for i in range(len(in_top_k))
        ]
        first_hit = next(
            (i + 1 for i, hit in enumerate(in_top_k) if hit), None
        )

        return {
            "cv_rmse": cv_rmse,
            "spearman_all": sr_all,
            "spearman_top_k": sr_top_k,
            "cumulative_best": cum_best,
            "top_k_cumulative": top_k_cumulative,
            "in_top_k": in_top_k,
            "first_top_k_hit": first_hit,
            "true_values": true_values,
            "random": {
                "cumulative_best": rand_cum_best,
                "top_k_cumulative": rand_top_k_cumulative,
                "in_top_k": rand_in_top_k,
                "first_top_k_hit": rand_first_hit,
            },
        }
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Benchmark orchestration
# ---------------------------------------------------------------------------


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    full_df = pd.read_csv(args.dataset)
    print(
        f"Loaded {args.dataset}: {full_df.shape[0]} rows, {full_df.shape[1]} columns",
        file=sys.stderr,
    )

    y_series = pd.to_numeric(full_df[args.target], errors="coerce")
    y_valid = y_series.dropna()
    n_top_k = max(1, int(len(y_valid) * args.top_k_pct / 100))

    if args.objective == "min":
        threshold = float(y_valid.nsmallest(n_top_k).max())
        top_k_set = set(full_df.index[y_series <= threshold].tolist())
        true_optimum = float(y_valid.min())
    else:
        threshold = float(y_valid.nlargest(n_top_k).min())
        top_k_set = set(full_df.index[y_series >= threshold].tolist())
        true_optimum = float(y_valid.max())

    print(
        f"True optimum: {true_optimum:.4f} | "
        f"Top-{args.top_k_pct}% threshold: {threshold:.4f} | "
        f"Top-K set size: {len(top_k_set)}",
        file=sys.stderr,
    )

    drop_cols = (
        [c.strip() for c in args.drop_cols.split(",") if c.strip()]
        if args.drop_cols
        else []
    )

    constraints: list[dict[str, Any]] | None = None
    if args.simplex_groups:
        constraints = []
        for group_str in args.simplex_groups:
            parts = group_str.split(":")
            cols = [c.strip() for c in parts[0].split(",") if c.strip()]
            total = float(parts[1]) if len(parts) > 1 else 1.0
            constraints.append({"type": "simplex", "cols": cols, "total": total})

    runs_root = Path(args.runs_root)
    valid_indices = list(full_df.index[y_series.notna()])

    repeat_results: list[dict[str, Any]] = []
    for rep in range(args.repeats):
        repeat_seed = args.seed + rep
        # Each repeat gets its own independent RNG so repeats are individually
        # reproducible regardless of how many repeats are run.
        repeat_rng = np.random.default_rng(repeat_seed)

        if args.stratified_seeds:
            seed_indices = _stratified_seed_indices(
                y_series, valid_indices, args.seed_count, args.n_strata, repeat_rng
            )
        else:
            seed_indices = list(
                repeat_rng.choice(
                    valid_indices,
                    size=min(args.seed_count, len(valid_indices)),
                    replace=False,
                )
            )

        print(
            f"Repeat {rep + 1}/{args.repeats} (seed={repeat_seed}, "
            f"stratified={args.stratified_seeds})...",
            file=sys.stderr,
        )
        result = _run_single_repeat(
            full_df=full_df,
            target_col=args.target,
            objective=args.objective,
            seed_indices=[int(i) for i in seed_indices],
            valid_indices=valid_indices,
            n_iterations=args.iterations,
            top_k_set=top_k_set,
            runs_root=runs_root,
            repeat_seed=repeat_seed,
            constraints=constraints,
            cv_folds=args.cv_folds,
            max_features=args.max_features,
            drop_cols=drop_cols,
            verbose=args.verbose,
        )
        repeat_results.append(result)
        first_hit_str = str(result["first_top_k_hit"]) if result["first_top_k_hit"] else "never"
        sr_topk_str = f"{result['spearman_top_k']:.3f}" if result["spearman_top_k"] is not None else "n/a"
        print(
            f"  CV RMSE: {result['cv_rmse']:.4f} | "
            f"Spearman (all): {result['spearman_all']:.3f} | "
            f"Spearman (top-K%): {sr_topk_str} | "
            f"Iters to first top-{args.top_k_pct}% hit: {first_hit_str}",
            file=sys.stderr,
        )

    # Aggregate across repeats
    cv_rmse_vals = [r["cv_rmse"] for r in repeat_results]
    spearman_all_vals = [r["spearman_all"] for r in repeat_results]
    spearman_top_k_vals = [r["spearman_top_k"] for r in repeat_results if r["spearman_top_k"] is not None]
    first_hits = [r["first_top_k_hit"] for r in repeat_results if r["first_top_k_hit"] is not None]
    final_recovery = [
        sum(r["in_top_k"]) / len(r["in_top_k"])
        for r in repeat_results
        if r["in_top_k"]
    ]

    rand_first_hits = [r["random"]["first_top_k_hit"] for r in repeat_results if r["random"]["first_top_k_hit"] is not None]
    rand_final_recovery = [
        sum(r["random"]["in_top_k"]) / len(r["random"]["in_top_k"])
        for r in repeat_results
        if r["random"]["in_top_k"]
    ]

    n_iters = min(len(r["cumulative_best"]) for r in repeat_results)
    n_rand_iters = min(len(r["random"]["cumulative_best"]) for r in repeat_results)

    cum_best_matrix = np.array([r["cumulative_best"][:n_iters] for r in repeat_results])
    top_k_matrix = np.array([r["top_k_cumulative"][:n_iters] for r in repeat_results])
    rand_cum_best_matrix = np.array([r["random"]["cumulative_best"][:n_rand_iters] for r in repeat_results])
    rand_top_k_matrix = np.array([r["random"]["top_k_cumulative"][:n_rand_iters] for r in repeat_results])

    output: dict[str, Any] = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "dataset": str(args.dataset),
        "target": args.target,
        "objective": args.objective,
        "seed_count": args.seed_count,
        "stratified_seeds": args.stratified_seeds,
        "n_strata": args.n_strata,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "top_k_pct": args.top_k_pct,
        "true_optimum": true_optimum,
        "top_k_threshold": threshold,
        "cv_rmse": {
            "mean": float(np.mean(cv_rmse_vals)),
            "std": float(np.std(cv_rmse_vals)),
            "values": cv_rmse_vals,
        },
        "spearman_all": {
            "mean": float(np.mean(spearman_all_vals)),
            "std": float(np.std(spearman_all_vals)),
            "values": spearman_all_vals,
        },
        "spearman_top_k": {
            "mean": float(np.mean(spearman_top_k_vals)) if spearman_top_k_vals else None,
            "std": float(np.std(spearman_top_k_vals)) if spearman_top_k_vals else None,
            "values": spearman_top_k_vals,
        },
        "final_top_k_recovery": {
            "mean": float(np.mean(final_recovery)),
            "std": float(np.std(final_recovery)),
            "values": final_recovery,
        },
        "first_top_k_hit_iteration": {
            "mean": float(np.mean(first_hits)) if first_hits else None,
            "std": float(np.std(first_hits)) if first_hits else None,
            "values": first_hits,
            "never_found_count": args.repeats - len(first_hits),
        },
        "cumulative_best_mean": cum_best_matrix.mean(axis=0).tolist(),
        "cumulative_best_std": cum_best_matrix.std(axis=0).tolist(),
        "top_k_recovery_mean": top_k_matrix.mean(axis=0).tolist(),
        "top_k_recovery_std": top_k_matrix.std(axis=0).tolist(),
        "random_baseline": {
            "final_top_k_recovery": {
                "mean": float(np.mean(rand_final_recovery)) if rand_final_recovery else None,
                "std": float(np.std(rand_final_recovery)) if rand_final_recovery else None,
                "values": rand_final_recovery,
            },
            "first_top_k_hit_iteration": {
                "mean": float(np.mean(rand_first_hits)) if rand_first_hits else None,
                "std": float(np.std(rand_first_hits)) if rand_first_hits else None,
                "values": rand_first_hits,
                "never_found_count": args.repeats - len(rand_first_hits),
            },
            "cumulative_best_mean": rand_cum_best_matrix.mean(axis=0).tolist(),
            "cumulative_best_std": rand_cum_best_matrix.std(axis=0).tolist(),
            "top_k_recovery_mean": rand_top_k_matrix.mean(axis=0).tolist(),
            "top_k_recovery_std": rand_top_k_matrix.std(axis=0).tolist(),
        },
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2))
        print(f"Results saved: {out_path}", file=sys.stderr)

    if args.plot_out:
        _save_single_plot(output, label=args.label or Path(args.dataset).stem, out_path=Path(args.plot_out))

    return output


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_COLORS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]
_RANDOM_COLOR = "#9E9E9E"


def _save_single_plot(results: dict[str, Any], label: str, out_path: Path) -> None:
    """Two-panel plot for a single benchmark run (oracle vs random baseline)."""
    _save_comparison_plot([results], [label], out_path)


def _save_comparison_plot(
    all_results: list[dict[str, Any]], labels: list[str], out_path: Path
) -> None:
    """Two-panel comparison plot. Random baseline from first result shown as reference."""
    fig, (ax_best, ax_topk) = plt.subplots(1, 2, figsize=(12, 5))

    objective = all_results[0]["objective"]
    target = all_results[0]["target"]
    true_optimum = all_results[0]["true_optimum"]
    top_k_pct = all_results[0]["top_k_pct"]

    # Random baseline from first result (same dataset → same reference)
    rand = all_results[0].get("random_baseline", {})
    if rand.get("cumulative_best_mean"):
        n_rand = len(rand["cumulative_best_mean"])
        rand_iters = np.arange(1, n_rand + 1)
        rand_mean_best = np.array(rand["cumulative_best_mean"])
        rand_std_best = np.array(rand["cumulative_best_std"])
        rand_mean_topk = np.array(rand["top_k_recovery_mean"]) * 100
        rand_std_topk = np.array(rand["top_k_recovery_std"]) * 100

        ax_best.plot(rand_iters, rand_mean_best, color=_RANDOM_COLOR, lw=1.5,
                     linestyle="--", label="Random (mean)", zorder=1)
        ax_best.fill_between(rand_iters,
                              rand_mean_best - rand_std_best,
                              rand_mean_best + rand_std_best,
                              color=_RANDOM_COLOR, alpha=0.12, zorder=1)

        ax_topk.plot(rand_iters, rand_mean_topk, color=_RANDOM_COLOR, lw=1.5,
                     linestyle="--", label="Random (mean)", zorder=1)
        ax_topk.fill_between(rand_iters,
                              np.clip(rand_mean_topk - rand_std_topk, 0, 100),
                              np.clip(rand_mean_topk + rand_std_topk, 0, 100),
                              color=_RANDOM_COLOR, alpha=0.12, zorder=1)

    for i, (results, label) in enumerate(zip(all_results, labels)):
        color = _COLORS[i % len(_COLORS)]
        n = len(results["cumulative_best_mean"])
        iters = np.arange(1, n + 1)

        mean_best = np.array(results["cumulative_best_mean"])
        std_best = np.array(results["cumulative_best_std"])
        mean_topk = np.array(results["top_k_recovery_mean"]) * 100
        std_topk = np.array(results["top_k_recovery_std"]) * 100

        if objective == "min":
            best_lo = np.clip(mean_best - std_best, true_optimum, None)
            best_hi = mean_best + std_best
        else:
            best_lo = mean_best - std_best
            best_hi = np.clip(mean_best + std_best, None, true_optimum)
        topk_lo = np.clip(mean_topk - std_topk, 0, 100)
        topk_hi = np.clip(mean_topk + std_topk, 0, 100)

        ax_best.plot(iters, mean_best, color=color, lw=2, label=f"{label} (mean)", zorder=2)
        ax_best.fill_between(iters, best_lo, best_hi, color=color, alpha=0.18,
                              label=f"{label} (±1 std)", zorder=2)

        ax_topk.plot(iters, mean_topk, color=color, lw=2, label=f"{label} (mean)", zorder=2)
        ax_topk.fill_between(iters, topk_lo, topk_hi, color=color, alpha=0.18,
                              label=f"{label} (±1 std)", zorder=2)

    ax_best.axhline(
        true_optimum, linestyle=":", color="black", linewidth=1,
        label=f"True optimum ({true_optimum:.3f})",
    )
    direction = "↓ min" if objective == "min" else "↑ max"
    ax_best.set_title(f"Cumulative Best True Value ({direction})", fontweight="bold")
    ax_best.set_xlabel("Iteration")
    ax_best.set_ylabel(target)
    ax_best.legend(fontsize=8)
    ax_best.grid(True, alpha=0.3, linestyle="--")

    ax_topk.set_title(f"Top-{top_k_pct:.0f}% Recovery Rate", fontweight="bold")
    ax_topk.set_xlabel("Iteration")
    ax_topk.set_ylabel(f"% of suggestions in top {top_k_pct:.0f}%")
    ax_topk.set_ylim(0, 100)
    ax_topk.legend(fontsize=8)
    ax_topk.grid(True, alpha=0.3, linestyle="--")

    fig.suptitle(
        f"Oracle Benchmark — {all_results[0]['target']} ({all_results[0]['dataset']})",
        fontsize=11,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {out_path}", file=sys.stderr)


def _print_summary(results: dict[str, Any], label: str) -> None:
    print(f"\n{'=' * 55}")
    print(f"  {label}")
    print(f"{'=' * 55}")
    rmse = results["cv_rmse"]
    print(f"  CV RMSE              : {rmse['mean']:.4f} ± {rmse['std']:.4f}")

    sr_all = results.get("spearman_all", {})
    if sr_all.get("mean") is not None:
        print(f"  Spearman (all)       : {sr_all['mean']:.3f} ± {sr_all['std']:.3f}")

    sr_topk = results.get("spearman_top_k", {})
    if sr_topk.get("mean") is not None:
        print(f"  Spearman (top-K%)    : {sr_topk['mean']:.3f} ± {sr_topk['std']:.3f}")

    rec = results["final_top_k_recovery"]
    rand_rec = results.get("random_baseline", {}).get("final_top_k_recovery", {})
    oracle_str = f"{rec['mean'] * 100:.1f}% ± {rec['std'] * 100:.1f}%"
    rand_str = (
        f"{rand_rec['mean'] * 100:.1f}% ± {rand_rec['std'] * 100:.1f}%"
        if rand_rec.get("mean") is not None
        else "n/a"
    )
    print(f"  Top-K% recovery      : {oracle_str}  (random: {rand_str})")

    hits = results["first_top_k_hit_iteration"]
    rand_hits = results.get("random_baseline", {}).get("first_top_k_hit_iteration", {})
    if hits["mean"] is not None:
        oracle_hit_str = f"iter {hits['mean']:.1f} ± {hits['std']:.1f}"
    else:
        oracle_hit_str = "never"
    rand_hit_str = (
        f"iter {rand_hits['mean']:.1f} ± {rand_hits['std']:.1f}"
        if rand_hits.get("mean") is not None
        else "never"
    )
    print(f"  Iters to first top-K%: {oracle_hit_str}  (random: {rand_hit_str})")

    never = hits.get("never_found_count", 0)
    if never:
        print(f"  Never found          : {never}/{results['repeats']} repeats")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Benchmark oracle performance and compare before/after upgrades.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="mode")

    # --- run mode ---
    run = sub.add_parser("run", help="Run benchmark on a dataset")
    run.add_argument("--dataset", type=Path, required=True)
    run.add_argument("--target", required=True, help="Target column name")
    run.add_argument(
        "--objective", required=True, choices=["min", "max"], help="Optimisation direction"
    )
    run.add_argument("--seed-count", type=int, default=50, help="Seed observations per repeat")
    run.add_argument("--iterations", type=int, default=50, help="BO iterations per repeat")
    run.add_argument("--repeats", type=int, default=10, help="Independent repeats")
    run.add_argument("--top-k-pct", type=float, default=10.0, help="Top-K%% threshold")
    run.add_argument("--cv-folds", type=int, default=5)
    run.add_argument("--max-features", type=int, default=None)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--drop-cols", type=str, default=None, help="Comma-separated cols to drop")
    run.add_argument(
        "--simplex-groups",
        nargs="+",
        default=None,
        help="Simplex groups e.g. 'col1,col2,col3:1.0' (repeat for multiple groups)",
    )
    run.add_argument(
        "--stratified-seeds",
        action="store_true",
        default=False,
        help="Sample seeds proportionally across quantile strata of the target "
             "(reduces variance for skewed targets)",
    )
    run.add_argument(
        "--n-strata",
        type=int,
        default=5,
        help="Number of quantile strata for stratified seeding",
    )
    run.add_argument("--runs-root", type=Path, default=Path("runs"))
    run.add_argument("--output", type=Path, default=None, help="Save results JSON here")
    run.add_argument("--plot-out", type=Path, default=None, help="Save single-run plot here")
    run.add_argument("--label", type=str, default=None, help="Label for plot legend")
    run.add_argument("--verbose", action="store_true")

    # --- compare mode ---
    cmp = sub.add_parser("compare", help="Compare two or more saved benchmark JSON files")
    cmp.add_argument("results", nargs="+", type=Path, help="Benchmark JSON files to compare")
    cmp.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Display labels (one per results file)",
    )
    cmp.add_argument(
        "--plot-out",
        type=Path,
        default=Path("results/oracle_comparison.pdf"),
        help="Output PDF path",
    )

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.mode == "run":
        results = run_benchmark(args)
        label = args.label or Path(args.dataset).stem
        _print_summary(results, label)
        print(json.dumps({"cv_rmse_mean": results["cv_rmse"]["mean"]}, indent=2))
        return 0

    if args.mode == "compare":
        all_results = []
        for path in args.results:
            if not path.exists():
                print(f"File not found: {path}", file=sys.stderr)
                return 1
            all_results.append(json.loads(path.read_text()))

        labels = args.labels or [p.stem for p in args.results]
        if len(labels) != len(all_results):
            print(
                f"--labels count ({len(labels)}) must match results count ({len(all_results)})",
                file=sys.stderr,
            )
            return 1

        # Validate that all results share the same benchmark setup so the
        # comparison plot is meaningful.
        _COMPARE_KEYS = ["dataset", "target", "objective", "top_k_pct", "iterations"]
        ref = all_results[0]
        for path, result in zip(args.results[1:], all_results[1:]):
            mismatches = [
                k for k in _COMPARE_KEYS if result.get(k) != ref.get(k)
            ]
            if mismatches:
                print(
                    f"Warning: {path.name} differs from {args.results[0].name} "
                    f"on: {', '.join(mismatches)}. Comparison plot may be misleading.",
                    file=sys.stderr,
                )

        for result, label in zip(all_results, labels):
            _print_summary(result, label)

        _save_comparison_plot(all_results, labels, args.plot_out)
        print(
            json.dumps({"plot": str(args.plot_out), "n_compared": len(all_results)}, indent=2)
        )
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
