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

Usage:
    # Benchmark on BH dataset (primary development dataset)
    # Outputs: results/bh/BH_synthesis_data.json, results/bh/BH_synthesis_data.pdf, results/bh/runs/
    uv run python -m bo_workflow.scripts.oracle_benchmark run \\
        --dataset data/BH_synthesis_data.csv \\
        --target yield --objective max \\
        --max-features 20 \\
        --seed-count 20 --iterations 50 --repeats 10 \\
        --output-dir results/bh
"""

import argparse
import json
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, dict[str, int]]]:
    """Return (norm_matrix, mean, std, cat_encoders) for nearest-neighbour search.

    Categorical columns are label-encoded (sort=True) so that HEBO suggestions
    (which are also ordinally encoded) can be matched by Euclidean distance.
    cat_encoders maps column name -> {category_string: ordinal_code}.
    """
    encoded = df[feature_cols].copy()
    cat_encoders: dict[str, dict[str, int]] = {}
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(encoded[col]):
            vals = encoded[col].fillna("no_value").astype(str)
            labels = sorted(set(vals))
            label_to_code = {label: i for i, label in enumerate(labels)}
            cat_encoders[col] = label_to_code
            encoded[col] = vals.map(label_to_code)
        else:
            encoded[col] = pd.to_numeric(encoded[col], errors="coerce").fillna(0.0)
    matrix = encoded.values.astype(float)
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std[std == 0] = 1.0
    return (matrix - mean) / std, mean, std, cat_encoders


def _find_nearest(
    suggestion_x: dict[str, Any],
    feature_cols: list[str],
    norm_matrix: np.ndarray,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    excluded: set[int],
    cat_encoders: dict[str, dict[str, int]] | None = None,
) -> int:
    """Return nearest non-excluded row index by normalised Euclidean distance."""
    cat_encoders = cat_encoders or {}
    query = []
    for c in feature_cols:
        val = suggestion_x.get(c)
        if c in cat_encoders:
            key = str(val) if val is not None else "__none__"
            query.append(float(cat_encoders[c].get(key, -1)))
        else:
            try:
                query.append(float(val) if val is not None else 0.0)
            except (TypeError, ValueError):
                query.append(0.0)
    query_norm = (np.array(query, dtype=float) - norm_mean) / norm_std
    distances = np.sqrt(((norm_matrix - query_norm) ** 2).sum(axis=1))
    for idx in np.argsort(distances):
        if int(idx) not in excluded:
            return int(idx)
    return int(np.argmin(distances))  # fallback: all rows excluded


# ---------------------------------------------------------------------------
# Single repeat
# ---------------------------------------------------------------------------


def _run_single_model(
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
    top_k_pct: float,
    drop_cols: list[str],
    model_name: str,
    verbose: bool,
) -> dict[str, Any]:
    """Run BO simulation for a single oracle model. Returns metrics dict."""
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

        build_proxy_oracle(
            run_dir,
            model_candidates=(model_name,),
            cv_folds=cv_folds,
            max_features=max_features,
            top_k_pct=top_k_pct,
            verbose=verbose,
        )

        # Re-read state after oracle build — active_features may have been pruned
        state = read_json(run_dir / "state.json")
        active_features: list[str] = list(state["active_features"])

        # CV scores are already computed and stored by build_proxy_oracle
        oracle_meta = read_json(run_dir / "oracle_meta.json")
        model_scores = oracle_meta["scores"][model_name]

        norm_matrix, norm_mean, norm_std, cat_encoders = _build_norm_matrix(full_df, active_features)

        seed_obs = [
            {
                "x": {
                    c: (
                        float(full_df.at[i, c])
                        if pd.api.types.is_numeric_dtype(full_df[c])
                        else ("__none__" if pd.isna(full_df.at[i, c]) else str(full_df.at[i, c]))
                    )
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
                    x, active_features, norm_matrix, norm_mean, norm_std, excluded,
                    cat_encoders=cat_encoders,
                )
                excluded.add(nn_idx)
                true_y = float(full_df.at[nn_idx, target_col])
                true_values.append(true_y)
                in_top_k.append(nn_idx in top_k_set)

                x_df = pd.DataFrame([{c: x.get(c) for c in active_features}])
                y_pred = float(predict_original_scale(run_dir, state, x_df)[0])
                obs_for_engine.append({"x": x, "y": y_pred})

            engine.observe(run_id, obs_for_engine, source="proxy-oracle", verbose=False)

        cum_best: list[float] = []
        running = float("inf") if objective == "min" else float("-inf")
        for v in true_values:
            running = min(running, v) if objective == "min" else max(running, v)
            cum_best.append(running)

        top_k_cumulative = [
            sum(in_top_k[: i + 1]) / (i + 1) for i in range(len(in_top_k))
        ]
        first_hit = next((i + 1 for i, hit in enumerate(in_top_k) if hit), None)

        return {
            "cv_rmse": model_scores["rmse"],
            "spearman_all": model_scores["spearman_all"],
            "spearman_top_k": model_scores["spearman_top_k"],
            "cumulative_best": cum_best,
            "top_k_cumulative": top_k_cumulative,
            "in_top_k": in_top_k,
            "first_top_k_hit": first_hit,
            "true_values": true_values,
        }
    finally:
        tmp_path.unlink(missing_ok=True)


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
    top_k_pct: float,
    drop_cols: list[str],
    model_candidates: list[str],
    verbose: bool,
) -> dict[str, Any]:
    """Run one benchmark repeat across all model candidates. Returns per-model results + random baseline."""
    # Random baseline is model-independent — compute once per repeat.
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
    rand_first_hit = next((i + 1 for i, hit in enumerate(rand_in_top_k) if hit), None)

    model_results: dict[str, dict[str, Any]] = {}
    for model_name in model_candidates:
        model_results[model_name] = _run_single_model(
            full_df=full_df,
            target_col=target_col,
            objective=objective,
            seed_indices=seed_indices,
            valid_indices=valid_indices,
            n_iterations=n_iterations,
            top_k_set=top_k_set,
            runs_root=runs_root,
            repeat_seed=repeat_seed,
            constraints=constraints,
            cv_folds=cv_folds,
            max_features=max_features,
            top_k_pct=top_k_pct,
            drop_cols=drop_cols,
            model_name=model_name,
            verbose=verbose,
        )

    return {
        "model_results": model_results,
        "random": {
            "cumulative_best": rand_cum_best,
            "top_k_cumulative": rand_top_k_cumulative,
            "in_top_k": rand_in_top_k,
            "first_top_k_hit": rand_first_hit,
        },
    }


# ---------------------------------------------------------------------------
# Benchmark orchestration
# ---------------------------------------------------------------------------


def _aggregate_model_repeats(
    repeat_results: list[dict[str, Any]],
    model_name: str,
    total_repeats: int,
) -> dict[str, Any]:
    """Aggregate per-model metrics across repeats."""
    reps = [r["model_results"][model_name] for r in repeat_results]

    cv_rmse_vals = [r["cv_rmse"] for r in reps]
    sr_all_vals = [r["spearman_all"] for r in reps if r["spearman_all"] is not None]
    sr_topk_vals = [r["spearman_top_k"] for r in reps if r["spearman_top_k"] is not None]
    first_hits = [r["first_top_k_hit"] for r in reps if r["first_top_k_hit"] is not None]
    final_recovery = [sum(r["in_top_k"]) / len(r["in_top_k"]) for r in reps if r["in_top_k"]]

    n_iters = min(len(r["cumulative_best"]) for r in reps)
    cum_best_matrix = np.array([r["cumulative_best"][:n_iters] for r in reps])
    top_k_matrix = np.array([r["top_k_cumulative"][:n_iters] for r in reps])

    return {
        "cv_rmse": {
            "mean": float(np.mean(cv_rmse_vals)),
            "std": float(np.std(cv_rmse_vals)),
            "values": cv_rmse_vals,
        },
        "spearman_all": {
            "mean": float(np.mean(sr_all_vals)) if sr_all_vals else None,
            "std": float(np.std(sr_all_vals)) if sr_all_vals else None,
            "values": sr_all_vals,
        },
        "spearman_top_k": {
            "mean": float(np.mean(sr_topk_vals)) if sr_topk_vals else None,
            "std": float(np.std(sr_topk_vals)) if sr_topk_vals else None,
            "values": sr_topk_vals,
        },
        "final_top_k_recovery": {
            "mean": float(np.mean(final_recovery)) if final_recovery else None,
            "std": float(np.std(final_recovery)) if final_recovery else None,
            "values": final_recovery,
        },
        "first_top_k_hit_iteration": {
            "mean": float(np.mean(first_hits)) if first_hits else None,
            "std": float(np.std(first_hits)) if first_hits else None,
            "values": first_hits,
            "never_found_count": total_repeats - len(first_hits),
        },
        "cumulative_best_mean": cum_best_matrix.mean(axis=0).tolist(),
        "cumulative_best_std": cum_best_matrix.std(axis=0).tolist(),
        "top_k_recovery_mean": top_k_matrix.mean(axis=0).tolist(),
        "top_k_recovery_std": top_k_matrix.std(axis=0).tolist(),
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    full_df = pd.read_csv(args.dataset)
    # Fill NaN in categorical columns so "no value" becomes a first-class category
    # that HEBO includes in its design space from the start.
    for _col in full_df.select_dtypes(include="object").columns:
        full_df[_col] = full_df[_col].fillna("no_value")
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

    model_candidates: list[str] = args.model_candidates or [
        "random_forest", "extra_trees", "gradient_boosting", "gaussian_process"
    ]
    runs_root = Path(args.output_dir) / "runs"
    valid_indices = list(full_df.index[y_series.notna()])

    repeat_results: list[dict[str, Any]] = []
    for rep in range(args.repeats):
        repeat_seed = args.seed + rep
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
            f"stratified={args.stratified_seeds}, models={model_candidates})...",
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
            top_k_pct=args.top_k_pct,
            drop_cols=drop_cols,
            model_candidates=model_candidates,
            verbose=args.verbose,
        )
        repeat_results.append(result)

        for model_name in model_candidates:
            mr = result["model_results"][model_name]
            sr_all_str = f"{mr['spearman_all']:.3f}" if mr["spearman_all"] is not None else "n/a"
            sr_topk_str = f"{mr['spearman_top_k']:.3f}" if mr["spearman_top_k"] is not None else "n/a"
            first_hit_str = str(mr["first_top_k_hit"]) if mr["first_top_k_hit"] else "never"
            print(
                f"  [{model_name}] RMSE={mr['cv_rmse']:.4f} | "
                f"Spearman(all)={sr_all_str} | "
                f"Spearman(top-K%)={sr_topk_str} | "
                f"First hit={first_hit_str}",
                file=sys.stderr,
            )

    # Aggregate per model across repeats
    aggregated_models: dict[str, Any] = {}
    for model_name in model_candidates:
        aggregated_models[model_name] = _aggregate_model_repeats(
            repeat_results, model_name, args.repeats
        )

    rand_reps = [r["random"] for r in repeat_results]
    rand_first_hits = [r["first_top_k_hit"] for r in rand_reps if r["first_top_k_hit"] is not None]
    rand_final_recovery = [
        sum(r["in_top_k"]) / len(r["in_top_k"]) for r in rand_reps if r["in_top_k"]
    ]
    n_rand_iters = min(len(r["cumulative_best"]) for r in rand_reps)
    rand_cum_best_matrix = np.array([r["cumulative_best"][:n_rand_iters] for r in rand_reps])
    rand_top_k_matrix = np.array([r["top_k_cumulative"][:n_rand_iters] for r in rand_reps])

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
        "models": aggregated_models,
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

    output_dir = Path(args.output_dir)
    dataset_stem = Path(args.dataset).stem
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{dataset_stem}.json"
    json_path.write_text(json.dumps(output, indent=2))
    print(f"Results saved: {json_path}", file=sys.stderr)

    _save_run_plot(output, out_path=output_dir / f"{dataset_stem}.pdf")

    return output


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_MODEL_COLORS = {
    "random_forest": "#2196F3",
    "extra_trees": "#4CAF50",
    "gradient_boosting": "#FF9800",
    "gaussian_process": "#9C27B0",
}
_FALLBACK_COLORS = ["#F44336", "#00BCD4", "#795548", "#607D8B"]
_RANDOM_COLOR = "#9E9E9E"


def _model_color(model_name: str, idx: int) -> str:
    return _MODEL_COLORS.get(model_name, _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)])


def _plot_convergence_panels(
    ax_best: Any,
    ax_topk: Any,
    results: dict[str, Any],
    objective: str,
    true_optimum: float,
    top_k_pct: float,
    target: str,
) -> None:
    """Plot per-model convergence lines + random baseline onto existing axes."""
    rand = results.get("random_baseline", {})
    if rand.get("cumulative_best_mean"):
        n_rand = len(rand["cumulative_best_mean"])
        rand_iters = np.arange(1, n_rand + 1)
        rand_mean_best = np.array(rand["cumulative_best_mean"])
        rand_mean_topk = np.array(rand["top_k_recovery_mean"]) * 100

        ax_best.plot(rand_iters, rand_mean_best, color=_RANDOM_COLOR, lw=1.5,
                     linestyle="--", label="random", zorder=1)
        ax_topk.plot(rand_iters, rand_mean_topk, color=_RANDOM_COLOR, lw=1.5,
                     linestyle="--", label="random", zorder=1)

    for idx, (model_name, model_data) in enumerate(results.get("models", {}).items()):
        color = _model_color(model_name, idx)
        n = len(model_data["cumulative_best_mean"])
        iters = np.arange(1, n + 1)

        mean_best = np.array(model_data["cumulative_best_mean"])
        mean_topk = np.array(model_data["top_k_recovery_mean"]) * 100

        ax_best.plot(iters, mean_best, color=color, lw=2, label=model_name, zorder=2)
        ax_topk.plot(iters, mean_topk, color=color, lw=2, label=model_name, zorder=2)

    ax_best.axhline(true_optimum, linestyle=":", color="black", linewidth=1,
                    label=f"true optimum ({true_optimum:.3f})")
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


def _build_summary_table_data(
    results: dict[str, Any],
    top_k_pct: float,
) -> tuple[list[list[str]], list[str]]:
    """Build (cell_data, col_labels) for the summary table."""
    col_labels = [
        "Model",
        "RMSE",
        f"Spearman (all)",
        f"Spearman (top-{top_k_pct:.0f}%)",
        f"Top-{top_k_pct:.0f}% Recovery",
        "First Hit (iter)",
    ]

    def _fmt(val: float | None, fmt: str = ".3f") -> str:
        return format(val, fmt) if val is not None else "n/a"

    def _fmt_pm(mean: float | None, std: float | None, fmt: str = ".3f") -> str:
        if mean is None:
            return "n/a"
        return f"{format(mean, fmt)} ± {format(std, fmt)}"

    rows: list[list[str]] = []
    for model_name, md in results.get("models", {}).items():
        rmse = md["cv_rmse"]
        sr_all = md["spearman_all"]
        sr_topk = md["spearman_top_k"]
        rec = md["final_top_k_recovery"]
        hits = md["first_top_k_hit_iteration"]
        never = hits.get("never_found_count", 0)
        hit_str = _fmt_pm(hits["mean"], hits["std"], ".1f") if hits["mean"] is not None else "never"
        if never:
            hit_str += f" ({never} never)"
        rows.append([
            model_name,
            _fmt_pm(rmse["mean"], rmse["std"], ".4f"),
            _fmt_pm(sr_all["mean"], sr_all["std"]),
            _fmt_pm(sr_topk["mean"], sr_topk["std"]) if sr_topk["mean"] is not None else "n/a",
            f"{rec['mean'] * 100:.1f}% ± {rec['std'] * 100:.1f}%" if rec["mean"] is not None else "n/a",
            hit_str,
        ])

    rand = results.get("random_baseline", {})
    rand_rec = rand.get("final_top_k_recovery", {})
    rand_hits = rand.get("first_top_k_hit_iteration", {})
    rand_rec_str = f"{rand_rec['mean'] * 100:.1f}% ± {rand_rec['std'] * 100:.1f}%" if rand_rec.get("mean") is not None else "n/a"
    rand_hit_str = _fmt_pm(rand_hits.get("mean"), rand_hits.get("std"), ".1f") if rand_hits.get("mean") is not None else "never"
    rows.append(["random", "—", "—", "—", rand_rec_str, rand_hit_str])

    return rows, col_labels


def _save_run_plot(results: dict[str, Any], out_path: Path) -> None:
    """Three-panel figure: cumulative best, top-K% recovery, summary table."""
    objective = results["objective"]
    target = results["target"]
    true_optimum = results["true_optimum"]
    top_k_pct = results["top_k_pct"]

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[2, 1], hspace=0.45, wspace=0.3)
    ax_best = fig.add_subplot(gs[0, 0])
    ax_topk = fig.add_subplot(gs[0, 1])
    ax_table = fig.add_subplot(gs[1, :])

    _plot_convergence_panels(ax_best, ax_topk, results, objective, true_optimum, top_k_pct, target)

    cell_data, col_labels = _build_summary_table_data(results, top_k_pct)
    ax_table.axis("off")
    tbl = ax_table.table(
        cellText=cell_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(col=list(range(len(col_labels))))

    # Colour header row
    for col_idx in range(len(col_labels)):
        tbl[(0, col_idx)].set_facecolor("#E3F2FD")
        tbl[(0, col_idx)].set_text_props(fontweight="bold")

    # Colour model rows to match convergence plot
    model_names = list(results.get("models", {}).keys())
    for row_idx, model_name in enumerate(model_names, start=1):
        color = _model_color(model_name, row_idx - 1)
        tbl[(row_idx, 0)].set_facecolor(color)
        tbl[(row_idx, 0)].set_text_props(color="white", fontweight="bold")

    ax_table.set_title("Summary (mean ± std across repeats)", fontweight="bold", pad=8)

    fig.suptitle(
        f"Oracle Benchmark — {target} ({results['dataset']})",
        fontsize=12,
        fontweight="bold",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {out_path}", file=sys.stderr)



def _print_summary(results: dict[str, Any], label: str) -> None:
    top_k_pct = results.get("top_k_pct", 3.0)
    repeats = results.get("repeats", "?")

    rand = results.get("random_baseline", {})
    rand_rec = rand.get("final_top_k_recovery", {})
    rand_hits = rand.get("first_top_k_hit_iteration", {})
    rand_rec_str = (
        f"{rand_rec['mean'] * 100:.1f}% ± {rand_rec['std'] * 100:.1f}%"
        if rand_rec.get("mean") is not None else "n/a"
    )
    rand_hit_str = (
        f"iter {rand_hits['mean']:.1f} ± {rand_hits['std']:.1f}"
        if rand_hits.get("mean") is not None else "never"
    )

    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")
    print(f"  {'Model':<22} {'RMSE':>10}  {'Spearman(all)':>14}  {'Spearman(top-K%)':>16}  {'Recovery':>10}  {'First hit':>12}")
    print(f"  {'-' * 66}")

    for model_name, md in results.get("models", {}).items():
        rmse = md["cv_rmse"]
        sr_all = md["spearman_all"]
        sr_topk = md["spearman_top_k"]
        rec = md["final_top_k_recovery"]
        hits = md["first_top_k_hit_iteration"]

        rmse_str = f"{rmse['mean']:.4f}"
        sr_all_str = f"{sr_all['mean']:.3f}" if sr_all.get("mean") is not None else "n/a"
        sr_topk_str = f"{sr_topk['mean']:.3f}" if sr_topk.get("mean") is not None else "n/a"
        rec_str = f"{rec['mean'] * 100:.1f}%" if rec.get("mean") is not None else "n/a"
        hit_str = f"iter {hits['mean']:.1f}" if hits.get("mean") is not None else "never"
        never = hits.get("never_found_count", 0)
        if never:
            hit_str += f" ({never}/{repeats} never)"

        print(f"  {model_name:<22} {rmse_str:>10}  {sr_all_str:>14}  {sr_topk_str:>16}  {rec_str:>10}  {hit_str:>12}")

    print(f"  {'random':<22} {'—':>10}  {'—':>14}  {'—':>16}  {rand_rec_str:>10}  {rand_hit_str:>12}")
    print(f"{'=' * 70}")


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
    run.add_argument("--top-k-pct", type=float, default=3.0, help="Top-K%% threshold")
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
    run.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for all outputs: JSON, PDF, and engine runs (default: results/)",
    )
    run.add_argument(
        "--model-candidates",
        nargs="+",
        default=None,
        help="Models to benchmark (default: all four). Choices: random_forest extra_trees gradient_boosting gaussian_process",
    )
    run.add_argument("--verbose", action="store_true")

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.mode == "run":
        results = run_benchmark(args)
        _print_summary(results, Path(args.dataset).stem)
        summary = {
            model: {
                "rmse_mean": results["models"][model]["cv_rmse"]["mean"],
                "spearman_top_k_mean": results["models"][model]["spearman_top_k"]["mean"],
            }
            for model in results["models"]
        }
        print(json.dumps(summary, indent=2))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
