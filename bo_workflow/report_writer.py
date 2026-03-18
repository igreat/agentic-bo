"""Narrative writer for Bayesian Optimization runs.

This module generates human-readable report sections from persisted BO run files.
It is intentionally separate from report/plot generation so the written content
can be reused across:
- LaTeX reports
- posters
- markdown summaries
- Claude skills such as bo-write-report

Inputs:
    runs/<run_id>/state.json
    runs/<run_id>/report.json
    runs/<run_id>/observations.jsonl

Output:
    runs/<run_id>/written_report.json
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import math

import numpy as np
import pandas as pd

from .utils import read_json, read_jsonl, utc_now_iso, write_json


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def _format_num(value: Any, decimals: int = 4, default: str = "N/A") -> str:
    try:
        if value is None:
            return default
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return default
        return f"{v:.{decimals}f}"
    except Exception:
        return default


def _objective_word(objective: str) -> str:
    return "maximize" if str(objective).lower() == "max" else "minimize"


def _engine_label(engine: str) -> str:
    mapping = {
        "hebo": "HEBO",
        "bo_lcb": "BO (LCB)",
        "random": "Random Search",
    }
    return mapping.get(str(engine).lower(), str(engine).upper())


def _oracle_quality_label(rmse: float | None) -> str:
    if rmse is None or math.isnan(rmse):
        return "unknown"
    if rmse < 0.1:
        return "high"
    if rmse < 0.5:
        return "moderate"
    return "low"


def _load_run(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], pd.DataFrame]:
    state = read_json(run_dir / "state.json")
    report = read_json(run_dir / "report.json")
    observations = read_jsonl(run_dir / "observations.jsonl")
    obs_df = pd.DataFrame(observations)
    return state, report, observations, obs_df


def _extract_rmse(state: dict[str, Any], report: dict[str, Any]) -> float | None:
    report_oracle = report.get("oracle", {})
    state_oracle = state.get("oracle", {})

    for source in (report_oracle, state_oracle):
        if isinstance(source, dict) and "selected_rmse" in source:
            val = _safe_float(source.get("selected_rmse"), default=float("nan"))
            if not math.isnan(val):
                return val

    cv_rmse = state_oracle.get("cv_rmse")
    if isinstance(cv_rmse, dict):
        for key in ("extra_trees", "random_forest"):
            if key in cv_rmse:
                val = _safe_float(cv_rmse.get(key), default=float("nan"))
                if not math.isnan(val):
                    return val

    return None


def _parameter_summary(design_parameters: list[dict[str, Any]], max_items: int = 6) -> str:
    if not design_parameters:
        return "a configurable design space"

    parts: list[str] = []
    for param in design_parameters[:max_items]:
        name = str(param.get("name", "parameter"))
        ptype = str(param.get("type", "num"))
        if ptype == "num":
            lb = _format_num(param.get("lb"), 2, "N/A")
            ub = _format_num(param.get("ub"), 2, "N/A")
            parts.append(f"{name} ({ptype}, range {lb} to {ub})")
        elif ptype == "cat":
            categories = param.get("categories", [])
            parts.append(f"{name} (categorical, {len(categories)} categories)")
        else:
            parts.append(f"{name} ({ptype})")

    if len(design_parameters) > max_items:
        parts.append(f"and {len(design_parameters) - max_items} additional parameters")

    return "; ".join(parts)


def _best_candidate_text(best_x: dict[str, Any], max_items: int = 6) -> str:
    if not isinstance(best_x, dict) or not best_x:
        return "No best candidate was available."

    parts: list[str] = []
    items = list(best_x.items())[:max_items]
    for k, v in items:
        if isinstance(v, (int, float)):
            parts.append(f"{k}={_format_num(v, 3, str(v))}")
        else:
            parts.append(f"{k}={v}")

    if len(best_x) > max_items:
        parts.append("...")

    return ", ".join(parts)


def _convergence_description(y_values: np.ndarray, best_idx: int) -> str:
    if len(y_values) == 0:
        return "No convergence behaviour could be assessed because no observations were recorded."

    if len(y_values) == 1:
        return "Only one observation was recorded, so convergence behaviour cannot yet be assessed."

    running_best = np.maximum.accumulate(y_values)
    final_best = running_best[-1]

    thirds = max(1, len(running_best) // 3)
    early_best = running_best[min(thirds - 1, len(running_best) - 1)]

    if best_idx <= max(2, len(y_values) // 5):
        timing = "The best value was found early in the run."
    elif best_idx >= max(1, int(0.8 * len(y_values))):
        timing = "The best value was found late in the run."
    else:
        timing = "The best value was found midway through the run."

    if np.isclose(final_best, early_best):
        trend = "Performance improved quickly and then largely plateaued."
    else:
        trend = "Performance continued to improve over multiple iterations rather than plateauing immediately."

    return f"{timing} {trend}"


def _compute_metrics(
    state: dict[str, Any],
    report: dict[str, Any],
    obs_df: pd.DataFrame,
) -> dict[str, Any]:
    objective = str(report.get("objective", state.get("objective", "max"))).lower()
    engine = str(report.get("engine") or state.get("default_engine") or state.get("engine") or "hebo")
    engine_name = _engine_label(engine)
    dataset_path = state.get("dataset_path", "")
    dataset_name = Path(dataset_path).name if dataset_path else "unknown dataset"
    target_column = report.get("target_column", state.get("target_column", "Target"))
    design_parameters = state.get("design_parameters", [])
    num_parameters = len(design_parameters)

    rmse = _extract_rmse(state, report)
    oracle_quality = _oracle_quality_label(rmse)

    if not obs_df.empty and "y" in obs_df.columns:
        y_series = pd.to_numeric(obs_df["y"], errors="coerce").dropna()
    else:
        y_series = pd.Series(dtype=float)

    if len(y_series) > 0:
        initial_value = float(y_series.iloc[0])
        if objective == "max":
            best_value_obs = float(y_series.max())
            best_row_idx = int(y_series.idxmax())
        else:
            best_value_obs = float(y_series.min())
            best_row_idx = int(y_series.idxmin())
    else:
        initial_value = 0.0
        best_value_obs = 0.0
        best_row_idx = 0

    best_value = _safe_float(report.get("best_value"), best_value_obs)
    best_iteration = int(report.get("best_iteration", best_row_idx))
    best_x = report.get("best_x", {})

    if objective == "max":
        improvement_abs = best_value - initial_value
        improvement_pct = (improvement_abs / abs(initial_value) * 100) if initial_value != 0 else 0.0
    else:
        improvement_abs = initial_value - best_value
        improvement_pct = (improvement_abs / abs(initial_value) * 100) if initial_value != 0 else 0.0

    num_observations = int(report.get("num_observations", len(obs_df)))

    y_np = y_series.to_numpy(dtype=float) if len(y_series) > 0 else np.array([], dtype=float)

    return {
        "objective": objective,
        "objective_word": _objective_word(objective),
        "engine": engine_name,
        "dataset_name": dataset_name,
        "dataset_path": dataset_path,
        "target_column": target_column,
        "num_parameters": num_parameters,
        "design_parameters": design_parameters,
        "parameter_summary": _parameter_summary(design_parameters),
        "num_observations": num_observations,
        "initial_value": initial_value,
        "best_value": best_value,
        "best_iteration": best_iteration,
        "best_x": best_x,
        "best_candidate_text": _best_candidate_text(best_x),
        "improvement_abs": improvement_abs,
        "improvement_pct": improvement_pct,
        "rmse": rmse,
        "oracle_quality": oracle_quality,
        "convergence_description": _convergence_description(y_np, best_iteration),
    }


def _build_sections(run_id: str, metrics: dict[str, Any]) -> dict[str, str]:
    abstract = (
        f"This report summarises Bayesian Optimization run {run_id}, performed on "
        f"{metrics['dataset_name']} using {metrics['engine']}. The goal was to "
        f"{metrics['objective_word']} {metrics['target_column']} across a "
        f"{metrics['num_parameters']}-parameter design space. Over "
        f"{metrics['num_observations']} evaluations, the best observed value reached "
        f"{_format_num(metrics['best_value'])}, representing a "
        f"{_format_num(metrics['improvement_pct'], 1)}% improvement over the initial observation. "
        f"{metrics['convergence_description']} The oracle RMSE was "
        f"{_format_num(metrics['rmse'], 4, 'N/A')}, indicating {metrics['oracle_quality']} predictive quality."
    )

    introduction = (
        f"The problem considered in this run is the optimisation of "
        f"{metrics['target_column']} on {metrics['dataset_name']}. "
        f"This matters because identifying strong parameter combinations efficiently can reduce "
        f"experimental or computational cost while improving performance. "
        f"Bayesian Optimization is a suitable approach because it is designed for sample-efficient "
        f"search in expensive, structured design spaces where each evaluation may be costly. "
        f"In this run, the search space consisted of {metrics['num_parameters']} parameters, including "
        f"{metrics['parameter_summary']}."
    )

    results = (
        f"The best observed objective value was {_format_num(metrics['best_value'])}, "
        f"found at iteration {metrics['best_iteration']}. "
        f"This corresponds to a {_format_num(metrics['improvement_pct'], 1)}% improvement relative "
        f"to the initial observed value of {_format_num(metrics['initial_value'])}. "
        f"The best candidate identified by the optimization was: {metrics['best_candidate_text']}. "
        f"{metrics['convergence_description']}"
    )

    discussion = (
        f"These results suggest that the optimisation workflow was able to identify promising regions "
        f"of the search space within {metrics['num_observations']} evaluations. "
        f"The best value of {_format_num(metrics['best_value'])} indicates that Bayesian Optimization "
        f"was effective at steering the search toward higher-performing candidates. "
        f"At the same time, interpretation should account for the oracle quality: the recorded RMSE of "
        f"{_format_num(metrics['rmse'], 4, 'N/A')} suggests {metrics['oracle_quality']} predictive fidelity. "
        f"This means the top candidate should be treated as a strong lead for validation, rather than "
        f"as a guaranteed optimum. Additional iterations or retraining with new data could further improve results."
    )

    summary = (
        f"In summary, this run used {metrics['engine']} to {metrics['objective_word']} "
        f"{metrics['target_column']} over {metrics['num_observations']} evaluations and achieved a best value "
        f"of {_format_num(metrics['best_value'])}. The optimisation improved substantially over the initial "
        f"baseline and found its best-performing candidate at iteration {metrics['best_iteration']}."
    )

    significance = (
        f"This is important because it demonstrates how Bayesian Optimization can reduce the effort needed "
        f"to discover strong parameter settings in a high-dimensional design space. "
        f"For research or applied optimisation workflows, this enables faster iteration, more informed "
        f"candidate selection, and better use of limited evaluation budgets. "
        f"The next steps should be to validate the top candidate, incorporate the new results into the dataset, "
        f"and continue optimisation or retrain the oracle if higher fidelity is needed."
    )

    poster_narrative = (
        f"This run applied Bayesian Optimization to {metrics['dataset_name']} with the aim of "
        f"{metrics['objective_word']}ing {metrics['target_column']}. "
        f"Using {metrics['engine']}, the workflow explored {metrics['num_observations']} candidate evaluations "
        f"across a {metrics['num_parameters']}-parameter design space. "
        f"The best observed value was {_format_num(metrics['best_value'])}, found at iteration "
        f"{metrics['best_iteration']}, which was a {_format_num(metrics['improvement_pct'], 1)}% improvement "
        f"over the initial result. {metrics['convergence_description']} "
        f"The oracle RMSE of {_format_num(metrics['rmse'], 4, 'N/A')} suggests "
        f"{metrics['oracle_quality']} predictive quality, so the highest-ranked candidates are strong priorities "
        f"for follow-up validation."
    )

    return {
        "abstract": abstract,
        "introduction": introduction,
        "results": results,
        "discussion": discussion,
        "summary": summary,
        "significance": significance,
        "poster_narrative": poster_narrative,
    }


def write_report_sections(run_id: str, runs_root: str | Path = "runs") -> dict[str, Any]:
    """Generate and save report-writing sections for a run."""
    run_dir = Path(runs_root) / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    state, report, observations, obs_df = _load_run(run_dir)
    metrics = _compute_metrics(state, report, obs_df)
    sections = _build_sections(run_id, metrics)

    payload = {
        "run_id": run_id,
        "generated_at": utc_now_iso(),
        "source_files": {
            "state": str(run_dir / "state.json"),
            "report": str(run_dir / "report.json"),
            "observations": str(run_dir / "observations.jsonl"),
        },
        "metrics": {
            "objective": metrics["objective"],
            "engine": metrics["engine"],
            "dataset_name": metrics["dataset_name"],
            "target_column": metrics["target_column"],
            "num_parameters": metrics["num_parameters"],
            "num_observations": metrics["num_observations"],
            "initial_value": metrics["initial_value"],
            "best_value": metrics["best_value"],
            "best_iteration": metrics["best_iteration"],
            "improvement_abs": metrics["improvement_abs"],
            "improvement_pct": metrics["improvement_pct"],
            "rmse": metrics["rmse"],
            "oracle_quality": metrics["oracle_quality"],
            "best_candidate": metrics["best_x"],
            "best_candidate_text": metrics["best_candidate_text"],
            "parameter_summary": metrics["parameter_summary"],
            "convergence_description": metrics["convergence_description"],
        },
        "sections": sections,
    }

    write_json(run_dir / "written_report.json", payload)
    return payload