#!/usr/bin/env python3
"""
Generate a LaTeX poster from Bayesian Optimization run artifacts.

Usage:
    uv run python latex/generate_poster.py <RUN_ID>
    uv run python latex/generate_poster.py <RUN_ID> --output poster.tex
    uv run python latex/generate_poster.py <RUN_ID> --template latex/poster_template.tex
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
LATEX_DIR = ROOT / "latex"
RUNS_DIR = ROOT / "runs"


def latex_escape(value: Any) -> str:
    """Escape LaTeX special characters in normal text."""
    if value is None:
        return ""
    s = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def format_number(value: Any, decimals: int = 6, default: str = "N/A") -> str:
    try:
        if value is None:
            return default
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return default
        return f"{v:.{decimals}f}"
    except Exception:
        return default


def load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_jsonl_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_json(path, lines=True)


def load_written_report(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "written_report.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_run_data(run_id: str) -> dict[str, Any]:
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    state_path = run_dir / "state.json"
    report_path = run_dir / "report.json"
    observations_path = run_dir / "observations.jsonl"
    suggestions_path = run_dir / "suggestions.jsonl"
    convergence_path = run_dir / "convergence.pdf"

    if not state_path.exists():
        raise FileNotFoundError(f"Missing state.json: {state_path}")
    if not report_path.exists():
        raise FileNotFoundError(f"Missing report.json: {report_path}")
    if not observations_path.exists():
        raise FileNotFoundError(f"Missing observations.jsonl: {observations_path}")

    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "state": load_json(state_path),
        "report": load_json(report_path),
        "observations": load_jsonl_dataframe(observations_path),
        "suggestions": load_jsonl_dataframe(suggestions_path) if suggestions_path.exists() else pd.DataFrame(),
        "convergence_path": convergence_path if convergence_path.exists() else None,
    }


def get_oracle_rmse(state: dict[str, Any], report: dict[str, Any]) -> float | None:
    for source in (report.get("oracle", {}), state.get("oracle", {})):
        if isinstance(source, dict) and "selected_rmse" in source:
            val = safe_float(source.get("selected_rmse"), float("nan"))
            if not math.isnan(val):
                return val

    cv_rmse = state.get("oracle", {}).get("cv_rmse")
    if isinstance(cv_rmse, dict):
        for key in ("extra_trees", "random_forest"):
            if key in cv_rmse:
                val = safe_float(cv_rmse.get(key), float("nan"))
                if not math.isnan(val):
                    return val
    return None


def get_fidelity_label(rmse: float | None) -> str:
    if rmse is None or math.isnan(rmse):
        return "unknown"
    if rmse < 0.1:
        return "high"
    if rmse < 0.5:
        return "moderate"
    return "low"


def compute_statistics(data: dict[str, Any]) -> dict[str, Any]:
    state = data["state"]
    report = data["report"]
    obs = data["observations"]

    num_observations = int(report.get("num_observations", len(obs)))

    y_series = pd.to_numeric(obs["y"], errors="coerce") if "y" in obs.columns else pd.Series(dtype=float)
    valid_y = y_series.dropna()

    if len(valid_y) > 0:
        initial_y = float(valid_y.iloc[0])
        if str(report.get("objective", state.get("objective", "max"))).lower() == "min":
            best_y_from_obs = float(valid_y.min())
            best_idx = int(valid_y.idxmin())
        else:
            best_y_from_obs = float(valid_y.max())
            best_idx = int(valid_y.idxmax())
        best_iter_from_obs = int(obs.loc[best_idx, "iteration"]) if "iteration" in obs.columns else best_idx
    else:
        initial_y = 0.0
        best_y_from_obs = 0.0
        best_iter_from_obs = 0

    best_y = safe_float(report.get("best_value"), best_y_from_obs)
    best_iteration = int(report.get("best_iteration", best_iter_from_obs))

    objective = str(report.get("objective", state.get("objective", "max"))).lower()
    if objective == "min":
        improvement_pct = ((initial_y - best_y) / abs(initial_y) * 100) if initial_y != 0 else 0.0
    else:
        improvement_pct = ((best_y - initial_y) / abs(initial_y) * 100) if initial_y != 0 else 0.0

    dataset_path = state.get("dataset_path", "")
    dataset_name = Path(dataset_path).name if dataset_path else "Unknown dataset"
    rmse = get_oracle_rmse(state, report)

    return {
        "num_observations": num_observations,
        "initial_y": initial_y,
        "best_y": best_y,
        "best_iteration": best_iteration,
        "improvement_pct": improvement_pct,
        "rmse": rmse,
        "fidelity": get_fidelity_label(rmse),
        "objective": objective,
        "dataset_name": dataset_name,
        "engine": str(report.get("engine") or state.get("default_engine") or state.get("engine") or "hebo").upper(),
        "target_column": report.get("target_column", state.get("target_column", "Target")),
        "design_parameters": state.get("design_parameters", []),
        "num_parameters": len(state.get("design_parameters", [])),
    }


def parameter_summary_text(design_parameters: list[dict[str, Any]], max_items: int = 6) -> str:
    if not design_parameters:
        return "a configurable numerical design space"

    parts: list[str] = []
    for param in design_parameters[:max_items]:
        name = str(param.get("name", "parameter"))
        ptype = str(param.get("type", "num"))
        lb = format_number(param.get("lb"), 2, "N/A")
        ub = format_number(param.get("ub"), 2, "N/A")
        parts.append(f"{name} ({ptype}, range {lb} to {ub})")

    if len(design_parameters) > max_items:
        parts.append(f"and {len(design_parameters) - max_items} additional parameters")

    return "; ".join(parts)


def top_candidates_from_observations(observations: pd.DataFrame, objective: str, top_n: int = 5) -> pd.DataFrame:
    if len(observations) == 0 or "y" not in observations.columns:
        return pd.DataFrame()

    temp = observations.copy()
    temp["_y_num"] = pd.to_numeric(temp["y"], errors="coerce")
    temp = temp.dropna(subset=["_y_num"])
    if len(temp) == 0:
        return pd.DataFrame()

    ascending = objective == "min"
    return temp.sort_values("_y_num", ascending=ascending).head(top_n).reset_index(drop=True)


def compact_parameter_dict(x: Any, max_items: int = 4, max_total_len: int = 80) -> str:
    if not isinstance(x, dict):
        return str(x)[:max_total_len]

    items = list(x.items())[:max_items]
    parts = [f"{k}={format_number(v, 2, str(v))}" for k, v in items]
    if len(x) > max_items:
        parts.append("...")
    text = ", ".join(parts)
    if len(text) > max_total_len:
        text = text[: max_total_len - 3] + "..."
    return text


def build_top_candidates_table(data: dict[str, Any], objective: str, top_n: int = 5) -> str:
    suggestions = data["suggestions"]
    observations = data["observations"]
    rows: list[str] = []

    if len(suggestions) > 0 and "y_pred" in suggestions.columns:
        temp = suggestions.copy()
        temp["_y_pred_num"] = pd.to_numeric(temp["y_pred"], errors="coerce")
        temp = temp.dropna(subset=["_y_pred_num"])
        temp = temp.sort_values("_y_pred_num", ascending=(objective == "min")).head(top_n)

        for rank, (_, row) in enumerate(temp.iterrows(), start=1):
            iteration = row.get("iteration", row.name)
            y_pred = format_number(row.get("y_pred"), 4, "TBD")
            params = latex_escape(compact_parameter_dict(row.get("x", {})))
            rows.append(
                f"{rank} & {latex_escape(iteration)} & {latex_escape(y_pred)} & \\texttt{{{params}}} \\\\"
            )
    else:
        top_obs = top_candidates_from_observations(observations, objective=objective, top_n=top_n)
        if len(top_obs) == 0:
            return r"\multicolumn{4}{c}{\textit{No candidates available}} \\"

        for rank, (_, row) in enumerate(top_obs.iterrows(), start=1):
            iteration = int(row["iteration"]) if "iteration" in row else rank - 1
            y_val = format_number(row.get("y"), 4, "N/A")
            params = latex_escape(compact_parameter_dict(row.get("x", {})))
            rows.append(
                f"{rank} & {latex_escape(iteration)} & {latex_escape(y_val)} & \\texttt{{{params}}} \\\\"
            )

    return "\n".join(rows)


def build_abstract(run_id: str, stats: dict[str, Any]) -> str:
    objective_word = "maximise" if stats["objective"] == "max" else "minimise"
    text = (
        f"This poster summarises Bayesian Optimization run {run_id}, carried out on "
        f"{stats['dataset_name']} using the {stats['engine']} engine. "
        f"The objective was to {objective_word} the target variable "
        f"{stats['target_column']} over a {stats['num_parameters']}-parameter search space. "
        f"Across {stats['num_observations']} evaluations, the best observed value reached "
        f"{format_number(stats['best_y'], 4)}, corresponding to a "
        f"{format_number(stats['improvement_pct'], 1)}% improvement over the initial observation. "
        f"The surrogate oracle achieved an RMSE of "
        f"{format_number(stats['rmse'], 4, 'N/A')}, indicating {stats['fidelity']} predictive fidelity."
    )
    return latex_escape(text)


def build_fallback_sections(run_id: str, stats: dict[str, Any]) -> dict[str, str]:
    objective_word = "maximise" if stats["objective"] == "max" else "minimise"
    parameter_summary = parameter_summary_text(stats["design_parameters"])

    introduction = (
        f"This run applied Bayesian Optimization to {stats['dataset_name']} in order to "
        f"{objective_word} {stats['target_column']}. Bayesian Optimization is suitable here because it "
        f"searches efficiently over expensive, structured parameter spaces. The design space contained "
        f"{stats['num_parameters']} parameters, including {parameter_summary}."
    )

    results = (
        f"The best observed value was {format_number(stats['best_y'], 4)}, found at iteration "
        f"{stats['best_iteration']}. Relative to the initial observation of "
        f"{format_number(stats['initial_y'], 4)}, this corresponds to a "
        f"{format_number(stats['improvement_pct'], 1)}% improvement."
    )

    discussion = (
        f"These results suggest that the optimizer was able to identify promising regions of the design "
        f"space within {stats['num_observations']} evaluations. The oracle RMSE of "
        f"{format_number(stats['rmse'], 4, 'N/A')} indicates {stats['fidelity']} predictive fidelity, "
        f"so the strongest candidates should be treated as high-priority leads for follow-up validation."
    )

    significance = (
        "This is important because it reduces the number of expensive evaluations needed to identify strong "
        "candidate configurations and helps focus follow-up work on the most promising regions of the space."
    )

    summary = (
        f"In summary, the run completed {stats['num_observations']} evaluations and identified a best value "
        f"of {format_number(stats['best_y'], 4)}."
    )

    return {
        "abstract": build_abstract(run_id, stats),
        "introduction": latex_escape(introduction),
        "results": latex_escape(results),
        "discussion": latex_escape(discussion),
        "significance": latex_escape(significance),
        "summary": latex_escape(summary),
    }


def build_convergence_figure_block(data: dict[str, Any]) -> str:
    convergence_path = data["convergence_path"]
    if convergence_path is None:
        return r"\textit{Convergence plot not available for this run.}"

    return (
        "\\section*{Convergence Plot}\n"
        "\\begin{center}\n"
        f"\\includegraphics[width=0.95\\linewidth]{{{latex_escape(convergence_path.name)}}}\n"
        "\\end{center}"
    )


def get_default_poster_template() -> str:
    return r"""
\documentclass[10pt,letterpaper]{article}
\usepackage{geometry}
\usepackage{multicol}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage{enumitem}
\usepackage{tcolorbox}
\geometry{margin=0.7in}
\setlength{\columnsep}{1cm}
\setlist[itemize]{leftmargin=*}
\setlength{\parskip}{0.8ex}
\setlength{\parindent}{0pt}

\title{Bayesian Optimization Poster: INSERT_RUN_ID}
\author{BO Agent}
\date{\today}

\begin{document}

\maketitle

\begin{abstract}
INSERT_ABSTRACT
\end{abstract}

\begin{multicols}{2}

\section*{Introduction}
INSERT_INTRODUCTION

\section*{Key Results}
\begin{itemize}
  \item \textbf{Best value:} INSERT_BEST_Y
  \item \textbf{Improvement:} INSERT_IMPROVEMENT_PERCENT\%
  \item \textbf{Iterations:} INSERT_ITERATIONS
  \item \textbf{Oracle RMSE:} INSERT_RMSE (INSERT_FIDELITY)
\end{itemize}

INSERT_CONVERGENCE_FIGURE

\section*{Top Candidates}
\small
\begin{tabular}{@{}lrlp{4.3cm}@{}}
\toprule
\textbf{Rank} & \textbf{Iter} & \textbf{Value} & \textbf{Parameters} \\
\midrule
INSERT_TOP_CANDIDATES
\bottomrule
\end{tabular}
\normalsize

\section*{Results}
INSERT_RESULTS_SECTION

\section*{Discussion}
INSERT_DISCUSSION

\section*{Why It Matters}
INSERT_SIGNIFICANCE

\section*{Summary}
INSERT_SUMMARY

\end{multicols}

\end{document}
""".strip()


def render_template(template: str, replacements: dict[str, str]) -> str:
    result = template
    for key in sorted(replacements, key=len, reverse=True):
        result = result.replace(key, replacements[key])
    return result


def generate_poster(run_id: str, output_file: str | None = None, template_file: str | None = None) -> Path:
    data = load_run_data(run_id)
    stats = compute_statistics(data)
    written_report = load_written_report(data["run_dir"])

    if template_file:
        template_path = Path(template_file)
        if not template_path.is_absolute():
            template_path = ROOT / template_path
        if not template_path.exists():
            raise FileNotFoundError(f"Template file not found: {template_path}")
        template = template_path.read_text(encoding="utf-8")
    else:
        template_path = LATEX_DIR / "poster_template.tex"
        template = template_path.read_text(encoding="utf-8") if template_path.exists() else get_default_poster_template()

    if written_report and "sections" in written_report:
        sections = written_report["sections"]
        final_sections = {
            "abstract": latex_escape(sections.get("abstract", "")),
            "introduction": latex_escape(sections.get("introduction", "")),
            "results": latex_escape(sections.get("results", "")),
            "discussion": latex_escape(sections.get("discussion", "")),
            "significance": latex_escape(sections.get("significance", "")),
            "summary": latex_escape(sections.get("summary", "")),
        }
    else:
        final_sections = build_fallback_sections(run_id, stats)

    replacements = {
        "INSERT_RUN_ID": latex_escape(run_id),
        "INSERT_ABSTRACT": final_sections["abstract"],
        "INSERT_INTRODUCTION": final_sections["introduction"],
        "INSERT_RESULTS_SECTION": final_sections["results"],
        "INSERT_DISCUSSION": final_sections["discussion"],
        "INSERT_SIGNIFICANCE": final_sections["significance"],
        "INSERT_SUMMARY": final_sections["summary"],
        "INSERT_BEST_Y": latex_escape(format_number(stats["best_y"], 6, "N/A")),
        "INSERT_IMPROVEMENT_PERCENT": latex_escape(format_number(stats["improvement_pct"], 1, "N/A")),
        "INSERT_ITERATIONS": latex_escape(str(stats["num_observations"])),
        "INSERT_RMSE": latex_escape(format_number(stats["rmse"], 6, "N/A")),
        "INSERT_FIDELITY": latex_escape(stats["fidelity"]),
        "INSERT_TOP_CANDIDATES": build_top_candidates_table(data, objective=stats["objective"], top_n=5),
        "INSERT_CONVERGENCE_FIGURE": build_convergence_figure_block(data),
    }

    rendered = render_template(template, replacements)

    run_dir = data["run_dir"]
    output_path = run_dir / "poster.tex" if output_file is None else Path(output_file)
    if not output_path.is_absolute():
        output_path = run_dir / output_path.name

    output_path.write_text(rendered, encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a LaTeX poster from BO run results")
    parser.add_argument("run_id")
    parser.add_argument("--output", "-o")
    parser.add_argument("--template", "-t")
    args = parser.parse_args()

    try:
        output_path = generate_poster(args.run_id, args.output, args.template)
        print(f"Poster written to: {output_path}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()