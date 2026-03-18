#!/usr/bin/env python3
"""
Generate polished LaTeX scientific reports from BO run results.

Usage:
    uv run python latex/generate_latex_report.py <RUN_ID>
    uv run python latex/generate_latex_report.py <RUN_ID> --output custom_report.tex
    uv run python latex/generate_latex_report.py <RUN_ID> --template latex/report_template.tex
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
LATEX_DIR = ROOT / "latex"
RUNS_DIR = ROOT / "runs"


def latex_escape(value: Any) -> str:
    """Escape LaTeX special characters in arbitrary text."""
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


def latex_tt(value: Any) -> str:
    """Return text formatted as \\texttt{...} with safe escaping."""
    return rf"\texttt{{{latex_escape(value)}}}"


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert to float safely."""
    try:
        if value is None:
            return default
        if isinstance(value, float) and math.isnan(value):
            return default
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def format_number(value: Any, decimals: int = 6, default: str = "N/A") -> str:
    """Format numeric values safely."""
    try:
        if value is None:
            return default
        if isinstance(value, float) and math.isnan(value):
            return default
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return default
        return f"{v:.{decimals}f}"
    except Exception:
        return default


def compact_dict_text(value: Any, max_len: int = 90) -> str:
    """Convert dict-like values to a short escaped string for table cells."""
    s = latex_escape(str(value))
    s = s.replace("\n", " ")
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s


def load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_run_data(run_id: str) -> dict[str, Any]:
    """Load all data from a BO run."""
    run_path = RUNS_DIR / run_id
    if not run_path.exists():
        raise FileNotFoundError(f"Run directory not found: {run_path}")

    state = load_json(run_path / "state.json")

    observations = pd.DataFrame()
    if (run_path / "observations.jsonl").exists():
        observations = pd.read_json(run_path / "observations.jsonl", lines=True)

    suggestions = pd.DataFrame()
    if (run_path / "suggestions.jsonl").exists():
        suggestions = pd.read_json(run_path / "suggestions.jsonl", lines=True)

    oracle_meta = {}
    if (run_path / "oracle_meta.json").exists():
        oracle_meta = load_json(run_path / "oracle_meta.json")

    report_data = {}
    if (run_path / "report.json").exists():
        report_data = load_json(run_path / "report.json")

    written_report = None
    if (run_path / "written_report.json").exists():
        written_report = load_json(run_path / "written_report.json")

    return {
        "run_path": run_path,
        "run_id": run_id,
        "state": state,
        "observations": observations,
        "suggestions": suggestions,
        "oracle_meta": oracle_meta,
        "report_data": report_data,
        "written_report": written_report,
    }


def get_oracle_rmse(data: dict[str, Any]) -> float | None:
    """Extract RMSE from report, state, or oracle_meta."""
    report = data["report_data"]
    state = data["state"]
    oracle_meta = data["oracle_meta"]

    for source in (
        report.get("oracle", {}),
        state.get("oracle", {}),
        oracle_meta,
    ):
        if isinstance(source, dict) and "selected_rmse" in source:
            val = safe_float(source.get("selected_rmse"), float("nan"))
            if not math.isnan(val):
                return val

    for source in (
        state.get("oracle", {}),
        oracle_meta,
    ):
        cv_rmse = source.get("cv_rmse") if isinstance(source, dict) else None
        if isinstance(cv_rmse, dict):
            for key in ("extra_trees", "random_forest"):
                if key in cv_rmse:
                    val = safe_float(cv_rmse.get(key), float("nan"))
                    if not math.isnan(val):
                        return val
    return None


def compute_statistics(data: dict[str, Any]) -> dict[str, Any]:
    """Compute key statistics for the report."""
    obs = data["observations"]
    report = data["report_data"]
    oracle_meta = data["oracle_meta"]
    state = data["state"]

    objective = str(report.get("objective", state.get("objective", "max"))).lower()
    n_iterations = len(obs) if len(obs) > 0 else int(state.get("iterations", 0))

    if len(obs) > 0 and "y" in obs.columns:
        y_series = pd.to_numeric(obs["y"], errors="coerce").dropna()
        if len(y_series) > 0:
            if objective == "min":
                best_y_obs = float(y_series.min())
                best_idx = int(y_series.idxmin())
            else:
                best_y_obs = float(y_series.max())
                best_idx = int(y_series.idxmax())
            best_iter = (
                int(obs.loc[best_idx, "iteration"]) + 1
                if "iteration" in obs.columns
                else best_idx + 1
            )
            initial_best = safe_float(y_series.iloc[0], 0.0)
        else:
            best_y_obs = 0.0
            best_iter = 0
            initial_best = 0.0
    else:
        best_y_obs = 0.0
        best_iter = 0
        initial_best = 0.0

    best_y = safe_float(report.get("best_value", report.get("best_y", best_y_obs)), best_y_obs)
    best_iter = int(report.get("best_iteration", report.get("best_y_idx", best_iter - 1)) or (best_iter - 1)) + 1

    if objective == "min":
        improvement = initial_best - best_y
    else:
        improvement = best_y - initial_best
    improvement_pct = (improvement / abs(initial_best)) * 100 if initial_best != 0 else 0.0

    rmse = get_oracle_rmse(data)
    rmse_safe = rmse if rmse is not None else float("nan")
    if not math.isnan(rmse_safe):
        fidelity = "high" if rmse_safe < 0.1 else "moderate" if rmse_safe < 0.5 else "low"
    else:
        fidelity = "unknown"

    design_parameters = state.get("design_parameters", [])
    n_features = len(design_parameters) if design_parameters else state.get("n_vars", len(state.get("bounds", [])))
    engine = str(report.get("engine") or state.get("default_engine") or state.get("engine") or "hebo").upper()
    batch_size = int(state.get("default_batch_size", state.get("batch_size", 1)))
    target_col = state.get("target_column", report.get("target_column", "Target"))
    dataset_path = state.get("dataset_path", "")
    dataset_name = Path(dataset_path).name if dataset_path else "Unknown dataset"

    best_x = report.get("best_x", {})
    if not best_x and len(obs) > 0:
        y_num = pd.to_numeric(obs["y"], errors="coerce")
        if y_num.notna().any():
            row = obs.loc[y_num.idxmin() if objective == "min" else y_num.idxmax()]
            best_x = row.get("x", {})

    oracle_train = (
        oracle_meta.get("n_train")
        or state.get("oracle", {}).get("rows_used")
        or "N/A"
    )
    cv_folds = oracle_meta.get("cv_folds", 5)

    return {
        "objective": objective,
        "n_iterations": n_iterations,
        "best_y": best_y,
        "best_iter": best_iter,
        "initial_best": initial_best,
        "improvement": improvement,
        "improvement_pct": improvement_pct,
        "rmse": rmse_safe,
        "fidelity": fidelity,
        "n_features": n_features,
        "n_train": oracle_train,
        "cv_folds": cv_folds,
        "engine": engine,
        "batch_size": batch_size,
        "target_col": target_col,
        "dataset_name": dataset_name,
        "best_x": best_x,
    }


def build_top_candidates_table(data: dict[str, Any], n: int = 5) -> str:
    """Build LaTeX table of top candidates."""
    suggestions = data["suggestions"]
    observations = data["observations"]
    objective = str(data["report_data"].get("objective", data["state"].get("objective", "max"))).lower()

    rows: list[str] = []

    if len(suggestions) > 0:
        n_show = min(n, len(suggestions))
        if "y_pred" in suggestions.columns and pd.to_numeric(suggestions["y_pred"], errors="coerce").notna().any():
            temp = suggestions.copy()
            temp["_y_pred_num"] = pd.to_numeric(temp["y_pred"], errors="coerce")
            top = temp.sort_values("_y_pred_num", ascending=(objective == "min")).head(n_show)
        else:
            top = suggestions.tail(n_show)

        for rank, (idx, row) in enumerate(top.iterrows(), 1):
            y_pred = row.get("y_pred", "TBD")
            y_pred_str = format_number(y_pred, decimals=4, default="TBD")
            x_val = row.get("x", {})
            x_str = compact_dict_text(x_val, max_len=90)
            iter_val = int(row.get("iteration", idx)) + 1 if "iteration" in row else idx + 1
            rows.append(
                f"{rank} & {latex_escape(iter_val)} & {latex_escape(y_pred_str)} & \\texttt{{{x_str}}} \\\\"
            )

    elif len(observations) > 0 and "y" in observations.columns:
        temp = observations.copy()
        temp["_y_num"] = pd.to_numeric(temp["y"], errors="coerce")
        temp = temp.dropna(subset=["_y_num"])
        temp = temp.sort_values("_y_num", ascending=(objective == "min")).head(n)

        for rank, (_, row) in enumerate(temp.iterrows(), 1):
            y_val = format_number(row.get("y"), decimals=4, default="N/A")
            x_str = compact_dict_text(row.get("x", {}), max_len=90)
            iter_val = int(row.get("iteration", rank - 1)) + 1
            rows.append(
                f"{rank} & {latex_escape(iter_val)} & {latex_escape(y_val)} & \\texttt{{{x_str}}} \\\\"
            )

    if not rows:
        return r"\multicolumn{4}{c}{\textit{No candidates available}} \\"

    out = []
    for i, row_text in enumerate(rows):
        out.append(row_text)
        if i < len(rows) - 1:
            out.append(r"\rowcolor{tablealt}")
    return "\n".join(out)


def build_iteration_table(observations: pd.DataFrame) -> str:
    """Build LaTeX table of all iterations."""
    if len(observations) == 0:
        return r"""
\begin{table}[htbp]
\centering
\caption{Complete Iteration History}
\label{tab:iterations}
\begin{tabular}{@{}rr@{}}
\toprule
\textbf{Iteration} & \textbf{Observed Value} \\
\midrule
\multicolumn{2}{c}{\textit{No observations available}} \\
\bottomrule
\end{tabular}
\end{table}
"""

    rows = []
    for idx, row in observations.iterrows():
        iter_num = int(row.get("iteration", idx)) + 1
        y = row.get("y", 0)
        y_str = format_number(y, decimals=6, default=latex_escape(y))
        rows.append(f"{iter_num} & {y_str} \\\\")
        if idx < len(observations) - 1:
            rows.append(r"\rowcolor{tablealt}")

    return rf"""
\begin{{table}}[htbp]
\centering
\caption{{Complete Iteration History}}
\label{{tab:iterations}}
\begin{{tabular}}{{@{{}}rr@{{}}}}
\toprule
\textbf{{Iteration}} & \textbf{{Observed Value}} \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def build_parameter_table(design_parameters: list[dict[str, Any]]) -> str:
    if not design_parameters:
        return r"\multicolumn{4}{c}{\textit{No parameter metadata available}} \\"

    rows = []
    for i, p in enumerate(design_parameters):
        name = latex_escape(p.get("name", "parameter"))
        ptype = latex_escape(p.get("type", "unknown"))
        lower = latex_escape(format_number(p.get("lb"), 3, "-")) if ptype == "num" else "-"
        upper = latex_escape(format_number(p.get("ub"), 3, "-")) if ptype == "num" else "-"
        row = f"{name} & {ptype} & {lower} & {upper} \\\\"
        rows.append(row)
        if i < len(design_parameters) - 1:
            rows.append(r"\rowcolor{tablealt}")
    return "\n".join(rows)


def get_written_sections(data: dict[str, Any], stats: dict[str, Any]) -> dict[str, str]:
    written = data.get("written_report")
    if written and "sections" in written:
        sections = written["sections"]
        return {
            "abstract": latex_escape(sections.get("abstract", "")),
            "introduction": latex_escape(sections.get("introduction", "")),
            "results": latex_escape(sections.get("results", "")),
            "discussion": latex_escape(sections.get("discussion", "")),
            "summary": latex_escape(sections.get("summary", "")),
            "significance": latex_escape(sections.get("significance", "")),
        }

    objective_word = "maximize" if stats["objective"] == "max" else "minimize"
    abstract = (
        f"This report summarises Bayesian Optimization run {latex_tt(data['run_id'])}, "
        f"performed on {latex_tt(stats['dataset_name'])} using {latex_escape(stats['engine'])}. "
        f"The goal was to {objective_word} {latex_tt(stats['target_col'])} across "
        f"{latex_escape(stats['n_features'])} parameters. The best observed value was "
        f"{latex_escape(format_number(stats['best_y'], 4))}, found at iteration "
        f"{latex_escape(stats['best_iter'])}, corresponding to a "
        f"{latex_escape(format_number(stats['improvement_pct'], 1))}\\% improvement over the initial observation."
    )
    introduction = (
        f"This study applies Bayesian Optimization to identify promising parameter settings for "
        f"{latex_tt(stats['target_col'])}. Bayesian Optimization is appropriate because it is sample-efficient "
        f"and well suited to expensive experimental or computational search problems."
    )
    results = (
        f"The optimisation completed {latex_escape(stats['n_iterations'])} evaluations and identified a best "
        f"value of {latex_escape(format_number(stats['best_y'], 4))} at iteration "
        f"{latex_escape(stats['best_iter'])}. The estimated oracle RMSE was "
        f"{latex_escape(format_number(stats['rmse'], 4, 'N/A'))}."
    )
    discussion = (
        "The results indicate that the workflow can identify stronger candidates within a limited "
        "evaluation budget. However, conclusions should be interpreted alongside oracle fidelity "
        "and validated with follow-up experiments."
    )
    summary = (
        "In summary, the run produced a measurable improvement and identified a best candidate "
        "for further validation."
    )
    significance = (
        "This is important because it supports faster and more targeted optimisation in high-dimensional "
        "design spaces."
    )

    return {
        "abstract": abstract,
        "introduction": introduction,
        "results": results,
        "discussion": discussion,
        "summary": summary,
        "significance": significance,
    }


def build_best_candidate_table(best_x: dict[str, Any]) -> str:
    if not isinstance(best_x, dict) or not best_x:
        return r"\multicolumn{2}{c}{\textit{Best candidate not available}} \\"

    rows = []
    items = list(best_x.items())
    for i, (k, v) in enumerate(items):
        if isinstance(v, (int, float)):
            val = format_number(v, 4, str(v))
        else:
            val = str(v)
        rows.append(f"{latex_escape(k)} & {latex_escape(val)} \\\\")
        if i < len(items) - 1:
            rows.append(r"\rowcolor{tablealt}")
    return "\n".join(rows)


def generate_latex_report(run_id: str, output_file: str | None = None, template_file: str | None = None) -> Path:
    """
    Generate LaTeX report from BO results.

    Args:
        run_id: BO run ID
        output_file: Output .tex path (default: runs/<RUN_ID>/report.tex)
        template_file: Custom template path
    """
    print(f"📊 Loading BO results from run: {run_id}")
    data = load_run_data(run_id)

    stats = compute_statistics(data)
    sections = get_written_sections(data, stats)

    print(f"   Best value: {format_number(stats['best_y'], 6, 'N/A')} (iter {stats['best_iter']})")
    print(f"   Improvement: {format_number(stats['improvement_pct'], 1, 'N/A')}%")
    print(f"   Oracle RMSE: {format_number(stats['rmse'], 6, 'N/A')}")

    top_candidates = build_top_candidates_table(data, n=5)
    iteration_log = build_iteration_table(data["observations"])
    parameter_table = build_parameter_table(data["state"].get("design_parameters", []))
    best_candidate_table = build_best_candidate_table(stats["best_x"])

    run_dir = data["run_path"]

    if output_file is None:
        output_path = run_dir / "report.tex"
    else:
        output_path = Path(output_file)
        if not output_path.is_absolute():
            output_path = run_dir / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if template_file:
        with open(template_file, encoding="utf-8") as f:
            template = f.read()
    else:
        template = _get_default_template()

    plot_block = r"\textit{Convergence plot not available.}"
    if (run_dir / "convergence.pdf").exists():
        plot_block = r"""
\begin{figure}[H]
\centering
\includegraphics[width=0.9\textwidth]{convergence.pdf}
\caption{Optimization convergence across recorded evaluations.}
\label{fig:convergence}
\end{figure}
""".strip()

    replacements = {
        "INSERT_RUN_ID": latex_escape(run_id),
        "INSERT_TIMESTAMP": latex_escape(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        "INSERT_TARGET_COL": latex_escape(stats["target_col"]),
        "INSERT_OBJECTIVE": latex_escape(stats["objective"].upper()),
        "INSERT_ENGINE": latex_escape(stats["engine"]),
        "INSERT_BATCH_SIZE": latex_escape(str(stats["batch_size"])),
        "INSERT_ITERATIONS": latex_escape(str(stats["n_iterations"])),
        "INSERT_DIM": latex_escape(str(stats["n_features"])),
        "INSERT_DATASET": latex_escape(stats["dataset_name"]),
        "INSERT_BEST_Y": latex_escape(format_number(stats["best_y"], 6, "N/A")),
        "INSERT_BEST_ITER": latex_escape(str(stats["best_iter"])),
        "INSERT_INITIAL_BEST": latex_escape(format_number(stats["initial_best"], 6, "N/A")),
        "INSERT_IMPROVEMENT": latex_escape(format_number(stats["improvement"], 6, "N/A")),
        "INSERT_IMPROVEMENT_PERCENT": latex_escape(format_number(stats["improvement_pct"], 1, "N/A")),
        "INSERT_RMSE": latex_escape(format_number(stats["rmse"], 6, "N/A")),
        "INSERT_FIDELITY": latex_escape(stats["fidelity"]),
        "INSERT_CV_FOLDS": latex_escape(str(stats["cv_folds"])),
        "INSERT_N_TRAIN": latex_escape(str(stats["n_train"])),
        "INSERT_ABSTRACT": sections["abstract"],
        "INSERT_INTRODUCTION": sections["introduction"],
        "INSERT_RESULTS_TEXT": sections["results"],
        "INSERT_DISCUSSION": sections["discussion"],
        "INSERT_SUMMARY": sections["summary"],
        "INSERT_SIGNIFICANCE": sections["significance"],
        "INSERT_TOP_CANDIDATES": top_candidates,
        "INSERT_ITERATION_LOG": iteration_log,
        "INSERT_PARAMETER_TABLE": parameter_table,
        "INSERT_BEST_CANDIDATE_TABLE": best_candidate_table,
        "INSERT_CONVERGENCE_PLOT": plot_block,
    }

    result = template
    for key in sorted(replacements, key=len, reverse=True):
        result = result.replace(key, str(replacements[key]))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result)

    print(f"\n✅ LaTeX report generated: {output_path}")
    return output_path


def _get_default_template() -> str:
    """Return a polished self-contained LaTeX template."""
    return r"""
% Bayesian Optimization Results Report
% Generated by generate_latex_report.py
% Compile with: xelatex

\documentclass[11pt,letterpaper]{report}

\usepackage[margin=1in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{setspace}
\usepackage{array}
\usepackage{booktabs}
\usepackage{colortbl}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{float}
\usepackage{longtable}
\usepackage{tabularx}
\usepackage{titlesec}
\usepackage{fancyhdr}
\usepackage[most]{tcolorbox}
\usepackage{enumitem}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{parskip}

\emergencystretch=3em

% ---------------------------------------------------------------------------
% COLOR PALETTE
% ---------------------------------------------------------------------------

\definecolor{primaryblue}{RGB}{0, 51, 102}
\definecolor{secondaryblue}{RGB}{74, 144, 226}
\definecolor{lightblue}{RGB}{220, 235, 252}
\definecolor{darkgreen}{RGB}{0, 128, 96}
\definecolor{lightgreen}{RGB}{220, 245, 240}
\definecolor{cautionorange}{RGB}{255, 140, 66}
\definecolor{lightorange}{RGB}{255, 243, 224}
\definecolor{criticalred}{RGB}{198, 40, 40}
\definecolor{lightred}{RGB}{255, 235, 238}
\definecolor{lightgray}{RGB}{245, 245, 245}
\definecolor{tablealt}{RGB}{248, 250, 252}
\definecolor{darkgray}{RGB}{66, 66, 66}

% ---------------------------------------------------------------------------
% PAGE STYLE
% ---------------------------------------------------------------------------

\setstretch{1.1}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\textit{Bayesian Optimization Results Report}}
\fancyhead[R]{\small\textit{Run ID: INSERT_RUN_ID}}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0pt}

% ---------------------------------------------------------------------------
% SECTION STYLING
% ---------------------------------------------------------------------------

\titleformat{\chapter}
{\normalfont\Huge\bfseries\color{primaryblue}}
{\thechapter}{1em}{}

\titleformat{\section}
{\normalfont\Large\bfseries\color{primaryblue}}
{\thesection}{1em}{}

\titleformat{\subsection}
{\normalfont\large\bfseries\color{secondaryblue}}
{\thesubsection}{1em}{}

% ---------------------------------------------------------------------------
% BOX STYLES
% ---------------------------------------------------------------------------

\newtcolorbox{summarybox}[1]{
    colback=lightblue,
    colframe=primaryblue,
    title=#1,
    fonttitle=\bfseries,
    boxrule=0.8pt,
    arc=3pt
}

\newtcolorbox{resultbox}[1]{
    colback=lightgreen,
    colframe=darkgreen,
    title=#1,
    fonttitle=\bfseries,
    boxrule=0.8pt,
    arc=3pt
}

\newtcolorbox{warningbox}[1]{
    colback=lightorange,
    colframe=cautionorange,
    title=#1,
    fonttitle=\bfseries,
    boxrule=0.8pt,
    arc=3pt
}

\newtcolorbox{importantbox}[1]{
    colback=lightred,
    colframe=criticalred,
    title=#1,
    fonttitle=\bfseries,
    boxrule=0.8pt,
    arc=3pt
}

% ---------------------------------------------------------------------------
% HYPERREF
% ---------------------------------------------------------------------------

\hypersetup{
    colorlinks=true,
    linkcolor=primaryblue,
    urlcolor=secondaryblue,
    citecolor=secondaryblue,
    pdftitle={Bayesian Optimization Report - INSERT_RUN_ID},
    pdfauthor={BO Agent},
    pdfkeywords={optimization, bayesian, surrogate, report}
}

% ---------------------------------------------------------------------------
% DOCUMENT
% ---------------------------------------------------------------------------

\begin{document}

% ---------------------------------------------------------------------------
% TITLE PAGE
% ---------------------------------------------------------------------------

\begin{titlepage}
\centering
\vspace*{1.5cm}

{\Huge\bfseries\color{primaryblue} Bayesian Optimization Results Report\par}
\vspace{0.5cm}
{\LARGE Run ID: \texttt{INSERT_RUN_ID}\par}

\vspace{1.2cm}

\begin{tcolorbox}[
    colback=lightblue,
    colframe=primaryblue,
    width=0.92\textwidth,
    arc=4pt,
    boxrule=1pt
]
\centering
{\Large\textbf{Executive Snapshot}}\par
\vspace{0.4cm}
\begin{tabular}{@{}ll@{}}
\textbf{Dataset:} & \texttt{INSERT_DATASET} \\
\textbf{Target column:} & \texttt{INSERT_TARGET_COL} \\
\textbf{Objective:} & INSERT_OBJECTIVE \\
\textbf{Engine:} & INSERT_ENGINE \\
\textbf{Best value:} & INSERT_BEST_Y \\
\textbf{Best iteration:} & INSERT_BEST_ITER \\
\textbf{Improvement:} & INSERT_IMPROVEMENT_PERCENT\% \\
\textbf{Oracle RMSE:} & INSERT_RMSE \\
\end{tabular}
\end{tcolorbox}

\vfill

{\Large\textbf{BO Agent}\par}
\vspace{0.3cm}
{\large BOGroupResearch\par}
\vspace{0.3cm}
{\large \today\par}

\end{titlepage}

\tableofcontents
\newpage

% ---------------------------------------------------------------------------
% EXECUTIVE SUMMARY
% ---------------------------------------------------------------------------

\chapter*{Executive Summary}
\addcontentsline{toc}{chapter}{Executive Summary}

\begin{summarybox}{Overview}
INSERT_SUMMARY
\end{summarybox}

\begin{summarybox}{Headline Metrics}
\begin{itemize}[leftmargin=*]
    \item \textbf{Dataset:} \texttt{INSERT_DATASET}
    \item \textbf{Objective:} INSERT_OBJECTIVE\ \texttt{INSERT_TARGET_COL}
    \item \textbf{Engine:} INSERT_ENGINE
    \item \textbf{Best observed value:} INSERT_BEST_Y
    \item \textbf{Best iteration:} INSERT_BEST_ITER
    \item \textbf{Improvement over initial result:} INSERT_IMPROVEMENT_PERCENT\%
    \item \textbf{Oracle RMSE:} INSERT_RMSE\ (\textit{INSERT_FIDELITY} fidelity)
\end{itemize}
\end{summarybox}

% ---------------------------------------------------------------------------
% INTRODUCTION
% ---------------------------------------------------------------------------

\chapter{Introduction}

INSERT_ABSTRACT

\vspace{0.8em}

INSERT_INTRODUCTION

% ---------------------------------------------------------------------------
% METHODOLOGY
% ---------------------------------------------------------------------------

\chapter{Methodology}

\section{Run Configuration}

\begin{summarybox}{Optimization Setup}
\begin{itemize}[leftmargin=*]
    \item \textbf{Run ID:} \texttt{INSERT_RUN_ID}
    \item \textbf{Dataset:} \texttt{INSERT_DATASET}
    \item \textbf{Target column:} \texttt{INSERT_TARGET_COL}
    \item \textbf{Objective:} INSERT_OBJECTIVE
    \item \textbf{Optimizer:} INSERT_ENGINE
    \item \textbf{Batch size:} INSERT_BATCH_SIZE
    \item \textbf{Total evaluations:} INSERT_ITERATIONS
    \item \textbf{Search-space dimensionality:} INSERT_DIM
\end{itemize}
\end{summarybox}

\section{Parameter Space}

\begin{table}[H]
\centering
\caption{Design parameters used in the optimization run}
\label{tab:parameters}
\begin{tabular}{@{}llll@{}}
\toprule
\textbf{Parameter} & \textbf{Type} & \textbf{Lower Bound} & \textbf{Upper Bound} \\
\midrule
INSERT_PARAMETER_TABLE
\bottomrule
\end{tabular}
\end{table}

\section{Oracle Model}

\begin{resultbox}{Oracle Quality}
The surrogate model was evaluated using cross-validation. Its reported RMSE was
\textbf{INSERT_RMSE}, indicating \textbf{INSERT_FIDELITY} predictive fidelity.
\end{resultbox}

\begin{table}[H]
\centering
\caption{Oracle model metadata}
\label{tab:oracle}
\begin{tabular}{@{}lr@{}}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
Training samples & INSERT_N_TRAIN \\
\rowcolor{tablealt} Cross-validation folds & INSERT_CV_FOLDS \\
Oracle RMSE & INSERT_RMSE \\
\rowcolor{tablealt} Fidelity label & INSERT_FIDELITY \\
\bottomrule
\end{tabular}
\end{table}

% ---------------------------------------------------------------------------
% RESULTS
% ---------------------------------------------------------------------------

\chapter{Results}

\section{Performance Summary}

\begin{table}[H]
\centering
\caption{Optimization performance summary}
\label{tab:summary}
\begin{tabular}{@{}lr@{}}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
Initial observation & INSERT_INITIAL_BEST \\
\rowcolor{tablealt} Best observed value & INSERT_BEST_Y \\
Absolute improvement & INSERT_IMPROVEMENT \\
\rowcolor{tablealt} Percent improvement & INSERT_IMPROVEMENT_PERCENT\% \\
Best iteration & INSERT_BEST_ITER \\
\rowcolor{tablealt} Total evaluations & INSERT_ITERATIONS \\
\bottomrule
\end{tabular}
\end{table}

\section{Convergence Behaviour}

INSERT_CONVERGENCE_PLOT

INSERT_RESULTS_TEXT

\section{Top Candidates}

\begin{table}[H]
\centering
\caption{Top candidate configurations for validation}
\label{tab:candidates}
\begin{tabular}{@{}lrlp{8cm}@{}}
\toprule
\textbf{Rank} & \textbf{Iteration} & \textbf{Value} & \textbf{Parameters} \\
\midrule
INSERT_TOP_CANDIDATES
\bottomrule
\end{tabular}
\end{table}

\section{Best Candidate Configuration}

\begin{table}[H]
\centering
\caption{Best candidate identified during the run}
\label{tab:best_candidate}
\begin{tabular}{@{}ll@{}}
\toprule
\textbf{Parameter} & \textbf{Value} \\
\midrule
INSERT_BEST_CANDIDATE_TABLE
\bottomrule
\end{tabular}
\end{table}

% ---------------------------------------------------------------------------
% DISCUSSION
% ---------------------------------------------------------------------------

\chapter{Discussion}

INSERT_DISCUSSION

\begin{summarybox}{Recommended Next Steps}
\begin{enumerate}[leftmargin=*]
    \item Experimentally validate the highest-ranked candidates.
    \item Incorporate new outcomes into the dataset and retrain the oracle.
    \item Extend the optimisation budget if additional improvement is required.
\end{enumerate}
\end{summarybox}

\begin{warningbox}{Practical Limitations}
\begin{itemize}[leftmargin=*]
    \item Reported performance depends on surrogate-model fidelity.
    \item Suggested candidates still require downstream validation.
    \item Limited iteration budgets may underexplore promising regions.
\end{itemize}
\end{warningbox}

\section{Why This Matters}

INSERT_SIGNIFICANCE

% ---------------------------------------------------------------------------
% APPENDIX
% ---------------------------------------------------------------------------

\appendix

\chapter{Appendix}

\section{Iteration Log}

INSERT_ITERATION_LOG

\section{Run Metadata}

\begin{itemize}[leftmargin=*]
    \item \textbf{Run ID:} \texttt{INSERT_RUN_ID}
    \item \textbf{Generated on:} INSERT_TIMESTAMP
    \item \textbf{Dataset:} \texttt{INSERT_DATASET}
    \item \textbf{Engine:} INSERT_ENGINE
    \item \textbf{Objective:} INSERT_OBJECTIVE
\end{itemize}

\end{document}
""".strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate LaTeX scientific reports from BO results"
    )
    parser.add_argument("run_id", help="BO run ID (e.g., vivid-heron-3397)")
    parser.add_argument(
        "--output",
        "-o",
        help="Output .tex file (default: runs/<RUN_ID>/report.tex)",
    )
    parser.add_argument(
        "--template",
        "-t",
        help="Custom LaTeX template file",
    )

    args = parser.parse_args()

    try:
        generate_latex_report(
            args.run_id,
            output_file=args.output,
            template_file=args.template,
        )
    except FileNotFoundError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)