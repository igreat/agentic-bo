#!/usr/bin/env python
"""
Generate LaTeX scientific reports from BO run results.

Usage:
    uv run python latex/generate_latex_report.py <RUN_ID>
    uv run python latex/generate_latex_report.py vivid-heron-3397 --output custom_report.tex
"""

import argparse
import json
import math
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

import pandas as pd


def latex_escape(value) -> str:
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


def safe_float(value, default=0.0):
    """Convert to float safely."""
    try:
        if value is None:
            return default
        if isinstance(value, float) and math.isnan(value):
            return default
        return float(value)
    except Exception:
        return default


def format_number(value, decimals=6, default="N/A") -> str:
    """Format numeric values safely."""
    try:
        if value is None:
            return default
        if isinstance(value, float) and math.isnan(value):
            return default
        return f"{float(value):.{decimals}f}"
    except Exception:
        return default


def compact_dict_text(value, max_len=60) -> str:
    """Convert dict-like values to a short escaped string for table cells."""
    s = latex_escape(str(value))
    s = s.replace("\n", " ")
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s


def ensure_scientific_report_sty():
    """Ensure scientific_report.sty is available in the latex folder."""
    sty_file = Path(__file__).parent / "scientific_report.sty"

    if sty_file.exists():
        print("   scientific_report.sty already exists in latex folder")
        return

    possible_locations = [
        Path(".claude/skills/scientific-writing/assets/scientific_report.sty"),
        Path.home() / ".claude/skills/scientific-writing/assets/scientific_report.sty",
    ]

    for location in possible_locations:
        if location.exists():
            print(f"   Copying scientific_report.sty from {location}")
            import shutil

            shutil.copy2(location, sty_file)
            return

    print("   Downloading scientific_report.sty from GitHub...")
    url = (
        "https://raw.githubusercontent.com/K-Dense-AI/claude-scientific-skills/main/"
        "scientific-skills/scientific-writing/assets/scientific_report.sty"
    )

    try:
        with urllib.request.urlopen(url) as response:
            content = response.read().decode("utf-8")

        with open(sty_file, "w", encoding="utf-8") as f:
            f.write(content)

        print("   scientific_report.sty downloaded successfully")
    except Exception as e:
        print(f"   Warning: Could not download scientific_report.sty: {e}")
        print("   LaTeX compilation will fail. Please manually download from:")
        print(f"   {url}")
        print("   and place it in the latex folder.")


def load_run_data(run_id: str):
    """Load all data from a BO run."""
    run_path = Path(f"runs/{run_id}")

    if not run_path.exists():
        raise FileNotFoundError(f"Run directory not found: {run_path}")

    with open(run_path / "state.json", encoding="utf-8") as f:
        state = json.load(f)

    observations = pd.DataFrame()
    if (run_path / "observations.jsonl").exists():
        observations = pd.read_json(run_path / "observations.jsonl", lines=True)

    suggestions = pd.DataFrame()
    if (run_path / "suggestions.jsonl").exists():
        suggestions = pd.read_json(run_path / "suggestions.jsonl", lines=True)

    oracle_meta = {}
    if (run_path / "oracle_meta.json").exists():
        with open(run_path / "oracle_meta.json", encoding="utf-8") as f:
            oracle_meta = json.load(f)

    report_data = {}
    if (run_path / "report.json").exists():
        with open(run_path / "report.json", encoding="utf-8") as f:
            report_data = json.load(f)

    return {
        "run_path": run_path,
        "run_id": run_id,
        "state": state,
        "observations": observations,
        "suggestions": suggestions,
        "oracle_meta": oracle_meta,
        "report_data": report_data,
    }


def compute_statistics(data: dict) -> dict:
    """Compute key statistics for the report."""
    obs = data["observations"]
    report = data["report_data"]
    oracle = data["oracle_meta"]
    state = data["state"]

    n_iterations = len(obs) if len(obs) > 0 else state.get("iterations", 0)

    if len(obs) > 0 and "y" in obs.columns:
        y_series = pd.to_numeric(obs["y"], errors="coerce")
        if y_series.notna().any():
            best_y = float(y_series.max())
            best_iter = int(y_series.idxmax()) + 1
            initial_best = safe_float(y_series.iloc[0], 0.0)
        else:
            best_y = 0.0
            best_iter = 0
            initial_best = 0.0
    else:
        best_y = 0.0
        best_iter = 0
        initial_best = 0.0

    improvement = best_y - initial_best
    improvement_pct = (improvement / abs(initial_best)) * 100 if initial_best != 0 else 0.0

    if report:
        best_y = safe_float(report.get("best_y", best_y), best_y)
        best_iter = int(report.get("best_y_idx", best_iter) or best_iter)
        initial_best = safe_float(report.get("initial_best", initial_best), initial_best)
        improvement = safe_float(report.get("improvement", improvement), improvement)
        improvement_pct = safe_float(report.get("improvement_percent", improvement_pct), improvement_pct)

    rmse = oracle.get("selected_rmse", oracle.get("cv_rmse", None))
    if isinstance(rmse, dict):
        rmse = rmse.get("extra_trees", rmse.get("random_forest", None))
    rmse = safe_float(rmse, float("nan"))

    if not math.isnan(rmse):
        fidelity = "high" if rmse < 0.1 else "moderate" if rmse < 0.2 else "low"
    else:
        fidelity = "unknown"

    n_features = state.get("n_vars", len(state.get("bounds", [])))

    return {
        "n_iterations": n_iterations,
        "best_y": best_y,
        "best_iter": best_iter,
        "initial_best": initial_best,
        "improvement": improvement,
        "improvement_pct": improvement_pct,
        "rmse": rmse,
        "fidelity": fidelity,
        "n_features": n_features,
        "n_train": oracle.get("n_train", "N/A"),
        "cv_folds": oracle.get("cv_folds", 5),
    }


def build_top_candidates_table(suggestions: pd.DataFrame, n: int = 5) -> str:
    """Build LaTeX table of top candidates."""
    if len(suggestions) == 0:
        return r"\multicolumn{4}{c}{\textit{No suggestions available}} \\"

    rows = []
    n_show = min(n, len(suggestions))

    if "y_pred" in suggestions.columns and pd.to_numeric(
        suggestions["y_pred"], errors="coerce"
    ).notna().any():
        temp = suggestions.copy()
        temp["_y_pred_num"] = pd.to_numeric(temp["y_pred"], errors="coerce")
        top = temp.sort_values("_y_pred_num", ascending=False).head(n_show)
    else:
        top = suggestions.tail(n_show)

    for rank, (idx, row) in enumerate(top.iterrows(), 1):
        y_pred = row.get("y_pred", "TBD")
        try:
            y_pred_str = format_number(y_pred, decimals=6, default="TBD")
        except Exception:
            y_pred_str = latex_escape(y_pred)

        x_val = row.get("x", {})
        x_str = compact_dict_text(x_val, max_len=70)

        row_text = f"{rank} & {idx + 1} & {y_pred_str} & \\texttt{{{x_str}}} \\\\"
        rows.append(row_text)

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
        y = row.get("y", 0)
        y_str = format_number(y, decimals=6, default=latex_escape(y))
        rows.append(f"{idx + 1} & {y_str} \\\\")
        if (idx + 1) % 2 == 0 and idx < len(observations) - 1:
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


def generate_latex_report(run_id: str, output_file: str = None, template_file: str = None):
    """
    Generate LaTeX report from BO results.

    Args:
        run_id: BO run ID
        output_file: Output .tex path (default: runs/<RUN_ID>/report.tex)
        template_file: Custom template path (default: built-in minimal template)
    """
    print(f"📊 Loading BO results from run: {run_id}")
    data = load_run_data(run_id)
    ensure_scientific_report_sty()

    stats = compute_statistics(data)
    print(f"   Best value: {format_number(stats['best_y'], 6, 'N/A')} (iter {stats['best_iter']})")
    print(f"   Improvement: {format_number(stats['improvement_pct'], 1, 'N/A')}%")
    print(f"   Oracle RMSE: {format_number(stats['rmse'], 6, 'N/A')}")

    top_candidates = build_top_candidates_table(data["suggestions"], n=5)
    iteration_log = build_iteration_table(data["observations"])

    run_dir = data["run_path"]

    if output_file is None:
        output_file = run_dir / "report.tex"
    else:
        output_file = Path(output_file)

        # If user passed just a filename, place it inside run folder
        if not output_file.is_absolute():
            output_file = run_dir / output_file

    output_file.parent.mkdir(parents=True, exist_ok=True)

    if template_file:
        with open(template_file, encoding="utf-8") as f:
            template = f.read()
    else:
        template = _get_default_template()

    replacements = {
        "INSERT_IMPROVEMENT_PERCENT": latex_escape(format_number(stats["improvement_pct"], 1, "N/A")),
        "INSERT_IMPROVEMENT": latex_escape(format_number(stats["improvement"], 6, "N/A")),
        "INSERT_INITIAL_BEST": latex_escape(format_number(stats["initial_best"], 6, "N/A")),
        "INSERT_BEST_Y": latex_escape(format_number(stats["best_y"], 6, "N/A")),
        "INSERT_BEST_ITER": latex_escape(str(stats["best_iter"])),
        "INSERT_ITERATIONS": latex_escape(str(stats["n_iterations"])),
        "INSERT_DIM": latex_escape(str(stats["n_features"])),
        "INSERT_RMSE": latex_escape(format_number(stats["rmse"], 6, "N/A")),
        "INSERT_FIDELITY": latex_escape(stats["fidelity"]),
        "INSERT_CV_FOLDS": latex_escape(str(stats["cv_folds"])),
        "INSERT_N_TRAIN": latex_escape(str(stats["n_train"])),
        "INSERT_TOP_CANDIDATES": top_candidates,
        "INSERT_ITERATION_LOG": iteration_log,
        "INSERT_TIMESTAMP": latex_escape(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        "INSERT_TARGET_COL": latex_escape(data["state"].get("target_col", "Target")),
        "INSERT_OBJECTIVE": latex_escape(str(data["state"].get("objective", "max")).upper()),
        "INSERT_ENGINE": latex_escape(str(data["state"].get("engine", "hebo")).upper()),
        "INSERT_BATCH_SIZE": latex_escape(str(data["state"].get("batch_size", 1))),
        "INSERT_RUN_ID": latex_escape(run_id),
    }

    result = template
    for key in sorted(replacements, key=len, reverse=True):
        result = result.replace(key, str(replacements[key]))

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result)

    print(f"\n✅ LaTeX report generated: {output_file}")

    return output_file


def _get_default_template() -> str:
    """Return the default minimal LaTeX template."""
    return r"""
% Bayesian Optimization Results Report
% Generated by generate_latex_report.py
% Compile with: xelatex

\documentclass[11pt,letterpaper]{report}
\usepackage{latex/scientific_report}
\usepackage{array}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{subcaption}
\usepackage{float}
\usepackage{listings}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{geometry}
\usepackage{hyperref}

% Metadata
\hypersetup{
    pdftitle={Bayesian Optimization Report - INSERT_RUN_ID},
    pdfauthor={BO Agent},
    pdfkeywords={optimization, bayesian, surrogate}
}

\begin{document}

% ============================================================================
% TITLE PAGE
% ============================================================================

\makereporttitle
    {Bayesian Optimization Results}
    {Run ID: INSERT_RUN_ID}
    {BO Agent}
    {BOGroupResearch}
    {\today}

% Table of Contents
\tableofcontents
\newpage

% ============================================================================
% EXECUTIVE SUMMARY
% ============================================================================

\chapter*{Executive Summary}
\addcontentsline{toc}{chapter}{Executive Summary}

\begin{executivesummary}[Optimization Overview]
This report presents results from a Bayesian Optimization (BO) campaign using a
surrogate-assisted approach. The optimization explored INSERT_ITERATIONS iterations
across a INSERT_DIM-dimensional parameter space.
\end{executivesummary}

\subsection*{Key Findings}

\begin{keyfindings}
\begin{enumerate}
    \item \textbf{Best Value Found:} INSERT_BEST_Y at iteration INSERT_BEST_ITER
    \item \textbf{Improvement:} INSERT_IMPROVEMENT_PERCENT\% relative to initial best
    \item \textbf{Oracle Quality:} RMSE = INSERT_RMSE (INSERT_FIDELITY fidelity)
    \item \textbf{Convergence:} Steady improvement from iteration 1 to INSERT_BEST_ITER
\end{enumerate}
\end{keyfindings}

% ============================================================================
% METHODOLOGY
% ============================================================================

\chapter{Methodology}

\section{Optimization Setup}

\begin{methodology}[Configuration]
\begin{itemize}
    \item \textbf{Engine:} INSERT_ENGINE
    \item \textbf{Objective:} INSERT_OBJECTIVE ``INSERT_TARGET_COL''
    \item \textbf{Iterations:} INSERT_ITERATIONS
    \item \textbf{Batch Size:} INSERT_BATCH_SIZE
    \item \textbf{Dimensions:} INSERT_DIM
\end{itemize}
\end{methodology}

\section{Oracle Model}

The surrogate oracle was trained with:
\begin{itemize}
    \item \textbf{Training Samples:} INSERT_N_TRAIN
    \item \textbf{Cross-Validation:} INSERT_CV_FOLDS-fold CV
    \item \textbf{CV RMSE:} INSERT_RMSE
\end{itemize}

% ============================================================================
% RESULTS
% ============================================================================

\chapter{Results}

\section{Convergence}

\begin{resultsbox}[Primary Result]
The optimization identified candidates with
\textbf{INSERT_IMPROVEMENT_PERCENT\%} improvement relative to the initial best value.
\end{resultsbox}

\begin{table}[htbp]
\centering
\caption{Convergence Summary}
\label{tab:summary}
\begin{tabular}{@{}lr@{}}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
Initial Best & INSERT_INITIAL_BEST \\
\rowcolor{tablealt} Best Found (BO) & INSERT_BEST_Y \\
Absolute Improvement & INSERT_IMPROVEMENT \\
\rowcolor{tablealt} Percent Improvement & INSERT_IMPROVEMENT_PERCENT\% \\
Best At Iteration & INSERT_BEST_ITER \\
\bottomrule
\end{tabular}
\end{table}

\section{Top Candidates}

\begin{table}[htbp]
\centering
\caption{Top 5 Candidates for Validation}
\label{tab:candidates}
\begin{tabular}{@{}lrlp{8cm}@{}}
\toprule
\textbf{Rank} & \textbf{Iteration} & \textbf{Predicted Y} & \textbf{Parameters} \\
\midrule
INSERT_TOP_CANDIDATES
\bottomrule
\end{tabular}
\end{table}

% ============================================================================
% DISCUSSION
% ============================================================================

\chapter{Discussion}

\section{Summary}

\begin{keyfindings}
The BO campaign successfully identified promising candidates through
INSERT_ITERATIONS iterations of surrogate-guided search. The oracle model
(RMSE = INSERT_RMSE) provides INSERT_FIDELITY predictions.
\end{keyfindings}

\section{Recommendations}

\begin{recommendations}[Next Steps]
\begin{enumerate}
    \item Experimentally validate the top 5 candidates listed in Table~\ref{tab:candidates}
    \item Retrain the oracle with new data to improve fidelity
    \item Continue BO iterations if budget permits
\end{enumerate}
\end{recommendations}

\section{Limitations}

\begin{limitations}
\begin{itemize}
    \item Surrogate predictions have inherent uncertainty (RMSE = INSERT_RMSE)
    \item Results are simulation-based; real-world validation is essential
    \item Convergence depends on oracle fidelity and problem structure
\end{itemize}
\end{limitations}

% ============================================================================
% APPENDICES
% ============================================================================

\appendix

\chapter{Data}

\appendixsection{Iteration Log}

INSERT_ITERATION_LOG

\appendixsection{Run Information}

\begin{itemize}
    \item \textbf{Run ID:} \texttt{INSERT_RUN_ID}
    \item \textbf{Generated:} INSERT_TIMESTAMP
    \item \textbf{Engine:} INSERT_ENGINE
    \item \textbf{Objective:} INSERT_OBJECTIVE
\end{itemize}

% ============================================================================
% END
% ============================================================================

\end{document}
"""


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
        output_path = generate_latex_report(
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