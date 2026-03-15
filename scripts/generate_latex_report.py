#!/usr/bin/env python
"""
Generate LaTeX scientific reports from BO run results.

Usage:
    uv run python scripts/generate_latex_report.py <RUN_ID>
    uv run python scripts/generate_latex_report.py vivid-heron-3397 --output custom_report.tex
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import argparse

import pandas as pd
from scipy import stats


def load_run_data(run_id: str):
    """Load all data from a BO run."""
    run_path = Path(f"runs/{run_id}")
    
    if not run_path.exists():
        raise FileNotFoundError(f"Run directory not found: {run_path}")
    
    # Load state
    with open(run_path / "state.json") as f:
        state = json.load(f)
    
    # Load observations and suggestions (both optional)
    observations = pd.DataFrame()
    if (run_path / "observations.jsonl").exists():
        observations = pd.read_json(run_path / "observations.jsonl", lines=True)
    
    suggestions = pd.DataFrame()
    if (run_path / "suggestions.jsonl").exists():
        suggestions = pd.read_json(run_path / "suggestions.jsonl", lines=True)
    
    # Load oracle metadata if available
    oracle_meta = {}
    if (run_path / "oracle_meta.json").exists():
        with open(run_path / "oracle_meta.json") as f:
            oracle_meta = json.load(f)
    
    # Load report if available
    report_data = {}
    if (run_path / "report.json").exists():
        with open(run_path / "report.json") as f:
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

    # Convergence metrics
    n_iterations = len(obs) if len(obs) > 0 else state.get("iterations", 0)
    best_y = obs["y"].max() if len(obs) > 0 else 0
    best_iter = (obs["y"].idxmax() + 1) if len(obs) > 0 else 0
    initial_best = obs["y"].iloc[0] if len(obs) > 0 else 0
    improvement = best_y - initial_best if best_y and initial_best else 0
    improvement_pct = (improvement / abs(initial_best)) * 100 if initial_best != 0 else 0

    # Use report data if available
    if report:
        best_y = report.get("best_y", best_y)
        best_iter = report.get("best_y_idx", best_iter)
        initial_best = report.get("initial_best", initial_best)
        improvement = report.get("improvement", improvement)
        improvement_pct = report.get("improvement_percent", improvement_pct)

    # Oracle quality
    rmse = oracle.get("selected_rmse", oracle.get("cv_rmse", None))
    if isinstance(rmse, dict):
        rmse = rmse.get("extra_trees", rmse.get("random_forest", None))
    fidelity = "high" if rmse and rmse < 0.1 else "moderate" if rmse and rmse < 0.2 else "low"

    # Dimensions
    n_features = state.get("n_vars", len(state.get("bounds", [])))

    # Additional statistics for more detailed report
    if len(obs) > 0:
        # Performance metrics
        mean_y = obs["y"].mean()
        std_y = obs["y"].std()
        median_y = obs["y"].median()
        min_y = obs["y"].min()
        max_y = obs["y"].max()

        # Convergence analysis
        rolling_best = obs["y"].expanding().max()
        convergence_rate = (rolling_best.iloc[-1] - rolling_best.iloc[0]) / len(rolling_best)

        # Statistical significance (if we have enough data)
        from scipy import stats
        if len(obs) >= 10:
            # Test if final performance is significantly better than initial
            initial_mean = obs["y"].iloc[:5].mean()  # First 5 iterations
            final_mean = obs["y"].iloc[-5:].mean()   # Last 5 iterations
            t_stat, p_value = stats.ttest_ind(obs["y"].iloc[:5], obs["y"].iloc[-5:])
        else:
            p_value = None

        # Parameter analysis (if x data is available)
        param_ranges = {}
        if "x" in obs.columns and len(obs["x"]) > 0:
            # Extract parameter names from first x dict
            first_x = obs["x"].iloc[0]
            if isinstance(first_x, dict):
                param_names = list(first_x.keys())
                for param in param_names:
                    values = obs["x"].apply(lambda x: x.get(param) if isinstance(x, dict) else None).dropna()
                    if len(values) > 0:
                        param_ranges[param] = {
                            "min": values.min(),
                            "max": values.max(),
                            "mean": values.mean(),
                            "std": values.std()
                        }
    else:
        mean_y = std_y = median_y = min_y = max_y = convergence_rate = p_value = None
        param_ranges = {}

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
        # Additional statistics
        "mean_y": mean_y,
        "std_y": std_y,
        "median_y": median_y,
        "min_y": min_y,
        "max_y": max_y,
        "convergence_rate": convergence_rate,
        "p_value": p_value,
        "param_ranges": param_ranges,
        "_observations": obs,  # Store observations for helper functions
    }


def build_top_candidates_table(suggestions: pd.DataFrame, n: int = 5) -> str:
    """Build LaTeX table of top candidates."""
    if len(suggestions) == 0:
        return "\\textit{No suggestions available}"
    
    rows = ""
    n_show = min(n, len(suggestions))
    
    # If y_pred exists, sort by it; otherwise just take the latest
    if "y_pred" in suggestions.columns and len(suggestions["y_pred"].dropna()) > 0:
        top = suggestions.nlargest(n_show, "y_pred")
    else:
        # Just take the latest suggestions
        top = suggestions.tail(n_show)
    
    for rank, (idx, row) in enumerate(top.iterrows(), 1):
        y_pred = row.get("y_pred", "TBD")
        x_str = str(row.get("x", {}))[:30]
        
        rows += f"{rank} & {idx+1} & {y_pred} & {x_str}... \\\\\n"
        if rank < n_show:
            rows += "\\rowcolor{tablealt} "
    
    return rows


def build_iteration_table(observations: pd.DataFrame) -> str:
    """Build LaTeX table of all iterations."""
    if len(observations) == 0:
        return """
\\begin{table}[htbp]
\\centering
\\caption{Complete Iteration History}
\\label{tab:iterations}
\\begin{tabular}{@{}rr@{}}
\\toprule
\\textbf{Iteration} & \\textbf{Observed Value} \\\\
\\midrule
\\multicolumn{2}{c}{\\textit{No observations available}} \\\\
\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    rows = ""
    for idx, row in observations.iterrows():
        y = row.get("y", 0)
        rows += f"{idx+1} & {y:.6f} \\\\\n"
        if (idx + 1) % 2 == 0 and idx < len(observations) - 1:
            rows += "\\rowcolor{tablealt} "
    
    return f"""
\\begin{{table}}[htbp]
\\centering
\\caption{{Complete Iteration History}}
\\label{{tab:iterations}}
\\begin{{tabular}}{{@{{}}rr@{{}}}}
\\toprule
\\textbf{{Iteration}} & \\textbf{{Observed Value}} \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def _format_p_value_text(p_value):
    """Format p-value for display in LaTeX."""
    if p_value is None:
        return "Not available (insufficient data)"
    elif p_value < 0.001:
        return "$p < 0.001$ (highly significant)"
    elif p_value < 0.01:
        return f"$p = {p_value:.3f}$ (very significant)"
    elif p_value < 0.05:
        return f"$p = {p_value:.3f}$ (significant)"
    else:
        return f"$p = {p_value:.3f}$ (not significant)"


def _analyze_convergence_pattern(stats):
    """Analyze the convergence pattern for description."""
    if stats.get('convergence_rate') is None:
        return "steady improvement throughout the optimization process"

    rate = stats['convergence_rate']
    if rate > 0.1:
        return "rapid initial improvement followed by refinement"
    elif rate > 0.05:
        return "steady improvement with consistent progress"
    else:
        return "gradual convergence requiring extensive exploration"


def _get_phase_stats(stats, phase_key):
    """Get phase statistics for the template."""
    obs = stats.get('_observations', pd.DataFrame())
    if len(obs) == 0:
        return "N/A"

    if phase_key == 'initial_mean':
        if len(obs) >= 5:
            return f"{obs['y'].iloc[:5].mean():.6f}"
        else:
            return f"{obs['y'].iloc[0]:.6f}" if len(obs) > 0 else "N/A"
    elif phase_key == 'initial_std':
        if len(obs) >= 5:
            return f"{obs['y'].iloc[:5].std():.6f}"
        else:
            return "0.000000"
    elif phase_key == 'middle_mean':
        n_iter = stats["n_iterations"]
        start_idx = 5
        end_idx = min(n_iter // 2, len(obs))
        if end_idx > start_idx:
            return f"{obs['y'].iloc[start_idx:end_idx].mean():.6f}"
        return "N/A"
    elif phase_key == 'middle_std':
        n_iter = stats["n_iterations"]
        start_idx = 5
        end_idx = min(n_iter // 2, len(obs))
        if end_idx > start_idx:
            return f"{obs['y'].iloc[start_idx:end_idx].std():.6f}"
        return "N/A"
    elif phase_key == 'final_mean':
        n_iter = stats["n_iterations"]
        start_idx = max(5, n_iter // 2)
        if start_idx < len(obs):
            return f"{obs['y'].iloc[start_idx:].mean():.6f}"
        return "N/A"
    elif phase_key == 'final_std':
        n_iter = stats["n_iterations"]
        start_idx = max(5, n_iter // 2)
        if start_idx < len(obs):
            return f"{obs['y'].iloc[start_idx:].std():.6f}"
        return "N/A"

    return "N/A"


def _get_middle_phase_end(stats):
    """Get the end iteration for middle phase."""
    n_iter = stats["n_iterations"]
    return str(max(5, n_iter // 2))


def _get_final_phase_start(stats):
    """Get the start iteration for final phase."""
    n_iter = stats["n_iterations"]
    return str(max(6, n_iter // 2 + 1))


def _build_statistical_significance_section(stats):
    """Build the statistical significance section."""
    p_value = stats.get('p_value')
    if p_value is None or stats["n_iterations"] < 10:
        return """
\\subsection{Statistical Analysis}

\\textit{Statistical significance testing requires at least 10 iterations of data.}
"""
    else:
        significance = "significant" if p_value < 0.05 else "not significant"
        return f"""
\\subsection{{Statistical Significance}}

A t-test comparing the initial performance (iterations 1-5, mean = {_get_phase_stats(stats, 'initial_mean')})
against final performance (iterations {_get_final_phase_start(stats)}-{stats["n_iterations"]}, mean = {_get_phase_stats(stats, 'final_mean')})
yielded $p = {p_value:.4f}$, indicating that the improvement was \\textit{{{significance}}}.
"""


def _build_parameter_analysis_section(stats):
    """Build parameter analysis section."""
    param_ranges = stats.get('param_ranges', {})
    if not param_ranges:
        return """
\\section{Parameter Space Exploration}

\\textit{Parameter analysis requires x data in observations.}
"""
    else:
        rows = ""
        for param, info in param_ranges.items():
            rows += f"{param} & {info['min']:.3f} & {info['max']:.3f} & {info['mean']:.3f} & {info['std']:.3f} \\\\\n"

        return f"""
\\section{{Parameter Space Exploration}}

The optimization explored the following parameter ranges:

\\begin{{table}}[H]
\\centering
\\caption{{Parameter Exploration Summary}}
\\label{{tab:param_ranges}}
\\begin{{tabular}}{{@{{}}lSSSS@{{}}}}
\\toprule
\\textbf{{Parameter}} & \\textbf{{Min}} & \\textbf{{Max}} & \\textbf{{Mean}} & \\textbf{{Std. Dev.}} \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def _build_enhanced_top_candidates_table(suggestions):
    """Build enhanced top candidates table."""
    if len(suggestions) == 0:
        return "\\textit{No suggestions available}"

    rows = ""
    n_show = min(5, len(suggestions))

    # Sort by predicted y if available
    if "y_pred" in suggestions.columns and len(suggestions["y_pred"].dropna()) > 0:
        top = suggestions.nlargest(n_show, "y_pred")
    else:
        top = suggestions.tail(n_show)

    for rank, (idx, row) in enumerate(top.iterrows(), 1):
        y_pred = row.get("y_pred", "TBD")
        x_str = str(row.get("x", {}))[:50]  # Truncate for table
        if len(x_str) > 47:
            x_str = x_str[:47] + "..."

        rows += f"{rank} & {idx+1} & {y_pred} & {x_str} \\\\\n"

    return rows


def _get_convergence_insights(stats):
    """Get convergence insights for discussion."""
    if stats.get('best_iter') and stats['best_iter'] > stats["n_iterations"] * 0.8:
        return "The optimal solution was found late in the optimization process, suggesting the need for extended exploration"
    elif stats.get('best_iter') and stats['best_iter'] < stats["n_iterations"] * 0.3:
        return "The optimal solution was found early, indicating efficient exploration of the parameter space"
    else:
        return "The optimization showed balanced exploration and exploitation throughout the process"


def _get_p_value_interpretation(stats):
    """Get p-value interpretation."""
    p_value = stats.get('p_value')
    if p_value is None:
        return "could not be assessed due to limited data"
    elif p_value < 0.05:
        return f"was statistically significant ($p = {p_value:.4f}$)"
    else:
        return f"was not statistically significant ($p = {p_value:.4f}$)"


def _get_oracle_reliability_statement(stats):
    """Get oracle reliability statement."""
    fidelity = stats.get('fidelity', 'moderate')
    rmse = stats.get('rmse')
    if rmse and rmse < 0.1:
        return "provides highly reliable predictions suitable for guiding optimization decisions"
    elif rmse and rmse < 0.2:
        return "provides moderately reliable predictions that should be validated experimentally"
    else:
        return "provides low-fidelity predictions requiring careful experimental validation"


def _get_middle_sample_size(stats):
    """Get sample size for middle phase."""
    n_iter = stats["n_iterations"]
    start_idx = 5
    end_idx = min(n_iter // 2, len(stats.get('_observations', [])))
    return str(max(0, end_idx - start_idx))


def _get_final_sample_size(stats):
    """Get sample size for final phase."""
    n_iter = stats["n_iterations"]
    start_idx = max(5, n_iter // 2)
    total_obs = len(stats.get('_observations', []))
    return str(max(0, total_obs - start_idx))


def generate_latex_report(run_id: str, output_file: str = None, template_file: str = None):
    """
    Generate LaTeX report from BO results.
    
    Args:
        run_id: BO run ID
        output_file: Output .tex path (default: runs/<RUN_ID>/report.tex)
        template_file: Custom template path (default: built-in minimal template)
    """
    
    # Load data
    print(f"📊 Loading BO results from run: {run_id}")
    data = load_run_data(run_id)
    
    # Compute statistics
    stats = compute_statistics(data)
    print(f"   Best value: {stats['best_y']:.6f} (iter {stats['best_iter']})")
    print(f"   Improvement: {stats['improvement_pct']:.1f}%")
    print(f"   Oracle RMSE: {stats['rmse'] if stats['rmse'] else 'N/A'}")
    
    # Build tables
    top_candidates = build_top_candidates_table(data["suggestions"], n=5)
    iteration_log = build_iteration_table(data["observations"])
    
    # Generate convergence plots
    try:
        import subprocess
        result = subprocess.run([
            sys.executable, "scripts/generate_convergence_plots.py", run_id
        ], capture_output=True, text=True)
        if result.returncode == 0:
            print("📊 Convergence plots generated successfully")
        else:
            print(f"⚠️  Warning: Could not generate plots: {result.stderr}")
    except Exception as e:
        print(f"⚠️  Warning: Could not generate plots: {e}")
    
    # Determine output file
    if output_file is None:
        output_file = data["run_path"] / "report.tex"
    else:
        output_file = Path(output_file)
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load template
    if template_file:
        with open(template_file) as f:
            template = f.read()
    else:
        # Use built-in minimal template
        template = _get_default_template()
    
    # Replace placeholders
    replacements = {
        "INSERT_RUN_ID": run_id,
        "INSERT_ITERATIONS": str(stats["n_iterations"]),
        "INSERT_DIM": str(stats["n_features"]),
        "INSERT_BEST_Y": f"{stats['best_y']:.6f}",
        "INSERT_BEST_ITER": str(stats["best_iter"]),
        "INSERT_IMPROVEMENT": f"{stats['improvement']:.6f}",
        "INSERT_IMPROVEMENT_PERCENT": f"{stats['improvement_pct']:.1f}",
        "INSERT_INITIAL_BEST": f"{stats['initial_best']:.6f}",
        "INSERT_RMSE": f"{stats['rmse']:.6f}" if stats['rmse'] else "N/A",
        "INSERT_FIDELITY": stats["fidelity"],
        "INSERT_CV_FOLDS": str(stats["cv_folds"]),
        "INSERT_N_TRAIN": str(stats["n_train"]),
        "INSERT_TOP_CANDIDATES": top_candidates,
        "INSERT_ITERATION_LOG": iteration_log,
        "INSERT_TIMESTAMP": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "INSERT_TARGET_COL": data["state"].get("target_col", "Target"),
        "INSERT_OBJECTIVE": data["state"].get("objective", "max").upper(),
        "INSERT_ENGINE": data["state"].get("engine", "hebo").upper(),
        "INSERT_BATCH_SIZE": str(data["state"].get("batch_size", 1)),
        "INSERT_SEED": str(data["state"].get("seed", "N/A")),
        # New placeholders for enhanced template
        "INSERT_MEAN_Y": f"{stats.get('mean_y', 0):.6f}" if stats.get('mean_y') else "N/A",
        "INSERT_STD_Y": f"{stats.get('std_y', 0):.6f}" if stats.get('std_y') else "N/A",
        "INSERT_MEDIAN_Y": f"{stats.get('median_y', 0):.6f}" if stats.get('median_y') else "N/A",
        "INSERT_MIN_Y": f"{stats.get('min_y', 0):.6f}" if stats.get('min_y') else "N/A",
        "INSERT_MAX_Y": f"{stats.get('max_y', 0):.6f}" if stats.get('max_y') else "N/A",
        "INSERT_CONVERGENCE_RATE": f"{stats.get('convergence_rate', 0):.6f}" if stats.get('convergence_rate') else "N/A",
        "INSERT_P_VALUE_TEXT": _format_p_value_text(stats.get('p_value')),
        "INSERT_CONVERGENCE_PATTERN": _analyze_convergence_pattern(stats),
        "INSERT_INITIAL_PHASE_MEAN": _get_phase_stats(stats, 'initial_mean'),
        "INSERT_INITIAL_PHASE_STD": _get_phase_stats(stats, 'initial_std'),
        "INSERT_MIDDLE_END": _get_middle_phase_end(stats),
        "INSERT_MIDDLE_PHASE_MEAN": _get_phase_stats(stats, 'middle_mean'),
        "INSERT_MIDDLE_PHASE_STD": _get_phase_stats(stats, 'middle_std'),
        "INSERT_MIDDLE_SAMPLE_SIZE": _get_middle_sample_size(stats),
        "INSERT_FINAL_START": _get_final_phase_start(stats),
        "INSERT_FINAL_PHASE_MEAN": _get_phase_stats(stats, 'final_mean'),
        "INSERT_FINAL_PHASE_STD": _get_phase_stats(stats, 'final_std'),
        "INSERT_FINAL_SAMPLE_SIZE": _get_final_sample_size(stats),
        "INSERT_N_ITERATIONS": str(stats["n_iterations"]),
        "INSERT_STATISTICAL_SIGNIFICANCE_SECTION": _build_statistical_significance_section(stats),
        "INSERT_PARAMETER_ANALYSIS_SECTION": _build_parameter_analysis_section(stats),
        "INSERT_TOP_CANDIDATES_TABLE": _build_enhanced_top_candidates_table(data["suggestions"]),
        "INSERT_CONVERGENCE_INSIGHTS": _get_convergence_insights(stats),
        "INSERT_P_VALUE_INTERPRETATION": _get_p_value_interpretation(stats),
        "INSERT_ORACLE_RELIABILITY_STATEMENT": _get_oracle_reliability_statement(stats),
        "INSERT_STATE_JSON": json.dumps(data["state"], indent=2),
    }
    
    result = template
    for key, value in replacements.items():
        result = result.replace(key, str(value))
    
    # Write output
    with open(output_file, "w") as f:
        f.write(result)
    
    print(f"\n✅ LaTeX report generated: {output_file}")
    print(f"\n📝 Next steps:")
    print(f"   1. cd {output_file.parent}")
    print(f"   2. xelatex {output_file.name}")
    print(f"   3. Open {output_file.with_suffix('.pdf').name}")
    
    return output_file


def _get_default_template() -> str:
    """Return the enhanced professional LaTeX template."""
    return r"""
% Bayesian Optimization Results Report
% Generated by bo_latex_report.py
% Compile with: xelatex

\documentclass[11pt,letterpaper]{report}
\usepackage{scientific_report}
\usepackage{array}
\usepackage{booktabs}
\usepackage{siunitx}
\usepackage{amsmath}
\usepackage{subcaption}
\usepackage{float}
\usepackage{listings}
\usepackage{graphicx}
\usepackage{xcolor}

% Configure siunitx for scientific notation
\sisetup{
    scientific-notation = true,
    round-mode = places,
    round-precision = 4,
}

% Configure code listings
\lstset{
    basicstyle=\ttfamily\footnotesize,
    breaklines=true,
    captionpos=b,
    numbers=left,
    numberstyle=\tiny,
    frame=single,
    backgroundcolor=\color{gray!10}
}

% Metadata
\hypersetup{
    pdftitle={Bayesian Optimization Report - INSERT_RUN_ID},
    pdfauthor={BO Agent},
    pdfkeywords={optimization, bayesian, surrogate, machine learning}
}

\begin{document}

% ============================================================================
% TITLE PAGE
% ============================================================================

\makereporttitle
    {Bayesian Optimization Results Report}
    {Run ID: INSERT_RUN_ID}
    {BO Agent}
    {BOGroupResearch}
    {\today}

% Table of Contents
\tableofcontents
\listoftables
\listoffigures
\newpage

% ============================================================================
% EXECUTIVE SUMMARY
% ============================================================================

\chapter*{Executive Summary}
\addcontentsline{toc}{chapter}{Executive Summary}

\begin{executivesummary}[Optimization Overview]
This comprehensive report presents the results from a Bayesian Optimization (BO) campaign
conducted on INSERT_TARGET_COL optimization using the INSERT_ENGINE algorithm.
The campaign explored INSERT_ITERATIONS iterations across a INSERT_DIM-dimensional
parameter space, achieving a \SI{INSERT_IMPROVEMENT_PERCENT}{\percent} improvement
over the baseline performance.
\end{executivesummary}

\subsection*{Key Performance Metrics}

\begin{keyfindings}
\begin{enumerate}
    \item \textbf{Optimal Value:} \SI{INSERT_BEST_Y}{} achieved at iteration INSERT_BEST_ITER
    \item \textbf{Performance Improvement:} \SI{INSERT_IMPROVEMENT_PERCENT}{\percent} relative to initial baseline (\SI{INSERT_INITIAL_BEST}{})
    \item \textbf{Oracle Fidelity:} RMSE = \SI{INSERT_RMSE}{} (\textit{INSERT_FIDELITY fidelity})
    \item \textbf{Statistical Significance:} INSERT_P_VALUE_TEXT
    \item \textbf{Convergence Rate:} \SI{INSERT_CONVERGENCE_RATE}{} per iteration
\end{enumerate}
\end{keyfindings}

\subsection*{Summary Statistics}

\begin{table}[H]
\centering
\caption{Summary Statistics}
\label{tab:summary_stats}
\begin{tabular}{@{}lS[table-format=1.4]@{}}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
Initial Best & INSERT_INITIAL_BEST \\
Final Best & INSERT_BEST_Y \\
Absolute Improvement & INSERT_IMPROVEMENT \\
Percent Improvement & \SI{INSERT_IMPROVEMENT_PERCENT}{\percent} \\
\addlinespace
Mean Performance & INSERT_MEAN_Y \\
Standard Deviation & INSERT_STD_Y \\
Median Performance & INSERT_MEDIAN_Y \\
Range (Min-Max) & INSERT_MIN_Y -- INSERT_MAX_Y \\
\addlinespace
Convergence Rate & INSERT_CONVERGENCE_RATE \\
Iterations to Convergence & INSERT_BEST_ITER \\
\bottomrule
\end{tabular}
\end{table}

% ============================================================================
% METHODOLOGY
% ============================================================================

\chapter{Methodology}

\section{Optimization Framework}

\begin{methodology}[Bayesian Optimization Setup]
The optimization campaign was conducted using a surrogate-assisted Bayesian Optimization
framework with the following configuration:

\begin{itemize}
    \item \textbf{Optimization Engine:} INSERT_ENGINE
    \item \textbf{Objective Function:} INSERT_OBJECTIVE maximization of \texttt{INSERT_TARGET_COL}
    \item \textbf{Total Iterations:} INSERT_ITERATIONS
    \item \textbf{Batch Size:} INSERT_BATCH_SIZE candidates per iteration
    \item \textbf{Parameter Dimensions:} INSERT_DIM continuous variables
    \item \textbf{Random Seed:} INSERT_SEED for reproducibility
\end{itemize}
\end{methodology}

\section{Surrogate Model}

\begin{methodology}[Oracle Configuration]
The surrogate oracle was constructed using machine learning regression models
trained on experimental data to approximate the expensive objective function:

\begin{itemize}
    \item \textbf{Training Dataset:} INSERT_N_TRAIN samples
    \item \textbf{Cross-Validation:} INSERT_CV_FOLDS-fold stratified CV
    \item \textbf{Model Selection:} Best performing model based on CV RMSE
    \item \textbf{Prediction Fidelity:} INSERT_FIDELITY (RMSE = \SI{INSERT_RMSE}{})
    \item \textbf{Feature Engineering:} Automatic feature selection and preprocessing
\end{itemize}
\end{methodology}

% ============================================================================
% RESULTS AND ANALYSIS
% ============================================================================

\chapter{Results and Analysis}

\section{Convergence Analysis}

\begin{resultsbox}[Optimization Trajectory]
The Bayesian Optimization campaign demonstrated strong convergence characteristics,
identifying the optimal solution at iteration INSERT_BEST_ITER with a
\SI{INSERT_IMPROVEMENT_PERCENT}{\percent} improvement over the initial baseline.
The optimization trajectory shows INSERT_CONVERGENCE_PATTERN.
\end{resultsbox}

\subsection{Performance Trajectory}

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{convergence_plot.png}
\caption{Optimization convergence plots showing objective function values over iterations (left) and cumulative best values with uncertainty bounds (right).}
\label{fig:convergence}
\end{figure}

\subsection{Statistical Analysis}

\begin{table}[H]
\centering
\caption{Statistical Analysis of Optimization Results}
\label{tab:statistical_analysis}
\begin{tabular}{@{}lSSSS@{}}
\toprule
\textbf{Phase} & \textbf{Mean Performance} & \textbf{Std. Deviation} & \textbf{Sample Size} \\
\midrule
Initial (Iterations 1-5) & INSERT_INITIAL_PHASE_MEAN & INSERT_INITIAL_PHASE_STD & 5 \\
Middle (Iterations 6-INSERT_MIDDLE_END) & INSERT_MIDDLE_PHASE_MEAN & INSERT_MIDDLE_PHASE_STD & INSERT_MIDDLE_SAMPLE_SIZE \\
Final (Iterations INSERT_FINAL_START-INSERT_N_ITERATIONS) & INSERT_FINAL_PHASE_MEAN & INSERT_FINAL_PHASE_STD & INSERT_FINAL_SAMPLE_SIZE \\
\addlinespace
Overall & INSERT_MEAN_Y & INSERT_STD_Y & INSERT_N_ITERATIONS \\
\bottomrule
\end{tabular}
\end{table}

INSERT_STATISTICAL_SIGNIFICANCE_SECTION

\section{Parameter Space Exploration}

INSERT_PARAMETER_ANALYSIS_SECTION

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{parameter_distributions.png}
\caption{Distributions of parameters explored during optimization, showing the range and frequency of values tested for each parameter.}
\label{fig:param_dist}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{correlation_matrix.png}
\caption{Correlation matrix showing relationships between parameters and the objective function. Red indicates positive correlation, blue indicates negative correlation.}
\label{fig:correlation}
\end{figure}

\section{Top Performing Candidates}

\begin{table}[H]
\centering
\caption{Top 5 Candidates for Experimental Validation}
\label{tab:top_candidates}
\begin{tabular}{@{}rlS[table-format=1.6]l@{}}
\toprule
\textbf{Rank} & \textbf{Iteration} & \textbf{Predicted Value} & \textbf{Parameter Configuration} \\
\midrule
INSERT_TOP_CANDIDATES_TABLE
\bottomrule
\end{tabular}
\end{table}

% ============================================================================
% DISCUSSION
% ============================================================================

\chapter{Discussion}

\section{Performance Evaluation}

\begin{keyfindings}
The Bayesian Optimization campaign successfully identified high-performing candidates
through INSERT_ITERATIONS iterations of surrogate-guided exploration. The surrogate
oracle demonstrated INSERT_FIDELITY predictive fidelity (RMSE = \SI{INSERT_RMSE}{}),
providing reliable guidance throughout the optimization process.

Key observations from the optimization trajectory:
\begin{enumerate}
    \item Rapid initial improvement during the exploration phase
    \item Steady convergence to optimal regions in the parameter space
    \item INSERT_CONVERGENCE_INSIGHTS
    \item Statistical significance INSERT_P_VALUE_INTERPRETATION
\end{enumerate}
\end{keyfindings}

\section{Oracle Model Assessment}

\begin{resultsbox}[Surrogate Model Quality]
The surrogate oracle achieved \SI{INSERT_RMSE}{} root mean square error on
INSERT_CV_FOLDS-fold cross-validation, indicating INSERT_FIDELITY predictive
performance. This level of fidelity INSERT_ORACLE_RELIABILITY_STATEMENT.
\end{resultsbox}

\section{Computational Efficiency}

The optimization required INSERT_ITERATIONS function evaluations to achieve
convergence, representing an efficient exploration of the INSERT_DIM-dimensional
parameter space. The surrogate-assisted approach reduced the experimental burden
by guiding the search toward promising regions.

% ============================================================================
% RECOMMENDATIONS
% ============================================================================

\chapter{Recommendations}

\section{Experimental Validation}

\begin{recommendations}[Priority Validation Candidates]
Based on the optimization results, the following candidates are recommended
for experimental validation in order of priority:

\begin{enumerate}
    \item \textbf{Primary Candidate:} The optimal solution identified at iteration INSERT_BEST_ITER
    \item \textbf{Secondary Candidates:} Top 5 predicted performers (Table~\ref{tab:top_candidates})
    \item \textbf{Diverse Candidates:} Representative samples from different parameter regions
\end{enumerate}
\end{recommendations}

\section{Oracle Refinement}

\begin{recommendations}[Model Improvement Strategies]
To enhance future optimization campaigns:

\begin{enumerate}
    \item Incorporate validation results to retrain the surrogate oracle
    \item Consider ensemble modeling approaches for improved prediction fidelity
    \item Evaluate alternative surrogate models (Gaussian Processes, Neural Networks)
    \item Implement active learning strategies for adaptive sampling
\end{enumerate}
\end{recommendations}

\section{Optimization Extensions}

\begin{recommendations}[Advanced Optimization Strategies]
\begin{enumerate}
    \item Multi-objective optimization for simultaneous target optimization
    \item Constrained optimization incorporating experimental feasibility
    \item Transfer learning from related optimization problems
    \item Robust optimization under uncertainty
\end{enumerate}
\end{recommendations}

% ============================================================================
% LIMITATIONS AND FUTURE WORK
% ============================================================================

\chapter{Limitations and Future Work}

\section{Current Limitations}

\begin{limitations}
\begin{itemize}
    \item \textbf{Surrogate Uncertainty:} Predictions have inherent uncertainty (RMSE = \SI{INSERT_RMSE}{})
    \item \textbf{Simulation vs. Reality:} Results are surrogate-based; experimental validation essential
    \item \textbf{Parameter Space Coverage:} Optimization may not have fully explored all promising regions
    \item \textbf{Model Assumptions:} Surrogate assumes smooth, well-behaved objective function
    \item \textbf{Computational Cost:} Each iteration requires oracle retraining and acquisition function optimization
\end{itemize}
\end{limitations}

\section{Future Research Directions}

\begin{criticalnotice}
Future work should focus on experimental validation of the predicted optimal
solutions and incorporation of real experimental data to improve surrogate model fidelity.
\end{criticalnotice}

% ============================================================================
% APPENDICES
% ============================================================================

\appendix

\chapter{Complete Results}

\appendixsection{Iteration-by-Iteration Results}

INSERT_ITERATION_LOG

\appendixsection{Oracle Model Details}

\begin{table}[H]
\centering
\caption{Oracle Model Performance Metrics}
\label{tab:oracle_metrics}
\begin{tabular}{@{}lS[table-format=1.4]@{}}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
Training Samples & INSERT_N_TRAIN \\
Cross-Validation Folds & INSERT_CV_FOLDS \\
CV RMSE & INSERT_RMSE \\
Prediction Fidelity & \textit{INSERT_FIDELITY} \\
\bottomrule
\end{tabular}
\end{table}

\appendixsection{Configuration Details}

\begin{lstlisting}[caption=BO Configuration JSON,label=lst:config]
INSERT_STATE_JSON
\end{lstlisting}

\appendixsection{Run Information}

\begin{itemize}
    \item \textbf{Run ID:} \texttt{INSERT_RUN_ID}
    \item \textbf{Generated:} INSERT_TIMESTAMP
    \item \textbf{Engine:} INSERT_ENGINE
    \item \textbf{Objective:} INSERT_OBJECTIVE \texttt{INSERT_TARGET_COL}
    \item \textbf{Dimensions:} INSERT_DIM
    \item \textbf{Iterations:} INSERT_ITERATIONS
    \item \textbf{Batch Size:} INSERT_BATCH_SIZE
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
        "--output", "-o",
        help="Output .tex file (default: runs/<RUN_ID>/report.tex)"
    )
    parser.add_argument(
        "--template", "-t",
        help="Custom LaTeX template file"
    )
    
    args = parser.parse_args()
    
    try:
        output_path = generate_latex_report(
            args.run_id,
            output_file=args.output,
            template_file=args.template
        )
    except FileNotFoundError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
