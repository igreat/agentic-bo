# Generating LaTeX Reports from BO Results: Complete Guide

## Overview

This guide shows how to use the **Scientific Writing** skill from the [claude-scientific-skills](https://github.com/K-Dense-AI/claude-scientific-skills) repository to automatically generate publication-ready LaTeX reports from your Bayesian Optimization (BO) results.

The workflow uses the `scientific_report.sty` LaTeX package—a professional style package with:
- **Helvetica typography** for clean, modern appearance
- **Colored box environments** for key findings, methodology, results, recommendations
- **Professional tables** with alternating row colors and statistics
- **Scientific commands** for p-values, effect sizes, confidence intervals
- **Ready-to-compile LaTeX** → PDF pipeline

---

## Step 1: Install Scientific Writing Skill

### Option A: Global Installation (Recommended)

Clone the scientific-skills repo and copy globally:

```bash
# Clone the repo
git clone https://github.com/K-Dense-AI/claude-scientific-skills.git

# Copy to global Claude skills directory (Windows)
robocopy claude-scientific-skills\scientific-skills\scientific-writing %USERPROFILE%\.claude\skills\scientific-writing /E

# Verify installation
ls %USERPROFILE%\.claude\skills\scientific-writing
# Should show: SKILL.md, assets/, references/
```

### Option B: Project-Level Installation

Copy to your `.claude/skills/` directory:

```bash
mkdir -p .claude\skills
robocopy ..\..\path\to\claude-scientific-skills\scientific-skills\scientific-writing .\.claude\skills\scientific-writing /E
```

### Verify Files

After installation, you should have:
```
.claude/skills/scientific-writing/
  SKILL.md                          # Skill documentation
  assets/
    scientific_report.sty           # LaTeX style package
    scientific_report_template.tex  # Complete template
    REPORT_FORMATTING_GUIDE.md      # Quick reference
  references/
    professional_report_formatting.md
    imrad_structure.md
    ...
```

---

## Step 2: Extract BO Results

Your BO run stores results in `runs/<RUN_ID>/`. Extract key data:

```python
# extract_bo_results.py
import json
import pandas as pd
from pathlib import Path

def extract_bo_results(run_id: str) -> dict:
    """Extract BO results for LaTeX report generation."""
    
    run_path = Path(f"runs/{run_id}")
    
    # Load state
    with open(run_path / "state.json") as f:
        state = json.load(f)
    
    # Load suggestions and observations
    suggestions = pd.read_json(run_path / "suggestions.jsonl", lines=True)
    observations = pd.read_json(run_path / "observations.jsonl", lines=True)
    
    # Load oracle metadata (if available)
    oracle_meta = None
    if (run_path / "oracle_meta.json").exists():
        with open(run_path / "oracle_meta.json") as f:
            oracle_meta = json.load(f)
    
    # Load report (if available)
    report_data = None
    if (run_path / "report.json").exists():
        with open(run_path / "report.json") as f:
            report_data = json.load(f)
    
    return {
        "state": state,
        "suggestions": suggestions,
        "observations": observations,
        "oracle_meta": oracle_meta,
        "report_data": report_data,
        "run_path": run_path
    }
```

**Key data from state.json:**
```json
{
  "run_id": "vivid-heron-3397",
  "engine": "hebo",
  "objective": "max",
  "target_col": "Target",
  "status": "completed",
  "seed": 42,
  "batch_size": 1,
  "iterations": 20
}
```

**Key data from report.json:**
```json
{
  "iterations": 20,
  "best_y": 0.89,
  "best_y_idx": 15,
  "initial_best": 0.45,
  "improvement": 0.44,
  "improvement_percent": 98.2,
  "oracle_rmse_cv": 0.12
}
```

---

## Step 3: Create BO-Specific LaTeX Template

Create a specialized template for BO reports:

```tex
% bo_report.tex
% Bayesian Optimization Results Report
% Compile with: xelatex bo_report.tex

\documentclass[11pt,letterpaper]{report}
\usepackage{scientific_report}
\usepackage{array}

% Document metadata
\hypersetup{
    pdftitle={Bayesian Optimization Report},
    pdfauthor={BO Agent},
    pdfkeywords={optimization, surrogate, results}
}

\begin{document}

% ============================================================================
% TITLE PAGE
% ============================================================================

\makereporttitle
    {Bayesian Optimization Results}
    {Surrogate-Assisted Experimental Design}
    {BO Agent}
    {BOGroupResearch}
    {\today}

% ============================================================================
% TABLE OF CONTENTS
% ============================================================================

\tableofcontents
\newpage

% ============================================================================
% EXECUTIVE SUMMARY
% ============================================================================

\chapter*{Executive Summary}
\addcontentsline{toc}{chapter}{Executive Summary}

\begin{executivesummary}[Optimization Overview]
This report presents results from a Bayesian Optimization (BO) campaign using a 
surrogate oracle trained on historical data. The optimization explored 
\textbf{INSERT_ITERATIONS} iterations across a \textbf{INSERT_DIM}-dimensional 
parameter space, identifying promising candidates through acquisition-function-guided 
search.
\end{executivesummary}

\subsection*{Key Results}

\begin{keyfindings}
\begin{enumerate}
    \item \textbf{Best Observed Value:} \textbf{INSERT_BEST_Y} 
          (improvement: \textbf{INSERT_IMPROVEMENT}\%)
    \item \textbf{Oracle Quality:} CV RMSE = \textbf{INSERT_RMSE} 
          on held-out validation data
    \item \textbf{Convergence:} Steady improvement from iteration 1 to 
          \textbf{INSERT_BEST_ITER}, plateauing thereafter
    \item \textbf{Candidates Identified:} \textbf{INSERT_TOP_N} high-value 
          candidates for validation
\end{enumerate}
\end{keyfindings}

% ============================================================================
% METHODOLOGY
% ============================================================================

\chapter{Methodology}

\section{Bayesian Optimization Framework}

\begin{methodology}[Optimization Setup]
A surrogate-assisted optimization approach was employed:
\begin{itemize}
    \item \textbf{Engine:} HEBO (Heteroscedastic BO)
    \item \textbf{Acquisition:} Expected Improvement (EI)
    \item \textbf{Objective:} Maximize ``\textit{INSERT_TARGET_COL}''
    \item \textbf{Iterations:} \textbf{INSERT_ITERATIONS}
    \item \textbf{Batch Size:} \textbf{INSERT_BATCH_SIZE} (sequential suggestions)
\end{itemize}
\end{methodology}

\subsection{Training Data}

The surrogate oracle was trained on INSERT_N_TRAIN samples from 
``INSERT_DATASET.csv'' with the following characteristics:

\begin{table}[htbp]
\centering
\caption{Training Dataset Summary}
\begin{tabular}{@{}lr@{}}
\toprule
\textbf{Attribute} & \textbf{Value} \\
\midrule
Total Samples & INSERT_N_TRAIN \\
\rowcolor{tablealt} Input Features & INSERT_N_FEATURES \\
Target Variable & INSERT_TARGET_COL \\
\rowcolor{tablealt} Objective & Maximize \\
Initial Best Value & \meansd{INSERT_INIT_BEST}{} \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Oracle Model}

The surrogate was built using:
\begin{itemize}
    \item \textbf{Algorithm:} Gradient Boosted Trees (via LightGBM)
    \item \textbf{Cross-Validation:} INSERT_CV_FOLDS-fold CV
    \item \textbf{Feature Selection:} Top INSERT_N_FEATURES features used
\end{itemize}

% ============================================================================
% RESULTS
% ============================================================================

\chapter{Results}

\section{Convergence Analysis}

\begin{resultsbox}[Primary Finding]
The optimization successfully identified candidates with 
\effectsize{$\Delta$}{INSERT_IMPROVEMENT_PERCENT\%} improvement relative to the 
initial best value in the training dataset.
\end{resultsbox}

\begin{table}[htbp]
\centering
\caption{Convergence Metrics}
\label{tab:convergence}
\begin{tabular}{@{}lrr@{}}
\toprule
\textbf{Metric} & \textbf{Value} & \textbf{Unit} \\
\midrule
Initial Best (from training) & \meansd{INSERT_INIT_BEST}{} & \\
\rowcolor{tablealt} Best Found (via BO) & \meansd{INSERT_BEST_Y}{} & \\
Absolute Improvement & INSERT_IMPROVEMENT & units \\
\rowcolor{tablealt} Percent Improvement & INSERT_IMPROVEMENT_PERCENT & \% \\
Iteration of Best & INSERT_BEST_ITER & iteration \\
\bottomrule
\end{tabular}
\end{table}

\section{Top Candidates}

The optimization identified the following high-value candidates for validation:

\begin{table}[htbp]
\centering
\caption{Top 5 Candidates Suggested by BO}
\label{tab:top_candidates}
\begin{tabular}{@{}lrrr@{}}
\toprule
\textbf{Rank} & \textbf{Iteration} & \textbf{Predicted Value} & \textbf{Parameter Set} \\
\midrule
INSERT_TOP_TABLE_ROWS
\bottomrule
\end{tabular}
\end{table}

\section{Oracle Fidelity}

\begin{resultsbox}[Model Quality]
The surrogate oracle achieved a cross-validation RMSE of 
\textbf{INSERT_RMSE}, indicating INSERT_FIDELITY_ASSESSMENT accuracy on held-out data.
\end{resultsbox}

The oracle model was evaluated via INSERT_CV_FOLDS-fold cross-validation. 
Residuals are presented in Figure~\ref{fig:residuals}.

\begin{figure}[htbp]
\centering
% \includegraphics[width=0.8\textwidth]{../INSERT_RESIDUAL_PLOT.png}
\caption{Oracle Prediction Residuals (CV Hold-Out Data)}
\figurenote{Residuals show the difference between oracle predictions and actual values.}
\label{fig:residuals}
\end{figure}

% ============================================================================
% DISCUSSION
% ============================================================================

\chapter{Discussion}

\section{Summary of Findings}

\begin{keyfindings}
\begin{enumerate}
    \item The BO campaign achieved sustained improvement across iterations.
    \item Top candidates show INSERT_IMPROVEMENT_PERCENT\% improvement over 
          training baseline.
    \item Oracle model fidelity (RMSE = INSERT_RMSE) supports confidence in 
          suggestions.
\end{enumerate}
\end{keyfindings}

\section{Recommendations}

\begin{recommendations}[Next Steps]
\begin{enumerate}
    \item \textbf{Validate Top Candidates:} Experimentally test the top 
          INSERT_TOP_N candidates identified by BO.
    \item \textbf{Retrain Oracle:} Include new validation results to improve 
          fidelity for subsequent rounds.
    \item \textbf{Expand Search:} If budget permits, run additional BO iterations 
          with the updated oracle.
    \item \textbf{Parameter Analysis:} Analyze the top candidates to identify 
          design principles.
\end{enumerate}
\end{recommendations}

\section{Limitations}

\begin{limitations}
\begin{itemize}
    \item Oracle quality depends on training data representativeness.
    \item Surrogate predictions have inherent uncertainty (CV RMSE = INSERT_RMSE).
    \item BO results are simulation-based—validation on real data is essential.
    \item Convergence may plateau if budget is insufficient for full exploration.
\end{itemize}
\end{limitations}

% ============================================================================
% APPENDICES
% ============================================================================

\appendix

\chapter{Supplementary Data}

\appendixsection{Complete Iteration Log}

INSERT_FULL_ITERATION_TABLE

\appendixsection{Technical Details}

\textbf{Compilation:}
\begin{verbatim}
xelatex bo_report.tex
\end{verbatim}

\textbf{Dataset:} \texttt{INSERT_DATASET_PATH}

\textbf{Run ID:} \texttt{INSERT_RUN_ID}

\textbf{Generated:} \today

% ============================================================================
% END DOCUMENT
% ============================================================================

\end{document}
```

---

## Step 4: Generate LaTeX Report Programmatically

Create a Python script to fill in the template:

```python
# generate_latex_report.py
import json
import pandas as pd
from pathlib import Path
from string import Template

def generate_latex_report(run_id: str, output_file: str = None):
    """
    Generate a LaTeX report from BO results.
    
    Args:
        run_id: The run ID (e.g., 'vivid-heron-3397')
        output_file: Output .tex file path (default: runs/<RUN_ID>/report.tex)
    """
    
    run_path = Path(f"runs/{run_id}")
    if not run_path.exists():
        raise FileNotFoundError(f"Run directory not found: {run_path}")
    
    if output_file is None:
        output_file = run_path / "report.tex"
    
    # Load data
    with open(run_path / "state.json") as f:
        state = json.load(f)
    
    with open(run_path / "report.json") as f:
        report = json.load(f)
    
    suggestions = pd.read_json(run_path / "suggestions.jsonl", lines=True)
    observations = pd.read_json(run_path / "observations.jsonl", lines=True)
    
    # Load oracle metadata if available
    oracle_meta = {}
    if (run_path / "oracle_meta.json").exists():
        with open(run_path / "oracle_meta.json") as f:
            oracle_meta = json.load(f)
    
    # Compute key statistics
    n_iterations = report.get("iterations", len(observations))
    best_y = report.get("best_y", observations["y"].max())
    best_iter = report.get("best_y_idx", observations["y"].idxmax())
    improvement = report.get("improvement", best_y - report.get("initial_best", observations["y"].min()))
    improvement_pct = report.get("improvement_percent", 
                                  (improvement / abs(report.get("initial_best", 1e-6))) * 100)
    
    # Build top candidates table
    top_n = 5
    top_suggestions = suggestions.nlargest(top_n, "y_pred")
    
    top_table_rows = ""
    for rank, (idx, row) in enumerate(top_suggestions.iterrows(), 1):
        y_pred = row.get("y_pred", "TBD")
        params = str(row.get("x", {}))[:50] + "..."
        top_table_rows += f"{rank} & {idx} & {y_pred:.4f} & {params} \\\\\n"
        if rank % 2 == 0:
            top_table_rows += "\\rowcolor{tablealt}"
    
    # Read template
    template_file = Path(__file__).parent / "bo_report.tex"  # Adjust path
    with open(template_file) as f:
        template_text = f.read()
    
    # Replace placeholders
    replacements = {
        "INSERT_ITERATIONS": str(n_iterations),
        "INSERT_DIM": str(state.get("n_vars", len(state.get("bounds", [])))),
        "INSERT_BEST_Y": f"{best_y:.4f}",
        "INSERT_IMPROVEMENT": f"{improvement:.4f}",
        "INSERT_IMPROVEMENT_PERCENT": f"{improvement_pct:.1f}",
        "INSERT_RMSE": f"{oracle_meta.get('cv_rmse', 'N/A')}",
        "INSERT_BEST_ITER": str(best_iter),
        "INSERT_TOP_N": str(top_n),
        "INSERT_TARGET_COL": state.get("target_col", "Target"),
        "INSERT_BATCH_SIZE": str(state.get("batch_size", 1)),
        "INSERT_N_TRAIN": str(oracle_meta.get("n_train", "N/A")),
        "INSERT_DATASET": "data.csv",  # From state
        "INSERT_N_FEATURES": str(oracle_meta.get("n_features", "N/A")),
        "INSERT_CV_FOLDS": str(oracle_meta.get("cv_folds", 5)),
        "INSERT_INIT_BEST": f"{report.get('initial_best', 0):.4f}",
        "INSERT_TOP_TABLE_ROWS": top_table_rows,
        "INSERT_FIDELITY_ASSESSMENT": "good" if oracle_meta.get("cv_rmse", 1) < 0.2 else "moderate",
        "INSERT_RUN_ID": run_id,
        "INSERT_FULL_ITERATION_TABLE": _build_iteration_table(observations),
    }
    
    result = template_text
    for key, value in replacements.items():
        result = result.replace(key, str(value))
    
    # Write output
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        f.write(result)
    
    print(f"✓ LaTeX report generated: {output_file}")
    print(f"  Compile with: xelatex {output_file}")
    
    return output_file


def _build_iteration_table(observations: pd.DataFrame) -> str:
    """Build LaTeX table of all iterations."""
    rows = ""
    for idx, row in observations.iterrows():
        y = row.get("y", "N/A")
        rows += f"{idx+1} & {y:.4f} \\\\\n"
        if (idx + 1) % 2 == 0:
            rows += "\\rowcolor{tablealt}"
    
    return f"""
\\begin{{table}}[htbp]
\\centering
\\caption{{All Iterations}}
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


if __name__ == "__main__":
    # Example usage
    import sys
    run_id = sys.argv[1] if len(sys.argv) > 1 else "vivid-heron-3397"
    generate_latex_report(run_id)
```

---

## Step 5: Compile to PDF

Once the LaTeX is generated:

```bash
# From the project root (so the style file is found automatically)
xelatex -output-directory="runs/vivid-heron-3397" "runs/vivid-heron-3397/report.tex"

# If using bibliography (for citations):
xelatex -output-directory="runs/vivid-heron-3397" "runs/vivid-heron-3397/report.tex"
bibtex runs/vivid-heron-3397/report
xelatex -output-directory="runs/vivid-heron-3397" "runs/vivid-heron-3397/report.tex"
xelatex -output-directory="runs/vivid-heron-3397" "runs/vivid-heron-3397/report.tex"

# Or use latexmk (handles all passes automatically)
latexmk -xelatex runs/vivid-heron-3397/report.tex
```

**Output:** `runs/vivid-heron-3397/report.pdf`

---

## Complete Workflow Integration

### Approach A: Integrated into BO Pipeline

Modify your reporting command to generate both JSON and LaTeX:

```bash
# 1. Run BO optimization
uv run python -m bo_workflow.cli init \
  --dataset data/HER_virtual_data.csv \
  --target Target --objective max --seed 42

# 2. Build oracle
uv run python -m bo_workflow.cli build-oracle --run-id <RUN_ID>

# 3. Run proxy BO
uv run python -m bo_workflow.cli run-proxy --run-id <RUN_ID> --iterations 20

# 4. Generate JSON report (as before)
uv run python -m bo_workflow.cli report --run-id <RUN_ID>

# 5. Generate LaTeX report (new step)
python latex/generate_latex_report.py <RUN_ID>

# 6. Compile to PDF
# From the project root, compile using the run directory as the output directory
xelatex -output-directory="runs/<RUN_ID>" "runs/<RUN_ID>/report.tex"
```

### Approach B: Add Skill Integration to CLI

Create a new CLI command `report-latex` that invokes the scientific-writing skill:

```python
# bo_workflow/reporting_cli.py
def register_commands(subparsers):
    parser = subparsers.add_parser(
        "report-latex",
        help="Generate LaTeX scientific report from BO results"
    )
    parser.add_argument(
        "--run-id", required=True, help="Run ID"
    )
    parser.add_argument(
        "--output", help="Output .tex file (default: runs/<RUN_ID>/report.tex)"
    )
    parser.set_defaults(handler=handle)

def handle(args):
    from generate_latex_report import generate_latex_report
    generate_latex_report(args.run_id, args.output)
```

Then use:

```bash
uv run python -m bo_workflow.cli report-latex --run-id vivid-heron-3397
```

---

## Key Templates for Different Report Styles

### 1. **Minimal Report** (2–3 pages)

For quick dissemination:

```latex
\chapter{Results}
\section{Best Candidate}
Value: \textbf{INSERT_BEST_Y} (improvement: INSERT_IMPROVEMENT_PERCENT\%)

\section{Top 3 Recommendations}
\begin{enumerate}
    \item INSERT_TOP_CANDIDATES
\end{enumerate}
```

### 2. **Extended Report** (10–15 pages)

Includes:
- Detailed methodology
- Hyperparameter specifications
- Convergence plots
- Feature importance analysis
- Parameter sensitivity

### 3. **Publication-Ready** (20+ pages)

With:
- Comprehensive literature review
- Theoretical justification
- Extensive appendices
- Multiple figure sets
- Citations and references

---

## Customization Options

### Change Color Scheme

Edit `scientific_report.sty`:

```latex
% Change primary color from navy blue to forest green
\definecolor{primaryblue}{RGB}{34, 139, 34}  % Forest green
```

### Add Your Institution Logo

Modify title page in template:

```latex
\makereporttitlewithimage
    {BO Results}
    {Subtitle}
    {path/to/your/logo.png}     % ← Add logo
    {Your Name}
    {Your Institution}
    {\today}
```

### Include Convergence Plot

Add after `\chapter{Results}`:

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=0.9\textwidth]{../convergence.pdf}
\caption{Convergence of Bayesian Optimization Across Iterations}
\label{fig:convergence}
\end{figure}
```

The `convergence.pdf` is already generated by `report` command!

---

## Troubleshooting

| Issue | Solution |
|-------|-----------|
| Font not found (Helvetica) | Use **XeLaTeX** or **LuaLaTeX**, not pdflatex |
| LaTeX compilation errors | Check `.tex` file for unescaped special chars (`_`, `&`, `%`) |
| Missing `scientific_report.sty` | Ensure skill is properly installed in `.claude/skills/` |
| Placeholder not replaced | Verify key names match exactly (case-sensitive) |
| PDF looks wrong | Ensure all `.png` figure paths are correct |

---

## Example Output

A generated report will contain:

✅ Executive summary with key findings
✅ Methodology section explaining BO approach
✅ Results table with convergence metrics
✅ Top candidates ranked by predicted value
✅ Oracle fidelity assessment
✅ Recommendations for validation
✅ Limitations discussion
✅ Appendices with full iteration logs

---

## Next Steps

1. **Install the scientific-writing skill** (Step 1)
2. **Test with a small run** (`initialize → build-oracle → run-proxy → generate-report`)
3. **Customize the BO template** for your specific domain
4. **Integrate into your CLI** (optional automation)
5. **Share reports** with collaborators as publication-quality PDFs

---

## References

- **Claude Scientific Skills**: https://github.com/K-Dense-AI/claude-scientific-skills
- **Scientific Writing Skill**: https://github.com/K-Dense-AI/claude-scientific-skills/tree/main/scientific-skills/scientific-writing
- **LaTeX Style Package**: `scientific_report.sty`
- **Quick Reference**: `assets/REPORT_FORMATTING_GUIDE.md`

