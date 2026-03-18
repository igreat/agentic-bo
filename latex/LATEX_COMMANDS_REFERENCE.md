# LaTeX Commands Reference for BO Reports

## 🎨 Box Environments (for organizing content)

### Key Findings
Highlight major discoveries and important results:
```latex
\begin{keyfindings}
\begin{enumerate}
    \item Finding 1
    \item Finding 2
\end{enumerate}
\end{keyfindings}
```

### Methodology
Describe methods and procedures:
```latex
\begin{methodology}[Study Design]
Your methodology description here.
\end{methodology}
```

### Results Box
Present key experimental results:
```latex
\begin{resultsbox}[Main Finding]
The intervention showed improvement of 45\%.
\end{resultsbox}
```

### Recommendations
Action items and next steps:
```latex
\begin{recommendations}[Next Steps]
\begin{enumerate}
    \item Validate findings
    \item Extend analysis
\end{enumerate}
\end{recommendations}
```

### Limitations
Caveats and constraints:
```latex
\begin{limitations}
\begin{itemize}
    \item Limited sample size
    \item Potential bias
\end{itemize}
\end{limitations}
```

### Critical Notice
Important warnings or alerts:
```latex
\begin{criticalnotice}
This is a critical finding requiring immediate attention.
\end{criticalnotice}
```

### Executive Summary
Overview section:
```latex
\begin{executivesummary}[Title]
Executive summary text goes here.
\end{executivesummary}
```

---

## 📊 Scientific Notation Commands

### P-Values
```latex
\pvalue{0.023}        % Outputs: p = 0.023
\psig{< 0.001}        % Outputs: p < 0.001 (bold, significant)
```

### Confidence Intervals
```latex
\CI{0.45}{0.72}       % Outputs: 95% CI [0.45, 0.72]
```

### Effect Sizes
```latex
\effectsize{d}{0.75}        % Outputs: d = 0.75
\effectsize{r}{0.42}        % Outputs: r = 0.42
\effectsize{F(2,97)}{12.45} % Outputs: F(2, 97) = 12.45
```

### Mean ± SD
```latex
\meansd{42.5}{8.3}    % Outputs: 42.5 ± 8.3
```

### Sample Size
```latex
\samplesize{250}      % Outputs: n = 250
```

### Significance Indicators (for tables)
```latex
Result\sigone         % * for p < 0.05
Result\sigtwo         % ** for p < 0.01
Result\sigthree       % *** for p < 0.001
Result\signs          % ns for not significant
```

### Quality Ratings
```latex
\qualityhigh          % HIGH (green)
\qualitymedium        % MEDIUM (orange)
\qualitylow           % LOW (red)
```

### Evidence Strength
```latex
\evidencestrong       % Strong (dark green)
\evidencemoderate     % Moderate (orange)
\evidenceweak         % Weak (red)
```

---

## 🎨 Colors Available

### Primary Colors
```latex
\textcolor{primaryblue}{Navy blue text}        % Headers
\textcolor{secondaryblue}{Medium blue text}    % Subsections
```

### Semantic Colors
```latex
\textcolor{sciencegreen}{Green text}           % Positive findings
\textcolor{cautionorange}{Orange text}         % Caution/limitations
\textcolor{criticalred}{Red text}              % Critical/negative
\textcolor{darkgreen}{Dark green text}         % Evidence strength
```

---

## 📋 Table Formatting

### Standard Table
```latex
\begin{table}[htbp]
\centering
\caption{Descriptive Statistics}
\label{tab:descriptives}
\begin{tabular}{@{}lccc@{}}
\toprule
\textbf{Variable} & \textbf{Group A} & \textbf{Group B} & \textbf{p} \\
\midrule
Age (years) & \meansd{42.5}{8.3} & \meansd{43.1}{7.9} & .58 \\
\rowcolor{tablealt} Score 1 & \meansd{15.2}{3.4} & \meansd{18.7}{4.1} & <.001\sigthree \\
Score 2 & \meansd{22.8}{5.1} & \meansd{23.4}{4.8} & .42 \\
\bottomrule
\end{tabular}
\end{table}
```

### Table Notes
```latex
% Add below table
\siglegend  % Outputs: *p < 0.05; **p < 0.01; ***p < 0.001; ns not significant
```

### Professional Table Commands
```latex
\tableheader{Header Text}        % Styled table header
\tablerowcolor                   % Alternate row color
```

---

## 🖼️ Figure Formatting

### Basic Figure
```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=0.8\textwidth]{../figures/convergence.png}
\caption{Convergence of Bayesian Optimization}
\label{fig:convergence}
\end{figure}
```

### Figure with Source
```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=0.9\textwidth]{../figures/results.png}
\caption{Primary Outcome Scores}
\figuresource{Smith et al. (2023)}
\label{fig:results}
\end{figure}
```

### Figure with Note
```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=0.8\textwidth]{../figures/model.png}
\caption{Conceptual Model}
\figurenote{Solid arrows indicate direct effects; dashed arrows indicate moderated effects.}
\label{fig:model}
\end{figure}
```

---

## 📄 Document Structure

### Appendix Sections
```latex
\appendix

\chapter{Supplementary Materials}

\appendixsection{Additional Tables}
Table content here...

\appendixsection{Instruments}
Instrument details here...
```

### Custom Title Page
```latex
\makereporttitle
    {Report Title}              % Title
    {Subtitle}                  % Subtitle
    {Author Name, PhD}          % Author(s)
    {Institution Name}          % Institution
    {January 2025}              % Date
```

### Title Page with Image
```latex
\makereporttitlewithimage
    {Report Title}
    {Subtitle}
    {path/to/logo.png}          % Logo/image
    {Author Name}
    {Institution}
    {Date}
```

---

## 🔗 Cross-References

### Reference Figures
```latex
As shown in Figure~\ref{fig:convergence}, the results...
```

### Reference Tables
```latex
Table~\ref{tab:descriptives} presents the statistics.
```

### Reference Sections
```latex
See Chapter~\ref{ch:results} for details.
```

---

## 🎯 Common BO Report Patterns

### Convergence Pattern
```latex
\begin{resultsbox}[Convergence Analysis]
The algorithm achieved steady improvement across iterations, 
reaching \textbf{\insertimprovement_percent\%} improvement 
(from \meansd{\initial_best}{} to \meansd{\best_y}{}) by iteration 
INSERT_BEST_ITER.
\end{resultsbox}
```

### Oracle Quality Pattern
```latex
\begin{resultsbox}[Model Fidelity]
The surrogate oracle achieved a cross-validation RMSE of 
INSERT_RMSE, indicating INSERT_FIDELITY accuracy on held-out data.
\end{resultsbox}
```

### Top Candidates Pattern
```latex
The optimization identified INSERT_TOP_N high-value candidates 
(Table~\ref{tab:candidates}) for experimental validation.

\begin{table}[htbp]
\centering
\caption{Top Candidates by Predicted Value}
\label{tab:candidates}
\begin{tabular}{@{}lrr@{}}
\toprule
\textbf{Rank} & \textbf{Iteration} & \textbf{Predicted Value} \\
\midrule
INSERT_TOP_CANDIDATES
\bottomrule
\end{tabular}
\end{table}
```

---

## 🔌 Template Placeholders for BO Reports

| Placeholder | Example | Description |
|------------|---------|-------------|
| `INSERT_RUN_ID` | `vivid-heron-3397` | Unique run identifier |
| `INSERT_ITERATIONS` | `20` | Total iterations run |
| `INSERT_DIM` | `10` | Number of parameters |
| `INSERT_BEST_Y` | `27.704` | Best observed value |
| `INSERT_BEST_ITER` | `15` | Iteration of best value |
| `INSERT_IMPROVEMENT` | `15.234` | Absolute improvement |
| `INSERT_IMPROVEMENT_PERCENT` | `98.2` | Percent improvement |
| `INSERT_RMSE` | `1.787` | Oracle CV RMSE |
| `INSERT_FIDELITY` | `moderate` | Oracle fidelity assessment |
| `INSERT_TARGET_COL` | `Target` | Objective column name |
| `INSERT_OBJECTIVE` | `MAX` | Maximize or minimize |
| `INSERT_ENGINE` | `HEBO` | BO engine used |
| `INSERT_BATCH_SIZE` | `1` | Batch size |
| `INSERT_TOP_CANDIDATES` | Table rows | Top N candidates table |
| `INSERT_ITERATION_LOG` | Table rows | Full iteration history |

---

## ✨ Style Tips

### Emphasis
```latex
\textbf{Bold text}
\textit{Italic text}
\texttt{Code/monospace}
```

### Highlights
```latex
\highlight{Important finding}
```

### Lists
```latex
\begin{itemize}
    \item Automatically blue bullets
    \item Second item
\end{itemize}

\begin{enumerate}
    \item Automatically blue numbers
    \item Second item
\end{enumerate}
```

### Pull Quotes
```latex
\begin{quote}
"This is an important statement that deserves emphasis."
\end{quote}
```

---

## 🚀 Compilation

### Basic Compilation
```bash
xelatex report.tex
```

### With Bibliography
```bash
xelatex report.tex
bibtex report
xelatex report.tex
xelatex report.tex
```

### Automatic (Recommended)
```bash
latexmk -xelatex report.tex
```

---

## 📚 Learn More

- **Full Guide:** LATEX_REPORT_GUIDE.md
- **Quick Start:** LATEX_QUICK_START.md
- **Scientific Skills:** https://github.com/K-Dense-AI/claude-scientific-skills
- **LaTeX Reference:** https://en.wikibooks.org/wiki/LaTeX
