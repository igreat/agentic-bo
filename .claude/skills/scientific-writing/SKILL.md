---
name: scientific-writing
description: Draft a concise IMRAD-style paper for a BO-backed chemistry or materials study from research-agent artifacts, or generate LaTeX technical reports.
---

# Scientific Writing

Generate professional `.tex` files from BO and research artifacts, or produce Markdown paper drafts.

**Key principle:** Read artifacts directly, write the document, do not build auxiliary report systems.

## Supported Outputs

This skill can produce two independent output formats:

1. **Markdown paper draft:** `research_runs/<research_id>/paper.md`
2. **LaTeX paper/report draft:** `research_runs/<research_id>/paper.tex` or `bo_runs/<run_id>/report.tex`

Choose the output format that best matches the request:
- for "paper draft" or "write a paper" with no format preference → Markdown (`research_runs/<research_id>/paper.md`)
- for requests that explicitly ask for `.tex`, LaTeX, compiled figures, or a figure-heavy paper artifact → LaTeX under `research_runs/<research_id>/paper.tex`
- for BO-only "report" or "tex" requests tied to a run directory rather than a research draft → LaTeX report (`bo_runs/<run_id>/report.tex`)

---

## Inputs

### Research workflow inputs
- `research_runs/<research_id>/research_state.json`
- `research_runs/<research_id>/research_plan.md`

### BO run inputs
- `bo_runs/<run_id>/report.json`
- `bo_runs/<run_id>/state.json`
- `bo_runs/<run_id>/observations.jsonl` (when trajectory details needed)
- `bo_runs/<run_id>/suggestions.jsonl` (optional; useful for phase or candidate plots)
- existing run-local figure assets such as `bo_runs/<run_id>/convergence.pdf` (optional; reuse only if still fit for purpose)

### Optional inputs
- literature sources gathered earlier
- repo docs describing run structure or oracle setup
- optional supporting files referenced in `research_state.json.run_artifacts.extra_paths` when they materially help explain the workflow or result

---

## General Writing Rule

Keep the writing tightly grounded in the actual artifacts.

**Do:**
- extract facts directly from JSON/JSONL artifacts
- use `report.json` as source of truth for best value and candidates
- use `observations.jsonl` for exact trajectory details only when needed
- treat figures as first-class outputs when they materially improve the paper or report
- clearly distinguish proxy-oracle/surrogate outcomes from real measurements

**Do not:**
- invent unsupported numeric details
- add generic domain boilerplate not in artifacts
- claim laboratory validation if only surrogate predictions exist
- create a separate software layer—just write the document

---

## Figure Generation Policy

Figures are part of the writing workflow, not a separate built-in subsystem.

**Default expectation:** if the artifacts support a useful visual story and a figure will clarify the point better than prose or a table, generate and embed it. If not, skip the figure and keep the draft lean.

**How to do this:**
- write a small, run-local plotting script for the specific paper or report
- place figure scripts beside the output, for example:
  - Markdown paper: `research_runs/<research_id>/figures/scripts/make_figures.py`
  - LaTeX report: `bo_runs/<run_id>/figures/scripts/make_figures.py`
- write figure files into a sibling `figures/` directory as `.pdf` and/or `.png`
- make the script read the actual run artifacts directly (`report.json`, `state.json`, `observations.jsonl`, calibration files, candidate tables, etc.)
- embed the generated figures in the paper or report with captions that explain what the reader should notice

**Figure selection heuristic:**
- use figures for patterns, uncertainty, calibration quality, trajectory shape, or shortlist comparison
- use tables when the reader mainly needs exact values, categorical side-by-side comparisons, or compact scorecards
- in short papers and reports, prefer 0--2 high-value figures by default; only go to 3 if each figure does distinct explanatory work
- for workflow or system papers, one clear architecture/workflow figure may be more valuable than several generic BO curves

**Prefer ad hoc plotting over built-in helpers:**
- do not rely on repo-level plotting helpers such as `bo_workflow/plotting.py`
- do not assume a single standard figure is sufficient for every study
- reuse an existing figure like `convergence.pdf` only if it already matches the story you need to tell

**Common useful figures for optimisation papers:**
- best-so-far convergence over BO iterations
- observed-value trajectory, especially when random and BO phases should be visually separated
- regret or gap-to-baseline curves, but only when the baseline is explicit and defensible
- evaluator calibration plots, parity plots, or residual summaries when calibration is part of the method
- candidate comparison plots for the final shortlist when multiple candidates are within the uncertainty band

**Guardrails for figures:**
- if a regret baseline is not explicit in the artifacts, do not invent one; use best-so-far or observed-value plots instead
- if the run is proxy-backed, label figure captions as simulated or surrogate-backed where relevant
- do not add a convergence plot just because BO was used; include it when the trajectory itself is part of the evidence
- figures should support the main claim, not compensate for weak prose; if a figure adds no new information beyond a table and one sentence, skip it
- if no figure would add real value, skip it and keep the prose tight rather than adding decorative plots

---

# Output Mode 1: Markdown Paper Draft

**Output:** `research_runs/<research_id>/paper.md`

**Structure:** Title → Abstract → Introduction → Methods → Results → Discussion → Conclusion

**Length:** target `1500–2000` words excluding title and abstract for a single-run draft; allow `1800–2500` for a broader paper or report.

## Required Structure

Write an IMRAD-style paper with these sections:
- Title
- Abstract
- Introduction
- Methods
- Results
- Discussion
- Conclusion

**Soft section budgets (compression targets):**
- Abstract: `150–200`
- Introduction: `150–250`
- Methods: `250–450`
- Results: `300–450`
- Discussion: `150–300`
- Conclusion: `50–120`
- If one section grows, another should usually shrink.
- Discussion should usually be the shortest major section unless the user explicitly asks for deeper interpretation.

## Section Guidance

### Abstract
- 150–200 words.
- One sentence each on: motivation, what was done, best result found, key takeaway.
- If the BO artifacts indicate proxy-backed or oracle-backed evaluation, say so explicitly in the abstract.

### Introduction
- State the research problem and why it matters.
- Summarize only the relevant literature context — 2–4 sentences, tied to the baselines in `literature_findings`.
- Motivate why optimization is needed rather than exhaustive screening.

**If literature review was skipped:**
- keep the Introduction minimal and artifact-scoped
- do not add generic domain background, historical context, or benchmark claims not present in `research_plan.md`, `research_state.json`, repo docs, or the run artifacts
- state plainly that no literature baseline comparison was performed for this run

### Methods
- Describe the search space actually used (design variables, bounds, any simplex constraints).
- Describe the BO engine and relevant configuration (surrogate model, acquisition function, batch size).
- State how observations were obtained based on the available artifacts.
- If `research_state.json.evaluator_assessment` is present, state the evaluator evidence class plainly in Methods.
- If `research_state.json.calibration_summary` is present, state the calibration scope plainly in Methods:
  - how many reference points were used
  - whether multiple facets/sites were covered
  - what metric or uncertainty band the calibration supports
- Describe oracle provenance only from what the artifacts explicitly say. If `report.json` exposes oracle metadata but not training timing, describe it as backend-reported or artifact-reported oracle metadata rather than claiming it was fitted post hoc.
- If run-local helper scripts or dependency installs materially affected the setup, state that briefly and describe only what was actually supported by the artifacts.
- If figures were generated from run artifacts, state briefly what files they were derived from.
- **If proxy-backed evaluation:** report the proxy oracle CV RMSE and note that outcomes reflect surrogate predictions, not direct measurements.
- If a figure materially helps explain the setup or trajectory, generate it and reference it in Methods or Results rather than only mentioning an existing `convergence.pdf`.

### Results
- Report the best value found and the corresponding candidate (composition, conditions, etc.).
- If multiple top candidates fall within the evaluator uncertainty band, present them as a shortlist of essentially indistinguishable screening candidates rather than a single settled winner.
- Mention convergence behavior — did the search plateau, was it still improving at the end?
- When the artifacts support it, include 0--3 figures, using the smallest number that materially improves the paper.
- For BO-focused studies, a convergence figure is often useful, but omit it if the trajectory is not central to the claim or if a table tells the story more clearly.
- Use `report.json` as the source of truth for best-value summary statistics.
- For human-facing numbering, prefer `report.json["best_observation_number"]` over the zero-based internal `best_iteration` field.
- Prefer `report.json["trajectory"]` for phase summaries, random-phase ranges, and best-observation numbering when available.
- If `report.json["trajectory"]` is present, use it directly rather than recomputing phase summaries from memory or ad hoc output.
- If you include exact random-phase ranges, phase breakpoints, or iteration-specific claims beyond the reported best and they are not already present in `report.json["trajectory"]`, verify them from `observations.jsonl`.
- If `observations.jsonl` is not provided or not read, avoid precise trajectory numbers and keep the narrative qualitative.
- **If proxy-backed evaluation:** do not present outcomes as measured values. Use phrasing like "the proxy oracle predicted…" or "the surrogate model identified…".

### Discussion
- Interpret the result chemically or materially only to the degree supported by the artifacts and evaluator strength.
- Compare against the literature baselines from `literature_findings.baselines` if available.
- State important caveats: oracle error when applicable, dataset coverage when applicable, and any gap between simulated/externally observed evidence and real experiments.
- If optional supporting artifacts in `run_artifacts.extra_paths` materially informed the workflow or the caveats, Claude may cite them, but they are not required inputs.
- If `research_state.json.evaluator_assessment` is present, match the strength of the claims to `claim_posture`.
- If the evaluator is `physics_inspired_heuristic` or another weak screening evaluator, cap unsupported mechanistic interpretation at `2–3 sentences` and frame it as an artifact-derived hypothesis or screening-level interpretation.
- Do not let the Discussion turn into a second Introduction or a long speculative chemistry essay.
- If the modeled structures are metastable simplifications or convenience phases, say that directly and do not frame the ranking as benchmark-quality truth.

**If literature review was skipped:**
- keep the Discussion grounded in the BO trajectory, candidate composition, oracle uncertainty, and simulation limitations
- do not introduce external mechanism claims
- any hypothesis must be labeled as tentative and artifact-derived, not literature-backed

### Conclusion
- One paragraph summarizing what was found.
- One sentence on the next practical step (e.g., experimental validation, broader search).

## Markdown Paper Guardrails

- Embed generated figures with Markdown image links where relevant.
- Keep figure captions factual and artifact-grounded.
- Clearly label proxy-oracle results as simulated throughout — in the abstract, methods, and results.
- Do not present simulated BO outcomes as real laboratory measurements.
- If `evaluator_assessment.evidence_class` is `physics_inspired_heuristic`, explicitly label the evaluator as a heuristic or simplified model and keep novelty/performance claims cautious.
- If `evaluator_assessment.evidence_class` is `physics_inspired_heuristic` and the top-candidate margin is inside the stated uncertainty band, do not use "surpasses benchmark" language. Use shortlist or DFT-validation-candidate language instead.
- If `evaluator_assessment.evidence_class` is `retrospective_lookup`, explicitly label the evaluator as retrospective/tabulated and do not frame the best candidate as a newly discovered material outside that tabulated space.
- If `evaluator_assessment.evidence_class` is `validated`, stronger claims are acceptable, but still keep normal uncertainty and scope caveats.
- If evidence is weak (high oracle RMSE, few iterations, narrow dataset), say so directly.
- For live structural screening runs backed by a heuristic or calibrated approximate evaluator, a safe framing is "screening-level hypothesis" or "DFT-validation candidate"; an unsafe framing is a strong claim of validated superiority over benchmark materials.
- Keep references lightweight in v1; plain links or compact citations are enough.
- Keep the writing tied to the actual artifacts rather than generic BO boilerplate.
- Do not assume any rigid supporting artifact set beyond `research_state.json`, `research_plan.md`, the final paper artifact (`paper.md` or `paper.tex`), and the BO artifacts actually provided.
- Do not invent fine-grained numeric trajectory details from memory. If exact ranges or iteration-level numbers are not explicitly supported by `report.json` or `observations.jsonl`, leave them out.
- Do not infer oracle training timing or methodology unless it is explicitly stated in the artifacts.
- If `report.json["oracle"]["source"]` says the metadata came from the evaluation backend, describe it as backend-reported oracle metadata rather than implying a fresh model fit after the run.

---

# Output Mode 2: LaTeX Paper or Report

**Output:** `research_runs/<research_id>/paper.tex` for a research paper draft, or `bo_runs/<run_id>/report.tex` for a BO-run technical report.

Generate a **self-contained, compile-ready** XeLaTeX document using the following guidelines.

## Style Requirements

**Preamble (XeLaTeX):**
```latex
\documentclass[11pt, a4paper]{article}
\usepackage[margin=1.0in]{geometry}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{caption}
\usepackage{booktabs}
\hypersetup{colorlinks=true, linkcolor=blue, urlcolor=blue}
```

**CRITICAL TABLE RULE:** Use `tabular` with fixed-width columns, **never** `tabularx`. The `tabularx` package requires at least one `X` (auto-width) column; fixed-width layouts must use plain `tabular`.

**Length:** target `1500–2000` words excluding title and abstract for a single-run draft; allow `1800–2500` for a broader paper or report.

**Typography:**
- Use Title Case for sections
- Escape special LaTeX characters: `_` → `\_`, `%` → `\%`, `&` → `\&`, `#` → `\#`, `{}` → `\{\}`
- Use `\texttt{...}` for run IDs, filenames, and technical identifiers

**Tables:**
- Use `booktabs` (clean lines, no vertical rules)
- Prefer 3–4 columns; wide tables should wrap text or use smaller font
- Include captions and labels
- **Use `tabular` for all fixed-width layouts** (column specs like `lccr`, `lll`, etc.)
- **Never use `tabularx` for fixed-width columns** — it requires at least one `X` column or will error
- Example (correct):
  ```latex
  \begin{tabular}{lccc}
    \toprule
    \textbf{Name} & \textbf{A} & \textbf{B} & \textbf{C} \\
    \midrule
    Row 1 & 1.2 & 3.4 & 5.6 \\
    Row 2 & 2.1 & 4.3 & 6.5 \\
    \bottomrule
  \end{tabular}
  ```

**Figures:**
- Generate fit-for-purpose figure files under the same artifact root as the LaTeX draft:
  - `research_runs/<research_id>/figures/` for `paper.tex`
  - `bo_runs/<run_id>/figures/` for `report.tex`
- Create one-off plotting scripts under the corresponding `figures/scripts/` directory.
- Prefer vector outputs (`.pdf`) for LaTeX figures.
- Add captions and labels.
- Prefer one or two high-value figures over a grab bag of generic plots.
- Reuse `convergence.pdf` only if it already matches the story being told; otherwise generate a new figure from the raw artifacts.

## Required Sections

1. **Title & Author** (or remove if not needed)
2. **Abstract** (150–200 words) — motivation, method, best result, key insight; explicitly state if proxy-backed
3. **Introduction** — problem statement, why optimization, research context
4. **Methodology** — search space definition, BO engine used, how observations were obtained, oracle/proxy details
5. **Results** — best value found, best candidate, convergence summary; use `report.json` as source of truth
6. **Discussion** — interpretation, caveats (oracle error, iteration budget, simulation limitations)
7. **Conclusion** — summary, one practical next step

**Soft section budgets (compression targets):**
- Abstract: `150–200`
- Introduction: `150–250`
- Methods: `250–450`
- Results: `300–450`
- Discussion: `150–300`
- Conclusion: `50–120`
- If one section grows, another should usually shrink.
- Discussion should usually be the shortest major section unless the user explicitly asks for deeper interpretation.

## LaTeX Section Guidance

### Abstract
- 150–200 words.
- One sentence each on: motivation, what was done, best result found, key takeaway.
- If the BO artifacts indicate proxy-backed or oracle-backed evaluation, say so explicitly.

### Introduction
- State the research problem and why it matters.
- Summarize only the relevant literature context — 2–4 sentences tied to the baselines in `literature_findings`.
- Motivate why optimization is needed rather than exhaustive screening.

**If literature review was skipped:**
- keep the Introduction minimal and artifact-scoped
- do not add generic domain history or broad textbook background
- state plainly that no literature baseline comparison was performed for this run

### Methodology
- Describe only the actual search space, evaluator, BO configuration, and observation path used in the run.
- If `research_state.json.evaluator_assessment` is present, state the evaluator evidence class plainly.
- If `research_state.json.calibration_summary` is present, describe the calibration scope briefly and factually.
- Mention generated figures only briefly and only in terms of the artifact files they were derived from.
- If proxy-backed evaluation was used, report the proxy oracle CV RMSE and state that the outcomes are surrogate-backed rather than direct measurements.

### Results
- Use `report.json` as the source of truth for best value, best candidate, and convergence summary.
- If multiple top candidates fall within the evaluator uncertainty band, present them as a shortlist rather than a single settled winner.
- Avoid exact phase ranges or iteration-specific numbers unless they are supported by `report.json["trajectory"]` or verified from `observations.jsonl`.
- Include figures only when they materially improve the argument.

### Discussion
- Tie claims to evaluator evidence class, uncertainty, and `claim_posture`.
- Do not upgrade screening evidence into benchmark truth.
- If the evaluator is `physics_inspired_heuristic` or another weak screening evaluator, cap unsupported mechanistic interpretation at `2–3 sentences` and frame it as an artifact-derived hypothesis or screening-level interpretation.
- If literature review was skipped, do not introduce external mechanism claims; keep the discussion artifact-grounded.
- If the modeled structures are metastable simplifications or convenience phases, say that directly.

### Conclusion
- One short paragraph summarizing what was found.
- One sentence on the next practical step.

## Compile Command

```bash
cd <output_dir>/
xelatex <paper.tex|report.tex>
```

Results in `paper.pdf` or `report.pdf` alongside the `.tex` file when compiled.

---

## Common LaTeX Pitfalls & Fixes

**Error: "! Package array Error: Illegal pream-token"**
- **Cause:** Using `tabularx` with only fixed-width columns (no `X` column)
- **Fix:** Replace `\begin{tabularx}{\textwidth}{lccr}` with `\begin{tabular}{lccr}`

**Error: "! Interruption" at `lastpage.sty`**
- **Cause:** MiKTeX `lastpage` package missing or corrupted
- **Fix:** Remove `\usepackage{lastpage}` from preamble; use simple `\cfoot{Page \thepage}` instead

**Error: Undefined control sequence `\thepage`**
- **Cause:** Missing `fancyhdr` package
- **Fix:** Ensure `\usepackage{fancyhdr}` and `\pagestyle{fancy}` are in preamble

**Error: Special characters like `_` in text break compilation**
- **Cause:** LaTeX interprets `_` as subscript; same for `%`, `&`, `#`
- **Fix:** Escape in text with backslash: `\_`, `\%`, `\&`, `\#`; use `\texttt{...}` for code/IDs

**Best practice:** Test XeLaTeX compilation early with a minimal preamble:
```bash
cd bo_runs/<run_id>/
xelatex -interaction=nonstopmode report.tex
```

---

## Artifact Priorities

When writing, use this priority order:

1. **`report.json`** for best value, candidates, convergence stats
2. **`state.json`** for search space, engine, objective, parameters
3. **`observations.jsonl`** for trajectory details only if needed for precision
4. **`research_state.json` / `research_plan.md`** for research workflow context
5. existing figure assets such as **`convergence.pdf`** only as optional reusable inputs

---

## Universal Guardrails

- ✓ Label proxy/surrogate results as **simulated** throughout all formats
- ✓ Extract facts directly from JSON/JSONL artifacts
- ✓ Use `report.json` as source of truth for best value and iteration count
- ✓ Include oracle/surrogate RMSE if available
- ✓ State constraints and search-space limits explicitly
- ✓ Keep LaTeX output clean and XeLaTeX-safe
- ✗ Do not present surrogate predictions as laboratory measurements
- ✗ Do not invent numeric details unsupported by artifacts
- ✗ Do not pad the document with generic motivation/method/result/takeaway boilerplate that is not tied to the artifacts

---

## Agent Workflow

**For LaTeX paper/report output:**
1. Decide whether the LaTeX artifact belongs under `research_runs/<research_id>/paper.tex` or `bo_runs/<run_id>/report.tex`
2. Read the relevant research and BO artifacts, including `observations.jsonl` when trajectory figures are needed
3. Decide which figures, if any, would materially help the document
4. Write a small ad hoc plotting script beside the output and generate the figure files
5. Write the `.tex` document with the structure and style guidelines above, embedding the generated figures
6. If user asks, compile the `.tex` file twice with XeLaTeX in its output directory

**For Markdown paper:**
1. Read `research_runs/<research_id>/research_state.json`, `research_plan.md`, and the relevant BO artifacts
2. Decide which figures, if any, would materially help the paper
3. Write a small ad hoc plotting script under `research_runs/<research_id>/figures/scripts/` and generate the figure files
4. Use IMRAD structure and embed the generated figures
5. Write to `research_runs/<research_id>/paper.md`

Do not build a separate reporting backend. Small one-off figure scripts are encouraged; write the document directly.
