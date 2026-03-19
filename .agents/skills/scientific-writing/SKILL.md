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
2. **LaTeX technical report:** `bo_runs/<run_id>/report.tex`

Choose the output format that best matches the request:
- for "paper draft" or "write a paper" → Markdown (`research_runs/<research_id>/paper.md`)
- for "report" or "tex" or "compile" → LaTeX report (`bo_runs/<run_id>/report.tex`)

---

## Inputs

### Research workflow inputs
- `research_runs/<research_id>/research_state.json`
- `research_runs/<research_id>/research_plan.md`

### BO run inputs
- `bo_runs/<run_id>/report.json`
- `bo_runs/<run_id>/state.json`
- `bo_runs/<run_id>/observations.jsonl` (when trajectory details needed)
- `bo_runs/<run_id>/convergence.pdf` (optional; reference if present)

### Optional inputs
- literature sources gathered earlier
- repo docs describing run structure or oracle setup

---

## General Writing Rule

Keep the writing tightly grounded in the actual artifacts.

**Do:**
- extract facts directly from JSON/JSONL artifacts
- use `report.json` as source of truth for best value and candidates
- use `observations.jsonl` for exact trajectory details only when needed
- clearly distinguish proxy-oracle/surrogate outcomes from real measurements

**Do not:**
- invent unsupported numeric details
- add generic domain boilerplate not in artifacts
- claim laboratory validation if only surrogate predictions exist
- create a separate software layer—just write the document

---

# Output Mode 1: Markdown Paper Draft

**Output:** `research_runs/<research_id>/paper.md`

**Structure:** Title → Abstract → Introduction → Methods → Results → Discussion → Conclusion

**Length:** 1500–2500 words, excluding title and abstract.

## Required Structure

Write an IMRAD-style paper with these sections:
- Title
- Abstract
- Introduction
- Methods
- Results
- Discussion
- Conclusion

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
- Describe oracle provenance only from what the artifacts explicitly say. If `report.json` exposes oracle metadata but not training timing, describe it as backend-reported or artifact-reported oracle metadata rather than claiming it was fitted post hoc.
- **If proxy-backed evaluation:** report the proxy oracle CV RMSE and note that outcomes reflect surrogate predictions, not direct measurements.
- Reference the convergence plot at `bo_runs/<run_id>/convergence.pdf` if it exists.

### Results
- Report the best value found and the corresponding candidate (composition, conditions, etc.).
- Mention convergence behavior — did the search plateau, was it still improving at the end?
- Use `report.json` as the source of truth for best-value summary statistics.
- For human-facing numbering, prefer `report.json["best_observation_number"]` over the zero-based internal `best_iteration` field.
- Prefer `report.json["trajectory"]` for phase summaries, random-phase ranges, and best-observation numbering when available.
- If `report.json["trajectory"]` is present, use it directly rather than recomputing phase summaries from memory or ad hoc output.
- If you include exact random-phase ranges, phase breakpoints, or iteration-specific claims beyond the reported best and they are not already present in `report.json["trajectory"]`, verify them from `observations.jsonl`.
- If `observations.jsonl` is not provided or not read, avoid precise trajectory numbers and keep the narrative qualitative.
- **If proxy-backed evaluation:** do not present outcomes as measured values. Use phrasing like "the proxy oracle predicted…" or "the surrogate model identified…".

### Discussion
- Interpret the result chemically or materially — why might this composition or condition work?
- Compare against the literature baselines from `literature_findings.baselines` if available.
- State important caveats: oracle error when applicable, dataset coverage when applicable, and any gap between simulated/externally observed evidence and real experiments.

**If literature review was skipped:**
- keep the Discussion grounded in the BO trajectory, candidate composition, oracle uncertainty, and simulation limitations
- do not introduce external mechanism claims
- any hypothesis must be labeled as tentative and artifact-derived, not literature-backed

### Conclusion
- One paragraph summarizing what was found.
- One sentence on the next practical step (e.g., experimental validation, broader search).

## Markdown Paper Guardrails

- Clearly label proxy-oracle results as simulated throughout — in the abstract, methods, and results.
- Do not present simulated BO outcomes as real laboratory measurements.
- If evidence is weak (high oracle RMSE, few iterations, narrow dataset), say so directly.
- Keep references lightweight in v1; plain links or compact citations are enough.
- Keep the writing tied to the actual artifacts rather than generic BO boilerplate.
- Do not invent fine-grained numeric trajectory details from memory. If exact ranges or iteration-level numbers are not explicitly supported by `report.json` or `observations.jsonl`, leave them out.
- Do not infer oracle training timing or methodology unless it is explicitly stated in the artifacts.
- If `report.json["oracle"]["source"]` says the metadata came from the evaluation backend, describe it as backend-reported oracle metadata rather than implying a fresh model fit after the run.

---

# Output Mode 2: LaTeX Report

**Output:** `bo_runs/<run_id>/report.tex`

Generate a **self-contained, compile-ready** XeLaTeX document using the following guidelines.

## Style Requirements

**Preamble:**
```latex
\documentclass{article}
\usepackage{geometry}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{hyperref}
\geometry{margin=1in}
\hypersetup{colorlinks=true, urlcolor=blue}
```

**Length:** 1500–2500 words, excluding title and abstract.

**Typography:**
- Use Title Case for sections
- Escape special LaTeX characters: `_` → `\_`, `%` → `\%`, `&` → `\&`, `#` → `\#`, `{}` → `\{\}`
- Use `\texttt{...}` for run IDs, filenames, and technical identifiers

**Tables:**
- Use `booktabs` (clean lines, no vertical rules)
- Prefer 3–4 columns; wide tables should wrap text or use smaller font
- Include captions and labels

**Figures:**
- Reference `convergence.pdf` if it exists with `\IfFileExists{convergence.pdf}{\includegraphics[width=0.8\textwidth]{convergence.pdf}}{}`
- Add captions and labels
- File may not always exist; use conditional inclusion

## Required Sections

1. **Title & Author** (or remove if not needed)
2. **Abstract** (150–200 words) — motivation, method, best result, key insight; explicitly state if proxy-backed
3. **Introduction** — problem statement, why optimization, research context
4. **Methodology** — search space definition, BO engine used, how observations were obtained, oracle/proxy details
5. **Results** — best value found, best candidate, convergence summary; use `report.json` as source of truth
6. **Discussion** — interpretation, caveats (oracle error, iteration budget, simulation limitations)
7. **Conclusion** — summary, one practical next step

## Compile Command

```bash
cd bo_runs/<run_id>/
xelatex report.tex
```

Results in `report.pdf` in the run directory.

---



## Artifact Priorities

When writing, use this priority order:

1. **`report.json`** for best value, candidates, convergence stats
2. **`state.json`** for search space, engine, objective, parameters
3. **`observations.jsonl`** for trajectory details only if needed for precision
4. **`convergence.pdf`** if present (reference it; do not regenerate)
5. **`research_state.json` / `research_plan.md`** for research workflow context

---

## Universal Guardrails

- ✓ Label proxy/surrogate results as **simulated** throughout all formats
- ✓ Extract facts directly from JSON/JSONL artifacts
- ✓ Use `report.json` as source of truth for best value and iteration count
- ✓ Include oracle/surrogate RMSE if available
- ✓ State constraints and search-space limits explicitly
- ✗ Do not present surrogate predictions as laboratory measurements
- ✗ Do not invent numeric details unsupported by artifacts
- ✗ Do not add generic BO boilerplate
  - motivation
  - what was done
  - best result found
  - key takeaway
- if the BO artifacts indicate proxy-backed or oracle-backed evaluation, say so explicitly
- if writing LaTeX, keep it clean and XeLaTeX-safe

---

## Agent Workflow

**For LaTeX report:**
1. Read `bo_runs/<run_id>/report.json`, `state.json`, `convergence.pdf` (if present)
2. Write `report.tex` with the structure and style guidelines above
3. If user asks, compile: `cd bo_runs/<run_id>/ && xelatex report.tex && xelatex report.tex`

**For Markdown paper:**
1. Use IMRAD structure
2. Write to `research_runs/<research_id>/paper.md`

Do not build a separate reporting backend. Write the document directly.
