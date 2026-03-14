---
name: scientific-writing
description: Draft a concise IMRAD-style paper for a BO-backed chemistry or materials study from research-agent artifacts.
---

# Scientific Writing

Use this skill to turn `research_state.json`, `research_plan.md`, and BO artifacts into a paper draft.

V1 output is markdown only: `research_runs/<research_id>/paper.md`. Target length is 1500–2500 words, excluding the title and abstract.

## Inputs

- `research_runs/<research_id>/research_state.json`
- `research_runs/<research_id>/research_plan.md`
- `bo_runs/<bo_run_id>/report.json`
- `bo_runs/<bo_run_id>/convergence.pdf` (reference in Methods; do not embed, just cite its path)
- optional literature sources gathered earlier

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
- If results are from a proxy simulation, say so explicitly in the abstract.

### Introduction
- State the research problem and why it matters.
- Summarize only the relevant literature context — 2–4 sentences, tied to the baselines in `literature_findings`.
- Motivate why optimization is needed rather than exhaustive screening.

If literature review was skipped:
- keep the Introduction minimal and artifact-scoped
- do not add generic domain background, historical context, or benchmark claims that are not present in `research_plan.md`, `research_state.json`, repo docs, or the run artifacts
- state plainly that no literature baseline comparison was performed for this run

### Methods
- Describe the search space actually used (design variables, bounds, any simplex constraints).
- Describe the BO engine and relevant configuration (surrogate model, acquisition function, batch size).
- State whether the workflow was simulation-backed or human-in-the-loop.
- If simulation: report the proxy oracle CV RMSE and note that outcomes reflect surrogate predictions, not direct measurements.
- Reference the convergence plot at `bo_runs/<bo_run_id>/convergence.pdf`.

### Results
- Report the best value found and the corresponding candidate (composition, conditions, etc.).
- Mention convergence behavior — did the search plateau, was it still improving at the end?
- If simulation: do not present outcomes as measured values. Use phrasing like "the proxy oracle predicted…" or "the surrogate model identified…".

### Discussion
- Interpret the result chemically or materially — why might this composition or condition work?
- Compare against the literature baselines from `literature_findings.baselines` if available.
- State important caveats: oracle error, dataset coverage, gap between simulation and real experiments.

If literature review was skipped:
- keep the Discussion grounded in the BO trajectory, candidate composition, oracle uncertainty, and simulation limitations
- do not introduce external mechanism claims
- any hypothesis must be labeled as tentative and artifact-derived, not literature-backed

### Conclusion
- One paragraph summarizing what was found.
- One sentence on the next practical step (e.g., experimental validation, broader search).

## Guardrails

- Clearly label proxy-oracle results as simulated throughout — in the abstract, methods, and results.
- Do not present simulated BO outcomes as real laboratory measurements.
- If evidence is weak (high oracle RMSE, few iterations, narrow dataset), say so directly.
- Keep references lightweight in v1; plain links or compact citations are enough.
- Keep the writing tied to the actual artifacts rather than generic BO boilerplate.
