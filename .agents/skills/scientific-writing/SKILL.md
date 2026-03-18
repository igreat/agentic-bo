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
- `bo_runs/<bo_run_id>/observations.jsonl` when you want exact trajectory or phase-specific numeric claims
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
- If the BO artifacts indicate proxy-backed or oracle-backed evaluation, say so explicitly in the abstract.

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
- State how observations were obtained based on the available artifacts.
- Describe oracle provenance only from what the artifacts explicitly say. If `report.json` exposes oracle metadata but not training timing, describe it as backend-reported or artifact-reported oracle metadata rather than claiming it was fitted post hoc.
- If the BO artifacts indicate proxy-backed evaluation: report the proxy oracle CV RMSE and note that outcomes reflect surrogate predictions, not direct measurements.
- Reference the convergence plot at `bo_runs/<bo_run_id>/convergence.pdf`.

### Results
- Report the best value found and the corresponding candidate (composition, conditions, etc.).
- Mention convergence behavior — did the search plateau, was it still improving at the end?
- Use `report.json` as the source of truth for best-value summary statistics.
- For human-facing numbering, prefer `report.json["best_observation_number"]` over the zero-based internal `best_iteration` field.
- Prefer `report.json["trajectory"]` for phase summaries, random-phase ranges, and best-observation numbering when it is available.
- If `report.json["trajectory"]` is present, use it directly rather than recomputing phase summaries from memory or ad hoc shell output.
- If you include exact random-phase ranges, phase breakpoints, or iteration-specific claims beyond the reported best and they are not already present in `report.json["trajectory"]`, verify them from `observations.jsonl`.
- If `observations.jsonl` is not provided or not read, avoid precise trajectory numbers and keep the narrative qualitative.
- If the BO artifacts indicate proxy-backed evaluation: do not present outcomes as measured values. Use phrasing like "the proxy oracle predicted…" or "the surrogate model identified…".

### Discussion
- Interpret the result chemically or materially — why might this composition or condition work?
- Compare against the literature baselines from `literature_findings.baselines` if available.
- State important caveats: oracle error when applicable, dataset coverage when applicable, and any gap between simulated/externally observed evidence and real experiments.

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
- Do not invent fine-grained numeric trajectory details from memory. If exact ranges or iteration-level numbers are not explicitly supported by `report.json` or `observations.jsonl`, leave them out.
- Do not infer oracle training timing or methodology unless it is explicitly stated in the artifacts.
- If `report.json["oracle"]["source"]` says the metadata came from the evaluation backend, describe it as backend-reported oracle metadata rather than implying a fresh model fit after the run.
