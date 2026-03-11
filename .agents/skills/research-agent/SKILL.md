---
name: research-agent
description: Orchestrate an end-to-end chemistry or materials optimization study from a plain-English research question to BO execution and a paper draft.
---

# Research Agent

Use this skill when the user wants a top-level research workflow rather than a raw BO command sequence.

V1 supports two modes only:
- `simulation`: retrospective dataset-backed proxy BO
- `warm_start_human`: user has some prior observations and then continues in a human-in-the-loop BO loop

Do not treat proxy evaluation as the default scientific workflow. It is a simulation backend for demos and retrospective testing.

## Inputs

- Research question in plain English
- Optional dataset path
- Optional prior observations path or inline observations

## State Files

Generate `research_id` as a short slug from the system and date, e.g. `oer_caltech_20240311`. Create and maintain these files under `research_runs/<research_id>/`:

- `research_state.json`: machine-readable phase state
- `research_plan.md`: human-readable lab notebook
- `paper.md`: final draft written in Phase 6

Use this `research_state.json` shape in v1:

```json
{
  "research_id": "string",
  "research_question": "string",
  "mode": "simulation | warm_start_human",
  "system": null,
  "objective_property": null,
  "objective_direction": null,
  "dataset_path": null,
  "prior_observations_path": null,
  "bo_run_id": null,
  "literature_findings": {
    "baselines": [],
    "key_variables": [],
    "known_constraints": [],
    "summary": ""
  },
  "experiment_spec": {
    "target_column": null,
    "design_variables": [],
    "constraints": [],
    "seed_observations_count": 0
  },
  "bo_results": {
    "best_value": null,
    "best_x": null,
    "oracle_rmse": null,
    "report_path": null
  },
  "paper_path": null,
  "phases": {
    "problem_framing": "pending | in_progress | completed",
    "literature_search": "pending | in_progress | completed | skipped",
    "experiment_setup": "pending | in_progress | completed",
    "bo_execution": "pending | in_progress | completed",
    "interpretation": "pending | in_progress | completed",
    "paper_writing": "pending | in_progress | completed"
  }
}
```

`research_plan.md` must contain these sections:
- Research Question
- Problem Framing
- Literature Context
- Experiment Design
- BO Results
- Interpretation
- Paper Draft Link

## Workflow

### 1. Problem Framing

Resolve and write:
- `system`
- `objective_property`
- `objective_direction`
- `mode`
- `dataset_path`
- `prior_observations_path`

Mode selection rules — apply exactly one:
- Dataset provided AND user wants a fully automated retrospective run → `simulation`
- User has prior observations OR intends to supply future observations manually → `warm_start_human`
- If neither fits clearly, ask before proceeding.

Also decide whether to run a literature search:
- If the user does not mention literature or asks to skip it, set `phases.literature_search` to `skipped` and proceed.
- If the user wants literature context or the problem is novel enough that baselines would inform setup, plan it.

Other rules:
- Do not infer the full BO schema from CSV columns alone.
- If the system, objective, or direction are ambiguous, clarify them before continuing.

### 2. Literature Search

If `phases.literature_search` is `skipped`, write empty-but-valid `literature_findings` and move to Phase 3.

Otherwise, delegate to the `literature-review` skill. Pass:
- `system`
- `objective_property`
- `objective_direction`
- `dataset_path` (if available)
- path: `research_runs/<research_id>/research_plan.md` (for the skill to write the Literature Context section)

Receive back the structured `literature_findings` JSON and write it into `research_state.json`.

### 3. Experiment Setup

Use the framed problem plus literature findings to define the experiment.

Rules:
- Treat the dataset as supporting evidence, not the canonical source of semantics.
- Use dataset columns to map or confirm an already-decided setup, not to invent the objective or constraints from scratch.
- Infer domain constraints from the problem description and literature.
- If composition variables must sum to a fixed total, pass `--simplex-groups` during BO init.
- If there is no dataset and no prior observations, explicitly ask the user for the search space: design variables, bounds or categories, target measurement, and known constraints. Do not start BO until that is resolved.
- In that fallback, do not ask an empty question. Propose a draft BO spec first, based on the research question plus any literature findings:
  - candidate design variables
  - tentative bounds or categorical options
  - target measurement to optimize
  - likely physical or chemical constraints
- Present that draft as a recommendation for the user to confirm or edit before BO init.

Delegate to BO skills:
- `bo-init-run`
- `bo-build-proxy-oracle` in `simulation` mode only
- `bo-record-observation` to seed prior observations in `warm_start_human` mode

Write the resulting BO run ID into `research_state.json.bo_run_id`.
Keep `research_state.json.experiment_spec.constraints` structured and machine-readable. Do not collapse constraints into prose strings if they were originally represented as typed objects or explicit column groups.

### 4. BO Execution

Delegate based on mode:
- `simulation`: use the existing `bo_run_id` created in Phase 3 and continue with `uv run python -m bo_workflow.cli run-proxy --run-id <BO_RUN_ID> --iterations <N> [--batch-size <N>]`, then finish with `bo-report-run`
- `warm_start_human`: iterative `bo-next-batch` plus `bo-record-observation`

Always finish with `bo-report-run` and write:
- `best_value`
- `best_x`
- `oracle_rmse` when applicable
- `report_path`

Do not delegate simulation mode to `bo-end-to-end-proxy` once Phase 3 has already run `bo-init-run` and `bo-build-proxy-oracle`; that would re-initialize the BO run and duplicate setup.

### 5. Interpretation

Summarize:
- best result found
- comparison to literature baselines if available
- brief chemical or materials reasoning for why the best condition may work
- whether the evidence is simulated or real
- important caveats such as oracle error or sparse evidence

Write this into the Interpretation section of `research_plan.md`.

If literature was skipped, keep interpretation artifact-grounded:
- describe patterns visible in the BO trajectory, best candidate, oracle quality, and convergence
- do not introduce external literature or mechanism claims
- any hypothesis must be explicitly labeled as a tentative interpretation from this simulation run only

### 6. Paper Writing

Delegate drafting to `scientific-writing`. Pass all of the following so the skill has everything it needs:
- `research_runs/<research_id>/research_state.json`
- `research_runs/<research_id>/research_plan.md`
- `bo_runs/<bo_run_id>/report.json`
- `bo_runs/<bo_run_id>/convergence.pdf` (reference path; skill will mention it in Methods)
- Any literature sources from Phase 2

Output:
- `research_runs/<research_id>/paper.md`
- `research_state.json.paper_path`

## Resuming

On resume:
1. Read `research_state.json`.
2. Find the first phase not marked `completed` or `skipped`.
3. Continue from that phase.
4. Do not re-run completed BO setup or rebuild an oracle unless the user explicitly asks.

## Guardrails

- Always label simulation results as proxy-oracle simulations.
- Include oracle CV RMSE whenever reporting simulation results.
- Never auto-record observations in `warm_start_human` mode.
- Do not call `bo-end-to-end-proxy` in `warm_start_human` mode.
- Keep `research_state.json` concise and structured; put narrative detail in `research_plan.md`.
- Fully prospective no-dataset mode is out of scope for v1.
