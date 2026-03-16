---
name: research-agent
description: Orchestrate an end-to-end chemistry or materials optimization study from a plain-English research question to BO execution and a paper draft.
---

# Research Agent

Use this skill when the user wants a top-level research workflow rather than a raw BO command sequence.

V1 is **observer-agnostic**. The agent frames the problem, defines a structured experiment spec, initializes a BO run, continues through `suggest` / `observe` / `report`, interprets the outcome, and drafts a paper.

The agent does **not** choose a mode like `simulation` vs `human_in_the_loop`. Observation values may come from:
- manual user reports
- prior observations supplied up front
- an external benchmark harness or other observer owned by the operator

Do not tell the agent to build its own proxy oracle as part of this workflow.

## Inputs

- Research question in plain English
- Optional dataset path
- Optional prior observations path or inline observations
- Optional search-space context already supplied by the user

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
    "design_parameters": [],
    "fixed_features": {},
    "constraints": [],
    "seed_observations_count": 0
  },
  "bo_results": {
    "best_value": null,
    "best_x": null,
    "best_iteration": null,
    "num_observations": null,
    "oracle_model": null,
    "oracle_rmse": null,
    "report_path": null,
    "convergence_plot_path": null
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
- `dataset_path`
- `prior_observations_path`

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
- `experiment_spec` is the canonical BO search-space object for the agent. Populate:
  - `target_column`
  - `design_parameters`
  - `fixed_features`
  - `constraints`
- Infer domain constraints from the problem description and literature.
- If composition variables must sum to a fixed total, keep those constraints structured and machine-readable.
- If there is no resolved search space yet, do not ask an empty question. Propose a draft experiment spec first, based on the research question plus any literature findings:
  - candidate design parameters
  - tentative bounds or categorical options
  - fixed features if any
  - target measurement to optimize
  - likely physical or chemical constraints
- Present that draft as a recommendation for the user to confirm or edit before BO init.

Delegate the BO-layer setup to `bo-execution-workflow`. That skill owns:
- dataset validation when a dataset is present
- simplex and `--drop-cols` execution config when relevant
- representation/encoding handoff to BO converters when the representation plan requires it
- `bo-init-run`
- `bo-record-observation` to seed prior observations when they exist

In Phase 3, call `bo-execution-workflow` in **setup-only** mode:
- stop once `init` and any seed observations are complete

Write the resulting BO run ID into `research_state.json.bo_run_id`.
Keep `research_state.json.experiment_spec.constraints` structured and machine-readable. Do not collapse constraints into prose strings.

### 4. BO Execution

Delegate BO execution to `bo-execution-workflow`, continuing from the existing `bo_run_id` from Phase 3 through iterative `suggest` / `observe` / `report`.

The observation source may be:
- the user
- a real experimental loop
- an external benchmark harness
- another operator-owned observer

The agent does not need to model those as separate modes.

Always finish with `bo-report-run` and write:
- `best_value`
- `best_x`
- `best_iteration`
- `num_observations`
- `oracle_model` when the BO artifacts report one
- `oracle_rmse` when the BO artifacts report one
- `report_path`
- `convergence_plot_path`

Do not re-run Phase 3 setup during Phase 4. In particular:
- do not call `bo-end-to-end-proxy`
- do not call `bo-build-proxy-oracle`
- do not call `build-oracle`
- do not call `run-proxy`
- do not re-run `init`
- always continue from the existing `bo_run_id`

### 5. Interpretation

Summarize:
- best result found
- comparison to literature baselines if available
- brief chemical or materials reasoning for why the best condition may work
- whether the evidence comes from recorded observations or a proxy/evaluator backend, if that is clear from the BO artifacts
- important caveats such as oracle error or sparse evidence

Write this into the Interpretation section of `research_plan.md`.

If literature was skipped or the BO artifacts indicate proxy-backed evaluation:
- keep interpretation artifact-grounded
- describe patterns visible in the BO trajectory, best candidate, oracle quality, and convergence
- do not introduce external literature or mechanism claims
- any hypothesis must be explicitly labeled as tentative and artifact-derived

### 6. Paper Writing

Delegate drafting to `scientific-writing`. Pass all of the following so the skill has everything it needs:
- `research_runs/<research_id>/research_state.json`
- `research_runs/<research_id>/research_plan.md`
- `bo_runs/<bo_run_id>/report.json`
- `bo_runs/<bo_run_id>/convergence.pdf` (reference path; skill will mention it in Methods)
- any literature sources from Phase 2

Output:
- `research_runs/<research_id>/paper.md`
- `research_state.json.paper_path`

## Resuming

On resume:
1. Read `research_state.json`.
2. Find the first phase not marked `completed` or `skipped`.
3. Continue from that phase.
4. Do not re-run completed BO setup unless the user explicitly asks.

## Guardrails

- Never invent observation values.
- Only record results provided by the user or an external observer/harness.
- If BO artifacts include oracle metadata, label results as simulations and include oracle CV RMSE.
- Keep `research_state.json` concise and structured; put narrative detail in `research_plan.md`.
- Do not call `bo-end-to-end-proxy`, `build-oracle`, or `run-proxy` as part of `research-agent`.
- A fully unresolved search space is out of scope for execution; resolve `experiment_spec` first.
