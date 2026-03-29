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
- Optional benchmark task bundle path or `task_manifest.json`

## State Files

Generate `research_id` as a short slug from the system and date, e.g. `oer_caltech_20240311`. Create and maintain these files under `research_runs/<research_id>/`:

- `research_state.json`: machine-readable phase state
- `research_plan.md`: human-readable lab notebook
- `paper.md` or `paper.tex`: final draft written in Phase 6, depending on the requested output format

These are the only required core artifacts. Any additional supporting files should be optional and discoverable through `research_state.json.run_artifacts`. When helper code is needed, the default run-local location is `research_runs/<research_id>/scripts/`.

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
    "source_urls": [],
    "summary": "",
    "computable_candidates": [],
    "evaluator_profile": {
      "mode": "lookup | surrogate | live_simulation | unknown",
      "evaluation_cost": "cheap | moderate | expensive",
      "stability_risk": "low | medium | high",
      "requires_run_local_setup": false,
      "why": ""
    }
  },
  "run_artifacts": {
    "scripts_dir": null,
    "extra_paths": [],
    "dependency_installs": []
  },
  "evaluator_assessment": {
    "evidence_class": "validated | physics_inspired_heuristic | retrospective_lookup | user_provided",
    "claim_posture": "strong | moderate | cautious",
    "why": ""
  },
  "calibration_summary": {
    "points_tested": [],
    "fit_scope": "",
    "metrics": {},
    "artifact_path": null
  },
  "experiment_spec": {
    "target_column": null,
    "design_parameters": [],
    "fixed_features": {},
    "constraints": [],
    "seed_observations_count": 0,
    "bo_engine": null
  },
  "bo_results": {
    "best_value": null,
    "best_x": null,
    "best_iteration": null,
    "num_observations": null,
    "oracle_model": null,
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

Interaction modes:
- **Default collaborative mode**: clarification-first
- **Explicit autonomy mode**: one-shot, no-questions, or fully open-world execution when the user explicitly asks for it

Rules:
- In default collaborative mode, begin with a short alignment step and ask a few high-value clarifying questions before committing to a search space, evaluator family, or BO budget.
- In explicit autonomy mode, do not pause for clarifying questions unless you are truly blocked. Make reasonable assumptions, record them in `research_plan.md`, and proceed.
- Treat fully open-world autonomy runs as an explicit benchmark mode, not the long-term default interaction pattern.
- For explicit autonomy benchmark runs that require a live structural evaluator, default to a scientifically credible screening story over a flashy breakthrough story.

### 1. Problem Framing

Resolve and write:
- `system`
- `objective_property`
- `objective_direction`
- `dataset_path`
- `prior_observations_path`

Also resolve the intended observation source as early as possible:
- the user will manually supply values back-and-forth
- an existing evaluator/backend/module already exists
- an external benchmark harness or operator-owned observer will supply values
- the agent needs to discover or build a local evaluator

Also decide whether to run a literature search:
- If the user does not mention literature or asks to skip it, set `phases.literature_search` to `skipped` and proceed.
- If the user wants literature context or the problem is novel enough that baselines would inform setup, plan it.

Other rules:
- Do not infer the full BO schema from CSV columns alone.
- In default collaborative mode, explicitly clarify what the user wants to use as the evaluator or observation source before committing to evaluator-design.
- If the user already has an evaluator, wants to act as the evaluator, or plans to provide values manually, treat that as the observation source and do not invoke `evaluator-design` unless they separately ask the agent to design a new evaluator.
- If the system, objective, or direction are ambiguous:
  - clarify them before continuing in default collaborative mode
  - in explicit autonomy mode, make the smallest reasonable set of assumptions, record them clearly, and continue unless truly blocked

### 2. Literature Search

If `phases.literature_search` is `skipped`, write empty-but-valid `literature_findings` and move to Phase 3.

Otherwise, delegate to the `literature-review` skill. Pass:
- `system`
- `objective_property`
- `objective_direction`
- `dataset_path` (if available)
- `local_packet_path` when the benchmark task bundle provides `literature.mode = local_packet`
- path: `research_runs/<research_id>/research_plan.md` (for the skill to write the Literature Context section)

Receive back the structured `literature_findings` JSON and write it into `research_state.json`.

By default, treat literature search as web-enabled and open-world:

- browse for computable evaluator paths in papers, equations, code, tutorials, repositories, docs, or reproducible algorithms
- identify the required inputs, design variables, and operational assumptions before worrying about broad baseline coverage
- use baselines as lightweight context, not the primary output of the search
- if the user explicitly asked for a real DFT-style or first-principles evaluator, use literature to recover the workflow and constraints, not to replace the evaluator with a lookup-table surrogate
- make sure `literature_findings.evaluator_profile` clearly says whether the path is a lookup, surrogate, or live simulation and how costly/risky it looks

Treat local-packet mode as the closed-world/control exception:

- use only the local packet from the task bundle when present
- do not browse the web
- treat the task bundle as the authoritative public context

### 3. Experiment Setup

Use the framed problem plus literature findings to define the experiment.

Treat Phase 3 as two internal steps when the evaluator path is nontrivial:

- **Phase 3A: Evaluator/Search-Space Design**
- **Phase 3B: BO Setup**

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
- Present that draft as a recommendation for the user to confirm or edit before BO init in default collaborative mode.
- In explicit autonomy mode, commit to the draft experiment spec yourself, record the key assumptions, and continue unless a real blocker remains.
- For structural screening problems, make an explicit candidate-family and representation choice and record it in `research_plan.md`.
- Do not default to a particular structural family, crystal prototype, or benchmark mode unless the user or benchmark materials explicitly require one.
- Prefer the smallest physically justified feasible set you can defend from the task materials and literature.
- Prefer existing repo tooling when it genuinely fits, but do not force a poor fit.
- If helper code is needed, create `research_runs/<research_id>/scripts/` and put ad hoc converters, preprocessors, evaluator modules, plotting scripts, scraping helpers, and other one-off utilities there.
- If a local evaluator is produced, use `research_runs/<research_id>/scripts/evaluator.py` by default.
- If a local BO search-space file is produced, use `research_runs/<research_id>/search_space.json` by default.
- If a minimal missing dependency is blocking progress, install it with `uv pip install <pkg>` rather than editing project dependency files.
- Record run-local helper code and supporting outputs in `run_artifacts.extra_paths`.
- Record dependency installs in `run_artifacts.dependency_installs` as objects with `packages`, `command`, and `reason`.
- Set `run_artifacts.scripts_dir` when a run-local scripts directory is used.
- Ensure `research_state.json.experiment_spec` mirrors the actual BO setup that will be passed to `init`; do not leave a stale draft schema or placeholder categorical options behind once setup is finalized.
- If the natural feasible region is not a clean Cartesian product, do not force `metal × facet × site` blindly. Use:
  - an explicit valid candidate catalog encoded as a single categorical `candidate_id`, or
  - another lossless constrained representation that only emits candidates the evaluator can actually instantiate
- For live structural evaluators, the search space must exclude known-invalid geometry/site combinations before BO starts.
- If an empirical calibration correction is used to map evaluator outputs onto a literature or DFT scale, persist a machine-readable `calibration_summary` in `research_state.json` or a sidecar artifact referenced from `run_artifacts.extra_paths`.
- If the user explicitly asked for a real DFT-style or first-principles evaluator, do not replace it with a literature lookup or pre-tabulated surrogate unless the user explicitly approves that fallback.

Routing rule:
- invoke the `evaluator-design` skill before BO init when any of these are true:
  - there is no user-provided evaluator or observation source and the agent needs to discover or build one
  - `literature_findings.evaluator_profile.mode` is `live_simulation`
  - `literature_findings.evaluator_profile.requires_run_local_setup` is `true`
  - the user explicitly asked for a first-principles or simulator-backed evaluator
  - the search space is still unresolved after literature review
  - `literature_findings.evaluator_profile.mode` is not `lookup` and either:
    - `literature_findings.evaluator_profile.stability_risk` is `high`
    - `literature_findings.evaluator_profile.evaluation_cost` is `expensive`

When `evaluator-design` is invoked:
- treat that as **Phase 3A**
- require a stabilized setup recommendation before BO init
- record calibration points, failures, pruned choices, and engine rationale in `research_plan.md`
- write the evaluator evidence class and claim posture into `research_state.json.evaluator_assessment`
- write the calibration scope/metrics into `research_state.json.calibration_summary` when an empirical calibration step is used
- write the recommended engine into `research_state.json.experiment_spec.bo_engine`

When `evaluator-design` is not needed:
- skip directly to **Phase 3B**
- this includes cases where the user is the evaluator, the user already has an evaluator/backend/module, or another external observer will provide the values
- still write a brief `research_state.json.evaluator_assessment` entry based on the actual observation source so later interpretation and writing stay calibrated

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

Do not re-run Phase 3 setup during Phase 4. In particular:
- do not call `build-oracle`
- do not call `run-proxy`
- do not re-run `init`
- always continue from the existing `bo_run_id`

If the user or operator explicitly provides a `backend_id` for external evaluation, `bo-run-evaluator` is an acceptable way to automate the suggest/observe loop. It is still not acceptable to build the backend from inside `research-agent`.

If a benchmark task bundle provides a prebuilt `evaluation.backend_id`, it is acceptable to automate Phase 4 directly with `run-evaluator` against the backend copied into the public workspace.

If Phase 3 produced a local evaluator module, automate Phase 4 with:

```bash
uv run python -m bo_workflow.cli run-python-evaluator \
  --run-id <BO_RUN_ID> \
  --module-path research_runs/<research_id>/scripts/evaluator.py \
  --iterations <BUDGET> \
  --batch-size <N>
```

### 5. Interpretation

Summarize:
- best result found
- comparison to literature baselines if available
- brief chemical or materials reasoning for why the best condition may work
- whether the evidence comes from recorded observations or a proxy/evaluator backend, if that is clear from the BO artifacts
- important caveats such as oracle error or sparse evidence
- whether the top candidate is actually distinguishable from nearby candidates given the evaluator uncertainty band

Write this into the Interpretation section of `research_plan.md`.

When updating `research_plan.md` after Phase 4:
- use `bo_runs/<bo_run_id>/report.json` as the source of truth for `best_value`, `best_iteration`, `num_observations`, `oracle_model`, and `oracle_rmse`
- use `report.json["best_observation_number"]` for human-facing iteration/observation references; treat `best_iteration` as a zero-based engine index
- prefer `report.json["trajectory"]` for human-facing trajectory summaries when it is available
- if `report.json["trajectory"]` is present, use it directly rather than reconstructing ranges or phase breakpoints from memory
- if you want to describe random-phase or iteration-specific trajectory details that are not present in `report.json["trajectory"]`, verify them against `observations.jsonl` rather than summarizing from memory
- if you did not read `observations.jsonl`, keep trajectory language qualitative rather than claiming exact phase ranges or per-iteration values

If literature was skipped or the BO artifacts indicate proxy-backed evaluation:
- keep interpretation artifact-grounded
- describe patterns visible in the BO trajectory, best candidate, oracle quality, and convergence
- do not introduce external literature or mechanism claims
- any hypothesis must be explicitly labeled as tentative and artifact-derived

If `research_state.json.evaluator_assessment` says the evaluator is `physics_inspired_heuristic` or if the calibration uncertainty is of the same order as the top-candidate gap:
- present the result as a shortlist of DFT-validation candidates rather than a fully settled winner
- do not claim a material "surpasses" an established benchmark unless the margin exceeds the stated uncertainty and the structural assumptions are physically plausible
- keep the headline at the level of screening guidance, not benchmark truth

### 6. Paper Writing

Delegate drafting to `scientific-writing`. Pass all of the following so the skill has everything it needs:
- `research_runs/<research_id>/research_state.json`
- `research_runs/<research_id>/research_plan.md`
- `bo_runs/<bo_run_id>/report.json`
- `bo_runs/<bo_run_id>/state.json`
- `bo_runs/<bo_run_id>/observations.jsonl`
- existing run-local figure assets such as `bo_runs/<bo_run_id>/convergence.pdf` if present
- any literature sources from Phase 2

Output:
- `research_runs/<research_id>/paper.tex` when the prompt explicitly asks for LaTeX; otherwise `research_runs/<research_id>/paper.md`
- `research_state.json.paper_path`

After the paper is written, perform a final consistency pass before declaring the workflow complete:
- reread `research_state.json`, `research_plan.md`, `bo_runs/<bo_run_id>/report.json`, and the final paper file referenced by `research_state.json.paper_path`
- ensure all phase states are correct
- ensure `research_state.json.paper_path` points to the final paper
- update the **Paper Draft Link** section in `research_plan.md` to the real paper path; do not leave placeholder text like "to be written in Phase 6"
- ensure key numeric claims in `research_plan.md` and the final paper match `report.json`
- ensure any detailed trajectory claims match `report.json["trajectory"]` when that field is present
- ensure human-facing iteration numbering in `research_plan.md` and the final paper uses `best_observation_number` when available rather than the zero-based `best_iteration`
- ensure oracle provenance language stays artifact-backed; do not let the paper imply a post-hoc fit unless an artifact explicitly says that
- ensure `research_state.json.experiment_spec` matches the final executed BO spec rather than an earlier draft
- ensure `research_plan.md` and the final paper reflect `evaluator_assessment.claim_posture` and any stated calibration uncertainty
- if optional supporting files, helper scripts, or dependency installs materially affected the run, ensure they are reflected in `run_artifacts` and described in `research_plan.md`
- if any artifact is stale or contradictory, fix it before marking the run complete

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
- Do not call `build-oracle` or `run-proxy` as part of `research-agent`.
- A fully unresolved search space is out of scope for execution; resolve `experiment_spec` first.
- In benchmark runs with a local literature packet, do not browse beyond the packet.
- Do not present a demo-quality live structural screening run as canonical benchmark truth when the calibration scope is narrow, the top candidates are within uncertainty, or the structures are metastable simplifications.
