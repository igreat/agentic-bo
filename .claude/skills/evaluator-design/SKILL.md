---
name: evaluator-design
description: Design and stabilize an expensive or fragile chemistry evaluator before BO setup.
---

# Evaluator Design

Use this skill when `research-agent` has identified a nontrivial evaluator path that still needs to be turned into a stable BO setup.

Typical triggers:
- first-principles or simulation-backed evaluators
- expensive or fragile custom pipelines
- unresolved search spaces that depend on what can actually be computed
- local evaluator code that still needs to be written and tested

This skill is **domain-general**. It applies to DFT, xTB, MD, custom simulators, and other chemistry evaluators that need setup and stabilization before BO.

## Goal

Turn a vague evaluable-chemistry idea into a stabilized experiment recommendation:

- choose an evaluator family
- choose a candidate family and initial search representation
- run a small calibration subset
- classify failures
- prune or revise unstable choices
- recommend a BO engine
- produce a runnable local evaluator plan for Phase 3B / Phase 4

When local code is needed, use these defaults:
- evaluator module: `research_runs/<research_id>/scripts/evaluator.py`
- search-space artifact: `research_runs/<research_id>/search_space.json`
- BO execution handoff: `uv run python -m bo_workflow.cli run-python-evaluator ...`

## Inputs

- `system`
- `objective_property`
- `objective_direction`
- `literature_findings`
- optional user-provided search-space hints
- path to `research_runs/<research_id>/research_state.json`
- path to `research_runs/<research_id>/research_plan.md`

## Output Contract

Return a stabilized setup recommendation that can be written back into `research_state.json`:

```json
{
  "experiment_spec": {
    "target_column": null,
    "design_parameters": [],
    "fixed_features": {},
    "constraints": [],
    "seed_observations_count": 0,
    "bo_engine": null
  },
  "run_artifacts": {
    "scripts_dir": null,
    "extra_paths": [],
    "dependency_installs": []
  },
  "calibration_summary": {
    "points_tested": [],
    "failures": [],
    "pruned_choices": [],
    "engine_recommendation": null,
    "why": ""
  }
}
```

Also write a short narrative into the **Experiment Design** section of `research_plan.md` covering:
- calibration points chosen
- failures and how they were classified
- what was pruned or revised
- final engine choice and rationale

## Pre-BO Checklist

Complete this checklist before BO setup:

1. Choose an evaluator candidate from literature or user context.
2. Define the smallest meaningful candidate family.
3. Choose a calibration subset of **at most 5 representative points**.
4. Check what chemistry packages are already available in the environment before committing to a stack.
5. Implement the minimum evaluator/setup needed to test those points.
   - prefer existing installed packages when they fit
   - if a local evaluator is needed, write it to `research_runs/<research_id>/scripts/evaluator.py`
   - if the environment is missing something essential, install the smallest missing dependency with `uv pip install ...`
6. Run the calibration subset.
7. Classify failures:
   - `candidate_local`: the specific candidate is bad or unsupported, but the evaluator family still looks sound
   - `systematic`: the evaluator/setup itself is unstable or misconfigured
8. If 2 or more calibration points fail for the same setup reason, treat that as systematic and revise the family/setup before BO.
9. Prune unstable choices and finalize the search space.
10. Recommend the BO engine.
11. Hand off the stabilized setup to BO.

## Calibration Budget

- Default calibration budget: **3–5 representative points**
- The goal is **pipeline stability and family pruning**, not mapping the space
- Representative means:
  - edge cases
  - likely winners
  - structurally distinct candidates
  - candidates most likely to expose evaluator fragility
- Going beyond 5 requires an explicit reason in `research_plan.md`

## What “Representative” Means

Prefer a calibration subset that reveals whether the evaluator can survive the family:

- extremes of composition or identity
- distinct geometry/site/family choices
- known difficult or fragile cases
- one or two chemically plausible candidates, not only pathological ones

Do not waste calibration budget on near-duplicates.

## Failure Handling

Use these rules:

- If a failure appears tied to one candidate only, record it as `candidate_local` and continue unless the candidate is central to the family.
- If failures repeat for the same reason across multiple points, record them as `systematic` and revise the setup before BO continues.
- Do not quietly hide systematic failures behind penalties.
- Penalty-based fallback is acceptable only after you have recorded that the failure is candidate-local rather than systematic.

## Search-Space Design Guidance

- Start with the smallest scientifically meaningful family.
- Prefer search-space choices that map cleanly into the evaluator.
- Avoid broadening the space until the evaluator survives the calibration subset.
- Prune dimensions that create lots of instability without adding much scientific value.
- Keep the design general: the right search space might be slabs, molecules, alloys, catalysts, solvents, or something else entirely.
- Prefer choices that the available software stack can actually support cleanly.

## Engine Recommendation Guidance

Follow the `bo-init-run` heuristic:

- Prefer `botorch` when the search space is mostly or entirely categorical, the all-categorical candidate count is still modest enough to reason about (default threshold `<= 2000`), and evaluations are expensive enough that sample efficiency matters.
- Prefer `hebo` when the space is broader and more mixed numeric/categorical, when numeric dimensions dominate, or when there is no strong reason to bias toward BoTorch.
- If `hebo --hebo-model gp` looks numerically unstable on a mixed space, recommend `hebo --hebo-model rf` before abandoning HEBO entirely.

## Guardrails

- Do not overfit this process to HER or surface catalysis examples.
- Do not treat literature lookup tables as a live evaluator unless the user explicitly allows a lookup fallback.
- Do not map the entire space during calibration.
- Do not start BO until the evaluator family, search space, and engine recommendation are stable enough to hand off.
- Do not assume a library is available; inspect the environment first.
