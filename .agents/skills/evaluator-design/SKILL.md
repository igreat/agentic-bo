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
- for non-Cartesian feasible sets, prefer a run-local candidate catalog artifact that the BO setup can encode as a single categorical `candidate_id`

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
  "evaluator_decision": {
    "mode": "local_executable_evaluator | retrospective_lookup_evaluator | hybrid",
    "why": ""
  },
  "evaluator_assessment": {
    "evidence_class": "validated | physics_inspired_heuristic | retrospective_lookup | user_provided",
    "claim_posture": "strong | moderate | cautious",
    "why": ""
  },
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
    "fit_scope": "",
    "metrics": {},
    "engine_recommendation": null,
    "why": "",
    "artifact_path": null
  }
}
```

Also write a short narrative into the **Experiment Design** section of `research_plan.md` covering:
- calibration points chosen
- failures and how they were classified
- what was pruned or revised
- final engine choice and rationale
- evaluator evidence class and what level of scientific claim it supports

## Pre-BO Checklist

Complete this checklist before BO setup:

1. Choose an evaluator candidate from literature or user context.
2. Make an explicit evaluator decision:
   - `local_executable_evaluator`
   - `retrospective_lookup_evaluator`
   - `hybrid`
   Record why this is the right mode for the run.
3. Define the smallest meaningful candidate family.
4. If this is a structural screening problem, choose and record the candidate family and search representation explicitly.
   - Examples include a filtered native-structure catalog, a constrained alloy family, a pre-pruned facet/site catalog, or another physically justified representation.
   - Do not default to a particular benchmark mode or crystal-family simplification unless the task materials explicitly require one.
5. Choose a calibration subset.
6. Check what chemistry packages are already available in the environment before committing to a stack.
7. Implement the minimum evaluator/setup needed to test those points.
   - prefer existing installed packages when they fit
   - if a local evaluator is needed, write it to `research_runs/<research_id>/scripts/evaluator.py`
   - if the environment is missing something essential, install the smallest missing dependency with `uv pip install ...`
   - if the evaluator uses a calibrated target, return the optimized target plus raw diagnostic fields (for example raw score, calibrated score, and failure metadata) so `observations.jsonl` preserves both
8. Run the calibration subset when the chosen mode includes a local executable path.
9. Classify failures:
   - `candidate_local`: the specific candidate is bad or unsupported, but the evaluator family still looks sound
   - `systematic`: the evaluator/setup itself is unstable or misconfigured
10. If 2 or more calibration points fail for the same setup reason, treat that as systematic and revise the family/setup before BO.
11. Prune unstable choices and finalize the search space.
    - the final BO search space must contain only candidates the evaluator can actually instantiate
    - if valid combinations are not a clean Cartesian product, emit an explicit candidate catalog and optimize over that catalog instead of exposing invalid cross-product combinations
12. Recommend the BO engine.
13. Assign an evaluator evidence class and matching claim posture:
   - `validated`: externally grounded local evaluator with real validation or a strongly established executable workflow
   - `physics_inspired_heuristic`: simplified or invented mechanistic model inspired by literature but not strongly validated
   - `retrospective_lookup`: tabulated or database-backed evaluator over precomputed values
   - `user_provided`: the user or another external observer supplies values directly
14. Hand off the stabilized setup to BO.

## Calibration Budget

- Default calibration budget: **3–5 representative points**
- The goal is **pipeline stability and family pruning**, not mapping the space
- Representative means:
  - edge cases
  - likely winners
  - structurally distinct candidates
  - candidates most likely to expose evaluator fragility
- Going beyond 5 requires an explicit reason in `research_plan.md`
- For structural screening runs that apply an empirical correction to compare against literature or DFT:
  - scale the reference set to the feasible-space size rather than hard-coding one count
  - for small finite spaces, prefer roughly **4–6 reference systems**
  - for medium spaces, prefer roughly **6–8 reference systems**
  - for larger or more heterogeneous spaces, prefer roughly **8–10 reference systems**
  - keep calibration to a modest fraction of a small finite catalog; do not spend a large share of the total benchmark budget on calibration alone
  - span at least **2 facets** and **2 site types** when the final search space spans multiple facets/sites
  - do not claim the correction is broadly valid outside the calibration scope

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
- Use literature, databases, and published values to orient the setup, identify descriptors, and calibrate the family, but do not reduce the final task to selecting the best row from an existing table when a local evaluator is feasible.
- For structural screening runs, prefer physically plausible structures over broader but metastable convenience families.
- When feasible, prefer a search space that can produce underexplored or non-tabulated candidates within a literature-grounded family rather than only replaying a fixed published candidate list.
- If the final oracle is a discrete lookup over tabulated candidates, prefer a search space that reflects that discrete structure rather than inventing an unjustified continuous interpolation scheme.
- If the feasible set is a filtered catalog of valid candidates, encode it explicitly rather than pretending the full factorized cross-product is valid.
- If you choose a continuous descriptor space, justify why interpolation or nearest-neighbor decoding is scientifically meaningful for that family.

## Engine Recommendation Guidance

Follow the `bo-init-run` heuristic:

- Prefer `botorch` when the search space is mostly or entirely categorical, the all-categorical candidate count is still modest enough to reason about (default threshold `<= 2000`), and evaluations are expensive enough that sample efficiency matters.
- Prefer `hebo` when the space is broader and more mixed numeric/categorical, when numeric dimensions dominate, or when there is no strong reason to bias toward BoTorch.
- If `hebo --hebo-model gp` looks numerically unstable on a mixed space, recommend `hebo --hebo-model rf` before abandoning HEBO entirely.

## Guardrails

- Do not overfit this process to HER or surface catalysis examples.
- Do not treat literature lookup tables as a live evaluator unless the user explicitly allows a lookup fallback.
- Do not choose a retrospective lookup evaluator just because it is convenient. Use it when the local executable path looks infeasible, unjustified, or clearly worse for the run's goal, and say so explicitly.
- Do not let external database integration consume the whole setup phase when the real goal is to decide whether a local executable evaluator can be built.
- Do not present a physics-inspired heuristic as if it were validated first-principles or experimental ground truth.
- Do not use narrow calibration data to justify strong comparative claims across out-of-scope facets, sites, or crystal structures.
- Do not claim a candidate exceeds a benchmark when the improvement is smaller than the stated calibration uncertainty.
- If you use metastable or convenience crystal structures (for example universal fcc slabs), label the run as screening-only and downgrade the claim posture accordingly.
- Use the evaluator evidence class to calibrate claims:
  - `validated`: stronger performance claims are acceptable, with normal caveats
  - `retrospective_lookup`: claims about ranking or recovery of known optima are acceptable, but not novelty beyond the tabulated space
  - `physics_inspired_heuristic`: keep claims cautious, hypothesis-like, and explicit about model simplifications
- Do not map the entire space during calibration.
- Do not start BO until the evaluator family, search space, and engine recommendation are stable enough to hand off.
- Do not assume a library is available; inspect the environment first.
