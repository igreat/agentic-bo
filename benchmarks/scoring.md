# Benchmark Scoring

This document locks the evidence package for the final report.

## Core evidence package

The required evidence package is four runs:

- `oer` skilled baseline: explicit `/research-agent`
- `oer` naive baseline: plain Claude Code, no explicit skill invocation
- `her_live_structural` skilled baseline: explicit `/research-agent`
- `her_live_structural` naive baseline: plain Claude Code, no explicit skill invocation

This package is intentionally split:

- `oer` is the formal scored benchmark
- `her_live_structural` is the autonomy/scientific case study

If time remains after these four runs are complete, a third case may be added as
support evidence. It is not required for the report core.

An optional support ablation may also include:

- `her_live_structural` lightly nudged baseline: soft workflow cue in the initial prompt, but no explicit skill invocation
- `her_live_structural` interactive rescue trace: one or more manual mid-conversation nudges, logged explicitly as operator intervention

## Main comparison

Primary headline comparison:

- full `research-agent` workflow
- naive Claude Code with the same model family, workspace, task materials, and budget

This isolates the value of the orchestration/skill layer, not just the model.

## Support comparisons

Support evidence may still include:

- BO engine quality (`hebo` / `botorch` / `random`) inside the packaged benchmark
- optimizer-only comparisons such as raw BO versus random search

Use these only as support evidence. They are not the headline claim.

## Baseline definitions

### Skilled baseline

- explicit `/research-agent` invocation is allowed
- benchmark workspace uses `skill_profile=full`
- project skills and workflow artifacts are part of the tested system

### Naive baseline

- no explicit skill invocation
- no custom project slash commands
- benchmark workspace uses `skill_profile=bo_only`
- same task bundle, same budget, and same evaluator constraints as the skilled baseline

The naive baseline may still use Claude Code's native capabilities and the
engine-level BO documentation surface. The point is to remove the explicit
research-layer orchestration, not to cripple the model or hide the BO engine.

### Lightly nudged baseline

- no explicit skill invocation
- no custom project slash commands
- uses the same workspace type as the corresponding naive baseline
- initial prompt may reference the intended workflow or artifact structure
- primarily used as an ablation on open-world case studies such as HER

### Interactive rescue trace

- starts from the naive or lightly nudged baseline
- includes one or more manual mid-conversation nudges from the operator
- every intervention should be logged with:
  - approximate timing or trigger condition
  - exact text of the nudge
  - reason for intervening
- treat this as qualitative support evidence, not as a primary benchmark condition

## Task budgets

### `oer`

- init mode: explicit `search_space.json`
- objective: minimize `overpotential_V`
- simplex constraint over all six molar fractions
- iterations: `100`
- batch size: `1`
- initial random suggestions: `10`
- evaluation mode: prebuilt hidden backend
- web search: disabled

### `her_live_structural`

- objective: minimize `abs_delta_g_h`
- evaluator: live local structural evaluator
- evaluation mode: open-world local Python evaluator
- web search: allowed if the prompt permits it
- budget: keep fixed across skilled and naive runs for a given comparison pair

Do not claim that HER is a hidden-optimum benchmark. It is a case study with a
different scoring model.

If interactive rescue is studied on HER, keep the trigger policy simple and
disclose it. Example acceptable triggers:

- no valid evaluator path chosen after a fixed amount of time
- repeated drift into lookup-oracle behavior after the prompt forbids it
- no BO run or research artifact created after a fixed amount of time

Do not compare an interactively rescued HER run directly against the clean
skilled-vs-naive headline result as if they were the same condition.

## OER benchmark metrics

For each scored `oer` run report:

- best-so-far value under budget
- absolute gap to hidden optimum
- percentile rank of the best found point
- normalized improvement over the initial random phase

Interpretation guidance:

- compare skilled vs naive directly under the same budget
- present BO-only engine comparisons separately from the skilled-vs-naive result
- treat run completion and workflow correctness as part of the score, not only the final optimum found

## Workflow correctness checklist

Score each item `pass` / `fail`:

- valid setup created from the public task bundle
- hard constraints respected
- required BO artifacts produced
- required research workflow artifacts produced under `research_runs/`
- prebuilt evaluator used without exposing labeled source datasets
- run completed without manual repair

For `her_live_structural`, replace the evaluator item with:

- live local structural evaluator was used without silent fallback to lookup, retrospective dataset, or hand-written heuristic oracle

## Qualitative review rubric

Use a `0/1/2` rubric with two-person consensus when possible.

### Problem framing

- `0`: objective or constraints are unclear or wrong
- `1`: mostly correct framing, but with missing nuance
- `2`: clear, correct, and well-scoped framing

### Workflow fidelity

- `0`: obvious workflow misuse, manual repair dependence, or broken orchestration
- `1`: mostly correct workflow, but with notable inconsistencies or recoveries
- `2`: clean phase progression with coherent artifacts and no avoidable repair loops

### Interpretation quality

- `0`: unsupported or confused conclusions
- `1`: partially grounded interpretation with some weak claims
- `2`: artifact-grounded interpretation that explains the result clearly

### Caveat honesty

- `0`: overclaims or hides important uncertainty
- `1`: mentions some caveats but misses key limitations
- `2`: states major limitations clearly and proportionately

### Paper usefulness

- `0`: poor structure or not useful as a research summary
- `1`: serviceable but thin or inconsistent
- `2`: clear, readable, and useful as a concise report draft

Present this rubric honestly as structured qualitative review, not as a hard
scientific metric.

## HER case-study rubric

Do not score HER primarily by hidden-optimum metrics. Instead, score it using
the checklist above plus the following case-study axes:

### Evaluator legitimacy

- `0`: evaluator falls back to lookup, retrospective oracle, or non-live heuristic
- `1`: evaluator is partly legitimate but has important validity gaps
- `2`: evaluator is clearly live/local, with limitations stated honestly

### Search-space validity

- `0`: invalid candidates or impossible combinations dominate the run
- `1`: mostly valid, but notable avoidable failures or thin calibration scope remain
- `2`: search space is feasible, pre-pruned where needed, and aligned with the evaluator

### Scientific setup quality

- `0`: calibration, literature context, or constraints are weak enough to undermine interpretation
- `1`: useful setup with real caveats
- `2`: setup is coherent, well-justified, and matched to the claims made

Use the HER rubric to place each run into one of three outcome categories:

- `fail`
- `demo-quality success`
- `benchmark-quality success`

The current expected bar for HER is that a good run may count as
`demo-quality success` without automatically qualifying as `benchmark-quality success`.

## Benchmark integrity notes

The report should state:

- scored OER runs used a stripped public workspace
- web search was disabled during scored OER runs
- labeled source datasets stayed outside the public workspace
- prebuilt evaluator assets were fixed before scoring
- task prompts, budgets, and backend ids were fixed before scoring
- HER runs were evaluated as open-world case studies, not hidden-optimum benchmark runs
- any lightly nudged support runs were analyzed as ablations, not as the main headline comparison
- any interactive rescue traces were reported as operator interventions, not as clean benchmark baselines

Also state the limitations:

- OER hidden evaluation is retrospective
- HER case-study quality depends on evaluator legitimacy and setup quality
- qualitative scoring is manual
- only a small number of end-to-end tasks are benchmarked
