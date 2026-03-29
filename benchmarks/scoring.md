# Benchmark Scoring

This document locks the primary evidence package for the final report and the scoring rules for supplementary support evidence.

## Core evidence package

The required primary evidence package is four runs:

- `oer` skilled baseline: explicit `/research-agent`
- `oer` naive baseline: plain Claude Code, no explicit research-layer skill invocation
- `her_live_structural` skilled baseline: explicit `/research-agent`
- `her_live_structural` naive baseline: plain Claude Code, no explicit research-layer skill invocation

This package is intentionally split:

- `oer` is the formal scored benchmark
- `her_live_structural` is the autonomy/scientific case study

After these four runs are complete, additional open-world support evidence may be added. It is not required for the report core.

An optional support ablation may also include:

- `her_live_structural` lightly nudged baseline: soft workflow cue in the initial prompt, but no explicit research-layer skill invocation
- `her_live_structural` interactive rescue trace: one or more manual mid-conversation nudges, logged explicitly as operator intervention
- `hea_live_structural` paired reruns: supplementary open-world comparisons judged with the same artifact-grounded procedure

## Main comparison

Primary comparison:

- full `research-agent` workflow
- naive Claude Code with the same model family, effort level, workspace, task materials, and budget

This isolates the value of the orchestration/skill layer, not just the model.

## Support comparisons

Support evidence may still include:

- BO engine quality (`hebo` / `botorch` / `random`) inside the packaged benchmark
- optimizer-only comparisons such as raw BO versus random search

Use these only as support evidence. They are not the primary claim.

## Baseline definitions

### Skilled baseline

- explicit `/research-agent` invocation is allowed
- for packaged benchmark runs such as `oer`, the workspace uses `skill_profile=full`
- for open-world runs such as `her_live_structural`, use a fresh clean repo workspace from the same branch/code state as the paired naive run
- project skills and workflow artifacts are part of the tested system

### Naive baseline

- no explicit research-layer skill invocation
- no research-layer slash commands such as `/research-agent`, `/literature-review`, or `/scientific-writing`
- for packaged benchmark runs such as `oer`, the workspace uses `skill_profile=bo_only`
- for open-world runs such as `her_live_structural`, use a fresh clean repo workspace from the same branch/code state as the paired skilled run
- same task bundle or repo state, same budget, and same evaluator constraints as the skilled baseline

The naive baseline may still use Claude Code's native capabilities, any BO engine helpers intentionally left available in the workspace, and the engine-level BO documentation surface. For open-world runs, ordinary project files and skill trees may still be present on disk unless the run protocol says otherwise. The intent is to remove explicit research-layer orchestration without hiding the BO engine or artificially weakening the model.

### Lightly nudged baseline

- no explicit research-layer skill invocation
- obeys the same no-research-layer-slash-command rule as the naive baseline
- uses the same workspace type as the corresponding naive baseline
- initial prompt may reference the intended workflow or artifact structure, but must not tell the model to call `/research-agent` or other research-layer slash commands
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
- effort level: keep fixed across compared runs
- initial random suggestions: `10`
- evaluation mode: prebuilt hidden backend
- web search: disabled

### `her_live_structural`

- objective: no predefined single scalar target; the run must choose and document a scientifically defensible optimization target, often `abs_delta_g_h`
- evaluator: live local structural evaluator
- evaluation mode: open-world local Python evaluator
- web search: allowed if the prompt permits it
- budget: no fixed iteration or query count is imposed by the prompt pack
- comparison policy: for a given naive-vs-skilled pair, either let both runs continue until they explicitly declare completion under the same stop policy, or apply the same external wall-clock cap and record it explicitly
- effort level: keep fixed across skilled and naive runs for a given comparison pair
- workspace state: fresh clean repo workspaces for both naive and skilled runs
- conversation state: both runs should start from a fresh empty chat
- operator intervention: none for the core four runs
- stop policy: let the run continue until the agent explicitly declares the workflow complete; if it is stopped early for clear stalling or looping, score it as incomplete and record the reason

Do not claim that HER is a hidden-optimum benchmark. It is a case study with a different scoring model.

If interactive rescue is studied on HER, keep the trigger policy simple and disclose it. Example acceptable triggers:

- no valid evaluator path chosen after a fixed amount of time
- repeated drift into lookup-oracle behavior after the prompt forbids it
- no BO run or research artifact created after a fixed amount of time

Do not compare an interactively rescued HER run directly against the clean skilled-vs-naive primary result as if they were the same condition.

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

## Two-layer judging model

Use two different judging layers and keep them separate:

1. **Deterministic artifact checks**
   - workflow correctness checklist
   - OER numeric metrics
   - artifact existence and path checks
   - factual verification that paper claims match `report.json` and `state.json`
2. **LLM-primary qualitative review**
   - problem framing
   - workflow fidelity as reflected in artifacts
   - interpretation quality
   - caveat honesty
   - paper usefulness
   - HER-specific evaluator legitimacy, search-space validity, and scientific setup quality

The qualitative layer should be judged primarily by an LLM using grounded artifact review, not by free-form human impressions. Human adjudication remains optional for disputed cases or final tie-breaks.

For the primary paper comparison, use:

- the final paper artifact (`paper.tex` or `paper.md`) as the main judged output
- `report.json` and `state.json` as required grounding artifacts
- `research_state.json` and calibration artifacts as optional supporting context when present

Do **not** use the raw full conversation transcript as the primary judging input. It is too noisy, too long, and too sensitive to irrelevant interaction details. If workflow trace evidence is needed, prefer `research_plan.md` or `research_state.json` over the raw chat/tool transcript.

For the core naive-vs-skilled comparisons, prefer **pairwise LLM judging** over isolated scoring when the goal is to decide which output is stronger. Single-run judging is still useful for archival records and sanity checks, but pairwise comparison is the primary qualitative comparison mode.

## Qualitative review rubric

Use a grounded `0/1/2` rubric with LLM-primary scoring and optional human adjudication when needed.

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

Present this rubric honestly as structured qualitative review, not as a hard scientific metric. Every LLM score should be anchored to the supplied artifacts with a brief justification and a direct evidence quote.

## HER case-study rubric

Do not score HER primarily by hidden-optimum metrics. Instead, score it using the checklist above plus the following case-study axes:

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

The current expected bar for HER is that a good run may count as `demo-quality success` without automatically qualifying as `benchmark-quality success`.

## Benchmark integrity notes

The report should state:

- scored OER runs used a stripped public workspace
- compared runs used the same model family and effort level
- web search was disabled during scored OER runs
- labeled source datasets stayed outside the public workspace
- prebuilt evaluator assets were fixed before scoring
- task prompts, budgets, and backend ids were fixed before scoring
- HER runs were evaluated as open-world case studies, not hidden-optimum benchmark runs
- any lightly nudged support runs were analyzed as ablations, not as the primary comparison
- any interactive rescue traces were reported as operator interventions, not as clean benchmark baselines

Also state the limitations:

- OER hidden evaluation is retrospective
- HER case-study quality depends on evaluator legitimacy and setup quality
- qualitative scoring is LLM-judged with optional human adjudication
- only a small number of end-to-end tasks are benchmarked
