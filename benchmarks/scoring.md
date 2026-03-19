# Benchmark Scoring

This document locks the evidence package for the final report.

## Main evaluation split

The project now has two evidence layers:

- **open-world AI-scientist runs**: the headline story
- **closed-world benchmark runs**: supporting/control evidence

For open-world runs:

- the prompt and nudge tier are fixed
- the operator keeps a hidden answer key
- the execution path is flexible
- scoring is based on the required evidence package

See [`open_world.md`](/Users/mujtabaalajmi/Documents/agentic-bo/bo-fun/benchmarks/open_world.md)
for the operator-side model.

## Main benchmark comparison

Primary scored comparison:

- full `research-agent`
- `research-agent` with `--engine random`

This isolates whether BO adds value **inside** the full research workflow.

## Support benchmark comparison

BO-only support evidence:

- raw BO
- random search

Use this only as support evidence for optimizer quality. It is not the headline
contribution.

## Open-world evidence checklist

For every scored open-world run, require:

- exact initial prompt saved to `initial_prompt.md`
- nudge tier recorded
- credible source URLs recorded
- final evaluator module exists
- final search-space artifact exists
- verification artifact exists
- helper scripts and dependency installs are recorded if used
- setup marked frozen before the reported BO run
- normal `bo_runs/<run_id>/` artifacts exist
- final paper/report explains the discovery and operationalization path

## Task budgets

Only `oer` is fully packaged in this PR. The `egfr_warm_start` and reaction
task sections below are planned benchmark targets so their intended budgets are
predeclared for follow-up work.

### `oer`

- init mode: explicit `search_space.json`
- objective: minimize `overpotential_V`
- simplex constraint over all six molar fractions
- iterations: `100`
- batch size: `1`
- initial random suggestions: `10`

### `egfr_warm_start`

- warm-start seed file: `egfr_seed50_mixed.csv`
- representation: fixed molecule descriptor pipeline
- new evaluations: `40`
- batch size: `4`

### reaction task

- default: Doyle amidation fixed-nucleophile categorical exact lookup
- initial budget: `8-10` evaluations
- keep this task small and clean; do not use it as the headline BO stress test

## Optimization metrics

For each scored task report:

- best-so-far value under budget
- absolute gap to hidden optimum
- percentile rank of the best found point
- normalized improvement over the initial random phase or seed set

Interpretation guidance:

- `oer`: compare against the hidden optimum and dataset percentile
- EGFR: compare against the best candidate in the hidden pool and improvement
  beyond the provided seed set
- reaction task: emphasize early efficiency under tight budget

## Workflow correctness checklist

Score each item `pass` / `fail`:

- valid setup created from the public task bundle
- hard constraints respected
- required BO artifacts produced
- required research workflow artifacts produced under `research_runs/`
- prebuilt evaluator used without exposing labeled source datasets
- run completed without manual repair

For open-world runs, also check:

- initial prompt saved exactly
- final evaluator/search space/verification artifact exist
- operationalization log exists and has a valid event structure
- any helper scripts and dependency installs are recorded if used
- the reported BO run corresponds to a frozen final setup

## Qualitative review rubric

Use a `0/1/2` rubric on four axes, with two-person consensus.

### Problem framing

- `0`: objective or constraints are unclear or wrong
- `1`: mostly correct framing, but with missing nuance
- `2`: clear, correct, and well-scoped framing

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

## Benchmark integrity notes

The report should state:

- scored runs used a stripped public workspace
- web search was disabled during scored runs
- labeled source datasets stayed outside the public workspace
- prebuilt evaluator assets were fixed before scoring
- local literature packets were frozen in advance
- task prompts, budgets, and backend ids were fixed before scoring

For open-world runs, state instead:

- the initial prompt and nudge tier were fixed before scoring
- a hidden operator spec defined the intended evaluator family, design-space
  family, constraints, and verification check
- the agent was allowed to browse, write ad hoc scripts, and install minimal
  dependencies
- benchmarkability came from fixed evidence requirements rather than a fixed
  execution path

Also state the limitations:

- hidden evaluation is retrospective
- boxed literature may reflect curator bias
- qualitative scoring is manual
- only a small number of end-to-end tasks are benchmarked
