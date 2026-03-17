# Benchmark Scoring

This document locks the evidence package for the final report.

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

## Task budgets

### `plate3496`

- init mode: explicit `search_space.json`
- objective: minimize `overpotential_V`
- simplex constraint over all six molar fractions
- iterations: `60`
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

- `plate3496`: compare against the hidden optimum and dataset percentile
- EGFR: compare against the best candidate in the hidden pool and improvement
  beyond the provided seed set
- reaction task: emphasize early efficiency under tight budget

## Workflow correctness checklist

Score each item `pass` / `fail`:

- valid setup created from the public task bundle
- hard constraints respected
- required artifacts produced
- hidden evaluation used without leaking raw evaluator assets
- run completed without manual repair

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
- hidden evaluators stayed outside the public workspace
- local literature packets were frozen in advance
- task prompts, budgets, and evaluator handles were fixed before scoring

Also state the limitations:

- hidden evaluation is retrospective
- boxed literature may reflect curator bias
- qualitative scoring is manual
- only a small number of end-to-end tasks are benchmarked
