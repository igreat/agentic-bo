# Benchmarks

This directory defines the benchmark packaging and scoring model for the final
report.

The benchmark is intentionally **closed-world**:

- scored runs happen in a stripped public workspace
- the agent can only see public task bundles, prebuilt evaluator artifacts, and
  repo code/docs needed to run
- labeled datasets and answer keys stay outside that workspace
- web search is disabled by protocol during scored runs

## Current benchmark suite

Locked task direction:

1. `oer`
   - flagship simplex/composition benchmark
   - search-space-only init
   - local boxed literature
   - prebuilt proxy backend copied into the public workspace
2. `egfr_warm_start`
   - warm-start molecule benchmark
   - real seed observations
   - fixed descriptor representation
   - exact hidden lookup
3. reaction task
   - default: Doyle amidation fixed-nucleophile categorical exact lookup
   - fallback/alternative: Buchwald fixed-substrate categorical exact lookup
   - do not use Buchwald DRFP decode as the scored benchmark path

Today, only `oer` is fully bundled. The other two tasks are locked at the
protocol level but are not packaged yet.

## Root repo vs public workspace

The benchmark only needs one extra directory:

- the **root repo** remains the operator/developer environment
- a separate **public benchmark workspace** is built for the agent

The root repo may contain:

- labeled source datasets
- full development history and notes
- reusable `evaluation_backends/`
- benchmark planning docs

The public workspace should contain only:

- repo code/docs/skills needed to run the workflow
- `tasks/<task_id>/...`
- sanitized prebuilt backends under `evaluation_backends/`
- fresh `bo_runs/` and `research_runs/`

## Task bundle contract

Every public task bundle should include:

- `brief.md`
- `task_manifest.json`
- optional `search_space.json`
- optional `seed_observations.csv`
- optional `literature/`

The visible manifest should stay minimal:

- `task_id`
- objective target and direction
- budget
- init mode/path
- representation info if fixed
- literature mode/path
- optional seed observations path
- `evaluation.backend_id`

Public task bundles may expose the backend id of a prebuilt evaluator that is
already present inside the public workspace. They should not expose the source
dataset or where that backend was originally built.

## Build the public workspace

From a clean repo checkout:

```bash
uv run python benchmarks/build_workspace.py \
  --output-dir /tmp/agentic-bo-benchmark \
  --tasks oer
```

Then inside the built workspace:

```bash
uv sync
uv pip install --no-deps "hebo @ git+https://github.com/huawei-noah/HEBO.git#subdirectory=HEBO"
```

If a task manifest names `evaluation.backend_id`, the builder will copy
`evaluation_backends/<backend_id>/` into the public workspace when that backend
already exists in the root repo.

Build the backend in the root repo first when needed:

```bash
uv run python -m bo_workflow.cli build-oracle \
  --dataset data/caltech_oer/plate_3496.csv \
  --target overpotential_V \
  --objective min \
  --backend-id oer_hidden
```

Then build the workspace. The resulting public workspace can use direct
`run-evaluator --backend-id <BACKEND_ID>` calls against the copied backend.

## Scored run rules

- no web search
- use only local literature packets when present
- observations come only from the prebuilt evaluator in the public workspace
- no direct access to labeled source datasets
- no manual artifact editing before scoring

See `benchmarks/scoring.md` for metrics, workflow checks, and qualitative review
criteria. That scoring document is intended for repo and operator use; it does
not need to be copied into the public benchmark workspace.
