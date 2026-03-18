# Benchmark Buildout

This directory defines the benchmark packaging and scoring model for the final
report.

The benchmark is intentionally **closed-world**:

- scored runs happen in a stripped public workspace
- the agent can only see public task bundles and repo code/docs needed to run
- hidden evaluators, labeled datasets, and answer keys stay outside that
  workspace
- web search is disabled by protocol during scored runs

## Current benchmark suite

Locked task direction:

1. `oer`
   - flagship simplex/composition benchmark
   - search-space-only init
   - local boxed literature
   - hidden proxy evaluator behind an opaque handle
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

## Public / private split

### Public benchmark workspace

Visible to the agent:

- repo code/docs/skills needed to run the workflow
- `benchmark_tasks/<task_id>/...`
- fresh `bo_runs/` and `research_runs/`
- the benchmark evaluator wrapper at `benchmarks/run_task_evaluator.py`

### Private operator side

Hidden from the agent:

- labeled source datasets
- `evaluation_backends/`
- exact lookup tables / answer keys
- evaluator handle map
- scoring sheets and benchmark logs

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
- `evaluation.handle`

The handle must stay opaque. Public task bundles must not expose raw backend ids
or backend file paths.

## Materialize a public workspace

From a clean repo checkout:

```bash
uv run python benchmarks/materialize_workspace.py \
  --output-dir /tmp/agentic-bo-benchmark \
  --tasks oer
```

Then inside the materialized workspace:

```bash
uv sync
uv pip install --no-deps "hebo @ git+https://github.com/huawei-noah/HEBO.git#subdirectory=HEBO"
```

## Hidden evaluator setup

The public evaluator wrapper resolves an opaque handle through operator-owned
environment variables:

```bash
export BENCHMARK_HANDLE_MAP=/abs/path/to/handle_map.json
export BENCHMARK_BACKENDS_ROOT=/abs/path/to/evaluation_backends
```

Optional:

```bash
export BENCHMARK_RUNS_ROOT=/abs/path/to/public_workspace/bo_runs
```

The handle map stays private. A minimal example lives in
operator-side config such as:

```json
{
  "oer_v1": {
    "backend_id": "oer_hidden"
  }
}
```

That file should stay outside the public benchmark workspace.

## Scored run rules

- no web search
- use only local literature packets when present
- observations come only from the hidden evaluator
- no direct access to hidden datasets or evaluator assets
- no manual artifact editing before scoring

See `benchmarks/scoring.md` for metrics, workflow checks, and qualitative review
criteria. That scoring document is intended for repo and operator use; it does
not need to be copied into the public benchmark workspace.
