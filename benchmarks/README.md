# Benchmarks

This directory holds the benchmark setup for the final report.

Right now, only one task is fully packaged:

- `oer`: the flagship OER composition benchmark

The benchmark model is intentionally simple:

- the **root repo** is the operator/developer environment
- a separate **built public workspace** is where the agent runs
- labeled source datasets stay in the root repo
- the public workspace contains only:
  - code/docs/skills needed to run
  - `tasks/<task_id>/...`
  - prebuilt evaluator assets under `evaluation_backends/`
  - fresh `bo_runs/` and `research_runs/`

## OER quick start

This is the main thing you should follow if you just want to run the benchmark.

### 1. In the root repo, install dependencies

```bash
uv sync
uv pip install --no-deps "hebo @ git+https://github.com/huawei-noah/HEBO.git#subdirectory=HEBO"
```

### 2. In the root repo, build the OER backend

```bash
uv run python -m bo_workflow.cli build-oracle \
  --dataset data/caltech_oer/plate_3496.csv \
  --target overpotential_V \
  --objective min \
  --backend-id oer_hidden
```

This creates:

- `evaluation_backends/oer_hidden/oracle.pkl`
- `evaluation_backends/oer_hidden/oracle_meta.json`

### 3. In the root repo, build the public workspace

```bash
uv run python benchmarks/build_workspace.py \
  --output-dir /tmp/agentic-bo-benchmark \
  --tasks oer \
  --overwrite
```

This also writes a benchmark-specific `.claude/settings.local.json` inside the
built workspace so Claude Code can run shell commands and write artifacts
without approval prompts while keeping Claude-native web/search disabled.

### 4. Switch into the built workspace and install dependencies there too

```bash
cd /tmp/agentic-bo-benchmark
uv sync
uv pip install --no-deps "hebo @ git+https://github.com/huawei-noah/HEBO.git#subdirectory=HEBO"
```

You need to install here as well because the built workspace is a separate working directory with its own environment.

### 5. Manual smoke run (skip to [6. Full agent run](/benchmarks/README.md#6-full-agent-run) if you want to jump straight to the agentic run)

Initialize a BO run from the public task bundle:

```bash
uv run python -m bo_workflow.cli init \
  --search-space-json tasks/oer/search_space.json \
  --target overpotential_V \
  --objective min \
  --simplex-groups 'Mn_molar_fraction,Fe_molar_fraction,Co_molar_fraction,Ni_molar_fraction,La_molar_fraction,Ce_molar_fraction:1' \
  --seed 42
```

Copy the returned `run_id`, then run the evaluator:

```bash
uv run python -m bo_workflow.cli run-evaluator \
  --run-id <RUN_ID> \
  --backend-id oer_hidden \
  --iterations 100 \
  --batch-size 1
```

Finish with:

```bash
uv run python -m bo_workflow.cli report --run-id <RUN_ID>
```

Artifacts will be written under `bo_runs/<RUN_ID>/`.

### 6. Full agent run

Run the agent from inside `/tmp/agentic-bo-benchmark` and point it at:

- `tasks/oer/brief.md`
- `tasks/oer/task_manifest.json`
- `tasks/oer/search_space.json`
- `tasks/oer/literature/`

Example initial prompt:

```text
Use the research-agent workflow with the benchmark task bundle at `tasks/oer/`.

This is a closed-world benchmark run. Do not use web search.

Please execute the full workflow end to end using the task bundle and draft the final report or paper.

Treat this as a scored benchmark run and do not pause between phases unless blocked.
```

## What `build_workspace.py` does

`benchmarks/build_workspace.py` copies a stripped set of files into the public
workspace:

- root files:
  - `AGENTS.md`
  - `README.md`
  - `pyproject.toml`
  - `uv.lock`
  - `.python-version`
  - `.gitignore`
- root directories:
  - `bo_workflow/`
  - `.agents/`
  - `.claude/`
- a benchmark-specific `.claude/settings.local.json` with:
  - `defaultMode: acceptEdits`
  - `permissions.allow: [Bash]`
  - `permissions.deny: [WebSearch, WebFetch]`
- selected benchmark task bundles from `benchmarks/tasks/`
- any prebuilt backend named by `evaluation.backend_id` in the task manifest,
  if it already exists under root `evaluation_backends/`

It also creates empty:

- `bo_runs/`
- `research_runs/`

## Task bundle shape

Each public task bundle may include:

- `brief.md`
- `task_manifest.json`
- optional `search_space.json`
- optional `seed_observations.csv`
- optional `literature/`

The manifest may also declare the intended workflow entrypoint, e.g.
`workflow.entrypoint = research-agent`.

For `oer`, the task bundle lives at:

- `benchmarks/tasks/oer` in the root repo

## Scored run rules

- no web search
- use only local literature packets when present
- observations come only from the prebuilt evaluator in the public workspace
- no direct access to labeled source datasets
- no manual artifact editing before scoring

See `benchmarks/scoring.md` for the metrics and review rubric used in the
report.
