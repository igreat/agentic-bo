# BO Workflow

This repository now has two layers:

1. A generic BO engine that works directly from tabular datasets and builds a proxy oracle automatically.
2. Problem-specific scripts (like HER) that can still be used as examples/benchmarks.

The generic engine is designed to be Claude-skill ready: deterministic CLI commands, persisted run state, and JSON outputs.

## Setup

```bash
uv sync
uv pip install --no-deps "hebo @ git+https://github.com/huawei-noah/HEBO.git#subdirectory=HEBO"
```

> **Why `--no-deps`?** HEBO's published metadata pins ancient NumPy/pymoo
> versions that conflict with modern stacks. All of HEBO's real runtime
> dependencies (torch, gpytorch, numpy, pandas, scikit-learn) are already
> declared in this project's `pyproject.toml`, so skipping HEBO's own
> dependency resolution is safe.

## Layout

```text
configs/
  her.toml
experiments/
  run_her.py
src/bo_workflow/
  cli.py
  engine.py
  config.py
  experiment.py
  plotting.py
  runner.py
  problems/
    her/
```

## Core Engine (Dataset -> Oracle -> BO)

The core engine is exposed via:

```bash
uv run python -m src.bo_workflow.cli --help
```

Main commands:

- `init`: create a run from a dataset and infer design space
- `init-from-spec`: create a run directly from a JSON spec
- `init-from-prompt`: parse a plain-language prompt into a JSON spec and initialize
- `build-oracle`: train and persist a proxy oracle from dataset rows
- `suggest`: propose next candidate experiments
- `observe`: record objective values from real or simulated evaluations
- `evaluate-last`: score pending suggestions with the proxy oracle
- `run-proxy`: run an end-to-end simulated BO loop
- `auto-proxy-from-prompt`: one-command prompt -> spec -> oracle -> BO -> report
- `status`: show best-so-far and run metadata
- `report`: emit JSON report and convergence plot

Example:

```bash
uv run python -m src.bo_workflow.cli init \
  --dataset src/bo_workflow/problems/her/data/HER_virtual_data.csv \
  --target Target \
  --objective max

uv run python -m src.bo_workflow.cli build-oracle --run-id <RUN_ID>
uv run python -m src.bo_workflow.cli run-proxy --run-id <RUN_ID> --iterations 80
uv run python -m src.bo_workflow.cli report --run-id <RUN_ID>
```

JSON spec-driven start:

```bash
uv run python -m src.bo_workflow.cli init-from-spec --spec configs/run_spec.example.json
```

One-command prompt-driven run:

```bash
uv run python -m src.bo_workflow.cli auto-proxy-from-prompt \
  --dataset src/bo_workflow/problems/her/data/HER_virtual_data.csv \
  --prompt "maximize HER yield with proxy BO" \
  --iterations 20 \
  --batch-size 2
```

Artifacts are written under `runs/<RUN_ID>/`:

- `state.json`
- `intent.json` (optional, if intent was provided)
- `oracle.pkl`
- `oracle_meta.json`
- `suggestions.jsonl`
- `observations.jsonl`
- `convergence.pdf`
- `report.json`

## JSON-First Runtime Config

The engine runtime is JSON-first. `state.json` is the source of truth for each run.

- Use `configs/run_spec.example.json` as a template for run creation.
- Agent/user intent can be attached and saved to `runs/<RUN_ID>/intent.json`.
- Existing TOML configs remain supported for legacy scripted experiments.

## Claude Skills

Project-local skills are under `.claude/skills/` and map to the engine CLI:

- `bo-init-run`
- `bo-build-oracle`
- `bo-next-batch`
- `bo-record-observation`
- `bo-report-run`
- `bo-end-to-end-proxy`
- `bo-auto-from-prompt`

These provide the agent interface layer; the Python engine remains the deterministic execution layer.

## Run HER

```bash
uv run experiments/run_her.py
```

Or with an explicit config path:

```bash
uv run experiments/run_her.py --config configs/her.toml
```

## Config Format (TOML)

This section applies to legacy `experiments/run_*.py` scripts.

Use an `[experiment]` table and an optional `[problem]` table.

```toml
[experiment]
title = "HER Optimization"
num_iterations = 200
num_seeds = 16
num_initial_random_samples = 20
result_path = "results/her/HER_bo_results.npz"
plot_path = "results/her/HER_bo_results.pdf"
objective = "min"  # "min" or "max"
y_scale = "linear" # "linear" or "log"
show_plot = false
error_style = "stderr" # "stderr" or "iqr"

[problem]
oracle_impl = "random_forest"
```

## Adding Another Experiment

1. Add `src/bo_workflow/problems/<name>/builder.py`.
2. Add `configs/<name>.toml`.
3. Add `experiments/run_<name>.py` that:
   - loads TOML via `load_experiment_file`
   - builds the problem from `problem_kwargs`
   - calls `run_experiment`
