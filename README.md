# BO Workflow

This repository is a lightweight, script-first framework for running Bayesian optimization experiments with HEBO, BO (LCB), and random search baselines.  
Each experiment is run explicitly from `experiments/run_<name>.py`, with parameters defined in TOML files under `configs/`.  
Core workflow logic lives in `src/bo_workflow/` and is shared across problems, while problem-specific setup lives in `src/bo_workflow/problems/`.

## Layout

```text
configs/
  her.toml
experiments/
  run_her.py
src/bo_workflow/
  config.py
  experiment.py
  plotting.py
  runner.py
  problems/
    her/
```

## Run HER

```bash
uv run experiments/run_her.py
```

Or with an explicit config path:

```bash
uv run experiments/run_her.py --config configs/her.toml
```

## Config Format (TOML)

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
