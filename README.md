# Bayesian Optimisation Workflow

A practical BO workflow for scientific discovery in chemistry.

This repository is intended to be an **agent-operable optimization engine**:

- define an optimization problem,
- build or plug in an objective evaluator (real experiment or proxy oracle),
- run iterative BO suggestions,
- track state and results for human-in-the-loop workflows.

One thing missing is a **data conversion and preprocessing layer** which would make this much more flexible to different problem types. This is the next major piece of work.

## Scope (current vs target)

- **Current MVP:** single-objective BO from tabular datasets with persisted run state and JSON CLI.
- **Target direction:** richer problem adapters (constraints/compositions/encodings), so different chemistry problems can be transformed into a common BO interface.

## Setup

```bash
uv sync
uv pip install --no-deps "hebo @ git+https://github.com/huawei-noah/HEBO.git#subdirectory=HEBO"
```

> **Why `--no-deps`?** HEBO's published metadata pins ancient NumPy/pymoo
> versions that conflict with modern stacks. This project's `pyproject.toml`
> already declares the real runtime dependencies, so skipping HEBO's own
> dependency resolution is safe.

## Quick start

```bash
uv run python -m bo_workflow.cli init \
  --dataset data/HER_virtual_data.csv \
  --target Target --objective max --seed 42

# grab the run_id from the JSON output, then:
uv run python -m bo_workflow.cli build-oracle --run-id <RUN_ID>
uv run python -m bo_workflow.cli run-proxy --run-id <RUN_ID> --iterations 20
uv run python -m bo_workflow.cli report --run-id <RUN_ID>
```

## CLI commands

```bash
uv run python -m bo_workflow.cli --help
```

or install/use as a program entrypoint:

```bash
uv run bo-workflow --help
```

| Command | Purpose |
|---------|---------|
| `init` | Create a run from a CSV dataset |
| `build-oracle` | Train a proxy oracle from dataset rows |
| `suggest` | Propose next candidate experiments |
| `observe` | Record objective values (real or simulated) |
| `run-proxy` | Run an end-to-end simulated BO loop |
| `status` | Show best-so-far and run metadata |
| `report` | Generate JSON report and convergence plot |
| `screen` | One-command screening from a SMILES dataset |
| `smiles-discovery` | Discovery pipeline from one SMILES or a SMILES file |
| `egfr-ic50-global` | Global EGFR IC50 experiment (train/lookup split + prescreen loop) |

Add `--verbose` to `init`, `build-oracle`, `suggest`, `observe`, `run-proxy`, and `report` to print progress logs (and a tqdm bar for `run-proxy`).

Engine options: `hebo` (default), `bo_lcb`, `random`. Set once at init with `--engine`.

## SMILES-file discovery (EGFR-style)

Run from one SMILES or a `.smi/.txt/.csv` SMILES file:

```bash
uv run bo-workflow smiles-discovery \
  --smiles-file data/my_seeds.smi \
  --dataset data/egfr_quinazoline.csv \
  --target pIC50 --objective max \
  --iterations 20 --top-k 10
```

Output is JSON with `run_id`, `oracle_cv_rmse`, and ranked `top_candidates`.
All returned scores are proxy-oracle simulations.

Unified EGFR global experiment entry:

```bash
uv run bo-workflow egfr-ic50-global \
  --dataset data/egfr_ic50.csv \
  --split-mode auto --seed-count 50 \
  --iterations 30 --rounds 20 --experiments-per-round 4
```

## Compare optimizers (demo)

For a single chart comparing `hebo`, `bo_lcb`, and `random`, run:

```bash
uv run python scripts/compare_optimizers.py \
  --dataset data/HER_virtual_data.csv \
  --target Target --objective max \
  --iterations 20 --batch-size 1 --repeats 1
```

Outputs:

- plot: `results/compare/optimizers.pdf`
- summary: `results/compare/optimizers_summary.json`

## Run artifacts

Each run writes to `runs/<RUN_ID>/`:

`state.json`, `oracle.pkl`, `oracle_meta.json`, `suggestions.jsonl`, `observations.jsonl`, `convergence.pdf`, `report.json`

## Design notes

- The engine is replay-first: it rebuilds optimizer state from logged observations. This makes runs easy to resume and audit.
- Proxy mode is a simulation workflow. Always present results as simulated outcomes and include oracle CV RMSE.
- `data/HER_virtual_data.csv` is included as an example dataset only. In real usage, users should provide problem-specific context (target meaning, constraints, objective direction, and valid operating domain).

## Layout

```text
bo_workflow/
  engine.py       # BOEngine — suggest/observe loop, no oracle knowledge
  engine_cli.py   # CLI subcommands: init, suggest, observe, status, report
  oracle.py       # standalone proxy oracle — train, load, predict on run_dir
  oracle_cli.py   # CLI subcommands: build-oracle, run-proxy
  workflows/      # domain workflows (egfr_ic50_global, etc.)
  cli.py          # top-level entrypoint — composes subparsers from each module
  plotting.py     # convergence plot generation
  utils.py        # RunPaths, JSON I/O, shared types
  observers/
    base.py       # Observer ABC — evaluate(suggestions) interface
    proxy.py      # ProxyObserver — self-contained, captures run_dir at init
    callback.py   # CallbackObserver — delegates to user callback
data/
  HER_virtual_data.csv  # example dataset (HER virtual screen)
scripts/
  compare_optimizers.py        # benchmark hebo/bo_lcb/random
  egfr_ic50_global_experiment.py  # thin wrapper -> bo_workflow.workflows.egfr_ic50_global
  egfr_smiles_discovery.py        # thin wrapper -> bo_workflow cli smiles-discovery
.claude/
  skills/         # Claude Code skills mapping to CLI commands
```

## Claude Skills

Skills in `.claude/skills/` provide the agent interface:

- `bo-init-run` — initialize a run
- `bo-build-proxy-oracle` — train proxy oracle
- `bo-next-batch` — suggest candidates
- `bo-record-observation` — record results
- `bo-report-run` — status and reports
- `bo-end-to-end-proxy` — full automated loop

## Credits

Much of the underlying HEBO and problem specific part of the code is taken from/inspired from [BO-Tutorial-for-Sci](https://github.com/zwyu-ai/BO-Tutorial-for-Sci).
