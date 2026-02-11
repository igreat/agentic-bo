# Bayesian Optimisation Workflow

Practical HEBO-based BO workflow for scientific discovery in chemistry. The system trains a proxy oracle from tabular data and runs BO against it. All CLI output is JSON. All run state lives under `runs/<run_id>/`.

## Setup

```bash
uv sync
uv pip install --no-deps "hebo @ git+https://github.com/huawei-noah/HEBO.git#subdirectory=HEBO"
```

HEBO's published metadata pins ancient NumPy/pymoo versions. Install it with `--no-deps`; this project's `pyproject.toml` already declares the real runtime dependencies.

## Architecture

```
src/bo_workflow/
  engine.py    # BOEngine class — all logic, deterministic, JSON-in/JSON-out
  cli.py       # argparse CLI wrapping BOEngine methods
  plotting.py  # convergence plot generation
data/
  HER_virtual_data.csv  # example dataset (HER virtual screen)
```

Skills in `.claude/skills/` map 1:1 to CLI subcommands. The engine is the source of truth; skills are the agent interface.

## Script-first policy

- Before writing ad-hoc one-off scripts, check `scripts/` and prefer existing scripts when they already cover the task.
- For explicit optimizer benchmarking/comparison requests, use:

```bash
uv run python scripts/compare_optimizers.py \
  --dataset data/HER_virtual_data.csv \
  --target Target --objective max \
  --iterations 20 --batch-size 1 --repeats 3 --verbose
```

- Only create a new script if no existing command/script fits the request. If creating one, keep it reusable and place it under `scripts/`.

## Run artifacts

Each run produces files under `runs/<run_id>/`:

| File | Created by |
|------|-----------|
| `state.json` | `init` |
| `intent.json` | `init` (when `--intent-json` is provided) |
| `oracle.pkl` | `build-oracle` |
| `oracle_meta.json` | `build-oracle` |
| `suggestions.jsonl` | `suggest` / `run-proxy` |
| `observations.jsonl` | `observe` / `run-proxy` |
| `convergence.pdf` | `report` / `run-proxy` |
| `report.json` | `report` / `run-proxy` |

## CLI quick reference

All commands: `uv run python -m src.bo_workflow.cli <command> [flags]`

| Command | Key flags | Purpose |
|---------|-----------|---------|
| `init` | `--dataset --target --objective` (req), `--engine --seed --init-random --batch-size` (opt) | Init run from CSV |
| `build-oracle` | `--run-id` (req), `--cv-folds --max-features` (opt) | Train proxy oracle |
| `suggest` | `--run-id` (req), `--batch-size` (opt) | Propose next candidates |
| `observe` | `--run-id --data` (req) | Record real/simulated results |
| `run-proxy` | `--run-id --iterations` (req), `--batch-size` (opt) | Full proxy BO loop |
| `status` | `--run-id` (req) | Quick run summary |
| `report` | `--run-id` (req) | Full report + convergence plot |

Engine options: `hebo` (default), `bo_lcb`, `random`. Note: `bo_lcb` currently supports batch-size 1 only.

## MVP demo (copy-paste)

```bash
uv run python -m src.bo_workflow.cli init \
  --dataset data/HER_virtual_data.csv \
  --target Target --objective max --seed 42

# grab the run_id from the JSON output, then:
uv run python -m src.bo_workflow.cli build-oracle --run-id <RUN_ID>
uv run python -m src.bo_workflow.cli run-proxy --run-id <RUN_ID> --iterations 20
```

Expected artifacts in `runs/<RUN_ID>/`: `state.json`, `oracle.pkl`, `oracle_meta.json`, `suggestions.jsonl`, `observations.jsonl`, `convergence.pdf`, `report.json`.

## Default dataset

`data/HER_virtual_data.csv` — HER virtual screen, target column is `Target`, objective is `max`.

## Guardrails

- **Always label proxy results as simulations.** The proxy oracle is a surrogate trained from data, not a real experiment.
- **Include oracle CV RMSE** when presenting optimization results so the user knows surrogate quality.
- **Prefer explicit `--target` and `--objective`.**
- **Never auto-evaluate with proxy oracle in human-in-the-loop mode.** If the user is recording real observations, do not call `evaluate-last`.

## Observation format

The `observe` command accepts `--data` as:
- Inline JSON: `'{"x": {"feat1": 1.0}, "y": 5.2}'` or a JSON list of such objects
- Path to `.json` file: list of `{"x": {...}, "y": ...}` objects
- Path to `.csv` file: must have a `y` column; all other columns become `x`
