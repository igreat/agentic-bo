# Bayesian Optimisation Workflow

A practical HEBO-based Bayesian Optimisation workflow for scientific discovery in chemistry. Point it at a tabular dataset, and it trains a proxy oracle, runs BO against it, and reports results — all through a JSON CLI.

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
uv run python -m src.bo_workflow.cli init \
  --dataset data/HER_virtual_data.csv \
  --target Target --objective max --seed 42

# grab the run_id from the JSON output, then:
uv run python -m src.bo_workflow.cli build-oracle --run-id <RUN_ID>
uv run python -m src.bo_workflow.cli run-proxy --run-id <RUN_ID> --iterations 20
uv run python -m src.bo_workflow.cli report --run-id <RUN_ID>
```

## CLI commands

```bash
uv run python -m src.bo_workflow.cli --help
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

Add `--verbose` to `init`, `build-oracle`, `suggest`, `observe`, `run-proxy`, and `report` to print progress logs (and a tqdm bar for `run-proxy`).

Engine options: `hebo` (default), `bo_lcb`, `random`. Set once at init with `--engine`.

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

## Layout

```text
src/bo_workflow/
  engine.py    # BOEngine — all logic, JSON-in/JSON-out
  cli.py       # argparse CLI wrapping engine methods
  plotting.py  # convergence plot generation
data/
  HER_virtual_data.csv  # example dataset (HER virtual screen)
.claude/
  skills/      # Claude Code skills mapping to CLI commands
```

## Claude Skills

Skills in `.claude/skills/` provide the agent interface:

- `bo-init-run` — initialize a run
- `bo-build-oracle` — train proxy oracle
- `bo-next-batch` — suggest candidates
- `bo-record-observation` — record results
- `bo-report-run` — status and reports
- `bo-end-to-end-proxy` — full automated loop
