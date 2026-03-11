# Bayesian Optimisation Workflow

A practical BO workflow for scientific discovery in chemistry.

This repository is intended to be an **agent-operable optimization engine**:

- define an optimization problem,
- build or plug in an objective evaluator (real experiment or proxy oracle),
- run iterative BO suggestions,
- track state and results for human-in-the-loop workflows.

## Scope

- Single-objective BO from tabular datasets with persisted run state and JSON CLI.
- **Converters** transform non-tabular inputs (e.g. reaction SMILES) into numerical features the BO engine can optimize over, and decode suggestions back to interpretable results.

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

| Command | Purpose |
|---------|---------|
| `init` | Create a run from a CSV dataset |
| `build-oracle` | Train a proxy oracle from dataset rows |
| `suggest` | Propose next candidate experiments |
| `observe` | Record objective values (real or simulated) |
| `run-proxy` | Run an end-to-end simulated BO loop |
| `status` | Show best-so-far and run metadata |
| `report` | Generate JSON report and convergence plot |
| `encode` | Encode reaction SMILES into DRFP fingerprint features |
| `decode` | Decode fingerprint suggestions back to nearest real reactions |

Converter commands use a separate entrypoint: `uv run python -m bo_workflow.converters.reaction_drfp <encode|decode> [flags]`

Converter commands use separate module entrypoints:

- `uv run python -m bo_workflow.converters.reaction_drfp <encode|decode> [flags]`
- `uv run python -m bo_workflow.converters.molecule_descriptors <encode|decode> [flags]`

Examples:

```bash
# Reaction SMILES -> DRFP bits
uv run python -m bo_workflow.converters.reaction_drfp encode \
  --input data/buchwald_hartwig_rxns.csv --output-dir data/bh_drfp

# Molecule SMILES -> RDKit descriptors + Morgan bits
uv run python -m bo_workflow.converters.molecule_descriptors encode \
  --input data/egfr_ic50.csv --output-dir data/egfr_desc --smiles-cols smiles
```

Add `--verbose` to `init`, `build-oracle`, `suggest`, `observe`, `run-proxy`, and `report` to print progress logs (and a tqdm bar for `run-proxy`).

Engine options: `hebo` (default), `bo_lcb`, `random`, `botorch`. `bo_lcb` supports batch-size 1 only. `botorch` now supports mixed numeric + categorical features via BoTorch's native mixed GP model, but `hebo` remains the default for categorical-heavy problems.

Constraints are explicit run configuration, not something inferred from the CSV. If the problem has composition variables that must sum to a fixed total, declare them at init time with `--simplex-groups 'col1,col2,...:total'`.

## Benchmark scripts

Compare `hebo`, `bo_lcb`, and `random` on any dataset:

```bash
uv run python -m bo_workflow.scripts.compare_optimizers \
  --dataset data/HER_virtual_data.csv \
  --target Target --objective max \
  --iterations 20 --batch-size 1 --repeats 1
```

Run the EGFR global simulation (descriptor-space BO against a real IC50 dataset):

```bash
uv run python -m bo_workflow.scripts.egfr_ic50_global_experiment \
  --dataset data/egfr_ic50.csv \
  --seed-count 50 --rounds 20 --batch-size 4
```

Each round suggests molecules in descriptor space, maps them to the nearest real molecule, looks up the true pIC50, and records it as an observation. Reports best found vs best in dataset.

## Run artifacts

Each run writes to `bo_runs/<RUN_ID>/`:

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
  cli.py          # top-level entrypoint — composes subparsers from each module
  plotting.py     # convergence plot generation
  utils.py        # RunPaths, JSON I/O, shared types
  constraints/
    base.py       # Constraint ABC — enforce search-space constraints at suggest time
    simplex.py    # SimplexConstraint — composition variables summing to a fixed total
  observers/
    base.py       # Observer ABC — evaluate(suggestions) interface
    proxy.py      # ProxyObserver — self-contained, captures run_dir at init
    callback.py   # CallbackObserver — delegates to user callback
  converters/
    molecule_descriptors.py  # RDKit descriptor encode/decode for molecule SMILES
    reaction_drfp.py  # DRFP fingerprint encode/decode for reaction SMILES
  scripts/
    compare_optimizers.py           # benchmark hebo/bo_lcb/random
    compare_representations.py      # benchmark descriptor/DRFP/combined representations
    egfr_ic50_global_experiment.py  # EGFR global simulation experiment
    egfr_utils.py                   # shared data loading helpers for EGFR scripts
data/
  HER_virtual_data.csv       # example dataset (HER virtual screen)
  buchwald_hartwig_rxns.csv  # Buchwald-Hartwig reaction SMILES dataset
  egfr_ic50.csv              # EGFR IC50 dataset (~10k molecules)
  egfr_seed50_mixed.csv      # EGFR seed set (50 labeled molecules)
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
- `bo-encode-drfp` — encode reaction SMILES to DRFP features
- `bo-decode-drfp` — decode suggestions back to real reactions
- `bo-encode-molecule-descriptors` — encode molecule SMILES to descriptor features
- `bo-decode-molecule-descriptors` — decode descriptor suggestions to real molecules

## Credits

Much of the underlying HEBO and problem specific part of the code is taken from/inspired from [BO-Tutorial-for-Sci](https://github.com/zwyu-ai/BO-Tutorial-for-Sci).
