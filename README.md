# Research Agent Workflow

This repo is evolving toward a **research-agent-first** workflow for chemistry and materials discovery.

The top-level goal is:

- start from a research problem in plain English,
- frame the problem and optional literature context,
- set up and execute an optimization campaign,
- interpret the outcome,
- draft a paper or report.

Bayesian optimization is an internal execution layer inside that larger workflow, not the whole product. BO run state lives under `bo_runs/<run_id>/`. Top-level research workflow artifacts live under `research_runs/<research_id>/`.

> **Note on existing runs:** Earlier versions of this project stored BO runs under `runs/<run_id>/`. To continue using those runs after upgrading, either pass `--runs-root runs` to the CLI, or move each run directory from `runs/<run_id>` into `bo_runs/<run_id>`.

## Scope

- Top-level research workflow orchestration via agent skills.
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

## Quick Start

### Full Research Workflow

Use `research-agent` when the user wants:

- problem framing
- optional literature review
- experiment setup
- BO execution
- interpretation
- paper drafting

`research-agent` v1 is observer-agnostic:
- it resolves a structured experiment spec
- initializes a run
- continues through `suggest` / `observe` / `report`
- does not need to know whether observations come from a user, a real experiment loop, or an external benchmark evaluator

### BO-Only Quick Start

```bash
uv run python -m bo_workflow.cli build-oracle \
  --dataset data/HER_virtual_data.csv \
  --target Target --objective max \
  --backend-id her-demo

uv run python -m bo_workflow.cli init \
  --dataset data/HER_virtual_data.csv \
  --target Target --objective max --seed 42

# grab the run_id from the JSON output, then:
uv run python -m bo_workflow.cli run-proxy \
  --run-id <RUN_ID> --backend-id her-demo --iterations 20
uv run python -m bo_workflow.cli report --run-id <RUN_ID>
```

`build-oracle` writes proxy assets under `evaluation_backends/<BACKEND_ID>/`. Reuse the same backend across multiple runs when the run features/objective match the backend.

## BO CLI Commands

```bash
uv run python -m bo_workflow.cli --help
```

| Command | Purpose |
|---------|---------|
| `init` | Create a run from a CSV dataset or explicit search-space JSON |
| `build-oracle` | Train a proxy backend directly from a labeled dataset |
| `suggest` | Propose next candidate experiments |
| `observe` | Record objective values (real or simulated) |
| `run-proxy` | Run an end-to-end simulated BO loop against a backend |
| `run-evaluator` | Run a hidden evaluation loop with an operator-owned backend |
| `status` | Show best-so-far and run metadata |
| `report` | Generate JSON report |

Converter commands use separate module entrypoints:

- `uv run python -m bo_workflow.converters.reaction_drfp <encode|decode> [flags]`
- `uv run python -m bo_workflow.converters.molecule_descriptors <encode|decode> [flags]`
- `uv run python -m bo_workflow.converters.column_transform <profile|transform> [flags]`

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

`state.json`, `input_spec.json`, `suggestions.jsonl`, `observations.jsonl`, `report.json`

Each evaluation backend writes to `evaluation_backends/<BACKEND_ID>/`:

`oracle.pkl`, `oracle_meta.json`

## Research Artifacts

Each top-level research workflow writes to `research_runs/<RESEARCH_ID>/`:

`research_state.json`, `research_plan.md`, `paper.md`

## Design notes

- `research-agent` is the top-level orchestration layer. Use BO skills directly only when the user wants the optimization subsystem without the surrounding research workflow.
- `research-agent` uses the `suggest` / `observe` / `report` loop and is agnostic to whether observations come from a person or an external evaluator.
- The engine is replay-first: it rebuilds optimizer state from logged observations. This makes runs easy to resume and audit.
- Proxy mode is a simulation workflow. Always present results as simulated outcomes and include oracle CV RMSE.
- For hidden benchmark runs, prefer `run-evaluator` over `run-proxy`.
- `data/HER_virtual_data.csv` is included as an example dataset only. In real usage, users should provide problem-specific context (target meaning, constraints, objective direction, and valid operating domain).

## Submission note

This repository may be submitted together with its staged result artifacts under
`results/`. The `data/` directory is intentionally retained in that submission
package even though not every dataset is a primary paper result. These datasets
are kept because:

- example commands in the README depend on them
- benchmark and control-task workflows depend on them
- local tests and validation scripts may expect them to be present

If a minimal redistribution is needed later, `data/` can be pruned selectively,
but the full submission package keeps it intact for reproducibility and to avoid
breaking test/demo workflows.

## Layout

```text
.
|-- bo_workflow/
|   |-- constraints/
|   |-- converters/
|   |-- evaluation/
|   |-- observers/
|   `-- scripts/
|-- data/
|   `-- caltech_oer/
|-- .agents/
|   `-- skills/
|-- .claude/
|   `-- skills/
|-- evaluation_backends/
|   `-- <backend_id>/
|-- bo_runs/
|   `-- <run_id>/
`-- research_runs/
    `-- <research_id>/
```

- `bo_workflow/` contains the BO engine, evaluation/oracle layer, converters, constraints, and reusable scripts.
- `data/` contains example and benchmark datasets used by the BO and research workflows.
- `.agents/skills/` and `.claude/skills/` contain the mirrored agent skill trees.
- `evaluation_backends/` stores reusable oracle/backend artifacts.
- `bo_runs/` stores BO run state and report artifacts.
- `research_runs/` stores top-level research workflow state, notes, and paper drafts.

## Skills

Skills in `.agents/skills/` and `.claude/skills/` provide the agent interface:

- `research-agent` — top-level research workflow orchestration
- `literature-review` — lightweight literature support for research-agent
- `scientific-writing` — IMRAD-style drafting from workflow artifacts

- `bo-execution-workflow` — BO-layer execution helper once problem framing is already resolved
- `bo-init-run` — initialize a run
- `bo-next-batch` — suggest candidates
- `bo-run-evaluator` — automate external evaluator observations for an existing run
- `bo-record-observation` — record results
- `bo-report-run` — status and reports
- `bo-encode-drfp` — encode reaction SMILES to DRFP features
- `bo-decode-drfp` — decode suggestions back to real reactions
- `bo-encode-molecule-descriptors` — encode molecule SMILES to descriptor features
- `bo-decode-molecule-descriptors` — decode descriptor suggestions to real molecules
