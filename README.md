# Beyond Optimisation: Research-Layer Orchestration for Autonomous Chemistry

`agentic-bo` is an agent-operable framework for end-to-end chemistry and materials optimisation studies. It separates a research orchestration layer (problem framing, evaluator design, calibration, interpretation, paper drafting) from a shared Bayesian optimisation execution layer, so that tool-using coding agents can construct scientifically defensible studies rather than running a BO loop alone.

It is designed for agentic coding environments such as Codex, Claude Code, OpenCode, and similar tools that can read repository instructions (`AGENTS.md` or `CLAUDE.md`) and local skill trees under `.agents/skills/` or `.claude/skills/`. The repo supports three closely related use cases:

- running an end-to-end research workflow from a plain-English problem statement
- using the Bayesian optimization layer directly as a lower-level subsystem
- reproducing benchmarked runs from the packaged task bundles under `benchmarks/`

At a high level, the system is meant to help an agent move from a research question to a defensible computational study:

1. start from a plain-English research problem
2. frame the system, objective, and constraints
3. set up and execute an optimization campaign
4. interpret the outcome
5. draft a paper or report

Bayesian optimization is one layer inside that larger workflow, not the whole product. BO run state lives under `bo_runs/<run_id>/`, while top-level research workflow artifacts live under `research_runs/<research_id>/`.

## Ways to use this repo

### Full research workflow

Use this mode when you want the agent to drive the whole study:

- problem framing
- optional literature review
- experiment setup
- BO execution
- interpretation
- report or paper drafting

If your tool supports slash-command skill invocation, start with `/research-agent`. If it does not, ask the agent to follow the workflow defined in [AGENTS.md](AGENTS.md).

In collaborative use, agents may ask clarifying questions before committing to a search space, evaluator strategy, or budget. If you prefer fully autonomous operation, explicitly tell the agent to proceed without questions (for example, "run end-to-end, no questions").

### BO-only workflow

Use this mode when the research framing is already settled and you only want the optimization subsystem, evaluators, or converters.

### Benchmarks and reproducibility

Use this mode when you want the packaged benchmark task bundles, prompt files, workspace builders, and scoring guidance under `benchmarks/`.

## Setup

```bash
uv sync
uv pip install --no-deps "hebo @ git+https://github.com/huawei-noah/HEBO.git#subdirectory=HEBO"
```

Why `--no-deps`:
- HEBO's published metadata pins very old NumPy and pymoo versions
- this repo already declares the intended runtime dependencies in `pyproject.toml`

## Quick start

### 1. Agent-driven research workflow

Open the repo in your agentic tool of choice and give it a plain-English goal, for example:

```text
Investigate a live local screening workflow for a HER catalyst family.
Frame the problem, choose a defensible evaluator path, run BO, and draft a concise report.
```

Useful repo guidance for the agent:
- [AGENTS.md](AGENTS.md)
- `.agents/skills/research-agent/`
- `.agents/skills/literature-review/`
- `.agents/skills/scientific-writing/`

### 2. BO-only CLI workflow

```bash
uv run python -m bo_workflow.cli build-oracle \
  --dataset data/HER_virtual_data.csv \
  --target Target --objective max \
  --backend-id her-demo

uv run python -m bo_workflow.cli init \
  --dataset data/HER_virtual_data.csv \
  --target Target --objective max --seed 42

# copy the run_id from the JSON output, then:
uv run python -m bo_workflow.cli run-proxy \
  --run-id <RUN_ID> --backend-id her-demo --iterations 20

uv run python -m bo_workflow.cli report --run-id <RUN_ID>
```

`build-oracle` writes proxy assets under `evaluation_backends/<backend_id>/`. Reuse the same backend across multiple runs when the run features and target match.

### 3. Hidden-evaluator or benchmark runs

For hidden evaluation loops, prefer `run-evaluator` over `run-proxy`.

For packaged benchmark workspaces and prompt bundles, start with:
- [benchmarks/README.md](benchmarks/README.md)

## Core concepts

Two layers matter:

- **Research layer**: problem framing, literature context, experiment setup, interpretation, paper drafting
- **BO layer**: concrete optimization engine, suggest/observe loop, constraints, converters, evaluators

Important design boundaries:

- the BO engine has no oracle knowledge
- evaluators and proxy backends live outside the core engine loop
- `research-agent` orchestrates the end-to-end workflow
- BO skills and CLI commands remain available for lower-level optimization tasks

For the full architecture and workflow contract, use:
- [AGENTS.md](AGENTS.md)

## CLI quick reference

Run all commands as:

```bash
uv run python -m bo_workflow.cli <command> [flags]
```

| Command | Purpose |
| --- | --- |
| `init` | Create a BO run from a dataset or explicit search-space JSON |
| `build-oracle` | Train a proxy backend from labeled data |
| `suggest` | Propose next candidate experiments |
| `observe` | Record objective values from an external observer |
| `run-proxy` | Run a simulated BO loop against a proxy backend |
| `run-evaluator` | Run a hidden evaluation loop with a prebuilt backend |
| `run-python-evaluator` | Run BO against a local Python evaluator module |
| `status` | Show best-so-far and run metadata |
| `report` | Generate a JSON report for a run |

Converter entrypoints:

- `uv run python -m bo_workflow.converters.reaction_drfp <subcommand> ...`
- `uv run python -m bo_workflow.converters.molecule_descriptors <subcommand> ...`
- `uv run python -m bo_workflow.converters.column_transform <subcommand> ...`

Engine options:
- `hebo` (default)
- `bo_lcb`
- `random`
- `botorch`

If a problem has composition variables that must sum to a fixed total, declare that explicitly at init time with `--simplex-groups 'col1,col2,...:total'`.

## Repository layout

```text
.
├── AGENTS.md
├── CLAUDE.md
├── .agents/
│   └── skills/
├── .claude/
│   └── skills/
├── benchmarks/
│   ├── README.md
│   ├── prompts/
│   ├── tasks/
│   └── templates/
├── bo_workflow/
│   ├── cli.py
│   ├── engine.py
│   ├── engine_cli.py
│   ├── plotting.py
│   ├── utils.py
│   ├── constraints/
│   ├── converters/
│   ├── evaluation/
│   ├── observers/
│   └── scripts/
├── data/
│   └── caltech_oer/
├── evaluation_backends/
│   └── <backend_id>/
├── bo_runs/
│   └── <run_id>/
├── research_runs/
│   └── <research_id>/
├── results/
│   ├── SUPPLEMENTARY_MATERIALS.md
│   ├── closed_world_reruns/
│   └── open_world_reruns/
└── tests/
```

What lives where:

- `bo_workflow/`: BO engine, evaluator integration, converters, constraints, reusable scripts
- `.agents/skills/` and `.claude/skills/`: mirrored agent skill trees
- `benchmarks/`: benchmark task bundles, prompts, workspace builders, and scoring docs
- `data/`: example and benchmark datasets
- `evaluation_backends/`: reusable oracle or evaluator backend artifacts
- `bo_runs/`: persisted BO state and reports
- `research_runs/`: research workflow notes and paper/report drafts
- `results/`: staged result bundles and supplementary-material packaging
- `tests/`: regression and workflow tests

## Artifact roots

BO run artifacts under `bo_runs/<run_id>/`:
- `state.json`
- `input_spec.json`
- `suggestions.jsonl`
- `observations.jsonl`
- `report.json`
- `convergence.png`

Evaluation backend artifacts under `evaluation_backends/<backend_id>/`:
- `oracle.pkl`
- `oracle_meta.json`

Research workflow artifacts under `research_runs/<research_id>/`:
- `research_state.json`
- `research_plan.md`
- `paper.md`

## Useful scripts

Compare optimizers on a dataset:

```bash
uv run python -m bo_workflow.scripts.compare_optimizers \
  --dataset data/HER_virtual_data.csv \
  --target Target --objective max \
  --iterations 20 --batch-size 1 --repeats 1
```

Run the EGFR global simulation:

```bash
uv run python -m bo_workflow.scripts.egfr_ic50_global_experiment \
  --dataset data/egfr_ic50.csv \
  --seed-count 50 --rounds 20 --batch-size 4
```

## Submission note

This repo may be submitted together with staged result artifacts under `results/`. The `data/` directory is intentionally retained in that submission package even though not every dataset is a primary paper result.

Those datasets are kept because:

- example commands in this README depend on them
- benchmark and control-task workflows depend on them
- local tests and validation scripts may expect them to be present

If a smaller redistribution is needed later, `data/` can be pruned selectively. The current submission-oriented package keeps it intact to avoid breaking test and demo workflows.
