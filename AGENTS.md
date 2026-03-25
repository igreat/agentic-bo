# Research Agent Workflow

This repo is evolving toward a **research-agent-first** workflow for chemistry and materials discovery.

The top-level goal is:
- start from a research problem in plain English,
- frame the problem and optional literature context,
- set up and execute an optimization campaign,
- interpret the outcome,
- draft a paper or report.

Bayesian optimization is an internal execution layer inside that larger workflow, not the whole product. BO run state lives under `bo_runs/<run_id>/`. Top-level research workflow artifacts live under `research_runs/<research_id>/`.
If you're resuming runs from an older checkout, their state may live under `runs/<run_id>/` instead — either pass `--runs-root runs` to the CLI or move those run directories into `bo_runs/` before resuming.

## Setup

```bash
uv sync
uv pip install --no-deps "hebo @ git+https://github.com/huawei-noah/HEBO.git#subdirectory=HEBO"
```

HEBO's published metadata pins ancient NumPy/pymoo versions. Install it with `--no-deps`; this project's `pyproject.toml` already declares the real runtime dependencies.

## Architecture

Two layers matter:

- **Research layer:** top-level orchestration, phase tracking, literature context, interpretation, paper drafting
- **BO layer:** concrete optimization engine, proxy oracle, suggest/observe loop, constraints, converters

```
bo_workflow/
  engine.py       # BOEngine class — suggest/observe loop, no oracle knowledge
  engine_cli.py   # CLI subcommands: init, suggest, observe, status, report
  cli.py          # top-level entrypoint — composes subparsers from each module
  utils.py        # RunPaths, JSON I/O, shared types
  evaluation/
    cli.py        # CLI subcommands: build-oracle, run-proxy, run-evaluator, run-python-evaluator
    oracle.py     # standalone proxy backend — train from run config, persist under evaluation_backends/
    proxy.py      # ProxyObserver — self-contained, captures backend_dir at init
    __main__.py   # optional evaluation-only module entrypoint
  observers/
    base.py       # Observer ABC — evaluate(suggestions) interface
    callback.py   # CallbackObserver — delegates to user callback
  constraints/
    base.py       # Constraint ABC — apply(suggestions) interface
    simplex.py    # SimplexConstraint — post-hoc normalization (upgradeable to bijection)
  converters/
    molecule_descriptors.py  # RDKit descriptor encode/decode for molecule SMILES
    reaction_drfp.py  # DRFP fingerprint encode/decode for reaction SMILES
  scripts/
    compare_optimizers.py           # benchmark hebo/bo_lcb/random
    compare_representations.py      # benchmark descriptor/DRFP/combined representations
    egfr_ic50_global_experiment.py  # EGFR global simulation (descriptor BO + real dataset lookup)
    egfr_utils.py                   # shared data loading helpers for EGFR scripts
data/
  HER_virtual_data.csv       # example dataset (HER virtual screen)
  buchwald_hartwig_rxns.csv  # Buchwald-Hartwig reaction SMILES dataset
  egfr_ic50.csv              # EGFR IC50 dataset (full, ~10k molecules)
  egfr_seed50_mixed.csv      # EGFR seed set (50 labeled molecules)
  OER_catalyst_data.csv      # OER catalyst dataset (multi-metal, synthesis conditions)
  caltech_oer/
    plate_3496.csv           # Mn-Fe-Co-Ni-La-Ce oxides, 2121 compositions (Rohr 2020, Chem Sci)
    plate_3851.csv           # Mn-Fe-Co-Ni-Cu-Ta oxides, 2119 compositions
    plate_3860.csv           # Mn-Fe-Co-Cu-Sn-Ta oxides, 2121 compositions
    plate_4098.csv           # Ca-Mn-Co-Ni-Sn-Sb oxides, 2121 compositions
    # Source: data.caltech.edu/records/7b106-nf257 (Gregoire Group / TRI)
    # Target: overpotential_V at 3 mA/cm² (lower is better → --objective min)
    # All 6 molar_fraction cols sum to 1.0 → use --simplex-groups at init
.agents/
  skills/
    research-agent/         # top-level research workflow orchestration
    literature-review/      # lightweight literature support for research-agent
    scientific-writing/     # IMRAD-style drafting from workflow artifacts
.claude/
  skills/
    research-agent/         # mirrored Claude skill tree
    literature-review/      # mirrored Claude literature helper
    scientific-writing/     # mirrored Claude writing helper
research_runs/
  <research_id>/
    research_state.json     # machine-readable research workflow state
    research_plan.md        # human-readable running notebook
    paper.md                # final paper/report draft
```

### Key design boundaries

- **Engine has zero oracle awareness.** It only knows the `Observer` ABC and calls `observer.evaluate(suggestions)`. No oracle imports in `engine.py`.
- **Oracle is standalone.** `evaluation/oracle.py` trains directly from labeled dataset inputs and persists oracle assets under `evaluation_backends/<backend_id>/`.
- **Observers are self-contained.** `evaluation/proxy.py` defines `ProxyObserver(backend_dir)`, which captures all context at construction. `observers/callback.py` provides `CallbackObserver` for locally discovered Python evaluators. `evaluate(suggestions)` takes no engine or run_id.
- **CLI is the wiring layer.** `build-oracle` trains a backend from a run and writes it under `evaluation_backends/`. `run-proxy` constructs `evaluation.proxy.ProxyObserver(backend_dir)` and passes it to `engine.run_optimization()`.
- **Each module owns its CLI surface.** `engine_cli.py` and `evaluation/cli.py` each define `register_commands()` + `handle()`. `cli.py` composes them.
- **Converters are standalone.** Each converter has its own `__main__`-style CLI (`python -m bo_workflow.converters.reaction_drfp`). They transform data before/after the BO loop but do not depend on the engine or oracle.
- **Constraints are search-space properties.** `constraints/` is the enforcement layer — each `Constraint` subclass receives raw suggestions from the optimizer and projects them into the feasible region via `apply()`. Constraints are stored in `state.json["constraints"]` and enforced at every `suggest` call. The agent is responsible for inferring constraints from the user's problem description (e.g. "proportions sum to 100%") and passing them via `--simplex-groups`; the engine never auto-detects them.

Skills in `.agents/skills/` and `.claude/skills/` are kept in sync. The BO engine is the source of truth for optimization behavior; skills are the agent-facing orchestration layer on top of it. `research-agent` is the top-level skill, while the BO skills are lower-level execution helpers.

## Top-Level Workflows

Use `research-agent` when the user wants an end-to-end study workflow:
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
- it may also run BO against a local Python evaluator module with `run-python-evaluator` when that evaluator already exists as part of the workflow

Use the BO skills directly when the user wants only the optimization subsystem:
- `bo-execution-workflow` for a resolved BO-layer setup/execution handoff
- init / suggest / observe / report
- build-oracle / run-proxy for low-level proxy demos or BO-only benchmarking
- reporting

## Script-first policy

- Before writing ad-hoc one-off scripts, check `bo_workflow/scripts/` and prefer existing scripts when they already cover the task.
- For explicit optimizer benchmarking/comparison requests, use:

```bash
uv run python -m bo_workflow.scripts.compare_optimizers \
  --dataset data/HER_virtual_data.csv \
  --target Target --objective max \
  --iterations 20 --batch-size 1 --repeats 3 --verbose
```

- For EGFR molecular optimization experiments (descriptor-space BO with real dataset lookup), use:

```bash
uv run python -m bo_workflow.scripts.egfr_ic50_global_experiment \
  --dataset data/egfr_ic50.csv \
  --rounds 20 --batch-size 4 [--seed-count 50]
```

- **Long-running scripts** (EGFR experiments, compare scripts with >1 repeat) can take 10–30+ minutes. Always run them with `run_in_background=true` in the Bash tool — do not use a fixed timeout, there is no safe upper bound.
- Only create a new script if no existing command/script fits the request. If creating one, keep it reusable and place it under `bo_workflow/scripts/`.

## Artifact Roots

- `research_runs/<research_id>/`: top-level research workflow state and writing artifacts
- `bo_runs/<run_id>/`: BO engine state, suggestions, observations, and reports
- `evaluation_backends/<backend_id>/`: reusable oracle/backend artifacts

## BO Run Artifacts

Each BO run produces files under `bo_runs/<run_id>/`:

| File | Created by |
|------|-----------|
| `state.json` | `init` |
| `input_spec.json` | `init` |
| `intent.json` | `init` (when `--intent-json` is provided) |
| `suggestions.jsonl` | `suggest` / `run-proxy` |
| `observations.jsonl` | `observe` / `run-proxy` |
| `report.json` | `report` / `run-proxy` |

## Evaluation Backend Artifacts

Each evaluation backend produces files under `evaluation_backends/<backend_id>/`:

| File | Created by |
|------|-----------|
| `oracle.pkl` | `build-oracle` |
| `oracle_meta.json` | `build-oracle` |

## Research Run Artifacts

Each top-level research workflow produces files under `research_runs/<research_id>/`:

| File | Created by |
|------|-----------|
| `research_state.json` | `research-agent` |
| `research_plan.md` | `research-agent` |
| `paper.md` | `research-agent` / `scientific-writing` |

## CLI quick reference

All commands: `uv run python -m bo_workflow.cli <command> [flags]`

| Command | Key flags | Purpose |
|---------|-----------|---------|
| `init` | `--dataset` or `--search-space-json` (req), `--target --objective` (req), `--engine --seed --init-random --batch-size --simplex-groups --drop-cols` (opt) | Init run from dataset inference or explicit search-space spec |
| `build-oracle` | `--dataset --target --objective --backend-id` (req), `--drop-cols --cv-folds --max-features --seed --engine` (opt) | Train proxy backend directly from a labeled dataset |
| `suggest` | `--run-id` (req), `--batch-size` (opt) | Propose next candidates |
| `observe` | `--run-id --data` (req) | Record real/simulated results |
| `run-proxy` | `--run-id --iterations` (req), `--backend-id --batch-size` (opt) | Full proxy BO loop |
| `run-evaluator` | `--run-id --backend-id --iterations` (req), `--batch-size` (opt) | Operator-owned hidden evaluation loop over `suggest` / `observe` |
| `run-python-evaluator` | `--run-id --module-path --iterations` (req), `--function --batch-size` (opt) | Run BO against a local Python evaluator module discovered or written during the workflow |
| `status` | `--run-id` (req) | Quick run summary |
| `report` | `--run-id` (req) | Full report JSON |

Converter commands (separate entrypoints):

- `uv run python -m bo_workflow.converters.reaction_drfp <subcommand> [flags]`
- `uv run python -m bo_workflow.converters.molecule_descriptors <subcommand> [flags]`
- `uv run python -m bo_workflow.converters.column_transform <subcommand> [flags]`

| Converter | Command | Key flags | Purpose |
|---------|---------|-----------|---------|
| `reaction_drfp` | `encode` | `--input --output-dir` (req), `--rxn-col --n-bits` (opt, default 128) | Encode reaction SMILES to DRFP features |
| `reaction_drfp` | `decode` | `--catalog --query` (req), `--k` (opt) | Decode fingerprint suggestions to nearest reactions |
| `molecule_descriptors` | `encode` | `--input --output-dir --smiles-cols` (req), `--morgan-bits` (opt, default 64) | Encode molecule SMILES columns to RDKit descriptor features |
| `molecule_descriptors` | `decode` | `--catalog --query` (req), `--k` (opt) | Decode descriptor suggestions to nearest molecules |
| `column_transform` | `profile` | `--input` (req), `--cols` (opt) | Analyse columns and recommend transforms |
| `column_transform` | `transform` | `--input --cols --transform --output` (req), `--keep-original` (opt) | Apply a named transform; renames column with prefix (e.g. `log10_ic50_nM`) |

Engine options: `hebo` (default), `bo_lcb`, `random`, `botorch`. Note: `bo_lcb` currently supports batch-size 1 only. `botorch` supports mixed numeric + categorical features via BoTorch's native mixed GP model, but `hebo` remains the default for strongly categorical problems.

Constraints are domain knowledge, not something the engine can reliably infer from a dataset alone. When the problem description includes composition variables that must sum to a fixed total, pass them explicitly during `init` with `--simplex-groups 'col1,col2,...:total'`.

## MVP demo (copy-paste)

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
```

Expected artifacts in `bo_runs/<RUN_ID>/`: `state.json`, `input_spec.json`, `suggestions.jsonl`, `observations.jsonl`, `report.json`.

Expected backend artifacts in `evaluation_backends/<RUN_ID>/`: `oracle.pkl`, `oracle_meta.json`.

## BO Suggest/Observe Workflow

The engine supports step-by-step usage without a proxy oracle. `suggest` accepts status `initialized` or `running` — no oracle needed for HEBO/BO/random.

```bash
uv run python -m bo_workflow.cli init --dataset ... --target ... --objective max
uv run python -m bo_workflow.cli suggest --run-id <RUN_ID>
# human runs experiment in the lab
uv run python -m bo_workflow.cli observe --run-id <RUN_ID> --data '{"x": {...}, "y": 5.2}'
# repeat suggest/observe
```

Within `research-agent`, this low-level pattern is the canonical execution flow. A human, hidden evaluator, or other external observer may supply the values that later get recorded with `observe`.

Low-level proxy commands (`build-oracle`, `run-proxy`) remain available for BO-only demos and retrospective benchmarking, but they are not the default research-agent path.

For hidden benchmark runs, prefer `run-evaluator` over `run-proxy`. The evaluator owns the oracle side externally and records observations back into the run without telling the agent to build or invoke its own proxy.

## Default dataset

`data/HER_virtual_data.csv` is provided as an example dataset.

Treat dataset semantics (what the target means, valid constraints, and success thresholds) as problem-specific context from the user or project docs.

## Resuming a completed run

`run-proxy` sets the run status to `completed` when it finishes. To continue optimizing from where it left off (appending more iterations without re-running earlier ones), flip the status back to `running` before calling `run-proxy` again:

```python
import json, pathlib
p = pathlib.Path("bo_runs/<RUN_ID>/state.json")
state = json.loads(p.read_text())
state["status"] = "running"
p.write_text(json.dumps(state, indent=2))
```

Then call `run-proxy` with the additional iterations desired. The engine naturally loads all existing observations, so the optimizer continues from the current best — no work is repeated.

This also applies to `suggest`: it accepts status `initialized` or `running`.

## Guardrails

- **Always label proxy results as simulations.** The proxy oracle is a surrogate trained from data, not a real experiment.
- **Include oracle CV RMSE** when presenting optimization results so the user knows surrogate quality.
- **Prefer explicit `--target` and `--objective`.**
- **Never auto-evaluate with proxy oracle when observations are meant to come from outside the BO engine.** If the user or an external observer is providing real values, do not call `run-proxy` or otherwise invoke the proxy oracle.

## Observation format

The `observe` command accepts `--data` as:
- Inline JSON: `'{"x": {"feat1": 1.0}, "y": 5.2}'` or a JSON list of such objects
- Path to `.json` file: list of `{"x": {...}, "y": ...}` objects
- Path to `.csv` file: must have a `y` column; all other columns become `x`
