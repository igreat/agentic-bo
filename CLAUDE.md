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
bo_workflow/
  engine.py       # BOEngine class — suggest/observe loop, no oracle knowledge
  engine_cli.py   # CLI subcommands: init, suggest, observe, status, report
  oracle.py       # standalone proxy oracle — train, load, predict on run_dir
  oracle_cli.py   # CLI subcommands: build-oracle, run-proxy
  cli.py          # top-level entrypoint — composes subparsers from each module
  plotting.py     # convergence plot generation
  utils.py        # RunPaths, JSON I/O, shared types
  observers/
    base.py       # Observer ABC — evaluate(suggestions) interface
    proxy.py      # ProxyObserver — self-contained, captures run_dir at init
    callback.py    # CallbackObserver — delegates to user callback
  converters/
    reaction_drfp.py  # DRFP fingerprint encode/decode for reaction SMILES
data/
  HER_virtual_data.csv       # example dataset (HER virtual screen)
  buchwald_hartwig_rxns.csv  # Buchwald-Hartwig reaction SMILES dataset
scripts/
  compare_optimizers.py  # benchmark hebo/bo_lcb/random
```

### Key design boundaries

- **Engine has zero oracle awareness.** It only knows the `Observer` ABC and calls `observer.evaluate(suggestions)`. No oracle imports in `engine.py`.
- **Oracle is standalone.** `oracle.py` operates on `run_dir: Path`, not `engine: BOEngine`. Reads/writes state.json and oracle files directly.
- **Observers are self-contained.** `ProxyObserver(run_dir)` captures all context at construction. `evaluate(suggestions)` takes no engine or run_id.
- **CLI is the wiring layer.** `build-oracle` calls `oracle.build_proxy_oracle(run_dir)` directly. `run-proxy` constructs `ProxyObserver(run_dir)` and passes it to `engine.run_optimization()`.
- **Each module owns its CLI surface.** `engine_cli.py` and `oracle_cli.py` each define `register_commands()` + `handle()`. `cli.py` composes them.
- **Converters are standalone.** Each converter has its own `__main__`-style CLI (`python -m bo_workflow.converters.reaction_drfp`). They transform data before/after the BO loop but do not depend on the engine or oracle.

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

All commands: `uv run python -m bo_workflow.cli <command> [flags]`

| Command | Key flags | Purpose |
|---------|-----------|---------|
| `init` | `--dataset --target --objective` (req), `--engine --seed --init-random --batch-size` (opt) | Init run from CSV |
| `build-oracle` | `--run-id` (req), `--cv-folds --max-features` (opt) | Train proxy oracle |
| `suggest` | `--run-id` (req), `--batch-size` (opt) | Propose next candidates |
| `observe` | `--run-id --data` (req) | Record real/simulated results |
| `run-proxy` | `--run-id --iterations` (req), `--batch-size` (opt) | Full proxy BO loop |
| `status` | `--run-id` (req) | Quick run summary |
| `report` | `--run-id` (req) | Full report + convergence plot |

Converter commands (separate entrypoint): `uv run python -m bo_workflow.converters.reaction_drfp <subcommand> [flags]`

| Command | Key flags | Purpose |
|---------|-----------|---------|
| `encode` | `--input --output-dir` (req), `--rxn-col --n-bits` (opt) | Encode reaction SMILES to DRFP features |
| `decode` | `--catalog --query` (req), `--k` (opt) | Decode fingerprint suggestions to nearest reactions |

Engine options: `hebo` (default), `bo_lcb`, `random`. Note: `bo_lcb` currently supports batch-size 1 only.

## MVP demo (copy-paste)

```bash
uv run python -m bo_workflow.cli init \
  --dataset data/HER_virtual_data.csv \
  --target Target --objective max --seed 42

# grab the run_id from the JSON output, then:
uv run python -m bo_workflow.cli build-oracle --run-id <RUN_ID>
uv run python -m bo_workflow.cli run-proxy --run-id <RUN_ID> --iterations 20
```

Expected artifacts in `runs/<RUN_ID>/`: `state.json`, `oracle.pkl`, `oracle_meta.json`, `suggestions.jsonl`, `observations.jsonl`, `convergence.pdf`, `report.json`.

## Human-in-the-loop workflow

The engine supports step-by-step usage without a proxy oracle. `suggest` accepts status `initialized`, `oracle_ready`, or `running` — no oracle needed for HEBO/BO/random.

```bash
uv run python -m bo_workflow.cli init --dataset ... --target ... --objective max
uv run python -m bo_workflow.cli suggest --run-id <RUN_ID>
# human runs experiment in the lab
uv run python -m bo_workflow.cli observe --run-id <RUN_ID> --data '{"x": {...}, "y": 5.2}'
# repeat suggest/observe
```

## Default dataset

`data/HER_virtual_data.csv` is provided as an example dataset.

Treat dataset semantics (what the target means, valid constraints, and success thresholds) as problem-specific context from the user or project docs.

## Resuming a completed run

`run-proxy` sets the run status to `completed` when it finishes. To continue optimizing from where it left off (appending more iterations without re-running earlier ones), flip the status back to `running` before calling `run-proxy` again:

```python
import json, pathlib
p = pathlib.Path("runs/<RUN_ID>/state.json")
state = json.loads(p.read_text())
state["status"] = "running"
p.write_text(json.dumps(state, indent=2))
```

Then call `run-proxy` with the additional iterations desired. The engine naturally loads all existing observations, so the optimizer continues from the current best — no work is repeated.

This also applies to `suggest`: it accepts status `initialized`, `oracle_ready`, or `running`.

## Guardrails

- **Always label proxy results as simulations.** The proxy oracle is a surrogate trained from data, not a real experiment.
- **Include oracle CV RMSE** when presenting optimization results so the user knows surrogate quality.
- **Prefer explicit `--target` and `--objective`.**
- **Never auto-evaluate with proxy oracle in human-in-the-loop mode.** If the user is recording real observations, do not call `run-proxy` or otherwise invoke the proxy oracle.

## Observation format

The `observe` command accepts `--data` as:
- Inline JSON: `'{"x": {"feat1": 1.0}, "y": 5.2}'` or a JSON list of such objects
- Path to `.json` file: list of `{"x": {...}, "y": ...}` objects
- Path to `.csv` file: must have a `y` column; all other columns become `x`
