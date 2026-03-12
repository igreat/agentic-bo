# Implementation Plan: Observer Extensions, Conversion Layer, and Skills

## Context

The BO engine is complete. What's missing is:
- Observer extensions for testing and agentic workflows
- Converter scripts to bridge chemistry-specific inputs (compositions, SMILES) into the tabular CSVs the engine expects
- Skills wiring so Claude Code can autonomously pick and run the right converter
- A meta-skill describing the full agentic orchestration workflow

The primary workflow: Claude acts as orchestrator between chemist (via chat) and HEBO (via CLI skills). Claude calls `suggest`/`observe` step-by-step -- the existing CLI already supports this. The new observers formalize testing (SimulatedHuman) and programmatic agent embedding (AgentObserver).

---

## Implementation Order

### Step 1: Composition Simplex Converter

**Priority: highest.** Pure math, no ML deps, directly unblocks `HEA_alloy_data.csv`. Matches next_steps.md recommendation to start here.

**Create** `scripts/converters/__init__.py` (empty) + `scripts/converters/composition_simplex.py`

Stick-breaking transform (each z_i in [0,1], matching HEBO's `num` param type):

- `stick_breaking_encode(compositions: ndarray) -> ndarray` -- (N,K) -> (N,K-1)
- `stick_breaking_decode(z: ndarray, n_components: int) -> ndarray` -- inverse
- `detect_composition_columns(df, tolerance, exclude_columns)` -- auto-detect columns summing to ~1
- `encode_csv(input, output, metadata, ...)` -- full CSV transform + metadata JSON. **Preserves original composition columns as passthrough** alongside reparameterized ones for interpretability.
- `decode_suggestions(input, metadata)` -- inverse transform suggestions back to compositions

CLI: `uv run python scripts/converters/composition_simplex.py encode|decode --input ... --output ... --metadata ...`

Data flow for HEA:
```
data/HEA_alloy_data.csv  [Co, Fe, Mn, V, Cu, target]
    |
    v  encode --composition-columns Co Fe Mn V Cu --target target
    |
data/HEA_encoded.csv  [z_0, z_1, z_2, z_3, target, Co, Fe, Mn, V, Cu]
                        ^ optimizable params ^        ^ passthrough IDs ^
data/HEA_simplex_meta.json
    |
    v  bo_workflow.cli init --dataset ... --target target --objective max
    |
HEBO optimizes over z_0..z_3 in [0,1]  (original cols are constant per-row, become fixed_features)
    |
    v  decode --metadata ...
    |
Original compositions  [Co=0.3, Fe=0.1, Mn=0.1, V=0.2, Cu=0.3]
```

---

### Step 2: SimulatedHumanObserver

**Create** `bo_workflow/observers/simulated_human.py`

- Wraps `ProxyObserver` (does not duplicate oracle logic)
- Adds configurable Gaussian noise (`noise_std`) to predictions
- Adds random refusal probability (`refusal_prob`) -- drops observations
- Uses `numpy.random.default_rng(seed)` for isolated, reproducible RNG
- Source: `"simulated-human"`

Key signatures:
```python
class SimulatedHumanObserver(Observer):
    def __init__(self, run_dir, *, noise_std=0.0, refusal_prob=0.0, seed=None)
    def evaluate(self, suggestions) -> list[dict]  # delegates to proxy, adds noise, filters refusals
```

**Modify** `bo_workflow/oracle_cli.py` -- add `run-simulated` subcommand:
- `--run-id`, `--iterations` (required), `--batch-size`, `--noise-std`, `--refusal-prob`, `--sim-seed`, `--verbose`
- Follows existing `run-proxy` pattern: constructs observer, passes to `engine.run_optimization()`

**Modify** `bo_workflow/observers/__init__.py` -- export new class

---

### Step 3: Mixed Catalog Validator

**Create** `scripts/converters/validate_mixed_catalog.py`

Diagnostic/validation script (engine handles mixed types natively):
- `validate_catalog(input, target_column, max_categories)` -> report dict
- Reports column types, ranges, missing values, cardinality
- Detects proportion columns that might need simplex treatment
- Flags issues (high cardinality, constant columns, missing target)
- Outputs `ready_for_engine` boolean

CLI: `uv run python scripts/converters/validate_mixed_catalog.py --input ... --target ...`

No new dependencies.

---

### Step 4: SMILES to Descriptors Converter

**Create** `scripts/converters/smiles_to_descriptors.py`

- `compute_descriptors(smiles_series, prefix)` -- RDKit 2D descriptors per SMILES
- `detect_smiles_columns(df, sample_size)` -- heuristic: parse sample values with RDKit
- `encode_csv(input, output, metadata, ...)` -- replace SMILES cols with descriptors, preserve originals as ID columns
- `decode_suggestions(suggestions, metadata, catalog)` -- KDTree nearest-neighbor lookup in descriptor space to recover original SMILES

**Modify** `pyproject.toml` -- add optional dependency group:
```toml
[dependency-groups]
chemistry = ["rdkit>=2024.3.0"]
```

CLI: `uv run python scripts/converters/smiles_to_descriptors.py encode|decode --input ... --output ... --metadata ...`

RDKit imports are deferred to function bodies for graceful failure when not installed.

---

### Step 5: AgentObserver

**Create** `bo_workflow/observers/agent.py`

More than a simple callback wrapper -- the AgentObserver adds agent-specific behavior on top of the Observer ABC:

```python
class AgentObserver(Observer):
    def __init__(
        self,
        callback: Callable[[list[dict]], list[dict]],
        *,
        source_label: str = "agent",
        validate_observations: bool = True,
        log_dir: Path | None = None,
    )
    def evaluate(self, suggestions) -> list[dict]
```

Distinguishing features vs CallbackObserver:
- **Observation validation**: verifies each returned observation has required fields (`x`, `y`) and `y` is numeric. Raises clear errors if the agent returns malformed data. CallbackObserver blindly passes through.
- **Interaction logging**: when `log_dir` is provided, writes each suggest/observe exchange to a JSONL file (`agent_interactions.jsonl`) with timestamps. Creates an audit trail of what the agent decided and why.
- **Suggestion formatting**: `format_suggestions(suggestions, state)` static method that produces a human-readable summary of HEBO's suggestions (feature names + values, decoded if metadata available). Agents can call this to present suggestions to the chemist.
- **Partial result handling**: explicitly documents and handles the case where the agent returns fewer observations than suggestions (some experiments refused/deferred).
- **Configurable source label**: defaults to `"agent"` but can be set to `"claude-agent"`, `"langchain-agent"`, etc. for provenance tracking.

Note: The primary Claude + chemist workflow uses the existing `suggest`/`observe` CLI skills step-by-step. This observer is for programmatic integration where an agent loop runs inside `run_optimization()`. The logging and validation make it safer for automated agent loops where errors need to be caught early.

**Modify** `bo_workflow/observers/__init__.py` -- export AgentObserver

---

### Step 6: Skills Wiring

**Create** 5 new skill directories:

| Skill | Directory | Maps to |
|-------|-----------|---------|
| `convert-composition` | `.claude/skills/convert-composition/SKILL.md` | `scripts/converters/composition_simplex.py` |
| `convert-smiles` | `.claude/skills/convert-smiles/SKILL.md` | `scripts/converters/smiles_to_descriptors.py` |
| `validate-catalog` | `.claude/skills/validate-catalog/SKILL.md` | `scripts/converters/validate_mixed_catalog.py` |
| `bo-run-simulated` | `.claude/skills/bo-run-simulated/SKILL.md` | `cli run-simulated` |
| `bo-execution-workflow` | `.claude/skills/bo-execution-workflow/SKILL.md` | Meta-skill: full orchestration pattern |

The **`bo-execution-workflow`** meta-skill describes the complete agentic orchestration pattern that Claude follows when a chemist asks to optimize something:

```
1. Understand the problem (what to optimize, what constraints exist)
2. Validate/convert the dataset:
   - Run validate-catalog to diagnose the CSV
   - If compositions detected -> run convert-composition
   - If SMILES columns detected -> run convert-smiles
   - Otherwise -> use dataset directly
3. Initialize: bo-init-run with converted dataset
4. Build surrogate: bo-build-proxy-oracle
5. Optimization loop (repeat):
   a. bo-next-batch -> get suggestions from HEBO
   b. Decode suggestions to chemist-readable form (run converter decode)
   c. Present to chemist via chat with context (literature, feasibility notes)
   d. Chemist runs experiment, reports result
   e. bo-record-observation -> feed result back to HEBO
6. bo-report-run -> final summary with convergence plot
```

This skill does not map to a single CLI command -- it's a workflow guide that tells Claude how to orchestrate the other skills together.

Each skill follows existing pattern: YAML frontmatter + markdown with Command, Return, Notes.

---

### Step 7: Tests

**Create** `tests/test_observers.py`:
- `TestSimulatedHumanObserver`: no-noise matches proxy, noise changes values, refusal reduces observations, source label
- `TestAgentObserver`: callback receives suggestions, custom source label, partial observations

**Create** `tests/test_converters.py`:
- `TestCompositionSimplex`: encode/decode roundtrip, full CSV encode on HEA, validation of non-summing compositions
- `TestValidateMixedCatalog`: OER validation passes, column type detection

---

### Step 8: Documentation

**Modify** `CLAUDE.md`:
- Add `scripts/converters/` to architecture section
- Add converter CLI commands to quick reference table
- Add `run-simulated` to CLI reference
- Add converter workflow examples

---

## Files Summary

### New files (13)
| File | Purpose |
|------|---------|
| `bo_workflow/observers/simulated_human.py` | SimulatedHumanObserver |
| `bo_workflow/observers/agent.py` | AgentObserver (with validation, logging, formatting) |
| `scripts/converters/__init__.py` | Package marker |
| `scripts/converters/composition_simplex.py` | Simplex encode/decode (preserves original cols) |
| `scripts/converters/smiles_to_descriptors.py` | SMILES to RDKit descriptors |
| `scripts/converters/validate_mixed_catalog.py` | Mixed catalog validator |
| `.claude/skills/convert-composition/SKILL.md` | Composition converter skill |
| `.claude/skills/convert-smiles/SKILL.md` | SMILES converter skill |
| `.claude/skills/validate-catalog/SKILL.md` | Catalog validator skill |
| `.claude/skills/bo-run-simulated/SKILL.md` | Simulated human run skill |
| `.claude/skills/bo-execution-workflow/SKILL.md` | Meta-skill: full agentic orchestration pattern |
| `tests/test_observers.py` | Observer tests |
| `tests/test_converters.py` | Converter tests |

### Modified files (4)
| File | Change |
|------|--------|
| `bo_workflow/observers/__init__.py` | Export new observer classes |
| `bo_workflow/oracle_cli.py` | Add `run-simulated` subcommand |
| `pyproject.toml` | Add `chemistry` dependency group (rdkit) |
| `CLAUDE.md` | Document converters, new commands, updated architecture |

---

## Dependency Graph

```
Step 1: Composition Simplex (start here -- pure math, unblocks HEA)
    depends on: nothing
    blocks: convert-composition skill

Step 2: SimulatedHumanObserver
    depends on: ProxyObserver (exists)
    blocks: run-simulated CLI, bo-run-simulated skill

Step 3: Mixed Catalog Validator (independent)
    depends on: nothing
    blocks: validate-catalog skill

Step 4: SMILES Converter (independent)
    depends on: rdkit (new optional dependency)
    blocks: convert-smiles skill

Step 5: AgentObserver (independent)
    depends on: Observer ABC (exists)
    blocks: nothing directly

Step 6: Skills (depends on Steps 1-5)
    includes bo-execution-workflow meta-skill

Steps 1-5 can be parallelized -- they are independent of each other.
```

---

## Verification

1. **Composition converter** (first): `uv run python scripts/converters/composition_simplex.py encode --input data/HEA_alloy_data.csv --output /tmp/hea_encoded.csv --metadata /tmp/hea_meta.json --composition-columns Co Fe Mn V Cu --target target` -- verify output has z_0..z_3 + original Co,Fe,Mn,V,Cu as passthrough + target. Then init + run-proxy on encoded CSV.
2. **SimulatedHumanObserver**: `uv run pytest tests/test_observers.py::TestSimulatedHumanObserver -v`
3. **Validator**: `uv run python scripts/converters/validate_mixed_catalog.py --input data/OER_catalyst_data.csv --target "Overpotential mV @10 mA cm-2"`
4. **SMILES converter**: requires rdkit installed (`uv sync --group chemistry`)
5. **AgentObserver**: `uv run pytest tests/test_observers.py::TestAgentObserver -v` -- verify validation catches malformed observations, logging writes to JSONL
6. **Full integration**: encode HEA with simplex -> init -> build-oracle -> run-simulated with noise -> report -> decode best suggestion back to compositions
