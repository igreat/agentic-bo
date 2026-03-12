# Project Scope, Status, and Next Steps

---

## What This Project Is

An **agent-operable Bayesian Optimization system for chemistry research**.

The goal: a chemistry researcher describes their problem in plain language, an AI agent (Claude Code) handles the entire BO workflow end-to-end, and the researcher receives actionable experiment suggestions. No Python knowledge required. No manual setup.

The system supports two modes:
- **Human-in-the-loop:** Claude suggests the next experiment → chemist runs it (wet lab or DFT) → chemist reports the result → repeat
- **Automated (proxy):** Claude runs a full BO loop against a surrogate oracle trained from existing data, for simulation and benchmarking

---

## Who This Is For

Two confirmed use cases from the chemistry team (form responses):

### Use Case 1 — Organic Synthesis Condition Optimization
- **Problem:** Which combination of catalyst, additive, solvent, temperature, and reaction time gives the highest yield or selectivity in a catalytic reaction?
- **Current method:** OFAT (one-factor-at-a-time) — inefficient, misses interaction effects between variables
- **Target:** Maximize yield and/or selectivity — straightforward, already supported
- **Data:** Usually no prior data for new reactions; starts from scratch
- **Features:** Mixed categorical (which reagent) + continuous (amounts, temperature, time)
- **Tools they already use:** EDBO+, Gryffin, Dragonfly for BO; RDKit, Mordred, morfeus for molecular descriptors
- **What they need from us:** Not another BO tool — they have those. They need the **agent interface** that removes setup friction entirely

### Use Case 2 — Computational Catalyst Design (DFT-based)
- **Problem:** Which Cu-alloy composition gives the lowest reaction barrier for Cu-based C-C coupling (targeting C2+ products in CO2 reduction)?
- **Current method:** High-throughput DFT screening — expensive, compute-limited
- **Target:** Minimize reaction barrier (free energy) — BO suggests which compositions to compute next
- **Data:** Can draw on Materials Project and Open Catalysis Project as starting datasets
- **Features:** DFT-computed properties (HOMO-LUMO, ESP, Mulliken charge, free energy) provided as plain numeric CSV — engine handles natively
- **Tools they use:** Gaussian, VASP, ASE, pymatgen
- **Open question:** Whether Cu-alloy element proportions sum to 1 (simplex constraint) or are independent concentrations — needs one clarifying question to Response 2 before the composition handling is finalized

### Out of Scope (confirmed)
- Automating Gaussian/VASP directly (submitting jobs, parsing output files) — the chemist runs these manually and reports results via chat. Python tools like AQME, cclib, and auto-qchem exist for output parsing but are not integrated.
- Multi-objective BO — HEBO is not well-suited for it; noted as future work
- Generative molecule design (inventing new SMILES via VAE) — current approach is nearest-neighbor lookup from a catalog, which is sufficient for library search problems

---

## What the Repository Currently Does

### Core Engine (`bo_workflow/engine.py`)
- Initializes a BO run from any tabular CSV, auto-infers numeric and categorical features
- Suggests next candidates via HEBO (default), BoTorch, BO-LCB, or random search
- Records observations (real or simulated)
- Persists all run state under `runs/<run_id>/` as JSON/JSONL — fully resumable
- Generates convergence reports and plots

### Proxy Oracle (`bo_workflow/oracle.py`)
- Trains a RandomForest or ExtraTrees surrogate from the dataset
- Used in automated proxy mode to simulate lab evaluations
- Supports feature selection (`--max-features`) for high-dimensional datasets

### CLI (`bo_workflow/cli.py`)
- `init` → `build-oracle` → `run-proxy` for automated mode
- `init` → `suggest` → `observe` (repeat) for human-in-the-loop mode
- All output is JSON; designed to be called by an AI agent

### Observers (`bo_workflow/observers/`)
- `ProxyObserver` — evaluates suggestions using the trained oracle
- `CallbackObserver` — delegates evaluation to a user-provided function

### Converters (`bo_workflow/converters/`)
- `molecule_descriptors.py` — encodes molecule SMILES into RDKit descriptors + Morgan fingerprint bits; decodes suggestions back to nearest real molecule via nearest-neighbor lookup
- `reaction_drfp.py` — encodes reaction SMILES into DRFP fingerprints (captures bond-breaking/forming patterns); decodes back to nearest real reaction
- `column_transform.py` — profiles columns and applies transforms (log10, sqrt, etc.) before BO
- `combined.py` — combined descriptor + fingerprint representations

### Scripts (`bo_workflow/scripts/`)
- `compare_optimizers.py` — benchmarks HEBO vs BO-LCB vs random on any dataset
- `compare_representations.py` — benchmarks descriptor vs DRFP vs combined representations
- `egfr_ic50_global_experiment.py` — full EGFR molecular optimization experiment (descriptor BO with real IC50 lookup)

### Claude Skills (`.claude/skills/`)
Skills exist for: `bo-init-run`, `bo-build-proxy-oracle`, `bo-next-batch`, `bo-record-observation`, `bo-report-run`, `bo-end-to-end-proxy`, `bo-encode-drfp`, `bo-decode-drfp`, `bo-encode-molecule-descriptors`, `bo-decode-molecule-descriptors`, `bo-transform-columns`

### Datasets (`data/`)
| Dataset | Status |
|---|---|
| `HER_virtual_data.csv` | Fully used — photocatalytic water splitting, 10 continuous features |
| `buchwald_hartwig_rxns.csv` | Used in DRFP converter and tests |
| `egfr_ic50.csv` | Fully used — EGFR molecular optimization experiment |
| `egfr_seed50_mixed.csv` | Used in EGFR experiment and tests |
| `HEA_alloy_data.csv` | In repo, has test — but simplex converter deferred (see open questions) |
| `OER_catalyst_data.csv` | In repo, has test — mixed categorical + numeric, engine handles natively |
| `BH_synthesis_data.csv` | In repo, has slow test — DFT features, feature selection with max_features=20 |

---

## What Is Missing

### 1. Agentic Orchestration Meta-Skill
**The most important gap.** All the individual skills exist but there is no skill telling Claude *how to handle a new problem from scratch end-to-end*. When a chemist says "I want to optimize the yield of my reaction, here is my dataset", Claude has to improvise the full workflow every time.

What's needed is a `bo-execution-workflow` skill that describes the orchestration pattern:
```
1. Understand the problem (what to optimize, objective direction, constraints)
2. Validate the dataset (check column types, target, missing values)
3. Convert if needed (SMILES columns → descriptors, etc.)
4. Initialize: bo-init-run
5. Build surrogate: bo-build-proxy-oracle (if using proxy mode)
6. Loop:
   a. bo-next-batch → get suggestions
   b. Decode to chemist-readable form (reagent names, not descriptor vectors)
   c. Present to chemist with context
   d. Chemist runs experiment, reports result
   e. bo-record-observation
7. bo-report-run → final summary
```

Without this, the agent interface that differentiates this project from existing BO tools (EDBO+, Gryffin) does not exist.

### 2. SimulatedHumanObserver
Before handing the system to real chemists, the full agentic loop needs to be tested without running real experiments. Currently the only end-to-end test uses `ProxyObserver` which evaluates with a perfect oracle — unrealistic.

`SimulatedHumanObserver` wraps `ProxyObserver` and adds:
- Configurable Gaussian noise on observations (simulates measurement error)
- Configurable refusal probability (simulates a chemist declining an infeasible suggestion)
- Reproducible via a seed

### 3. Mixed Catalog Validator
A diagnostic script that runs before `init` on any new dataset — checks column types, cardinality, missing values, detects potential issues (constant columns, high-cardinality categoricals, suspicious proportions). Outputs a `ready_for_engine` verdict with warnings.

Particularly needed for OER-style datasets with mixed metal categoricals and numeric process conditions.

### 4. AgentObserver
For fully automated agent loops where Claude evaluates suggestions programmatically (no human in the loop). Unlike `CallbackObserver` which blindly passes through, `AgentObserver` adds:
- Observation validation (checks `x`, `y` fields are present and `y` is numeric)
- Interaction logging (writes each suggest/observe exchange to JSONL for auditing)
- Partial result handling (handles cases where agent returns fewer observations than suggestions)

### 5. Tests
`tests/test_observers.py` and `tests/test_converters.py` do not exist. New observers need unit tests before being used in real workflows.

---

## Open Questions

| Question | Why it matters | How to resolve |
|---|---|---|
| Do Cu-alloy element proportions sum to 1? | Determines whether simplex converter is needed for Response 2 use case | Ask Response 2 directly: "Are your element fractions always proportions summing to 100%, or is Cu a fixed base with independent dopant concentrations?" |
| What is the exact optimization target for Response 2? | Their form answer listed DFT method/basis set settings instead of catalyst variables | Ask: "What value do you compute and optimize — is it reaction barrier energy (minimize), overpotential (minimize), or something else?" |

---

## Next Steps (Priority Order)

### Step 1 — `bo-execution-workflow` execution-layer skill
**No code required. Highest impact.**
Write `.claude/skills/bo-execution-workflow/SKILL.md` describing the full orchestration pattern for a new chemistry problem. This immediately makes the system usable by Use Case 1 chemists. Test by giving Claude a new dataset and problem description and verifying it runs the workflow correctly end-to-end.

### Step 2 — `SimulatedHumanObserver`
Create `bo_workflow/observers/simulated_human.py`. Add `run-simulated` CLI subcommand to `oracle_cli.py`. Export from `bo_workflow/observers/__init__.py`. Enables realistic end-to-end testing of the agentic loop before real chemistry experiments.

### Step 3 — Mixed Catalog Validator
Create `bo_workflow/converters/validate_catalog.py`. Diagnostic script with no ML dependencies — pure pandas column profiling. Add a corresponding `validate-catalog` skill.

### Step 4 — `AgentObserver`
Create `bo_workflow/observers/agent.py`. Add validation, interaction logging, and partial result handling. Export from `__init__.py`. Needed before any automated agent loop runs with real chemistry data.

### Step 5 — Tests
Create `tests/test_observers.py` covering `SimulatedHumanObserver` and `AgentObserver`. Verify noise, refusal, validation, and logging behaviour.
Create `tests/test_converters.py` covering the mixed catalog validator. Verify column type detection, missing value flagging, and the `ready_for_engine` verdict.

### Step 6 — Follow up with chemistry team
Ask the two open questions above. Resolving the composition question determines whether the simplex converter needs to come back on the roadmap.

---

## Deferred Items

- **Composition simplex converter** — needed only if Cu-alloy compositions sum to 1 (Case A). Deferred until composition representation is confirmed. Full math is in `A_Tutorial_for_Bayesian_Optimization_in_Scientific_Discovery.md` Appendix A.1 when ready.
- **Gaussian/VASP automation** — explicitly out of scope. Future work using tools like AQME or cclib for parsing DFT output files.
- **Multi-objective BO** — HEBO not well-suited. Future work.
- **Generative molecule decode** (latent vector → novel SMILES via VAE) — current nearest-neighbor lookup is sufficient for library search problems.
