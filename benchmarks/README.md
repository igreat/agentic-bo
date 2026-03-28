# Benchmarks

This directory holds the benchmark setup for the final report.

The exact required report runs are locked in [`core_runs.md`](core_runs.md).

Currently, one task is fully packaged:

- `oer`: the flagship OER composition benchmark

The benchmark model is intentionally simple:

- the **root repo** is the operator/developer environment
- a separate **built public workspace** is where the agent runs
- labeled source datasets stay in the root repo
- the public workspace contains only:
  - code/docs/skills needed to run
  - `tasks/<task_id>/...`
  - prebuilt evaluator assets under `evaluation_backends/`
  - fresh `bo_runs/` and `research_runs/`

## OER quick start

### 1. In the root repo, install dependencies

```bash
uv sync
uv pip install --no-deps "hebo @ git+https://github.com/huawei-noah/HEBO.git#subdirectory=HEBO"
```

### 2. In the root repo, build the OER backend

```bash
uv run python -m bo_workflow.cli build-oracle \
  --dataset data/caltech_oer/plate_3496.csv \
  --target overpotential_V \
  --objective min \
  --backend-id oer_hidden
```

This creates:

- `evaluation_backends/oer_hidden/oracle.pkl`
- `evaluation_backends/oer_hidden/oracle_meta.json`

### 3. In the root repo, build the public workspace

Build one workspace per baseline condition:

- `full` skill profile for the skilled baseline
- `bo_only` skill profile for the naive baseline

In the examples below, replace:

- `<SKILLED_WORKSPACE>` with a clean output directory for the skilled run
- `<NAIVE_WORKSPACE>` with a clean output directory for the naive run

```bash
uv run python benchmarks/build_workspace.py \
  --output-dir <SKILLED_WORKSPACE> \
  --tasks oer \
  --skill-profile full \
  --overwrite

uv run python benchmarks/build_workspace.py \
  --output-dir <NAIVE_WORKSPACE> \
  --tasks oer \
  --skill-profile bo_only \
  --overwrite
```

This also writes a benchmark-specific `.claude/settings.local.json` inside the built workspace so Claude Code can run shell commands and write artifacts without approval prompts while keeping Claude-native web/search disabled.

### 4. Switch into the built workspace and install dependencies there too

```bash
cd <SKILLED_WORKSPACE>
uv sync
uv pip install --no-deps "hebo @ git+https://github.com/huawei-noah/HEBO.git#subdirectory=HEBO"
```

Repeat the same install inside `<NAIVE_WORKSPACE>`.

You need to install in each built workspace because they are separate working directories with their own environments.

### 5. Manual smoke run

Initialize a BO run from the public task bundle:

```bash
uv run python -m bo_workflow.cli init \
  --search-space-json tasks/oer/search_space.json \
  --target overpotential_V \
  --objective min \
  --simplex-groups 'Mn_molar_fraction,Fe_molar_fraction,Co_molar_fraction,Ni_molar_fraction,La_molar_fraction,Ce_molar_fraction:1' \
  --seed 42
```

Copy the returned `run_id`, then run the evaluator:

```bash
uv run python -m bo_workflow.cli run-evaluator \
  --run-id <RUN_ID> \
  --backend-id oer_hidden \
  --iterations 100 \
  --batch-size 1
```

Finish with:

```bash
uv run python -m bo_workflow.cli report --run-id <RUN_ID>
```

Artifacts will be written under `bo_runs/<RUN_ID>/`.

### 6. Full agent run

For the report, this should be executed in two variants:

- **skilled baseline**: `full` skill-profile workspace + explicit `/research-agent`
- **naive baseline**: `bo_only` skill-profile workspace + plain Claude Code, no explicit research-layer skill invocation

The task bundle, budget, and evaluator must stay fixed between the two variants. The only intended difference is the research-layer orchestration surface.

Run the agent from inside the corresponding built workspace:

- `<SKILLED_WORKSPACE>`
- `<NAIVE_WORKSPACE>`

In either case, point it at:

- `tasks/oer/brief.md`
- `tasks/oer/task_manifest.json`
- `tasks/oer/search_space.json`
- `tasks/oer/literature/`

Use the fixed prompt files under [`prompts/`](prompts/):

- [`oer_skilled.md`](prompts/oer_skilled.md)
- [`oer_naive.md`](prompts/oer_naive.md)

For the skilled baseline, invoke `/research-agent` first and then paste [`oer_skilled.md`](prompts/oer_skilled.md).

For the naive baseline, paste [`oer_naive.md`](prompts/oer_naive.md) with no explicit research-layer skill invocation.

### 7. HER case-study prompts

The HER open-world case study uses the fixed prompt pack:

- [`her_live_structural_naive.md`](prompts/her_live_structural_naive.md)
- [`her_live_structural_light.md`](prompts/her_live_structural_light.md)
- [`her_live_structural_strong.md`](prompts/her_live_structural_strong.md)

Use these as:

- **naive**: paste `her_live_structural_naive.md`
- **lightly nudged**: paste `her_live_structural_light.md`
- **strong**: invoke `/research-agent`, then paste `her_live_structural_strong.md`

The report core requires the naive and strong HER runs. The light prompt is an optional nudging-ablation support run.

### 8. HEA open-world support prompts

An optional third open-world support case can use:

- [`hea_live_structural_naive.md`](prompts/hea_live_structural_naive.md)
- [`hea_live_structural_strong.md`](prompts/hea_live_structural_strong.md)

Use these as:

- **naive**: paste `hea_live_structural_naive.md`
- **strong**: invoke `/research-agent`, then paste `hea_live_structural_strong.md`

Unlike HER, this HEA support case deliberately leaves the family, objective, search space, and live evaluator unresolved at prompt time. The agent is expected to narrow the problem itself, justify the resulting HEA study design from literature, and avoid using the tutorial proxy dataset or any tabulated HEA score as the final evaluator.

As with HER, run the naive and strong HEA pair from fresh clean repo workspaces on the same branch or commit, starting from empty chats and allowing no operator nudges during the core run.

For the HER core pair, use two fresh clean repo workspaces from the same branch or commit. Keep the normal project skill trees present in both workspaces. The only intended intervention difference is whether a research-layer slash command is invoked. The naive and light conditions should not call `/research-agent` or other research-layer slash commands. Start each HER run from a fresh empty chat and let it continue uninterrupted until the agent explicitly declares completion, unless it is clearly stalled or looping. If you apply an external wall-clock cap, use the same cap for both runs and record it in the scorecard.

For fairness, all HER prompt variants now require the same minimum persisted review artifacts under `research_runs/<research_id>/`:

- `paper.tex`
- `research_plan.md`
- `research_state.json`

The prompt may still produce richer artifacts in the strong condition, but the baseline should not be marked down for missing a report artifact that it was never explicitly asked to create.

Mid-conversation nudging should be treated as a separate interactive rescue trace rather than a normal baseline:

- start from the naive or light HER prompt
- log the exact follow-up message(s) sent by the operator
- record why and when the intervention occurred
- do not fold the rescued run into the main skilled-vs-naive benchmark table as if it were a clean baseline condition

## What `build_workspace.py` does

`benchmarks/build_workspace.py` copies a stripped set of files into the public workspace:

- root files:
  - `AGENTS.md`
  - `README.md`
  - `pyproject.toml`
  - `uv.lock`
  - `.python-version`
  - `.gitignore`
- root directories:
  - `bo_workflow/`
  - `.agents/`
  - `.claude/`
- a benchmark-specific `.claude/settings.local.json` with:
  - `defaultMode: acceptEdits`
  - `permissions.allow: [Bash]`
  - `permissions.deny: [WebSearch, WebFetch]`
- selected benchmark task bundles from `benchmarks/tasks/`
- any prebuilt backend named by `evaluation.backend_id` in the task manifest, if it already exists under root `evaluation_backends/`

With `--skill-profile bo_only`, the copied workspace keeps BO/converter skills but strips the top-level research-layer skills:

- `.agents/skills/research-agent`
- `.agents/skills/literature-review`
- `.agents/skills/scientific-writing`
- `.claude/skills/research-agent`
- `.claude/skills/literature-review`
- `.claude/skills/scientific-writing`
- `.claude/skills/evaluator-design`

It also creates empty:

- `bo_runs/`
- `research_runs/`

## Task bundle shape

Each public task bundle may include:

- `brief.md`
- `task_manifest.json`
- optional `search_space.json`
- optional `seed_observations.csv`
- optional `literature/`

The manifest may also declare the intended workflow entrypoint, e.g. `workflow.entrypoint = research-agent`.

For `oer`, the task bundle lives at:

- `benchmarks/tasks/oer` in the root repo

## Scored run rules

- no web search
- use only local literature packets when present
- observations come only from the prebuilt evaluator in the public workspace
- no direct access to labeled source datasets
- no manual artifact editing before scoring

## Relationship to HER

The HER live-structural work is not the packaged public benchmark in this directory. Treat HER as a separate open-world case study:

- same skilled-vs-naive comparison idea
- different scoring model
- no hidden-optimum claim

See:

- [`scoring.md`](scoring.md) for the benchmark metrics and case-study rubric
- [`llm_judging.md`](llm_judging.md) for the LLM-primary qualitative judging spec
- [`core_runs.md`](core_runs.md) for the exact 4-run core evidence package
- [`prompts/`](prompts/) for the fixed prompt files used by those runs
