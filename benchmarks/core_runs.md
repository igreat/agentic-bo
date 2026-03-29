# Core Run Matrix

This file locks the exact **4-run primary evidence package** for the report.

These runs precede any supplementary support evidence such as `HER light`, interactive rescue traces, or HEA paired reruns.

## Core runs

| Run | Task | Baseline | Workspace | Prompt | Skill invocation | What it proves |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `oer` | naive | public benchmark workspace built with `--skill-profile bo_only` | `benchmarks/prompts/oer_naive.md` | none | plain Claude Code under fixed benchmark conditions |
| 2 | `oer` | skilled | public benchmark workspace built with `--skill-profile full` | `benchmarks/prompts/oer_skilled.md` | explicit `/research-agent` before pasting prompt | effect of the research-layer orchestration in a closed-world benchmark |
| 3 | `her_live_structural` | naive | fresh clean repo workspace from the same branch/code state as run 4; keep the normal project skill trees present | `benchmarks/prompts/her_live_structural_naive.md` | no explicit research-layer slash command invocation | open-world baseline without explicit orchestration |
| 4 | `her_live_structural` | skilled | fresh clean repo workspace from the same branch/code state as run 3; keep the normal project skill trees present | `benchmarks/prompts/her_live_structural_strong.md` | explicit `/research-agent` before pasting prompt | effect of the research workflow in the main open-world case study |

## Required fixed conditions

### OER

Keep these fixed across the naive and skilled runs:

- same task bundle: `tasks/oer/`
- same hidden backend: `oer_hidden`
- same budget: `100` iterations, batch size `1`
- same no-web rule
- same public benchmark workspace build process
- same model family and Claude Code version when practical
- same effort level
- same clean starting chat
- no mid-run operator intervention

Only the skill surface and explicit orchestration choice should differ.

### HER

Keep these fixed across the naive and skilled runs:

- same branch / code state
- same model family
- same effort level
- same initial prompt family
- same live-structural evaluator requirement
- same access to local code and web research
- no fixed iteration/query budget in the prompt; either let both runs self-terminate under the same stop policy or apply the same external wall-clock cap and record it
- same clean starting chat
- no mid-run operator intervention
- same operator stop policy:
  - let the run continue uninterrupted until the agent explicitly declares the workflow complete, or
  - stop it only if it is clearly stalled or looping without forward progress, and record that as incomplete in the scorecard

Use **separate fresh workspaces** for the HER naive and HER skilled runs so package installs and artifacts do not contaminate the comparison. Unlike OER, the HER naive workspace should still contain the ordinary project skill trees on disk; the intervention is the absence of explicit research-layer slash-command invocation, not a stripped repo.

## Workspace setup

### OER skilled workspace

Build with:

```bash
uv run python benchmarks/build_workspace.py \
  --output-dir /tmp/agentic-bo-oer-skilled \
  --tasks oer \
  --skill-profile full \
  --overwrite
```

Then inside that workspace:

```bash
uv sync
uv pip install --no-deps "hebo @ git+https://github.com/huawei-noah/HEBO.git#subdirectory=HEBO"
```

### OER naive workspace

Build with:

```bash
uv run python benchmarks/build_workspace.py \
  --output-dir /tmp/agentic-bo-oer-naive \
  --tasks oer \
  --skill-profile bo_only \
  --overwrite
```

Then inside that workspace:

```bash
uv sync
uv pip install --no-deps "hebo @ git+https://github.com/huawei-noah/HEBO.git#subdirectory=HEBO"
```

### HER naive workspace

Use a fresh clean repo workspace, for example:

- `/tmp/agentic-bo-her-naive`

Requirements before the run starts:

- same branch / commit as the skilled HER workspace
- empty `bo_runs/`
- empty `research_runs/`
- fresh `.venv`
- no preinstalled `ase`, `mace`, `chgnet`, or `gpaw`
- keep the normal `.agents/` and `.claude/` skill trees present

### HER skilled workspace

Use a second fresh clean repo workspace, for example:

- `/tmp/agentic-bo-her-skilled`

Apply the same cleanliness rules as the HER naive workspace.

## Recommended execution order

1. **OER naive**
2. **OER skilled**
3. **HER naive**
4. **HER skilled**

This order keeps the clean benchmark pair together and then moves to the open-world case study.

An **unscored manual OER smoke run** may be used to validate the benchmark path before collecting report evidence. Do not count that smoke run as one of the four core evidence runs.

## Logging and scoring

For each of the four core runs:

- for OER, fill one copy of [`oer_scorecard_template.md`](oer_scorecard_template.md)
- for HER, fill one copy of [`her_scorecard_template.md`](her_scorecard_template.md)
- record the exact prompt file used
- record the effort level used for the session
- record whether `/research-agent` was invoked
- record whether manual repair was needed
- score against the task-appropriate rubric in [`scoring.md`](scoring.md)

Do not include optional `HER light` or interactive rescue traces in the main four-run comparison table.
