# BO-Only Workspace

This workspace is prepared for BO-focused chemistry and materials optimisation
runs without the top-level research-agent workflow.

Use the local code and tools in this workspace directly. Do not assume that a
project-defined research-layer workflow, slash command, or orchestration skill
is available here.

The BO execution layer lives under `bo_workflow/`. BO run state should be
written under `bo_runs/<run_id>/`. If the prompt asks for a paper or report,
place those artifacts under `research_runs/<research_id>/`.

## Setup

```bash
uv sync
uv pip install --no-deps "hebo @ git+https://github.com/huawei-noah/HEBO.git#subdirectory=HEBO"
```

## Available capabilities

- `uv run python -m bo_workflow.cli init`
- `uv run python -m bo_workflow.cli suggest`
- `uv run python -m bo_workflow.cli observe`
- `uv run python -m bo_workflow.cli report`
- `uv run python -m bo_workflow.cli run-python-evaluator`

## Guardrails

- If the prompt requires a live local evaluator, do not silently replace it
  with a lookup table or proxy oracle.
- Use literature and databases for orientation, calibration, and validation
  only when the prompt allows that.
- Persist any requested report artifacts exactly where the prompt specifies.
