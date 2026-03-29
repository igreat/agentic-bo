# BO-Only Workspace

This workspace is intentionally stripped down for BO-only autonomous chemistry
or materials runs.

There is no project-defined research-agent workflow in this workspace. Work
directly with the code, prompts, and local tools that remain available. The
main optimization entrypoints live under `bo_workflow/`.

Use `bo_runs/<run_id>/` for BO run state. If the prompt requires a paper or
other review artifacts, write them under `research_runs/<research_id>/`.

## Setup

```bash
uv sync
uv pip install --no-deps "hebo @ git+https://github.com/huawei-noah/HEBO.git#subdirectory=HEBO"
```

## Guardrails

- Do not assume any hidden research-layer slash command exists.
- If a live structural evaluator is required, build an executable local path or
  stop and explain why it is infeasible.
- Keep output artifacts concise, machine-readable where requested, and rooted in
  the directories named in the prompt.
