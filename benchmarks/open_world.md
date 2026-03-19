# Open-World Benchmark Model

This document defines the new open-world benchmark style for the AI-scientist
story.

The benchmark is:

- **agent-open**: the agent sees a vague prompt and may browse, write helper
  scripts, install minimal dependencies, and revise its setup during the run
- **operator-grounded**: the operator keeps a hidden answer key that defines
  what families of solutions count as valid
- **evidence-driven**: scoring is based on fixed required artifacts rather than
  a fixed execution path

## Operator-side artifacts

Each open-world task should define three artifacts:

### 1. Agent prompt

The exact prompt shown to the agent.

This should be:

- plain English
- realistically vague
- paired with a nudge tier if applicable

### 2. Hidden operator spec

This is **not** shown to the agent. It exists so the run can be judged
afterward.

It should define:

- canonical evaluator family
- canonical design-parameter family
- canonical constraints
- expected verification artifact
- acceptable alternate solution families
- evaluation window / time budget
- nudge definitions `N0/N1/N2`

Use [`templates/open_world_operator_spec.template.json`](/Users/mujtabaalajmi/Documents/agentic-bo/bo-fun/benchmarks/templates/open_world_operator_spec.template.json)
as the starting point.

A concrete example now lives under:

- [`open_world_cases/her/agent_prompt.md`](/Users/mujtabaalajmi/Documents/agentic-bo/bo-fun/benchmarks/open_world_cases/her/agent_prompt.md)
- [`open_world_cases/her/operator_spec.json`](/Users/mujtabaalajmi/Documents/agentic-bo/bo-fun/benchmarks/open_world_cases/her/operator_spec.json)

### 3. Run evidence package

Every scored open-world run must leave behind:

- `research_runs/<research_id>/research_state.json`
- `research_runs/<research_id>/research_plan.md`
- `research_runs/<research_id>/paper.md`
- `research_runs/<research_id>/initial_prompt.md`
- `research_runs/<research_id>/discovered_search_space.json`
- `research_runs/<research_id>/evaluator.py`
- `research_runs/<research_id>/verification_artifacts/`
- `research_runs/<research_id>/operationalization_log.jsonl`
- normal BO artifacts under `bo_runs/<run_id>/`

## Scoring principle

The benchmark does **not** require a fixed path.

The agent may:

- use repo tooling
- bypass repo tooling
- write ad hoc converters or helper scripts
- install minimal new dependencies
- revise its setup mid-run

The requirements are:

- those actions are recorded
- the final setup is frozen before the reported BO run
- the resulting solution family is acceptable relative to the hidden operator
  spec

## Open-world success checklist

A run counts as structurally complete when:

- the exact prompt is saved
- the nudge tier is recorded
- credible source URLs are recorded
- the final evaluator module exists
- the final search-space artifact exists
- a verification artifact exists
- helper scripts and dependency installs are recorded if used
- the setup is marked frozen before the reported BO run
- BO completes and produces normal `bo_runs/<run_id>/` artifacts
- the final paper/report explains the discovery and operationalization path

## Notes

- Closed-world benchmark infrastructure remains useful as supporting/control
  evidence and does not need to disappear.
- Open-world runs should not be forced through the old task-bundle/public-
  workspace flow unless that flow is specifically useful for a given study.
