# Benchmark Prompt Pack

These prompt files are the fixed run prompts for the benchmark and case-study conditions described in [`../scoring.md`](../scoring.md).

## OER closed-world benchmark

- `oer_skilled.md`: use inside a `skill_profile=full` benchmark workspace after explicitly invoking `/research-agent`
- `oer_naive.md`: use inside a `skill_profile=bo_only` benchmark workspace with plain Claude Code and no explicit research-layer skill invocation

## HER open-world case study

- `her_live_structural_naive.md`: no orchestration hint and no research-layer slash-command invocation
- `her_live_structural_light.md`: soft workflow hint in the initial prompt, but no explicit `/research-agent` or other research-layer slash command
- `her_live_structural_strong.md`: use after explicitly invoking `/research-agent`

For the HER core pair, use fresh clean repo workspaces from the same branch or commit and keep the ordinary project skill trees present in both workspaces. The only intended intervention difference is the explicit research-layer skill invocation.

All HER prompt variants require the same minimum review artifacts under `research_runs/<research_id>/`:

- `paper.tex`
- `research_plan.md`
- `research_state.json`

This keeps artifact-quality comparisons fair across the naive, light, and strong conditions.

The OER benchmark requires only `oer_skilled.md` and `oer_naive.md`.

The HER case-study core evidence package requires:

- `her_live_structural_naive.md`
- `her_live_structural_strong.md`

`her_live_structural_light.md` is an optional nudging-ablation support run.

## HEA open-world support case

- `hea_live_structural_naive.md`: no orchestration hint and no research-layer slash-command invocation
- `hea_live_structural_strong.md`: use after explicitly invoking `/research-agent`

The HEA support case is intentionally broader than HER: the run must choose and justify the HEA family, property, search space, and live evaluator for itself. Both prompt variants require the same minimum review artifacts under `research_runs/<research_id>/`:

- `paper.tex`
- `research_plan.md`
- `research_state.json`

Use fresh clean repo workspaces from the same branch or commit for the naive and strong runs. The intended intervention difference remains the explicit research-layer skill invocation, not a different code state or different evaluator permissions.

Manual mid-conversation nudges are not represented as fixed prompt files. When used, treat them as interactive rescue traces and log the follow-up messages separately in the HER scorecard.
