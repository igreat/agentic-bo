I want you to investigate hydrogen evolution reaction (HER) catalyst optimization in this workspace.

This is a one-shot, no-questions, open-world run. You may use web research and local code. Do not pause for clarifying questions unless you are truly blocked.

Use the research workflow in this workspace, and keep the standard workflow phases explicit in the run artifacts. Do not invoke `/research-agent` or other project-defined research-layer slash commands.

There is no predefined search space, evaluator, dataset, or budget. Figure out a scientifically defensible computational path and act on it end to end.

For this run, the final evaluator must be a live executable structural simulator. Build actual candidate structures, place the relevant adsorbate(s), and evaluate them with a live local calculator.

Use literature and databases only for orientation, calibration, and validation, not as the final black-box evaluator. If a live structural evaluator proves infeasible in this workspace, stop and explain why rather than silently switching to a heuristic or lookup-based oracle.

Before you conclude, persist a concise research artifact package for review. At minimum:

- write a short LaTeX report draft under `research_runs/<research_id>/paper.tex`
- write a brief running plan / notebook under `research_runs/<research_id>/research_plan.md`
- write machine-readable workflow state under `research_runs/<research_id>/research_state.json`

If you generate figures, place them under `research_runs/<research_id>/figures/` and embed them in the LaTeX draft. If feasible without derailing the run, also compile `research_runs/<research_id>/paper.pdf`, but the required artifact is the `.tex` source.

The report should summarize:

- the chosen evaluator and why it is scientifically defensible
- the search space and objective actually used
- any calibration, uncertainty treatment, or major caveats
- the main results or shortlist
- recommended next steps
