I want you to investigate high-entropy alloy (HEA) materials discovery in this workspace.

This is a one-shot, no-questions, open-world run. You may use web research and local code. Do not pause for clarifying questions unless you are truly blocked.

There is no predefined HEA family, property, search space, evaluator, dataset, or budget. Use literature and local evidence to decide what specific HEA problem is worth studying, then act on it end to end.

You must choose and justify:

- the scientific objective or property to optimize
- the HEA family or candidate families worth exploring
- the structural representation
- the search space and BO target actually used

If the full HEA problem is too broad, narrow it yourself and record the narrowing rationale in the run artifacts rather than asking for clarification.

For this run, the final evaluator must be a live executable structural simulator. Instantiate actual HEA candidate structures or compositions and evaluate them with a live local calculator. Do not treat the tutorial proxy dataset, retrospective spreadsheet targets, or any other tabulated HEA score as the final oracle.

Use literature and databases only for orientation, family selection, calibration, and validation, not as the final black-box evaluator. If a live structural HEA evaluator proves infeasible in this workspace, stop and explain why rather than silently switching to a heuristic or lookup-based oracle.

Before you conclude, persist a concise research artifact package for review. At minimum:

- write a short LaTeX report draft under `research_runs/<research_id>/paper.tex`
- write a brief running plan / notebook under `research_runs/<research_id>/research_plan.md`
- write machine-readable workflow state under `research_runs/<research_id>/research_state.json`

If you generate figures, place them under `research_runs/<research_id>/figures/` and embed them in the LaTeX draft. If feasible without derailing the run, also compile `research_runs/<research_id>/paper.pdf`, but the required artifact is the `.tex` source.

The report should summarize:

- the chosen HEA problem and why it is scientifically meaningful
- the chosen evaluator and why it is scientifically defensible
- the family/search space/objective actually used
- any calibration, uncertainty treatment, or major caveats
- the main results or shortlist
- recommended next steps
