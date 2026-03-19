# HER Open-World Discovery

You are investigating hydrogen evolution reaction (HER) catalyst design as an
open-world AI chemistry scientist task.

The task is intentionally not pre-specified with a frozen search space or a
provided evaluator. You may use web search to:

- identify a credible, computable HER evaluator or analytical model
- decide a useful catalyst family or composition space to explore
- determine any variables and constraints needed to run optimization

Your success criteria are:

- operationalize the evaluator locally as a Python function or module
- generate a verification graph that demonstrates the evaluator behaves as
  expected for the chosen HER model
- run Bayesian optimization end to end on the discovered search space
- produce the normal research workflow artifacts under `research_runs/` and
  `bo_runs/`

If you need a hint, nudge tiers may be introduced separately and should be
recorded explicitly.
