You are working with the closed-world benchmark task bundle at `tasks/oer/`.

Use only the public task materials in this workspace:

- `tasks/oer/brief.md`
- `tasks/oer/task_manifest.json`
- `tasks/oer/search_space.json`
- `tasks/oer/literature/`

Do not use web search.

Do not access or assume any labeled source dataset outside this public workspace.

Use the prebuilt evaluator specified by the task bundle. Do not build your own oracle or replace the evaluator path.

Respect the fixed task budget and the simplex constraint over all six molar fractions.

Please execute the task end to end, produce the BO artifacts under `bo_runs/` and the research artifacts under `research_runs/`, and draft the final report or paper.

Treat this as a scored benchmark run and do not pause between phases unless blocked.
