# OER Run Scorecard

Use one copy of this template per scored `oer` benchmark run.

## Run metadata

- run label:
- task: `oer`
- baseline type: `skilled` / `naive`
- model:
- effort level:
- workspace type: `public benchmark workspace`
- prompt file or prompt id:
- intervention type: `none` / `interactive_rescue`
- clean starting chat: `yes` / `no`
- wall-clock duration:
- run id:
- bo run id:
- research run id:
- commit or branch:
- completed: `yes` / `no`
- manual repair required: `yes` / `no`

## Grounding artifact paths

- primary paper artifact:
- compiled PDF path (optional):
- `report.json` path:
- `state.json` path:
- `research_state.json` path (optional):

## Intervention log

Leave blank for clean baseline runs.

- trigger or timing:
- exact intervention message(s):
- reason for intervention:

## OER benchmark metrics

- best value under budget:
- absolute gap to hidden optimum:
- percentile rank of best found point:
- normalized improvement over initial random phase:

## Workflow correctness checklist

- valid setup from task materials: `pass` / `fail`
- hard constraints respected: `pass` / `fail`
- required BO artifacts produced: `pass` / `fail`
- required research artifacts produced: `pass` / `fail`
- evaluator path matched the task rules: `pass` / `fail`
- run completed without manual repair: `pass` / `fail`

## Judge outputs

- deterministic check reviewer or script:
- single-run judge model:
- single-run judge output path:
- pairwise judge model (optional):
- pairwise judge output path (optional):
- paper claims match `report.json`: `pass` / `fail` / `partial`
- paper setup matches `state.json`: `pass` / `fail` / `partial`

## Qualitative rubric (`0/1/2`, copied from judge output)

- problem framing:
- workflow fidelity:
- interpretation quality:
- caveat honesty:
- paper usefulness:

## Reviewer notes

- strongest positive signal:
- strongest limitation:
- merge/report significance:
