# HER Run Scorecard

Use one copy of this template per `her_live_structural` case-study run.

## Run metadata

- run label:
- task: `her_live_structural`
- baseline type: `skilled` / `light` / `naive`
- model:
- effort level:
- workspace type: `open-world local workspace`
- prompt file or prompt id:
- intervention type: `none` / `prompt_only` / `interactive_rescue`
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
- calibration / support artifact paths (optional):

## Intervention log

Leave blank for clean baseline runs.

- trigger or timing:
- exact intervention message(s):
- reason for intervention:

## HER case-study metrics

- evaluator type:
- evidence class:
- claim posture:
- best reported result:
- calibration scope summary:
- notable failures:
- outcome category: `fail` / `demo-quality success` / `benchmark-quality success`

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

## HER case-study rubric (`0/1/2`, copied from judge output)

- evaluator legitimacy:
- search-space validity:
- scientific setup quality:

## Reviewer notes

- strongest positive signal:
- strongest limitation:
- merge/report significance:
