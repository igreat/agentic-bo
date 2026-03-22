# Run Scorecard Template

Use one copy of this template per scored or case-study run.

## Run metadata

- run label:
- task: `oer` / `her_live_structural` / other
- baseline type: `skilled` / `light` / `naive`
- model:
- workspace type: `public benchmark workspace` / `open-world local workspace`
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

## Intervention log

Leave blank for clean baseline runs.

- trigger or timing:
- exact intervention message(s):
- reason for intervention:

## Core metrics

Fill the task-appropriate section below.

### OER benchmark metrics

- best value under budget:
- absolute gap to hidden optimum:
- percentile rank of best found point:
- normalized improvement over initial random phase:

### HER case-study metrics

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

## Qualitative rubric (`0/1/2`)

- problem framing:
- workflow fidelity:
- interpretation quality:
- caveat honesty:
- paper usefulness:

## HER case-study rubric (`0/1/2`)

Leave blank for non-HER tasks.

- evaluator legitimacy:
- search-space validity:
- scientific setup quality:

## Reviewer notes

- strongest positive signal:
- strongest limitation:
- merge/report significance:
