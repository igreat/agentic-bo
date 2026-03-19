# Open-World Cases

This directory holds **operator-side** assets for open-world AI-scientist
benchmarks.

These files are intended for:

- benchmark setup
- hidden answer keys
- nudge design
- post-run scoring

They are **not** meant to be given directly to the agent during a scored
open-world run.

Each case should contain:

- `agent_prompt.md`
- `operator_spec.json`

Optional supporting notes are fine, but the benchmark-critical fields should
live in the operator spec.
