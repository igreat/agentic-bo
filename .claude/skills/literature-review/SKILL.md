---
name: literature-review
description: Produce a lightweight chemistry or materials literature summary for research-agent, focused on baselines, key variables, and known constraints.
---

# Literature Review

Use this skill as a focused helper for `research-agent`, not as a full systematic-review workflow.

## Goal

Given a framed research problem, collect only the literature context needed to set up and interpret the optimization study.

## Inputs

- System or material class
- Objective property
- Objective direction
- Optional dataset context
- Optional `local_packet_path` for a benchmark-frozen literature packet
- Path to `research_runs/<research_id>/research_plan.md` (to write the Literature Context section)

## Output Contract

Return findings in this structure so they can be written into `research_state.json.literature_findings`:

```json
{
  "baselines": [],
  "key_variables": [],
  "known_constraints": [],
  "source_urls": [],
  "summary": ""
}
```

Also write a short narrative summary into the **Literature Context** section of `research_runs/<research_id>/research_plan.md`.

## What to Extract

- `baselines`: best or representative prior values for the target property, with source attribution
- `key_variables`: variables that the literature repeatedly treats as important
- `known_constraints`: physical, chemical, or experimental constraints that should inform BO setup
- `source_urls`: links or source identifiers for the papers, docs, or repositories actually used
- `summary`: 1–3 short paragraphs linking the literature to the experiment design

## Local Packet Mode

If `local_packet_path` is provided, treat it as the authoritative literature
environment for this run.

In that mode:

1. Read only the local markdown files in that packet.
2. Extract baselines, key variables, and constraints from those files only.
3. Write the Literature Context section as a summary of that boxed packet.
4. Do not browse the web.

This is the preferred mode for closed-world benchmark runs.

## Search Strategy

If no local packet is provided:

1. Start from the exact system and objective property — e.g. "OER overpotential Mn-Fe-Co oxides" not just "electrocatalysis".
2. Use web search to find recent reviews or benchmark studies for this system. Prefer sources from the last 5 years unless a classic baseline is widely cited.
3. For each relevant result: extract the best reported value, the conditions it was achieved under, and what design variables were varied.
4. Stop once you have 2–4 baselines and a clear picture of what variables the community treats as important. This is not a systematic review.

## Sparse Results Fallback

If the system is too narrow, novel, or obscure to find direct baselines:
- Search the broader material class (e.g., if no results for Mn-Fe-Co-Ni oxides, search multi-element oxide OER catalysts generally).
- Note explicitly that baselines are from an adjacent system, not the exact one.
- If still nothing useful is found, return an empty-but-valid structure and note in the summary that no relevant baselines were located. Do not invent numbers.

## Guardrails

- Do not invent baselines; cite or clearly mark uncertainty.
- If `local_packet_path` is provided, do not browse beyond that packet.
- If browsing is used, include links or clear source attribution in the final summary.
- Do not over-specify design variables from literature if the user already provided stronger domain knowledge.
- If literature search is skipped, return an empty-but-valid structure and let `research-agent` proceed.
