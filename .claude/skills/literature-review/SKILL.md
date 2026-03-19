---
name: literature-review
description: Produce a lightweight chemistry or materials literature summary for research-agent, focused on baselines, key variables, and known constraints.
---

# Literature Review

Use this skill as a focused helper for `research-agent`, not as a full systematic-review workflow.

## Goal

Given a framed research problem, collect the literature context needed to operationalize the study:

- find a computable evaluator path if one exists
- identify the inputs, design variables, and assumptions it requires
- collect only the baseline context needed to interpret the optimization study

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
  "summary": "",
  "computable_candidates": []
}
```

Also write a short narrative summary into the **Literature Context** section of `research_runs/<research_id>/research_plan.md`.

## What to Extract

- `baselines`: best or representative prior values for the target property, with source attribution
- `key_variables`: variables that the literature repeatedly treats as important
- `known_constraints`: physical, chemical, or experimental constraints that should inform BO setup
- `source_urls`: links or source identifiers for the papers, docs, repositories, or code artifacts actually used
- `summary`: 1–3 short paragraphs linking the literature to the experiment design
- `computable_candidates`: operationalizable evaluator/code/equation/tutorial/paper candidates that Claude could turn into a working local setup

## Local Packet Mode

If `local_packet_path` is provided, treat it as the authoritative literature environment for this run.

In that mode:

1. Read only the local markdown files in that packet.
2. Extract baselines, key variables, and constraints from those files only.
3. Write the Literature Context section as a summary of that boxed packet.
4. Do not browse the web.

This is the preferred mode for closed-world or control runs.

## Search Strategy

If no local packet is provided:

1. Start from the exact system and objective property — e.g. "OER overpotential in Mn-Fe-Co oxides" rather than a broad field label.
2. Search the web first for explicit evaluators in papers, equations, code, tutorials, repositories, docs, or simulators that expose a computable path.
3. For each promising candidate, identify:
   - what kind of source it is
   - what inputs it requires
   - what assumptions or simplifications are needed to operationalize it
   - whether it is explicit and reproducible enough to use in the workflow
4. Prefer the most explicit, reproducible, operationalizable candidate rather than the most prestigious source.
5. Only after that, gather lightweight baseline context: representative values, conditions, and variables the literature treats as important.
6. Stop once you have:
   - 1–3 viable computable candidates, or a clear statement that none were found
   - enough baseline context to interpret the eventual optimization result

Each `computable_candidates` item should be an object with:

```json
{
  "label": "",
  "kind": "paper | equation | code | tutorial | repository | simulator",
  "source_url": "",
  "inputs": [],
  "notes": ""
}
```

## Sparse Results Fallback

If the system is too narrow, novel, or obscure to find direct baselines:
- Search the broader material class (e.g., if no results for Mn-Fe-Co-Ni oxides, search multi-element oxide OER catalysts generally).
- Note explicitly that baselines are from an adjacent system, not the exact one.
- If still no useful baseline context is found, return an empty-but-valid structure and note that clearly in the summary.
- If no operationalizable evaluator path is found, leave `computable_candidates` empty and say so directly. Do not invent one.

## Guardrails

- Do not invent baselines; cite or clearly mark uncertainty.
- If `local_packet_path` is provided, do not browse beyond that packet.
- If browsing is used, include links or clear source attribution in the final summary.
- Do not over-specify design variables from literature if the user already provided stronger domain knowledge.
- If literature search is skipped, return an empty-but-valid structure and let `research-agent` proceed.
- Prefer explicit, reproducible sources over vague or prestige-only sources.
