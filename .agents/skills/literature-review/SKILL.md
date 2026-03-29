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
  "computable_candidates": [],
  "evaluator_profile": {
    "mode": "lookup | surrogate | live_simulation | unknown",
    "evaluation_cost": "cheap | moderate | expensive",
    "stability_risk": "low | medium | high",
    "requires_run_local_setup": false,
    "why": ""
  }
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
- `evaluator_profile`: the routing summary that tells `research-agent` whether this looks like a lookup, surrogate, or live-simulation setup and how risky/expensive it appears

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
   - whether it implies a lookup, surrogate, or live-simulation evaluator path
   - whether it likely requires run-local setup code
   - whether the evaluator is cheap, moderate, or expensive to run
   - whether the setup looks low, medium, or high risk for stability
4. Prefer the most explicit, reproducible, operationalizable candidate rather than the most prestigious source.
5. If the user explicitly asked for a real DFT-style or first-principles evaluator, use the literature to recover the descriptor family, workflow, and stability constraints, but do not collapse the task into a tabulated literature lookup unless the user explicitly allows a lookup or surrogate fallback.
6. Only after that, gather lightweight baseline context: representative values, conditions, and variables the literature treats as important.
7. Stop once you have:
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

Set `evaluator_profile` using the best-supported path you actually found:

- `mode`
  - `lookup`: tabulated or database values are the intended evaluator
  - `surrogate`: a fitted or prebuilt predictive model is the intended evaluator
  - `live_simulation`: the evaluator is meant to be computed by running chemistry code or another simulator
  - `unknown`: no clear evaluator mode was established
- `evaluation_cost`
  - `cheap`: seconds or otherwise easy to probe repeatedly
  - `moderate`: nontrivial but still practical for repeated BO calls
  - `expensive`: each call is materially costly and should shape BO/search-space design
- `stability_risk`
  - `low`: evaluator path looks straightforward and stable
  - `medium`: some setup or convergence risk exists
  - `high`: evaluator path looks fragile enough that setup stabilization is likely needed
- `requires_run_local_setup`
  - `true` when Claude will likely need to write local scripts or otherwise operationalize the evaluator directly
- `why`
  - one short explanation of why you chose that profile

## Sparse Results Fallback

If the system is too narrow, novel, or obscure to find direct baselines:
- Search the broader material class (e.g., if no results for Mn-Fe-Co-Ni oxides, search multi-element oxide OER catalysts generally).
- Note explicitly that baselines are from an adjacent system, not the exact one.
- If still no useful baseline context is found, return an empty-but-valid structure and note that clearly in the summary.
- If no operationalizable evaluator path is found, leave `computable_candidates` empty and say so directly. Do not invent one.
- If no clear evaluator path is found, set `evaluator_profile.mode` to `unknown` and explain why.

## Guardrails

- Do not invent baselines; cite or clearly mark uncertainty.
- If `local_packet_path` is provided, do not browse beyond that packet.
- If browsing is used, include links or clear source attribution in the final summary.
- Do not over-specify design variables from literature if the user already provided stronger domain knowledge.
- If literature search is skipped, return an empty-but-valid structure and let `research-agent` proceed.
- Prefer explicit, reproducible sources over vague or prestige-only sources.
- If the user explicitly requires a real first-principles evaluator, do not quietly substitute a literature-value lookup or fitted surrogate as the main evaluator path.
- Fill `evaluator_profile` from the path you actually found; do not leave it vague when the literature clearly implies a live simulator or a lookup-based path.
