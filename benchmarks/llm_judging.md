# LLM Judging Guide

This document defines how to use an independent LLM as the **primary qualitative judge** for the benchmark artifacts.

## Purpose

Use LLM judging for the qualitative layer only:

- problem framing
- workflow fidelity as reflected in artifacts
- interpretation quality
- caveat honesty
- paper usefulness
- HER-specific evaluator legitimacy, search-space validity, and scientific setup quality

Do **not** use the LLM as the primary source of truth for:

- OER numeric benchmark metrics
- artifact existence checks
- whether required files were produced
- whether numbers in the paper match `report.json`

Those remain deterministic checks.

## Judge Inputs

### Required grounding artifacts

For every judged run, provide:

- final paper artifact: `paper.tex` or `paper.md`
- `report.json`
- `state.json`

### Optional supporting artifacts

Provide these only when they materially improve grounding:

- `research_state.json`
- calibration or residual artifacts
- compiled PDF generated from the LaTeX source

### Do not use as primary judging inputs

- full raw conversation transcripts
- raw tool logs
- long unstructured shell output

If workflow trace evidence is needed, prefer `research_plan.md` or `research_state.json` over the raw transcript.

## Artifact-to-Criterion Mapping

### Common criteria

- **Problem framing**
  - read: Introduction, Methods, `state.json`
  - check whether the objective, constraints, and scope are coherent
- **Workflow fidelity**
  - read: Methods, artifact structure, `research_state.json` if present
  - check whether the run artifacts reflect a coherent end-to-end workflow
  - **important for asymmetric pairs:** assess each run relative to what it claimed to do, not relative to the richer artifact set of the other run. A naive run that produces a lean but internally consistent workflow should score higher than one that is incoherent, even if a paired orchestrated run produced more structured artifacts. Do not penalize a run for not producing artifacts it was not designed to produce.
- **Interpretation quality**
  - read: Results, Discussion, `report.json`
  - check whether the conclusions match the actual run outputs
- **Caveat honesty**
  - read: Discussion, Conclusion, `report.json`, `research_state.json` if present
  - check whether uncertainty and evaluator limits are stated proportionately
- **Paper usefulness**
  - read: the full paper artifact
  - check whether it is a credible, readable research output

### HER-specific criteria

- **Evaluator legitimacy**
  - read: Methods, `state.json`, `research_state.json`
  - check whether the evaluator is genuinely live/local and whether limitations are disclosed
- **Search-space validity**
  - read: Methods, `state.json`, `report.json`
  - check whether the described candidate space is feasible and aligned with the evaluator
- **Scientific setup quality**
  - read: Methods, Discussion, `research_state.json`, calibration artifacts if present
  - check whether the setup is coherent, calibrated where needed, and matched to the claims made

## Judging Modes

### Single-run mode

Use for:

- archival scoring of one run
- sanity-checking one paper artifact
- generating a structured score summary per run

### Pairwise mode

Use for:

- naive vs skilled comparison
- deciding which paper artifact is stronger
- HER primary qualitative comparison

Pairwise mode is the preferred mode for the paper narrative because relative judgment is more stable than isolated scoring.

## Required Judge Output Schema

The judge should return JSON only, using this shape:

```json
{
  "judge_model": "string",
  "task": "oer | her_live_structural",
  "mode": "single_run | pairwise",
  "runs": [
    {
      "run_label": "string",
      "paper_path": "string",
      "report_json_path": "string",
      "state_json_path": "string",
      "research_state_path": "string or null"
    }
  ],
  "factual_checks": {
    "paper_claims_match_report_json": "pass | fail | partial",
    "paper_setup_matches_state_json": "pass | fail | partial"
  },
  "scores": {
    "problem_framing": {
      "score": 0,
      "justification": "string",
      "evidence_quote": "string",
      "evidence_artifact": "string"
    },
    "workflow_fidelity": {
      "score": 0,
      "justification": "string",
      "evidence_quote": "string",
      "evidence_artifact": "string"
    },
    "interpretation_quality": {
      "score": 0,
      "justification": "string",
      "evidence_quote": "string",
      "evidence_artifact": "string"
    },
    "caveat_honesty": {
      "score": 0,
      "justification": "string",
      "evidence_quote": "string",
      "evidence_artifact": "string"
    },
    "paper_usefulness": {
      "score": 0,
      "justification": "string",
      "evidence_quote": "string",
      "evidence_artifact": "string"
    }
  },
  "her_scores": {
    "evaluator_legitimacy": {
      "score": 0,
      "justification": "string",
      "evidence_quote": "string",
      "evidence_artifact": "string"
    },
    "search_space_validity": {
      "score": 0,
      "justification": "string",
      "evidence_quote": "string",
      "evidence_artifact": "string"
    },
    "scientific_setup_quality": {
      "score": 0,
      "justification": "string",
      "evidence_quote": "string",
      "evidence_artifact": "string"
    }
  },
  "overall_outcome_category": "fail | demo-quality success | benchmark-quality success | not_applicable",
  "overall_summary": "string",
  "pairwise_comparison": {
    "winner": "run_a | run_b | tie | inconclusive",
    "why": "string"
  }
}
```

For OER runs, `her_scores` may be omitted or set to `null`. For single-run mode, `pairwise_comparison` may be omitted or set to `null`.

## Practical Process

1. Run deterministic checks first.
2. Run a **single-run judge** on each artifact set and store the JSON output.
3. Run a **pairwise judge** on the naive-vs-skilled pair and use that as the primary qualitative comparison.
4. Copy the structured results into the run scorecards and keep the raw judge JSON for auditability.

See:

- [`llm_judge_single_run_prompt.md`](llm_judge_single_run_prompt.md)
- [`llm_judge_pairwise_prompt.md`](llm_judge_pairwise_prompt.md)
