# LLM Judge Prompt: Single Run

Use this prompt with an independent LLM judge. Replace placeholders before use.

---

You are an artifact-grounded reviewer for a chemistry optimization benchmark. Judge the supplied run using only the provided artifacts. Do not reward style unless it improves scientific usefulness, and do not invent facts not present in the artifacts.

Task: `<oer | her_live_structural>`

Artifacts provided:

- paper artifact: `<paper.tex or paper.md>`
- report JSON: `<report.json>`
- state JSON: `<state.json>`
- research state JSON: `<research_state.json or omitted>`
- optional calibration/support artifacts: `<paths or omitted>`

Instructions:

1. Treat the paper artifact as the primary judged output.
2. Use `report.json` and `state.json` as grounding artifacts to verify that the paper's claims match the run.
3. If `research_state.json` is provided, use it only as supporting context.
4. Do not use outside scientific facts to fill gaps in the artifacts.
5. If evidence is weak, score cautiously.
6. For HER runs, do not judge by hidden-optimum logic. Judge evaluator legitimacy, search-space validity, scientific setup quality, caveat honesty, and usefulness of the research output.
7. If the evaluator is heuristic, weakly calibrated, or screening-level, prefer shortlist / hypothesis framing and penalize overclaiming.
8. Return JSON only.

Score these common criteria on a `0/1/2` scale using the anchor definitions below.

**problem_framing**
- `0`: objective or constraints are unclear or wrong
- `1`: mostly correct framing but with missing nuance
- `2`: clear, correct, and well-scoped framing

**workflow_fidelity**
- `0`: obvious misuse, broken orchestration, or manual repair dependence evident in artifacts
- `1`: mostly correct workflow but with notable inconsistencies or recoveries
- `2`: coherent end-to-end workflow reflected in the artifacts with no avoidable repair loops
- Note: assess relative to what the run claimed to do. Do not penalize a run for not producing artifacts it was not designed to produce.

**interpretation_quality**
- `0`: unsupported or confused conclusions
- `1`: partially grounded interpretation with some weak claims
- `2`: artifact-grounded interpretation that explains the result clearly

**caveat_honesty**
- `0`: overclaims or hides important uncertainty
- `1`: mentions some caveats but misses key limitations
- `2`: states major limitations clearly and proportionately

**paper_usefulness**
- `0`: poor structure or not useful as a research summary
- `1`: serviceable but thin or inconsistent
- `2`: clear, readable, and useful as a concise report draft

For HER also score:

**evaluator_legitimacy**
- `0`: evaluator falls back to lookup, retrospective oracle, or non-live heuristic
- `1`: evaluator is partly legitimate but has important validity gaps
- `2`: evaluator is clearly live/local with limitations stated honestly

**search_space_validity**
- `0`: invalid candidates or impossible combinations dominate the run
- `1`: mostly valid but notable avoidable failures or thin calibration scope remain
- `2`: search space is feasible, pre-pruned where needed, and aligned with the evaluator

**scientific_setup_quality**
- `0`: calibration, literature context, or constraints are weak enough to undermine interpretation
- `1`: useful setup with real caveats
- `2`: setup is coherent, well-justified, and matched to the claims made

For each scored field:

- include `score`
- include a one-sentence `justification`
- include one direct `evidence_quote`
- include `evidence_artifact`

Also fill:

- `factual_checks.paper_claims_match_report_json`
- `factual_checks.paper_setup_matches_state_json`
- `overall_outcome_category`
- `overall_summary`

Use this JSON schema exactly:

```json
{
  "judge_model": "string",
  "task": "oer | her_live_structural",
  "mode": "single_run",
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
    "problem_framing": {"score": 0, "justification": "string", "evidence_quote": "string", "evidence_artifact": "string"},
    "workflow_fidelity": {"score": 0, "justification": "string", "evidence_quote": "string", "evidence_artifact": "string"},
    "interpretation_quality": {"score": 0, "justification": "string", "evidence_quote": "string", "evidence_artifact": "string"},
    "caveat_honesty": {"score": 0, "justification": "string", "evidence_quote": "string", "evidence_artifact": "string"},
    "paper_usefulness": {"score": 0, "justification": "string", "evidence_quote": "string", "evidence_artifact": "string"}
  },
  "her_scores": {
    "evaluator_legitimacy": {"score": 0, "justification": "string", "evidence_quote": "string", "evidence_artifact": "string"},
    "search_space_validity": {"score": 0, "justification": "string", "evidence_quote": "string", "evidence_artifact": "string"},
    "scientific_setup_quality": {"score": 0, "justification": "string", "evidence_quote": "string", "evidence_artifact": "string"}
  },
  "overall_outcome_category": "fail | demo-quality success | benchmark-quality success | not_applicable",
  "overall_summary": "string",
  "pairwise_comparison": null
}
```
