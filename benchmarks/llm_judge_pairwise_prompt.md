# LLM Judge Prompt: Pairwise Comparison

Use this prompt with an independent LLM judge. Replace placeholders before use.

---

You are an artifact-grounded reviewer comparing two chemistry optimization run outputs. Judge the pair using only the supplied artifacts. Do not reward verbosity. Prefer the run that is more scientifically credible, better grounded, and more useful as a research output.

Task: `<oer | her_live_structural>`

Run A artifacts:

- paper artifact: `<run_a paper.tex or paper.md>`
- report JSON: `<run_a report.json>`
- state JSON: `<run_a state.json>`
- research state JSON: `<run_a research_state.json or omitted>`

Run B artifacts:

- paper artifact: `<run_b paper.tex or paper.md>`
- report JSON: `<run_b report.json>`
- state JSON: `<run_b state.json>`
- research state JSON: `<run_b research_state.json or omitted>`

Instructions:

1. Treat the paper artifacts as the primary judged outputs.
2. Use `report.json` and `state.json` to ground and verify each paper.
3. For HER, do not compare objective values as if this were a hidden-optimum benchmark when the runs operationalized different scientific problems.
4. Judge which output is stronger as a research artifact.
5. If both runs are weak, you may still choose a winner, but say why cautiously.
6. Return JSON only.

Score each run on the `0/1/2` criteria below. Apply the same anchor definitions to both runs so the scores are on a common scale.

**problem_framing**
- `0`: objective or constraints are unclear or wrong
- `1`: mostly correct framing but with missing nuance
- `2`: clear, correct, and well-scoped framing

**workflow_fidelity**
- `0`: obvious misuse, broken orchestration, or manual repair dependence evident in artifacts
- `1`: mostly correct workflow but with notable inconsistencies or recoveries
- `2`: coherent end-to-end workflow reflected in the artifacts with no avoidable repair loops
- Note: assess each run relative to what it claimed to do. Do not penalize a naive run for not producing orchestrated-workflow artifacts it was not asked to produce. Workflow fidelity measures internal coherence, not artifact richness.

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

For each score:

- include `score`
- include a one-sentence `justification`
- include one direct `evidence_quote`
- include `evidence_artifact`

Then provide a final pairwise decision:

- `winner`
- `why`

Use this JSON schema exactly:

```json
{
  "judge_model": "string",
  "task": "oer | her_live_structural",
  "mode": "pairwise",
  "runs": [
    {
      "run_label": "run_a",
      "paper_path": "string",
      "report_json_path": "string",
      "state_json_path": "string",
      "research_state_path": "string or null"
    },
    {
      "run_label": "run_b",
      "paper_path": "string",
      "report_json_path": "string",
      "state_json_path": "string",
      "research_state_path": "string or null"
    }
  ],
  "factual_checks": {
    "run_a_paper_claims_match_report_json": "pass | fail | partial",
    "run_b_paper_claims_match_report_json": "pass | fail | partial",
    "run_a_paper_setup_matches_state_json": "pass | fail | partial",
    "run_b_paper_setup_matches_state_json": "pass | fail | partial"
  },
  "scores_by_run": {
    "run_a": {
      "problem_framing": {"score": 0, "justification": "string", "evidence_quote": "string", "evidence_artifact": "string"},
      "workflow_fidelity": {"score": 0, "justification": "string", "evidence_quote": "string", "evidence_artifact": "string"},
      "interpretation_quality": {"score": 0, "justification": "string", "evidence_quote": "string", "evidence_artifact": "string"},
      "caveat_honesty": {"score": 0, "justification": "string", "evidence_quote": "string", "evidence_artifact": "string"},
      "paper_usefulness": {"score": 0, "justification": "string", "evidence_quote": "string", "evidence_artifact": "string"}
    },
    "run_b": {
      "problem_framing": {"score": 0, "justification": "string", "evidence_quote": "string", "evidence_artifact": "string"},
      "workflow_fidelity": {"score": 0, "justification": "string", "evidence_quote": "string", "evidence_artifact": "string"},
      "interpretation_quality": {"score": 0, "justification": "string", "evidence_quote": "string", "evidence_artifact": "string"},
      "caveat_honesty": {"score": 0, "justification": "string", "evidence_quote": "string", "evidence_artifact": "string"},
      "paper_usefulness": {"score": 0, "justification": "string", "evidence_quote": "string", "evidence_artifact": "string"}
    }
  },
  "her_scores_by_run": {
    "run_a": {
      "evaluator_legitimacy": {"score": 0, "justification": "string", "evidence_quote": "string", "evidence_artifact": "string"},
      "search_space_validity": {"score": 0, "justification": "string", "evidence_quote": "string", "evidence_artifact": "string"},
      "scientific_setup_quality": {"score": 0, "justification": "string", "evidence_quote": "string", "evidence_artifact": "string"}
    },
    "run_b": {
      "evaluator_legitimacy": {"score": 0, "justification": "string", "evidence_quote": "string", "evidence_artifact": "string"},
      "search_space_validity": {"score": 0, "justification": "string", "evidence_quote": "string", "evidence_artifact": "string"},
      "scientific_setup_quality": {"score": 0, "justification": "string", "evidence_quote": "string", "evidence_artifact": "string"}
    }
  },
  "pairwise_comparison": {
    "winner": "run_a | run_b | tie | inconclusive",
    "why": "string"
  },
  "overall_summary": "string"
}
```
