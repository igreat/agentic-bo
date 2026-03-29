This is a HEA base-run recovery prompt. Reproduce the earlier base HEA study
setup in this workspace as closely as practical rather than choosing a new HEA
family from scratch.

This is still a one-shot, no-questions run. Use local code and web research if
needed, but do not pause for clarifying questions unless you are truly blocked.

Use this scientific setup:

- family: quinary FCC noble-metal HEA `Ag-Au-Cu-Pd-Pt`
- evaluator: live local ASE/EMT structural simulation
- representation: explicit random FCC supercells
- objective: minimize a screening free-energy proxy
  `g_mix_1000K = delta_h_mix - T * delta_s_mix_ideal` in eV/atom
- composition regime: true five-principal-element HEA, with each element kept
  between `5` and `35` at.% so the study does not collapse into a dilute alloy

Expectations for the run:

- instantiate actual alloy compositions and evaluate them with a live local
  structural calculator
- use BO end to end rather than a lookup table or retrospective oracle
- keep the study explicitly screening-level and caveat-heavy rather than
  overclaiming
- if useful, sanity-check the setup against an equimolar Ag-Au-Cu-Pd-Pt FCC
  reference and report any obvious lattice or stability mismatch
- if useful, quantify ordering/disorder sensitivity for representative
  compositions so the final shortlist is not presented as a single exact truth

Do not switch to a different alloy family, lattice family, or foundation-model
evaluator unless the specified setup proves genuinely infeasible in this
workspace. If it does prove infeasible, stop and explain why rather than
silently changing the problem.

Before you conclude, persist a concise research artifact package for review. At
minimum:

- write a short LaTeX report draft under `research_runs/<research_id>/paper.tex`
- write a brief running plan / notebook under `research_runs/<research_id>/research_plan.md`
- write machine-readable workflow state under `research_runs/<research_id>/research_state.json`

If you generate figures, place them under `research_runs/<research_id>/figures/`
and embed them in the LaTeX draft. If feasible without derailing the run, also
compile `research_runs/<research_id>/paper.pdf`, but the required artifact is
the `.tex` source.

The report should summarize:

- the fixed HEA family and why it remains scientifically defensible
- the live ASE/EMT evaluator and its major limitations
- the structural representation, composition constraints, and BO search space
- any calibration or disorder/uncertainty treatment that was used
- the main results or shortlist
- recommended next steps for higher-fidelity follow-up
