# Benchmark Task: OER Composition Optimization

You are optimizing a six-component mixed-metal oxide catalyst composition for
the oxygen evolution reaction in alkaline media.

Goal:

- minimize `overpotential_V`
- treat lower overpotential at fixed current density as better performance

System:

- mixed Mn-Fe-Co-Ni-La-Ce oxide compositions
- retrospective hidden-evaluator benchmark
- intended story: high-throughput catalyst discovery under a compositional
  simplex constraint

Design variables:

- `Mn_molar_fraction`
- `Fe_molar_fraction`
- `Co_molar_fraction`
- `Ni_molar_fraction`
- `La_molar_fraction`
- `Ce_molar_fraction`

Hard constraints:

- each variable is in `[0, 1]`
- all six molar fractions must sum to `1.0`

Benchmark rules:

- this is a retrospective benchmark with external hidden evaluation
- do not use web search
- use only the local boxed literature packet in `literature/`
- do not request or use a labeled dataset
- do not build your own oracle or proxy evaluator
- use the external evaluator handle from `task_manifest.json` when observations
  are to be automated

Expected workflow:

1. frame the catalyst-discovery problem
2. use the local literature packet to identify sensible baselines, variables,
   and caveats
3. initialize from `search_space.json`
4. execute BO under the stated budget
5. interpret the outcome honestly as hidden-evaluator evidence
6. draft the paper or report
