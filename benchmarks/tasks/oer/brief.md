# Benchmark Task: OER Composition Optimization

You are optimizing a six-component mixed-metal oxide catalyst composition for the oxygen evolution reaction in alkaline media.

Goal:

- minimize `overpotential_V`
- treat lower overpotential at fixed current density as better performance

System:

- mixed Mn-Fe-Co-Ni-La-Ce oxide compositions
- retrospective evaluator-backed benchmark
- intended story: high-throughput catalyst discovery under a compositional simplex constraint

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

Public task materials:

- `task_manifest.json`
- `search_space.json`
- local literature packet in `literature/`

Notes:

- this is a retrospective benchmark with external evaluation
- interpret results honestly as evaluator-backed benchmark evidence
