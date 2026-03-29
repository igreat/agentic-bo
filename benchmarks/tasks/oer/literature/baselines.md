# Baselines

- Treat OER overpotential at fixed current density as the optimization target.
- Lower values are better.
- In the Gregoire-group Ce-rich OER study, electrodeposited `Ni0.2Co0.3Ce0.5Ox` reached about `310 mV` overpotential at `10 mA cm^-2`, which is presented there as among the lowest reported values for that experimental setting.
- The same paper states that known `(Ni-Fe)Ox` and `(Ni-Co)Ox` regions are strong baselines, but broader composition searches uncovered a distinct high-Ce activity region that local refinement would likely miss.

What the agent should take from this:

- there are credible composition baselines in Ni-Fe and Ni-Co oxide families
- broader multimetal searches can reveal disconnected high-performing regions
- performance should be framed against literature baselines qualitatively, not against any hidden optimum in this benchmark

This packet intentionally avoids giving task-specific hidden optimum values.
