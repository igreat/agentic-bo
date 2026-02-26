from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path

    @property
    def state(self) -> Path:
        return self.run_dir / "state.json"

    @property
    def intent(self) -> Path:
        return self.run_dir / "intent.json"

    @property
    def input_spec(self) -> Path:
        return self.run_dir / "input_spec.json"

    @property
    def suggestions(self) -> Path:
        return self.run_dir / "suggestions.jsonl"

    @property
    def observations(self) -> Path:
        return self.run_dir / "observations.jsonl"

    @property
    def oracle_model(self) -> Path:
        return self.run_dir / "oracle.pkl"

    @property
    def oracle_meta(self) -> Path:
        return self.run_dir / "oracle_meta.json"

    @property
    def report(self) -> Path:
        return self.run_dir / "report.json"

    @property
    def convergence_plot(self) -> Path:
        return self.run_dir / "convergence.pdf"

    @property
    def scaffold_spec(self) -> Path:
        return self.run_dir / "scaffold_spec.json"

    @property
    def feasibility_log(self) -> Path:
        return self.run_dir / "feasibility.jsonl"

    @property
    def descriptor_cache(self) -> Path:
        return self.run_dir / "descriptor_cache.json"

    @property
    def energy_cache(self) -> Path:
        return self.run_dir / "energy_cache.json"

    @property
    def crem_candidates(self) -> Path:
        return self.run_dir / "crem_candidates.json"

    @property
    def crem_feasibility(self) -> Path:
        return self.run_dir / "crem_feasibility.json"

    @property
    def validation_results(self) -> Path:
        return self.run_dir / "validation_results.json"
