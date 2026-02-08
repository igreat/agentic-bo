from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal, Mapping
import tomllib


@dataclass(frozen=True)
class ExperimentConfig:
    title: str
    num_iterations: int = 100
    num_seeds: int = 1
    num_initial_random_samples: int = 10
    y_label: str = "y"
    result_path: str | None = None
    plot_path: str | None = None
    objective: Literal["min", "max"] = "min"
    regret_baseline: float | None = None
    y_scale: Literal["linear", "log"] = "linear"
    show_plot: bool = False
    error_style: Literal["stderr", "iqr"] = "stderr"

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ExperimentConfig":
        valid_keys = {f.name for f in fields(cls)}
        unknown = sorted(set(raw) - valid_keys)
        if unknown:
            raise ValueError(f"Unknown experiment config keys: {', '.join(unknown)}")
        return cls(**dict(raw))


def load_experiment_file(
    path: str | Path,
) -> tuple[ExperimentConfig, dict[str, Any]]:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        data = tomllib.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Expected TOML table at root in {config_path}")

    experiment_raw = data.get("experiment", data)
    if not isinstance(experiment_raw, dict):
        raise ValueError(
            f"Expected [experiment] to be a TOML table in {config_path}"
        )

    problem_raw = data.get("problem", {})
    if not isinstance(problem_raw, dict):
        raise ValueError(f"Expected [problem] to be a TOML table in {config_path}")

    return ExperimentConfig.from_mapping(experiment_raw), dict(problem_raw)
