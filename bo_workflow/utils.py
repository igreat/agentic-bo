"""Shared utility helpers for run state, serialization, and naming."""

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import secrets
from typing import Any, Literal

import numpy as np
import pandas as pd

type Objective = Literal["min", "max"]
type OptimizerName = Literal["hebo", "bo_lcb", "random", "botorch"]


@dataclass(frozen=True)
class RunPaths:
    """Canonical file locations under a BO run directory (`<runs_root>/<run_id>/`)."""

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
    def report(self) -> Path:
        return self.run_dir / "report.json"

    @property
    def convergence_plot(self) -> Path:
        return self.run_dir / "convergence.png"


@dataclass(frozen=True)
class EvaluationBackendPaths:
    """Canonical file locations under an evaluation backend directory."""

    backend_dir: Path

    @property
    def oracle_model(self) -> Path:
        return self.backend_dir / "oracle.pkl"

    @property
    def oracle_meta(self) -> Path:
        return self.backend_dir / "oracle_meta.json"

_RUN_ADJECTIVES = (
    "amber",
    "brisk",
    "crisp",
    "daring",
    "eager",
    "fuzzy",
    "gentle",
    "jolly",
    "lively",
    "nimble",
    "rapid",
    "steady",
    "sunny",
    "vivid",
)

_RUN_NOUNS = (
    "otter",
    "falcon",
    "heron",
    "lynx",
    "fox",
    "orca",
    "panda",
    "sparrow",
    "badger",
    "koala",
    "wolf",
    "tiger",
    "eagle",
    "whale",
)


def utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def generate_run_id() -> str:
    """Generate a human-readable run id like `amber-otter-0421`."""
    adjective = secrets.choice(_RUN_ADJECTIVES)
    noun = secrets.choice(_RUN_NOUNS)
    suffix = secrets.randbelow(10000)
    return f"{adjective}-{noun}-{suffix:04d}"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def to_python_scalar(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_python_scalar(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_python_scalar(item) for item in value]
    if isinstance(value, np.ndarray):
        return [to_python_scalar(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def row_to_python_dict(row: pd.Series) -> dict[str, Any]:
    return {str(k): to_python_scalar(v) for k, v in row.to_dict().items()}
