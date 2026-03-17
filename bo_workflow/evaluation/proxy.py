"""ProxyObserver — evaluates suggestions using a persisted evaluation backend."""

from pathlib import Path
from typing import Any

import pandas as pd

from .oracle import predict_original_scale, read_backend_meta
from ..utils import EvaluationBackendPaths
from ..observers.base import Observer


class ProxyObserver(Observer):
    """Evaluates suggestions using the trained proxy oracle.

    Self-contained: captures all needed context (backend dir, features,
    objective, oracle metadata) at construction time.
    """

    def __init__(self, backend_dir: str | Path) -> None:
        self._backend_dir = Path(backend_dir)
        paths = EvaluationBackendPaths(backend_dir=self._backend_dir)
        if not paths.oracle_model.exists():
            raise FileNotFoundError(
                f"Oracle not found at {paths.oracle_model}. "
                "Run 'build-oracle' first."
            )
        meta = read_backend_meta(self._backend_dir)
        self._active_features = list(meta["active_features"])
        self._objective = str(meta["objective"])
        self._default_engine = str(meta.get("default_engine", "hebo"))

    @property
    def source(self) -> str:
        return "proxy-oracle"

    def evaluate(self, suggestions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        x_df = pd.DataFrame([row["x"] for row in suggestions])[self._active_features]
        y_pred = predict_original_scale(self._backend_dir, self._objective, x_df)

        payloads = []
        for row, y_val in zip(suggestions, y_pred, strict=True):
            payloads.append(
                {
                    "x": row["x"],
                    "y": float(y_val),
                    "engine": row.get("engine", self._default_engine),
                    "suggestion_id": row.get("suggestion_id"),
                }
            )
        return payloads
