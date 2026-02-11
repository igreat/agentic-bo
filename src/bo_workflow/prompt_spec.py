from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd


_MAX_WORDS = ("maximize", "maximise", "max", "highest", "increase")
_MIN_WORDS = ("minimize", "minimise", "min", "lowest", "decrease", "reduce")


def _normalize_text(value: str) -> str:
    lowered = value.lower()
    return re.sub(r"[^a-z0-9]+", " ", lowered).strip()


def _infer_objective(prompt: str, target_column: str | None) -> str:
    prompt_l = prompt.lower()

    max_hits = [prompt_l.find(w) for w in _MAX_WORDS if w in prompt_l]
    min_hits = [prompt_l.find(w) for w in _MIN_WORDS if w in prompt_l]

    if max_hits and min_hits:
        return "max" if min(max_hits) < min(min_hits) else "min"
    if max_hits:
        return "max"
    if min_hits:
        return "min"

    if target_column is not None:
        t = target_column.lower()
        if any(k in t for k in ("loss", "error", "cost", "regret", "rmse", "mae")):
            return "min"
    return "max"


def _infer_target_column(
    prompt: str,
    columns: list[str],
    explicit_target: str | None,
) -> str:
    if explicit_target is not None:
        if explicit_target not in columns:
            raise ValueError(
                f"target_column '{explicit_target}' is not in dataset columns"
            )
        return explicit_target

    normalized_prompt = _normalize_text(prompt)
    best_match = None
    best_len = -1
    for col in columns:
        alias = _normalize_text(col)
        if alias and alias in normalized_prompt and len(alias) > best_len:
            best_match = col
            best_len = len(alias)

    if best_match is not None:
        return best_match

    priority_names = (
        "target",
        "yield",
        "objective",
        "score",
        "response",
        "label",
    )
    lowered = {c.lower(): c for c in columns}
    for key in priority_names:
        if key in lowered:
            return lowered[key]

    return columns[-1]


def build_run_spec_from_prompt(
    *,
    dataset_path: str | Path,
    prompt: str,
    target_column: str | None = None,
    objective: str | None = None,
    default_engine: str = "hebo",
    seed: int = 0,
    num_initial_random_samples: int = 10,
    default_batch_size: int = 1,
    max_categories: int = 64,
    max_features: int | None = None,
) -> dict[str, Any]:
    dataset_path = Path(dataset_path).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    frame = pd.read_csv(dataset_path, nrows=5)
    columns = list(frame.columns)
    if not columns:
        raise ValueError("Dataset has no columns.")

    inferred_target = _infer_target_column(prompt, columns, target_column)
    inferred_objective = objective or _infer_objective(prompt, inferred_target)

    if default_engine not in {"hebo", "bo_lcb", "random"}:
        raise ValueError("default_engine must be one of: hebo, bo_lcb, random")

    spec: dict[str, Any] = {
        "dataset_path": str(dataset_path),
        "target_column": inferred_target,
        "objective": inferred_objective,
        "default_engine": default_engine,
        "seed": int(seed),
        "num_initial_random_samples": int(num_initial_random_samples),
        "default_batch_size": int(default_batch_size),
        "max_categories": int(max_categories),
        "intent": {
            "user_prompt": prompt,
            "parser": "heuristic-v1",
        },
    }
    if max_features is not None:
        spec["max_features"] = int(max_features)
    return spec
