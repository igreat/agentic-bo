"""Shared cross-layer primitives and helpers."""

from .json_io import append_jsonl, read_json, read_jsonl, write_json
from .run_id import generate_run_id, utc_now_iso
from .run_paths import RunPaths
from .types import Objective, OptimizerName

__all__ = [
    "Objective",
    "OptimizerName",
    "utc_now_iso",
    "generate_run_id",
    "read_json",
    "write_json",
    "append_jsonl",
    "read_jsonl",
    "RunPaths",
]
