"""Compatibility facade for shared helpers.

New code should prefer imports from `bo_workflow.shared.*`.
"""

from typing import Any

import numpy as np
import pandas as pd

from .shared.json_io import append_jsonl, read_json, read_jsonl, write_json
from .shared.run_id import generate_run_id, utc_now_iso
from .shared.run_paths import RunPaths
from .shared.types import Objective, OptimizerName


def to_python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def row_to_python_dict(row: pd.Series) -> dict[str, Any]:
    return {str(k): to_python_scalar(v) for k, v in row.to_dict().items()}
