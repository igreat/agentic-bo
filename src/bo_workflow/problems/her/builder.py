from pathlib import Path

import numpy as np
import pandas as pd
from hebo.design_space.design_space import DesignSpace

from src.bo_workflow.problems.base import ProblemContext


def _get_design_space() -> DesignSpace:
    params = [
        {"name": "AcidRed871_0gL", "type": "num", "lb": 0, "ub": 5},
        {"name": "L-Cysteine-50gL", "type": "num", "lb": 0, "ub": 5},
        {"name": "MethyleneB_250mgL", "type": "num", "lb": 0, "ub": 5},
        {"name": "NaCl-3M", "type": "num", "lb": 0, "ub": 5},
        {"name": "NaOH-1M", "type": "num", "lb": 0, "ub": 5},
        {"name": "P10-MIX1", "type": "num", "lb": 0, "ub": 5},
        {"name": "PVP-1wt", "type": "num", "lb": 0, "ub": 5},
        {"name": "RhodamineB1_0gL", "type": "num", "lb": 0, "ub": 5},
        {"name": "SDS-1wt", "type": "num", "lb": 0, "ub": 5},
        {"name": "Sodiumsilicate-1wt", "type": "num", "lb": 0, "ub": 5},
    ]
    return DesignSpace().parse(params)


def build_problem(
    data_path: str | None = None,
    oracle_impl: str = "random_forest",
) -> ProblemContext:
    if data_path is None:
        data_path = str(
            Path(__file__).resolve().parent / "data" / "HER_virtual_data.csv"
        )

    data = pd.read_csv(data_path)
    data["Target"] = data["Target"].max() - data["Target"]

    target = data["Target"]
    features = data.drop(columns=["Target"])

    if oracle_impl != "random_forest":
        raise ValueError(f"Unsupported implementation: {oracle_impl}")

    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(n_estimators=100)
    model.fit(features, target)

    def oracle(x: pd.DataFrame) -> np.ndarray:
        return model.predict(x).reshape(-1, 1)

    return ProblemContext(
        name="HER",
        design_space=_get_design_space(),
        oracle=oracle,
        metadata={"data_path": data_path, "oracle_impl": oracle_impl},
    )
