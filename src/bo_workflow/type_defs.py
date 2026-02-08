from typing import Callable
import numpy as np
import pandas as pd

type OracleFn = Callable[[pd.DataFrame], np.ndarray]
type ExperimentResult = dict[str, np.ndarray]
