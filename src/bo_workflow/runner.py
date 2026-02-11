from pathlib import Path

import numpy as np
from hebo.optimizers.bo import BO
from hebo.optimizers.hebo import HEBO
from tqdm import tqdm

from .problems.base import ProblemContext
from .type_defs import ExperimentResult


def load_result(path: str) -> ExperimentResult:
    loaded = np.load(path, allow_pickle=True)
    return {
        "HEBO": loaded["HEBO"],
        "BO (LCB)": loaded["BO (LCB)"],
        "Random Search": loaded["Random Search"],
    }


def save_result(result: ExperimentResult, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **result)


def run_hebo(
    problem: ProblemContext,
    num_iterations: int,
    random_seeds: list[int],
    num_initial_random_samples: int,
) -> np.ndarray:
    hebo_config = {
        "lr": 5e-3,
        "num_epochs": 100,
        "verbose": False,
        "noise_lb": 1e-5,
        "optimizer": "lbfgs",
        "pred_likeli": True,
        "warp": True,
    }
    history = []

    for seed in random_seeds:
        np.random.seed(seed)
        hebo = HEBO(
            problem.design_space,
            model_name="gp",
            rand_sample=num_initial_random_samples,
            model_config=hebo_config,
            scramble_seed=seed,
        )
        trace = []
        for i in tqdm(range(num_iterations)):
            try:
                rec_x = hebo.suggest()
            except Exception as e:
                print(
                    f"⚠️ HEBO suggest() failed at iteration {i} with error: {e}. Abort"
                )
                break
            rec_y = problem.oracle(rec_x)
            trace += rec_y.flatten().tolist()
            hebo.observe(rec_x, rec_y)

        history.append(trace)
        print(f"✓ HEBO seed {seed} completed. Final best: {hebo.y.min():.2f}")

    return np.array(history)


def run_bo_lcb(
    problem: ProblemContext,
    num_iterations: int,
    random_seeds: list[int],
    num_initial_random_samples: int,
) -> np.ndarray:
    history = []

    for seed in random_seeds:
        np.random.seed(seed)
        bo = BO(
            problem.design_space,
            model_name="gp",
            rand_sample=num_initial_random_samples,
        )
        trace = []

        for i in tqdm(range(num_iterations)):
            try:
                rec_x = bo.suggest()
            except Exception as e:
                print(f"⚠️ BO suggest() failed at iteration {i} with error: {e}. Abort")
                break
            rec_y = problem.oracle(rec_x)
            trace += rec_y.flatten().tolist()
            bo.observe(rec_x, rec_y)

        history.append(trace)
        print(f"✓ BO seed {seed} completed. Final best: {bo.y.min():.2f}")

    return np.array(history)


def run_random_search(
    problem: ProblemContext,
    num_iterations: int,
    random_seeds: list[int],
) -> np.ndarray:
    history = []

    for seed in random_seeds:
        np.random.seed(seed)
        trace = []

        for _ in tqdm(range(num_iterations)):
            rec_x = problem.design_space.sample(1)
            rec_y = problem.oracle(rec_x)
            trace += rec_y.flatten().tolist()

        history.append(trace)
        print(f"✓ Random Search seed {seed} completed. Final best: {min(trace):.2f}")

    return np.array(history)
