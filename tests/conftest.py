"""Shared fixtures for BO workflow integration tests."""

from pathlib import Path

import pytest

from bo_workflow.engine import BOEngine

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


@pytest.fixture()
def engine(tmp_path: Path) -> BOEngine:
    """BOEngine with an isolated runs_root under tmp_path."""
    return BOEngine(runs_root=tmp_path)


def _dataset_fixture(filename: str):
    """Factory for dataset path fixtures that skip when file is missing."""
    path = DATA_DIR / filename

    @pytest.fixture()
    def _fixture() -> Path:
        if not path.exists():
            pytest.skip(f"{filename} not found in {DATA_DIR}")
        return path

    return _fixture


her_csv = _dataset_fixture("HER_virtual_data.csv")
hea_csv = _dataset_fixture("HEA_alloy_data.csv")
oer_csv = _dataset_fixture("OER_catalyst_data.csv")
bh_csv = _dataset_fixture("BH_synthesis_data.csv")
