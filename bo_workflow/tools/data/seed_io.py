from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

SmilesRow = tuple[str, str, float]


def load_seed_smiles(
    path: Path,
    column: str,
    seed_count: int,
    *,
    canonicalize: Callable[[str], str | None],
    filter_fn: Callable[[str], bool] | None = None,
) -> list[tuple[str, str]]:
    with open(path) as f:
        rows = list(csv.DictReader(f))

    seeds: list[tuple[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        smi = row.get(column, "")
        if filter_fn is not None and not filter_fn(smi):
            continue
        can = canonicalize(smi)
        if not can or can in seen:
            continue
        seen.add(can)
        seeds.append((smi, can))
        if len(seeds) >= seed_count:
            break

    if not seeds:
        raise ValueError("No valid seed smiles found")
    return seeds


def write_smiles_target_csv(path: Path, rows: list[SmilesRow], *, target_name: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles", target_name])
        for smi, _, y in rows:
            writer.writerow([smi, y])


def append_labeled_rows(path: Path, rows: list[tuple[str, float]]) -> None:
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        for smi, y in rows:
            writer.writerow([smi, y])
