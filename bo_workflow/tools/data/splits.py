from __future__ import annotations

from typing import Callable

SmilesRow = tuple[str, str, float]


def split_from_seed_smiles(
    full_data: list[SmilesRow],
    seed_smiles: list[tuple[str, str]],
    *,
    seed_count: int,
    min_overlap: int = 5,
) -> tuple[list[SmilesRow], list[SmilesRow]]:
    full_by_can: dict[str, float] = {}
    full_smi_by_can: dict[str, str] = {}
    for smi, can, y in full_data:
        full_by_can[can] = y
        full_smi_by_can[can] = smi

    train_pairs: list[SmilesRow] = []
    for _, can in seed_smiles:
        if can in full_by_can:
            train_pairs.append((full_smi_by_can[can], can, full_by_can[can]))

    if len(train_pairs) < min_overlap:
        raise ValueError(
            f"Only {len(train_pairs)} seed molecules overlap with dataset; need at least {min_overlap}"
        )

    train_pairs = train_pairs[:seed_count]
    train_can = {can for _, can, _ in train_pairs}
    lookup_pairs = [row for row in full_data if row[1] not in train_can]
    return train_pairs, lookup_pairs


def select_representative_split(
    full_data: list[SmilesRow],
    seed_count: int,
    seed: int,
    *,
    select_diverse_train_fn: Callable[[list[str], list[float], int, int], tuple[list[int], list[int]]],
    top_fraction: float,
    quality_mode: str,
    decent_low_q: float,
    decent_high_q: float,
    filter_fn: Callable[[str], bool] | None = None,
) -> tuple[list[SmilesRow], list[SmilesRow]]:
    candidates = [r for r in full_data if (filter_fn is None or filter_fn(r[0]))]
    if len(candidates) < max(10, seed_count // 2):
        candidates = list(full_data)

    candidates_by_can: dict[str, SmilesRow] = {}
    for smi, can, y in candidates:
        if can not in candidates_by_can or y > candidates_by_can[can][2]:
            candidates_by_can[can] = (smi, can, y)
    candidates = list(candidates_by_can.values())

    candidates.sort(key=lambda x: x[2], reverse=True)
    ys_asc = sorted([y for _, _, y in candidates])

    def _q(vals: list[float], q: float) -> float:
        if not vals:
            return 0.0
        q = min(1.0, max(0.0, float(q)))
        idx = int(round(q * (len(vals) - 1)))
        return vals[idx]

    low_thr = _q(ys_asc, decent_low_q)
    high_thr = _q(ys_asc, decent_high_q)
    decent_pool = [r for r in candidates if low_thr <= r[2] <= high_thr]
    if len(decent_pool) < max(5, seed_count // 4):
        decent_pool = candidates[:]

    top_n = int(seed_count * top_fraction)
    if quality_mode == "top":
        top_n = max(0, min(top_n, seed_count, len(candidates)))
        top_part = candidates[:top_n]
    elif quality_mode == "decent":
        top_n = max(0, min(top_n, seed_count, len(decent_pool)))
        top_part = decent_pool[:top_n]
    else:
        top_n = max(0, min(top_n, seed_count))
        n_top = min(len(candidates), top_n // 2)
        n_decent = min(len(decent_pool), top_n - n_top)
        top_part = candidates[:n_top] + [
            r for r in decent_pool if r[1] not in {c for _, c, _ in candidates[:n_top]}
        ][:n_decent]

    top_cans = {c for _, c, _ in top_part}
    rest = [r for r in candidates if r[1] not in top_cans]

    need = max(0, seed_count - len(top_part))
    diverse_part: list[SmilesRow] = []
    if need > 0 and rest:
        all_smiles = [s for s, _, _ in rest]
        all_targets = [y for _, _, y in rest]
        n_pick = min(need, len(rest))
        train_idx, _ = select_diverse_train_fn(all_smiles, all_targets, n_pick, seed)
        diverse_part = [rest[i] for i in train_idx]

    train_pairs = top_part + diverse_part
    train_cans = {c for _, c, _ in train_pairs}

    if len(train_pairs) < seed_count:
        for row in candidates:
            if row[1] in train_cans:
                continue
            train_pairs.append(row)
            train_cans.add(row[1])
            if len(train_pairs) >= seed_count:
                break

    lookup_pairs = [row for row in full_data if row[1] not in train_cans]
    return train_pairs, lookup_pairs


def build_lookup_table(*datasets: list[SmilesRow]) -> dict[str, float]:
    lookup: dict[str, float] = {}
    for data in datasets:
        for _, can, y in data:
            lookup[can] = y
    return lookup
