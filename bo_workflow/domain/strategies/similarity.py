from __future__ import annotations

import random

from .chem import _tanimoto_fp, _tanimoto_similarity


def select_diverse_train(
    all_smiles: list[str],
    all_targets: list[float],
    n_train: int,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """MaxMin diversity pick for training set, ensuring target coverage."""
    from rdkit import DataStructs

    rng = random.Random(seed)
    fps = [_tanimoto_fp(s) for s in all_smiles]
    valid_idx = [i for i, fp in enumerate(fps) if fp is not None]

    sorted_by_target = sorted(valid_idx, key=lambda i: all_targets[i])
    n = len(sorted_by_target)

    quartile_picks = set()
    for q in range(4):
        lo = n * q // 4
        hi = n * (q + 1) // 4
        quartile_picks.add(sorted_by_target[rng.randint(lo, hi - 1)])

    selected = set(quartile_picks)
    remaining = [i for i in valid_idx if i not in selected]

    while len(selected) < n_train and remaining:
        best_idx = None
        best_min_dist = -1
        sel_fps = [fps[i] for i in selected]
        for idx in remaining:
            min_sim = min(DataStructs.TanimotoSimilarity(fps[idx], sfp) for sfp in sel_fps)
            dist = 1.0 - min_sim
            if dist > best_min_dist:
                best_min_dist = dist
                best_idx = idx
        if best_idx is not None:
            selected.add(best_idx)
            remaining.remove(best_idx)

    train_idx = sorted(selected)
    lookup_idx = sorted(set(valid_idx) - selected)
    return train_idx, lookup_idx


def knn_tanimoto_predict(
    candidate_smi: str,
    observed_results: dict[str, float],
    observed_fps: dict,
    k: int = 5,
    sim_threshold: float = 0.25,
) -> float | None:
    """Predict pIC50 using k-nearest neighbors by Tanimoto similarity."""
    cand_fp = _tanimoto_fp(candidate_smi)
    if cand_fp is None:
        return None

    similarities = []
    for smi, fp in observed_fps.items():
        sim = _tanimoto_similarity(cand_fp, fp)
        if sim > sim_threshold:
            similarities.append((smi, sim))

    if not similarities:
        return None

    similarities.sort(key=lambda x: -x[1])
    top_k = similarities[:k]

    weighted_sum = 0.0
    weight_sum = 0.0
    for smi, sim in top_k:
        w = sim**3
        weighted_sum += w * observed_results[smi]
        weight_sum += w

    return weighted_sum / weight_sum if weight_sum > 0 else None


def _select_diverse_seeds(
    observed_results: dict[str, float],
    observed_fps: dict,
    n_seeds: int = 10,
    pIC50_threshold: float = 7.0,
) -> list[tuple[str, float, object]]:
    candidates = []
    for smi, pic50 in observed_results.items():
        if pic50 < pIC50_threshold:
            continue
        fp = observed_fps.get(smi) or _tanimoto_fp(smi)
        if fp is not None:
            candidates.append((smi, pic50, fp))

    if not candidates:
        sorted_hits = sorted(observed_results.items(), key=lambda x: -x[1])
        result = []
        for smi, pic50 in sorted_hits[:n_seeds]:
            fp = observed_fps.get(smi) or _tanimoto_fp(smi)
            if fp is not None:
                result.append((smi, pic50, fp))
        return result

    candidates.sort(key=lambda x: -x[1])
    selected = [candidates[0]]
    remaining = candidates[1:]

    while len(selected) < n_seeds and remaining:
        best_idx = -1
        best_score = -1.0
        for i, (_, pic50, fp) in enumerate(remaining):
            min_sim = min(_tanimoto_similarity(fp, sel_fp) for _, _, sel_fp in selected)
            diversity = 1.0 - min_sim
            pic50_weight = max(0.1, pic50 - 6.0)
            score = diversity * pic50_weight
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx >= 0:
            selected.append(remaining.pop(best_idx))
        else:
            break

    return selected
