from __future__ import annotations

import numpy as np

from .chem import _canonical


def _build_canonical_descriptor_lookup(
    descriptor_lookup: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    canonical_lookup: dict[str, dict[str, float]] = {}
    for smi, desc in descriptor_lookup.items():
        can = _canonical(smi)
        if can is not None and can not in canonical_lookup:
            canonical_lookup[can] = desc
    return canonical_lookup


def _descriptor_distance_to_observed(
    candidate_canonical: str,
    canonical_desc_lookup: dict[str, dict[str, float]],
    observed_smiles: set[str],
    active_features: list[str],
) -> float | None:
    if candidate_canonical not in canonical_desc_lookup:
        return None

    cand_desc = canonical_desc_lookup[candidate_canonical]
    candidate_vec = np.array(
        [float(cand_desc.get(f, 0.0) or 0.0) for f in active_features],
        dtype=float,
    )

    observed_vecs = []
    for smi in observed_smiles:
        desc = canonical_desc_lookup.get(smi)
        if desc is None:
            continue
        vec = [float(desc.get(f, 0.0) or 0.0) for f in active_features]
        observed_vecs.append(vec)

    if not observed_vecs:
        return None

    obs_mat = np.array(observed_vecs, dtype=float)
    dists = np.linalg.norm(obs_mat - candidate_vec, axis=1)
    return float(np.min(dists))


def _pick_uncertainty_exploit_from_pool(
    all_pool_smiles: list[str],
    observed_smiles: set[str],
    canonical_desc_lookup: dict[str, dict[str, float]],
    active_features: list[str],
) -> tuple[str, float] | None:
    best_smiles: str | None = None
    best_dist = -1.0

    for smi in all_pool_smiles:
        can = _canonical(smi)
        if can is None or can in observed_smiles:
            continue
        dist = _descriptor_distance_to_observed(
            can, canonical_desc_lookup, observed_smiles, active_features
        )
        if dist is None:
            continue
        if dist > best_dist:
            best_dist = dist
            best_smiles = smi

    if best_smiles is None:
        return None
    return best_smiles, float(best_dist)
