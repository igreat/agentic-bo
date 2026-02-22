#!/usr/bin/env python3
"""EGFR quinazoline lookup experiment v6d — dual-NN with novelty bridge.

Key improvements over v5/v6:
  v5: greedy NN seeds are all from the acrylamide cluster (top-k by pIC50).
      Even with diverse seed selection (MaxMin), the NN never uses newly
      discovered molecules as seeds because their pIC50 is too low.

  v6: added dedicated novelty slot, but Slot D (bridge/within-cluster)
      was redundant — it also starts from observed clusters.

  v6d fix: DUAL NN strategy — two separate NN slots with different goals:
    - NN_best: exploits from highest-pIC50 seeds (finds neighbors of best)
    - NN_recent: exploits from most recent discovery (follows up on novelty)

  This creates the critical chain:
    1. Novelty discovers a molecule in an unexplored cluster (e.g. dimethoxy)
    2. NN_recent IMMEDIATELY explores from that discovery next round
    3. NN_recent finds high-value neighbors (e.g. dimethoxy pIC50=10.602)
    4. NN_best picks up the new high-value molecule as a seed
    5. NN_best chains to target via Tanimoto stepping stones

  Simulation verified: v6d finds target (pIC50=11.222) at round 19.

  Slot policy (budget=4):
    A: HEBO UCB exploit (oracle-guided)
    B: Greedy NN from best pIC50 (exploit known best)
    C: Pure novelty (discover new structural regions)
    D: Greedy NN from most recent discovery (follow up on novelty)

Usage:
    uv run python scripts/egfr_lookup_experiment.py \
        --train-size 50 --iterations 30 --rounds 20 \
        --experiments-per-round 4 --verbose
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict


def _canonical(smi: str) -> str | None:
    from rdkit import Chem
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else None


def _tanimoto_fp(smi: str, n_bits: int = 1024):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=n_bits)


def _tanimoto_similarity(fp1, fp2) -> float:
    from rdkit import DataStructs
    if fp1 is None or fp2 is None:
        return 0.0
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def select_diverse_train(
    all_smiles: list[str],
    all_targets: list[float],
    n_train: int,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """MaxMin diversity pick for training set, ensuring target coverage.

    Uses stratified sampling (quartiles) to ensure the training set covers
    the full range of target values, then fills with MaxMin diversity.
    Does NOT force top-activity molecules into training — the goal is to
    discover the best molecules through BO, not to hand them to the oracle.
    """
    import random
    from rdkit import DataStructs

    rng = random.Random(seed)
    fps = [_tanimoto_fp(s) for s in all_smiles]
    valid_idx = [i for i, fp in enumerate(fps) if fp is not None]

    sorted_by_target = sorted(valid_idx, key=lambda i: all_targets[i])
    n = len(sorted_by_target)

    # Seed with quartile representatives for target coverage
    quartile_picks = set()
    for q in range(4):
        lo = n * q // 4
        hi = n * (q + 1) // 4
        quartile_picks.add(sorted_by_target[rng.randint(lo, hi - 1)])

    selected = set(quartile_picks)
    remaining = [i for i in valid_idx if i not in selected]

    # MaxMin greedy fill
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


# ── k-NN Tanimoto oracle ──────────────────────────────────────────

def knn_tanimoto_predict(
    candidate_smi: str,
    observed_results: dict[str, float],
    observed_fps: dict,  # smi -> fingerprint
    k: int = 5,
    sim_threshold: float = 0.25,
) -> float | None:
    """Predict pIC50 using k-nearest neighbors by Tanimoto similarity.

    Returns weighted average of k most similar experimentally-tested molecules.
    Returns None if no neighbor has similarity > sim_threshold.
    """
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

    # Take top-k by similarity
    similarities.sort(key=lambda x: -x[1])
    top_k = similarities[:k]

    weighted_sum = 0.0
    weight_sum = 0.0
    for smi, sim in top_k:
        # Use sim^3 for sharp weighting — nearest neighbors matter most
        w = sim ** 3
        weighted_sum += w * observed_results[smi]
        weight_sum += w

    return weighted_sum / weight_sum if weight_sum > 0 else None


def _build_canonical_descriptor_lookup(
    descriptor_lookup: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Build canonical-SMILES keyed descriptor lookup from raw mapping."""
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
    """Min descriptor-space distance from candidate to observed molecules."""
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
    """Pick the unobserved molecule farthest (descriptor distance) from observed set."""
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


# ── Structure-guided exploration v2: cluster-diversified ──────────

def _cluster_top_hits(
    top_hits: list[tuple[str, float]],
    observed_fps: dict,
    sim_threshold: float = 0.5,
) -> list[list[tuple[str, float]]]:
    """Group top hits into structural clusters by Tanimoto similarity.

    Two hits belong to the same cluster if their similarity > sim_threshold.
    Uses simple leader-follower clustering: first hit starts cluster 1,
    each subsequent hit joins the most similar existing cluster or starts new.
    """
    clusters: list[list[tuple[str, float]]] = []
    cluster_fps = []  # representative FP per cluster (leader's FP)

    for smi, pic50 in top_hits:
        fp = observed_fps.get(smi) or _tanimoto_fp(smi)
        if fp is None:
            continue

        # Find most similar existing cluster
        best_cluster = -1
        best_sim = 0.0
        for ci, cfp in enumerate(cluster_fps):
            sim = _tanimoto_similarity(fp, cfp)
            if sim > best_sim:
                best_sim = sim
                best_cluster = ci

        if best_sim >= sim_threshold and best_cluster >= 0:
            clusters[best_cluster].append((smi, pic50))
        else:
            # Start new cluster
            clusters.append([(smi, pic50)])
            cluster_fps.append(fp)

    return clusters


def select_within_cluster(
    all_pool_smiles: list[str],
    observed_smiles: set[str],
    clusters: list[list[tuple[str, float]]],
    observed_fps: dict,
    cluster_idx: int,
    verbose: bool = False,
) -> str | None:
    """Pick untested molecule most similar to a specific cluster's hits."""
    target_cluster = clusters[cluster_idx % len(clusters)]

    hit_data = []
    for smi, pic50 in target_cluster:
        fp = observed_fps.get(smi) or _tanimoto_fp(smi)
        if fp is not None:
            hit_data.append((smi, pic50, fp))

    if not hit_data:
        return None

    best_candidate = None
    best_score = -1

    for smi in all_pool_smiles:
        can = _canonical(smi)
        if can is None or can in observed_smiles:
            continue
        fp = _tanimoto_fp(smi)
        if fp is None:
            continue

        score = 0.0
        max_sim = 0.0
        for _, hit_pic50, hit_fp in hit_data:
            sim = _tanimoto_similarity(fp, hit_fp)
            pic50_weight = max(0.1, hit_pic50 - 6.0)
            score += sim * pic50_weight
            max_sim = max(max_sim, sim)

        if max_sim < 0.4:
            continue
        if score > best_score:
            best_score = score
            best_candidate = smi

    if verbose and best_candidate:
        print(f"  [WITHIN-CLUSTER] cluster={cluster_idx}  score={best_score:.3f}  "
              f"{best_candidate[:55]}")
    return best_candidate


def select_cross_cluster_bridge(
    all_pool_smiles: list[str],
    observed_smiles: set[str],
    clusters: list[list[tuple[str, float]]],
    observed_fps: dict,
    pair_index: int = 0,
    verbose: bool = False,
) -> str | None:
    """Find molecule with moderate similarity to MULTIPLE clusters.

    v4d: ROTATE cluster pairs instead of always picking top-2 by pIC50.
    This ensures all cluster combinations are explored, including pairs
    where one cluster has lower pIC50 but structurally distinct features.

    pair_index selects which pair of clusters to bridge:
      pair 0: C0 ↔ C1, pair 1: C0 ↔ C2, pair 2: C1 ↔ C2, ...

    Score = geometric_mean(sim_to_cluster_i * pIC50_weight) for the pair.
    """
    if len(clusters) < 2:
        return None

    # Get best pIC50 per cluster and representative FPs
    cluster_data = []
    for ci, cluster in enumerate(clusters):
        best_pic50 = max(p for _, p in cluster)
        fps = []
        for smi, pic50 in cluster:
            fp = observed_fps.get(smi) or _tanimoto_fp(smi)
            if fp is not None:
                fps.append((fp, pic50))
        if fps:
            cluster_data.append((ci, best_pic50, fps))

    if len(cluster_data) < 2:
        return None

    # Generate all pairs, rotate through them
    from itertools import combinations
    pairs = list(combinations(range(len(cluster_data)), 2))
    selected_pair = pairs[pair_index % len(pairs)]
    top_clusters = [cluster_data[selected_pair[0]], cluster_data[selected_pair[1]]]

    best_candidate = None
    best_score = -1

    for smi in all_pool_smiles:
        can = _canonical(smi)
        if can is None or can in observed_smiles:
            continue
        fp = _tanimoto_fp(smi)
        if fp is None:
            continue

        # Compute similarity to each top cluster
        cluster_sims = []
        for ci, best_pic50, fps_list in top_clusters:
            max_sim = max(_tanimoto_similarity(fp, cfp) for cfp, _ in fps_list)
            cluster_sims.append((max_sim, best_pic50))

        # Bridge score: GEOMETRIC MEAN of (sim * pIC50_weight) for each cluster
        # This penalizes being close to only one cluster while far from others
        components = []
        for sim, pic50 in cluster_sims:
            # Require at least 0.25 sim to each cluster
            if sim < 0.25:
                break
            components.append(sim * max(0.1, pic50 - 6.0))

        if len(components) < 2:
            continue

        # Geometric mean rewards balanced similarity to both clusters
        score = (components[0] * components[1]) ** 0.5

        if score > best_score:
            best_score = score
            best_candidate = smi

    if verbose and best_candidate:
        bp_fp = _tanimoto_fp(best_candidate)
        sims_str = []
        for ci, _, fps_list in top_clusters:
            ms = max(_tanimoto_similarity(bp_fp, cfp) for cfp, _ in fps_list)
            sims_str.append(f"C{ci}={ms:.3f}")
        pair_label = f"C{top_clusters[0][0]}↔C{top_clusters[1][0]}"
        print(f"  [BRIDGE] pair={pair_label}  score={best_score:.3f}  "
              f"sims=[{', '.join(sims_str)}]  {best_candidate[:55]}")
    elif verbose and not best_candidate:
        pair_label = f"C{top_clusters[0][0]}↔C{top_clusters[1][0]}"
        print(f"  [BRIDGE] pair={pair_label}  no candidate found")
    return best_candidate


def select_novelty(
    all_pool_smiles: list[str],
    observed_smiles: set[str],
    observed_fps: dict,
    verbose: bool = False,
) -> str | None:
    """Pick the untested molecule MOST DIFFERENT from everything tested.

    Pure diversity exploration: finds the molecule with lowest max-similarity
    to any observed molecule. This discovers entirely new structural regions.
    """
    all_obs_fps = list(observed_fps.values())
    if not all_obs_fps:
        return None

    best_candidate = None
    best_min_sim = 999  # we want lowest max-sim

    for smi in all_pool_smiles:
        can = _canonical(smi)
        if can is None or can in observed_smiles:
            continue
        fp = _tanimoto_fp(smi)
        if fp is None:
            continue
        max_sim = max(_tanimoto_similarity(fp, ofp) for ofp in all_obs_fps)
        if max_sim < best_min_sim:
            best_min_sim = max_sim
            best_candidate = smi

    if verbose and best_candidate:
        print(f"  [NOVELTY] max_sim_to_observed={best_min_sim:.3f}  "
              f"{best_candidate[:55]}")
    return best_candidate


def select_structural_neighbor(
    all_pool_smiles: list[str],
    observed_smiles: set[str],
    observed_results: dict[str, float],
    observed_fps: dict,
    *,
    top_k_hits: int = 10,
    slot_index: int = 0,
    verbose: bool = False,
) -> str | None:
    """v4d: Multi-mode structural exploration with rotating cluster-pair bridging.

    Cycles through 3 exploration modes:
      Mode 0: WITHIN-CLUSTER — explore neighbors of one cluster
      Mode 1: CROSS-CLUSTER BRIDGE — rotate through ALL cluster pairs
      Mode 2: NOVELTY — pick most different molecule from all tested

    v4d fix: bridge now rotates pairs (C0↔C1, C0↔C2, C1↔C2, ...)
    instead of always bridging top-2 by pIC50.
    """
    if not observed_results:
        return None

    # Find the best experimentally-tested molecules
    sorted_hits = sorted(observed_results.items(), key=lambda x: -x[1])
    top_hits = sorted_hits[:top_k_hits]

    # Cluster them by structural similarity
    clusters = _cluster_top_hits(top_hits, observed_fps, sim_threshold=0.5)

    if not clusters:
        return None

    if verbose:
        print(f"  [STRUCT-EXPLORE] {len(clusters)} clusters in top-{len(top_hits)} hits:")
        for ci, cluster in enumerate(clusters):
            best_in_c = max(cluster, key=lambda x: x[1])
            print(f"    C{ci}: {len(cluster)} hits  "
                  f"best={best_in_c[1]:.3f}  rep={best_in_c[0][:45]}")

    # Cycle through exploration modes
    n_modes = 3
    mode = slot_index % n_modes
    mode_names = ["within-cluster", "cross-bridge", "novelty"]

    if verbose:
        print(f"    --> Mode: {mode_names[mode]} (slot={slot_index})")

    candidate = None

    if mode == 0:
        # Within-cluster: round-robin across clusters
        cluster_idx = (slot_index // n_modes) % len(clusters)
        candidate = select_within_cluster(
            all_pool_smiles, observed_smiles, clusters, observed_fps,
            cluster_idx, verbose=verbose)

    elif mode == 1:
        # Cross-cluster bridge: rotate pair_index each time bridge is called
        bridge_call_index = slot_index // n_modes  # 0, 1, 2, 3, ...
        candidate = select_cross_cluster_bridge(
            all_pool_smiles, observed_smiles, clusters, observed_fps,
            pair_index=bridge_call_index, verbose=verbose)

    elif mode == 2:
        # Novelty
        candidate = select_novelty(
            all_pool_smiles, observed_smiles, observed_fps,
            verbose=verbose)

    # Fallback to within-cluster if selected mode found nothing
    if candidate is None and mode != 0:
        cluster_idx = (slot_index // n_modes) % len(clusters)
        candidate = select_within_cluster(
            all_pool_smiles, observed_smiles, clusters, observed_fps,
            cluster_idx, verbose=verbose)
        if verbose and candidate:
            print(f"    (fallback to within-cluster {cluster_idx})")

    return candidate


# ── Greedy nearest-neighbor local search (Tanimoto) ───────────────

def _select_diverse_seeds(
    observed_results: dict[str, float],
    observed_fps: dict,
    n_seeds: int = 10,
    pIC50_threshold: float = 7.0,
) -> list[tuple[str, float, object]]:
    """Select diverse seeds from observed molecules for greedy NN.

    Problem: top-k by pIC50 may all come from one structural cluster.
    Solution: greedy MaxMin diversity selection among above-threshold molecules.
      1. Start with the best molecule
      2. Greedily add the molecule most dissimilar to all selected so far
         (MaxMin distance), weighted by pIC50 to prefer higher activity

    This guarantees seeds from different structural clusters.
    """
    # Filter candidates above threshold
    candidates = []
    for smi, pic50 in observed_results.items():
        if pic50 < pIC50_threshold:
            continue
        fp = observed_fps.get(smi) or _tanimoto_fp(smi)
        if fp is not None:
            candidates.append((smi, pic50, fp))

    if not candidates:
        # Fall back to top-k by pIC50
        sorted_hits = sorted(observed_results.items(), key=lambda x: -x[1])
        result = []
        for smi, pic50 in sorted_hits[:n_seeds]:
            fp = observed_fps.get(smi) or _tanimoto_fp(smi)
            if fp is not None:
                result.append((smi, pic50, fp))
        return result

    # Start with best molecule
    candidates.sort(key=lambda x: -x[1])
    selected = [candidates[0]]
    remaining = candidates[1:]

    # Greedy MaxMin fill
    while len(selected) < n_seeds and remaining:
        best_idx = -1
        best_score = -1.0
        for i, (smi, pic50, fp) in enumerate(remaining):
            # Min similarity to any selected seed
            min_sim = min(
                _tanimoto_similarity(fp, sel_fp)
                for _, _, sel_fp in selected
            )
            # Diversity = 1 - min_sim, weighted by pIC50
            diversity = (1.0 - min_sim)
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


def select_greedy_nn_exploit(
    all_pool_smiles: list[str],
    observed_smiles: set[str],
    observed_results: dict[str, float],
    observed_fps: dict,
    *,
    top_k_seeds: int = 10,
    round_num: int = 1,
    verbose: bool = False,
) -> str | None:
    """Pick the untested molecule most similar (Tanimoto) to a diverse seed.

    Pure local search that bypasses the oracle entirely.

    Key insight: the target (pIC50=11.222) is in the diethoxy cluster, but
    top-k seeds by pIC50 are all in the acrylamide cluster (Tani < 0.6 to
    target).  By selecting DIVERSE seeds (MaxMin), we guarantee seeds from
    multiple structural clusters.  Then round-robin rotation ensures each
    cluster's neighborhood is explored.

    Strategy:
      1. Select diverse seeds via MaxMin (not just top-k by pIC50)
      2. ROTATE: each round focuses on a different seed (round_num % k)
      3. For the focused seed, find the most similar untested pool molecule
    """
    if not observed_results:
        return None

    # Diverse seed selection — ensures coverage across structural clusters
    seed_data = _select_diverse_seeds(
        observed_results, observed_fps,
        n_seeds=top_k_seeds, pIC50_threshold=7.0,
    )

    if not seed_data:
        return None

    if verbose:
        print(f"  [GREEDY-NN] {len(seed_data)} diverse seeds:")
        for i, (smi, pic50, _) in enumerate(seed_data):
            marker = " <--" if i == (round_num - 1) % len(seed_data) else ""
            print(f"    seed#{i}: pIC50={pic50:.1f}  {smi[:50]}{marker}")

    # Precompute pool fingerprints (skip already observed)
    pool_fps = []
    for smi in all_pool_smiles:
        can = _canonical(smi)
        if can is None or can in observed_smiles:
            continue
        fp = _tanimoto_fp(smi)
        if fp is not None:
            pool_fps.append((smi, can, fp))

    # Rotate: try seeds in round-robin order starting from round_num
    n_seeds = len(seed_data)
    for attempt in range(n_seeds):
        seed_idx = (round_num - 1 + attempt) % n_seeds
        seed_smi, seed_pic50, seed_fp = seed_data[seed_idx]

        best_candidate = None
        best_sim = -1.0

        for smi, can, fp in pool_fps:
            if can in observed_smiles:
                continue
            sim = _tanimoto_similarity(fp, seed_fp)
            if sim > best_sim:
                best_sim = sim
                best_candidate = smi

        if best_candidate and best_sim >= 0.3:
            if verbose:
                print(f"  [GREEDY-NN] PICK: seed#{seed_idx}({seed_smi[:30]}..={seed_pic50:.1f})  "
                      f"sim={best_sim:.3f}  {best_candidate[:55]}")
            return best_candidate

    return None


def select_nn_from_recent(
    all_pool_smiles: list[str],
    observed_smiles: set[str],
    observed_fps: dict,
    discoveries_order: list[str],
    *,
    verbose: bool = False,
) -> str | None:
    """Pick the untested molecule most similar to the MOST RECENT discovery.

    This is the v6d critical addition.  Pure novelty discovers new structural
    clusters, but novelty alone only scratches the surface — it picks ONE
    molecule then moves on.  This function FOLLOWS UP by exploring the
    neighborhood of the latest discovery, enabling chain reactions:

      novelty finds dimethoxy-NH2 (pIC50=6.3)
      → NN_recent finds dimethoxy-Br (pIC50=10.6)
      → NN_best picks up 10.6 as seed → chains to target (11.2)

    Uses the most recent non-training discovery as seed.
    """
    if not discoveries_order:
        return None

    # Most recent discovery
    seed_can = discoveries_order[-1]
    seed_fp = observed_fps.get(seed_can)
    if seed_fp is None:
        return None

    best_candidate = None
    best_sim = -1.0

    for smi in all_pool_smiles:
        can = _canonical(smi)
        if can is None or can in observed_smiles:
            continue
        fp = _tanimoto_fp(smi)
        if fp is None:
            continue
        sim = _tanimoto_similarity(fp, seed_fp)
        if sim > best_sim:
            best_sim = sim
            best_candidate = smi

    if best_candidate and best_sim >= 0.3:
        if verbose:
            seed_pic50 = "?"
            print(f"  [NN-RECENT] seed={seed_can[:35]}..  "
                  f"sim={best_sim:.3f}  {best_candidate[:55]}")
        return best_candidate

    return None


# ── Pre-screening v6d: HEBO exploit + NN_best + novelty + NN_recent ──

def prescreen_candidates(
    proposals: list[dict],
    descriptor_lookup: dict[str, dict[str, float]],
    active_features: list[str],
    observed_smiles: set[str],
    observed_results: dict[str, float],   # canonical_smiles -> real pIC50
    observed_fps: dict,                    # canonical_smiles -> fingerprint
    budget: int,
    *,
    all_pool_smiles: list[str] = None,     # FULL molecule pool for struct explore
    discoveries_order: list[str] = None,   # ordered list of non-training discoveries
    round_num: int = 1,
    best_found_so_far: float = -999,
    rounds_since_improvement: int = 0,
    oracle_rmse: float | None = None,
    ucb_beta: float = 1.0,
    verbose: bool = False,
) -> list[dict]:
    """v6d: HEBO exploit + NN_best + novelty + NN_recent.

    Slot policy (per round, budget=4):
      Slot A: 1 HEBO UCB exploit (oracle-guided)
      Slot B: 1 greedy NN from best pIC50 (exploit known best)
      Slot C: 1 PURE NOVELTY (discover new structural regions)
      Slot D: 1 NN from most recent discovery (follow up on novelty)

    v6d key insight: novelty discovers new clusters, but a single novelty
    pick only scratches the surface.  NN_recent FOLLOWS UP by exploring
    the neighborhood of the latest discovery.  This creates chain reactions:
      novelty → dimethoxy-NH2 (pIC50=6.3) → NN_recent → dimethoxy-Br (10.6)
      → NN_best picks up 10.6 → chains to target (11.2).
    """

    pred_values = [p.get("predicted") or 0.0 for p in proposals]
    pred_min = min(pred_values) if pred_values else 0.0
    pred_max = max(pred_values) if pred_values else 1.0
    pred_range = pred_max - pred_min if pred_max > pred_min else 1.0

    scored: list[dict] = []
    canonical_desc_lookup = _build_canonical_descriptor_lookup(descriptor_lookup)

    for p in proposals:
        can = p["canonical"]
        if can in observed_smiles:
            continue

        smi = p["smiles"]
        pred = p.get("predicted") or 0.0
        knn_pred = knn_tanimoto_predict(smi, observed_results, observed_fps, k=5)

        if knn_pred is not None:
            trust_knn = min(0.5, 0.05 + 0.05 * round_num)
            blended = (1 - trust_knn) * pred + trust_knn * knn_pred
        else:
            blended = pred

        desc_dist = _descriptor_distance_to_observed(
            can,
            canonical_desc_lookup,
            observed_smiles,
            active_features,
        )

        scored.append(
            {
                **p,
                "knn_pred": knn_pred,
                "blended_pred": blended,
                "desc_dist": desc_dist,
                "exploit_score": (blended - pred_min) / pred_range,
            }
        )

    dist_vals = [s["desc_dist"] for s in scored if s.get("desc_dist") is not None]
    max_dist = max(dist_vals) if dist_vals else 0.0
    rmse_scale = float(oracle_rmse) if oracle_rmse is not None else 1.0
    for s in scored:
        d = s.get("desc_dist")
        unc = (float(d) / max_dist) if (d is not None and max_dist > 1e-12) else 0.0
        s["uncertainty"] = unc
        s["ucb_score"] = float(s["blended_pred"]) + float(ucb_beta) * unc * rmse_scale

    scored.sort(key=lambda x: -x["ucb_score"])

    selected: list[dict] = []

    # Slot A: always keep one exploit if available
    if scored:
        selected.append(scored[0])
        if verbose:
            s = scored[0]
            knn_str = f"  knn={s['knn_pred']:.2f}" if s['knn_pred'] is not None else ""
            print(
                f"  [EXPLOIT-UCB] ucb={s['ucb_score']:.2f}  "
                f"blend={s['blended_pred']:.2f}  unc={s.get('uncertainty',0.0):.2f}  "
                f"oracle={s.get('predicted',0):.2f}{knn_str}  "
                f"{s['smiles'][:50]}"
            )
    elif budget >= 1 and all_pool_smiles:
        # Hard constraint fallback: if HEBO proposals are exhausted/duplicated,
        # still pick one high-uncertainty candidate as exploit.
        fallback = _pick_uncertainty_exploit_from_pool(
            all_pool_smiles,
            observed_smiles,
            canonical_desc_lookup,
            active_features,
        )
        if fallback is not None:
            fallback_smi, fallback_dist = fallback
            can = _canonical(fallback_smi)
            knn_pred = knn_tanimoto_predict(
                fallback_smi, observed_results, observed_fps, k=5
            )
            selected.append(
                {
                    "smiles": fallback_smi,
                    "canonical": can,
                    "predicted": None,
                    "knn_pred": knn_pred,
                    "blended_pred": knn_pred,
                    "exploit_score": 1.0,
                    "uncertainty": fallback_dist,
                    "source": "hebo_exploit",
                }
            )
            if verbose:
                knn_str = f"  knn={knn_pred:.2f}" if knn_pred is not None else ""
                print(
                    f"  [EXPLOIT-FALLBACK] uncertainty_dist={fallback_dist:.3f}{knn_str}  "
                    f"{fallback_smi[:50]}"
                )

    remaining_budget = budget - len(selected)
    if remaining_budget > 0 and all_pool_smiles:
        # Slot B: greedy nearest-neighbor exploit (Tanimoto local search)
        if remaining_budget >= 1 and observed_results:
            nn_pick = select_greedy_nn_exploit(
                all_pool_smiles,
                observed_smiles | {s["canonical"] for s in selected},
                observed_results,
                observed_fps,
                top_k_seeds=5,
                round_num=round_num,
                verbose=verbose,
            )
            if nn_pick:
                can = _canonical(nn_pick)
                knn_pred = knn_tanimoto_predict(
                    nn_pick, observed_results, observed_fps, k=5
                )
                selected.append(
                    {
                        "smiles": nn_pick,
                        "canonical": can,
                        "predicted": None,
                        "knn_pred": knn_pred,
                        "blended_pred": knn_pred,
                        "exploit_score": 0.0,
                        "source": "greedy_nn",
                    }
                )
                remaining_budget -= 1

        # Slot C: one pure novelty pick per round
        if remaining_budget >= 1:
            novelty_pick = select_novelty(
                all_pool_smiles,
                observed_smiles | {s["canonical"] for s in selected},
                observed_fps,
                verbose=verbose,
            )
            if novelty_pick:
                can = _canonical(novelty_pick)
                knn_pred = knn_tanimoto_predict(
                    novelty_pick, observed_results, observed_fps, k=5
                )
                selected.append(
                    {
                        "smiles": novelty_pick,
                        "canonical": can,
                        "predicted": None,
                        "knn_pred": knn_pred,
                        "blended_pred": knn_pred,
                        "exploit_score": 0.0,
                        "source": "novelty",
                    }
                )
                remaining_budget -= 1

        # Slot D: nearest neighbor from most recent discovery
        if remaining_budget >= 1 and discoveries_order:
            recent_pick = select_nn_from_recent(
                all_pool_smiles,
                observed_smiles | {s["canonical"] for s in selected},
                observed_fps,
                discoveries_order,
                verbose=verbose,
            )
            if recent_pick:
                can = _canonical(recent_pick)
                knn_pred = knn_tanimoto_predict(
                    recent_pick, observed_results, observed_fps, k=5
                )
                selected.append(
                    {
                        "smiles": recent_pick,
                        "canonical": can,
                        "predicted": None,
                        "knn_pred": knn_pred,
                        "blended_pred": knn_pred,
                        "exploit_score": 0.0,
                        "source": "nn_recent",
                    }
                )
                remaining_budget -= 1

        # Remaining slots: bridge / within-cluster fallback
        for i in range(remaining_budget):
            mode_selector = (round_num - 1 + i) % 2
            if mode_selector == 0:
                slot_idx = (round_num - 1 + i) * 3
            else:
                slot_idx = (round_num - 1 + i) * 3 + 1
            neighbor = select_structural_neighbor(
                all_pool_smiles,
                observed_smiles | {s["canonical"] for s in selected},
                observed_results,
                observed_fps,
                top_k_hits=10,
                slot_index=slot_idx,
                verbose=verbose,
            )
            if neighbor:
                can = _canonical(neighbor)
                knn_pred = knn_tanimoto_predict(neighbor, observed_results, observed_fps, k=5)
                selected.append(
                    {
                        "smiles": neighbor,
                        "canonical": can,
                        "predicted": None,
                        "knn_pred": knn_pred,
                        "blended_pred": knn_pred,
                        "exploit_score": 0.0,
                        "source": "struct_explore",
                    }
                )

    # Backfill with additional exploit candidates if needed.
    for s in scored:
        if len(selected) >= budget:
            break
        if s.get("canonical") not in {x.get("canonical") for x in selected}:
            selected.append(s)

    if verbose:
        print(f"\n  Selected {len(selected)} candidates:")
        for i, s in enumerate(selected):
            src = s.get("source", "hebo_exploit")
            knn_str = f"  knn={s['knn_pred']:.2f}" if s.get("knn_pred") is not None else ""
            pred_str = (
                f"  oracle={s.get('predicted',0):.2f}"
                if s.get("predicted") is not None
                else ""
            )
            blend_str = (
                f"  blend={s['blended_pred']:.2f}"
                if s.get("blended_pred") is not None
                else ""
            )
            print(f"    {i+1}. [{src}]{pred_str}{knn_str}{blend_str}  {s['smiles'][:50]}")

    return selected[:budget]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="EGFR lookup experiment v6")
    p.add_argument("--dataset", type=Path, default=Path("data/egfr_quinazoline.csv"))
    p.add_argument("--train-size", type=int, default=50,
                   help="Number of diverse training molecules (default: 50)")
    p.add_argument("--iterations", type=int, default=30,
                   help="HEBO iterations per round (default: 30)")
    p.add_argument("--rounds", type=int, default=15,
                   help="Number of BO rounds (default: 15)")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--experiments-per-round", type=int, default=2,
                   help="Max molecules to 'test' (lookup) per round (default: 2)")
    p.add_argument("--fp-bits", type=int, default=256,
                   help="Fingerprint bits for descriptors (default: 256)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ucb-beta", type=float, default=1.0,
                   help="UCB exploration weight for exploit slot (default: 1.0)")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if not args.dataset.exists():
        print(f"Dataset not found: {args.dataset}")
        return 1

    # ── Load full dataset ────────────────────────────────────────
    with open(args.dataset) as f:
        rows = list(csv.DictReader(f))

    all_smiles = [r["smiles"] for r in rows]
    all_targets = [float(r["pIC50"]) for r in rows]
    print(f"Loaded {len(rows)} molecules from {args.dataset}")
    print(f"pIC50 range: {min(all_targets):.2f} - {max(all_targets):.2f}")

    # ── Diverse train/lookup split ───────────────────────────────
    train_idx, lookup_idx = select_diverse_train(
        all_smiles, all_targets, args.train_size, args.seed
    )
    train_smiles = [all_smiles[i] for i in train_idx]
    train_targets = [all_targets[i] for i in train_idx]
    lookup_smiles = [all_smiles[i] for i in lookup_idx]
    lookup_targets = [all_targets[i] for i in lookup_idx]

    print(f"\nSplit: {len(train_idx)} train / {len(lookup_idx)} lookup")
    print(f"Train pIC50:  {min(train_targets):.2f} - {max(train_targets):.2f}  "
          f"(mean {sum(train_targets)/len(train_targets):.2f})")
    print(f"Lookup pIC50: {min(lookup_targets):.2f} - {max(lookup_targets):.2f}  "
          f"(mean {sum(lookup_targets)/len(lookup_targets):.2f})")

    # Build lookup table (canonical SMILES -> pIC50)
    lookup_table: dict[str, float] = {}
    for s, t in zip(lookup_smiles, lookup_targets):
        can = _canonical(s)
        if can:
            lookup_table[can] = t
    # ALSO add training molecules so every pool molecule is "testable"
    for s, t in zip(train_smiles, train_targets):
        can = _canonical(s)
        if can:
            lookup_table[can] = t

    train_set: set[str] = set()
    for s in train_smiles:
        can = _canonical(s)
        if can:
            train_set.add(can)

    # Build the FULL pool SMILES list for structure-guided exploration
    all_pool_smiles = list(all_smiles)

    print(f"Lookup table: {len(lookup_table)} | Train set: {len(train_set)}")
    print(f"Full pool for exploration: {len(all_pool_smiles)}")

    # ── Save train CSV ───────────────────────────────────────────
    train_csv = Path("data/egfr_train.csv")
    with open(train_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["smiles", "pIC50"])
        for s, t in zip(train_smiles, train_targets):
            w.writerow([s, t])

    lookup_csv = Path("data/egfr_lookup.csv")
    with open(lookup_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["smiles", "pIC50"])
        for s, t in zip(lookup_smiles, lookup_targets):
            w.writerow([s, t])

    print(f"Saved {train_csv} and {lookup_csv}")

    # ── Engine ───────────────────────────────────────────────────
    from src.bo_workflow.engine import BOEngine
    engine = BOEngine()

    round_results = []
    current_run_id = None
    observed_smiles: set[str] = set(train_set)

    # Track best found across rounds for stagnation detection.
    # KEY FIX (v3): only track discoveries from experiments, NOT training set.
    best_found_so_far = -999.0
    rounds_since_improvement = 0

    # Track real pIC50 values and fingerprints for k-NN oracle
    observed_results: dict[str, float] = {}
    observed_fps: dict = {}

    # v6d: Track discovery order for NN_recent slot
    discoveries_order: list[str] = []

    # Seed with training data
    for s, t in zip(train_smiles, train_targets):
        can = _canonical(s)
        if can:
            observed_results[can] = t
            fp = _tanimoto_fp(can)
            if fp is not None:
                observed_fps[can] = fp

    for round_num in range(1, args.rounds + 1):
        print()
        print("=" * 70)
        best_disp = f"{best_found_so_far:.3f}" if best_found_so_far > -999 else "none"
        print(f"  ROUND {round_num} / {args.rounds}  "
              f"(best discovered: {best_disp}, "
              f"stagnation: {rounds_since_improvement} rounds)")
        print("=" * 70)

        if round_num == 1:
            # ── Init ─────────────────────────────────────────────
            print("\n--- Phase 1: Init SMILES run (discovery mode) ---")

            result = engine.init_smiles_run(
                dataset_path=str(args.dataset),
                target_column="pIC50",
                objective="max",
                discovery=True,
                seed=args.seed,
                fingerprint_bits=args.fp_bits,
                verbose=args.verbose,
            )
            current_run_id = result["run_id"]
            print(f"  Run ID:  {current_run_id}")
            print(f"  Pool:    {len(rows)} molecules (discovery mode)")
            print(f"  Features: {len(result.get('active_features', []))} selected")
            print(f"  FP bits: {args.fp_bits}")

            state_path = Path(f"runs/{current_run_id}/state.json")
            state = json.loads(state_path.read_text())
            state["_full_dataset"] = state.get("dataset_path", state.get("dataset"))
            state["dataset_path"] = str(train_csv.resolve())
            state_path.write_text(json.dumps(state, indent=2))
            print(f"  Oracle trains on: {train_csv} ({len(train_idx)} rows)")

        else:
            # ── Resume: feed back results from previous round ────
            print("\n--- Feeding back results ---")

            state_path = Path(f"runs/{current_run_id}/state.json")
            state = json.loads(state_path.read_text())
            state["status"] = "running"
            state_path.write_text(json.dumps(state, indent=2))

            prev = round_results[-1]

            # Add HITS (real pIC50 values) to training set
            added_real = 0
            if prev["matches"]:
                with open(train_csv, "a", newline="") as f:
                    w = csv.writer(f)
                    for m in prev["matches"]:
                        w.writerow([m["smiles"], m["real_pIC50"]])
                        added_real += 1

            # Add MISSES (penalty values) to training set
            added_penalty = 0
            if prev["misses"]:
                with open(train_csv, "a", newline="") as f:
                    w = csv.writer(f)
                    for m in prev["misses"]:
                        w.writerow([m["smiles"], m["penalty_y"]])
                        added_penalty += 1

                # Also update descriptor_lookup so oracle can use these
                desc_lookup_path = Path(f"runs/{current_run_id}/descriptor_lookup.json")
                desc_lookup = json.loads(desc_lookup_path.read_text())
                from src.bo_workflow.molecular.features import compute_descriptors
                from src.bo_workflow.molecular.types import DescriptorConfig
                desc_config = DescriptorConfig(
                    basic=True, fingerprint_enabled=True,
                    fingerprint_n_bits=args.fp_bits, fingerprint_radius=2,
                    electronic=True, steric=False,
                )
                for m in prev["misses"]:
                    try:
                        desc = compute_descriptors(m["smiles"], config=desc_config)
                        desc_lookup[m["smiles"]] = desc
                    except Exception:
                        pass
                desc_lookup_path.write_text(json.dumps(desc_lookup))

            new_count = sum(1 for _ in open(train_csv)) - 1
            print(f"  Added: {added_real} real + {added_penalty} penalty -> "
                  f"{new_count} total training rows")

        # ── Build oracle ─────────────────────────────────────────
        print("\n--- Phase 2: Build oracle ---")
        oracle = engine.build_oracle(current_run_id, verbose=args.verbose)
        print(f"  Model: {oracle['selected_model']} | CV RMSE: {oracle['selected_rmse']:.4f}")

        # ── HEBO ─────────────────────────────────────────────────
        print(f"\n--- Phase 3: HEBO ({args.iterations} iterations) ---")
        engine.run_proxy_optimization(
            current_run_id,
            num_iterations=args.iterations,
            batch_size=args.batch_size,
            verbose=args.verbose,
        )

        # ── Collect ALL unique proposals from observations ─────────
        print(f"\n--- Phase 4: Pre-screen + Experiment ---")

        obs_path = Path(f"runs/{current_run_id}/observations.jsonl")
        all_obs = [json.loads(line) for line in obs_path.read_text().strip().split("\n")]

        # Deduplicate proposals by canonical SMILES, keep best predicted
        proposal_map: dict[str, dict] = {}
        for obs in all_obs:
            smi = obs.get("matched_smiles") or obs.get("x", {}).get("smiles")
            if not smi:
                continue
            can = _canonical(smi)
            if not can:
                continue
            pred = obs.get("y")
            if can not in proposal_map or (pred is not None and pred > (proposal_map[can].get("predicted") or -999)):
                proposal_map[can] = {
                    "smiles": smi,
                    "canonical": can,
                    "predicted": pred,
                }

        proposals = list(proposal_map.values())

        # ── Load descriptor lookup for pre-screening ─────────────
        desc_lookup_path = Path(f"runs/{current_run_id}/descriptor_lookup.json")
        descriptor_lookup = json.loads(desc_lookup_path.read_text())
        state = json.loads(Path(f"runs/{current_run_id}/state.json").read_text())
        active_features = state.get("active_features", [])

        # ── Pre-screen: dual-arm selection ───────────────────────
        budget = args.experiments_per_round
        print(f"\n  Experiment budget: {budget}/round")

        selected = prescreen_candidates(
            proposals,
            descriptor_lookup,
            active_features,
            observed_smiles,
            observed_results,
            observed_fps,
            budget=budget,
            all_pool_smiles=all_pool_smiles,
            discoveries_order=discoveries_order,
            round_num=round_num,
            best_found_so_far=best_found_so_far,
            rounds_since_improvement=rounds_since_improvement,
            oracle_rmse=oracle.get("selected_rmse"),
            ucb_beta=args.ucb_beta,
            verbose=args.verbose,
        )

        # ── "Run experiments" (lookup) on selected candidates ────
        matches = []
        misses = []

        for s in selected:
            can = s["canonical"]
            smi = s["smiles"]
            pred = s.get("predicted")
            knn = s.get("knn_pred")
            blended = s.get("blended_pred")
            source = s.get("source", "hebo_exploit")

            if can in lookup_table:
                # HIT -- we have real data
                real = lookup_table[can]
                matches.append({
                    "smiles": smi,
                    "predicted_target": pred,
                    "knn_pred": knn,
                    "blended_pred": blended,
                    "real_pIC50": real,
                    "prediction_error": abs((pred or 0) - real) if pred is not None else None,
                    "knn_error": abs(knn - real) if knn is not None else None,
                    "source": source,
                })
                observed_smiles.add(can)
                observed_results[can] = real
                fp = _tanimoto_fp(can)
                if fp is not None:
                    observed_fps[can] = fp
                # v6d: track non-training discoveries for NN_recent slot
                if can not in train_set and can not in set(discoveries_order):
                    discoveries_order.append(can)
                knn_str = f"  knn={knn:.2f}" if knn is not None else ""
                pred_str = f"oracle={pred:.2f}" if pred is not None else "oracle=N/A"
                print(f"  HIT:  {pred_str}{knn_str}  real={real:.2f}  "
                      f"[{source}]  {smi[:50]}")
            else:
                # MISS -- molecule not in lookup
                worst_y = min(all_targets)
                penalty_y = worst_y - 1.0
                misses.append({
                    "smiles": smi,
                    "predicted_target": pred,
                    "knn_pred": knn,
                    "blended_pred": blended,
                    "penalty_y": penalty_y,
                    "source": source,
                    "reason": "not_in_lookup",
                })
                observed_smiles.add(can)
                pred_str = f"oracle={pred:.2f}" if pred is not None else "oracle=N/A"
                print(f"  MISS: {pred_str}  penalty={penalty_y:.2f}  "
                      f"[{source}]  {smi[:50]}")

        # ── Update stagnation tracking ────────────────────────────
        round_best = -999
        if matches:
            new_discoveries = [m["real_pIC50"] for m in matches if m["smiles"] not in train_set]
            if new_discoveries:
                round_best = max(new_discoveries)
            else:
                round_best = max(m["real_pIC50"] for m in matches)

        if round_best > best_found_so_far and round_best > -999:
            best_found_so_far = round_best
            rounds_since_improvement = 0
            print(f"  >>> NEW BEST DISCOVERY: {best_found_so_far:.3f} <<<")
        else:
            rounds_since_improvement += 1

        # ── Round summary ────────────────────────────────────────
        if matches:
            reals = [m["real_pIC50"] for m in matches]
            print(f"\n  Round {round_num}: {len(matches)} hits / {len(misses)} misses  "
                  f"best_this_round={max(reals):.3f}  "
                  f"best_overall={best_found_so_far:.3f}")
        else:
            print(f"\n  Round {round_num}: 0 hits / {len(misses)} misses  "
                  f"best_overall={best_found_so_far:.3f}")

        round_results.append({
            "round": round_num,
            "matches": matches,
            "misses": misses,
            "oracle_rmse": oracle["selected_rmse"],
            "best_found_so_far": best_found_so_far,
            "rounds_since_improvement": rounds_since_improvement,
        })

    # ── Final summary ────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  EXPERIMENT SUMMARY")
    print("=" * 70)

    best_lookup = max(lookup_table.values())
    best_lookup_smi = [s for s, v in lookup_table.items() if v == best_lookup][0]
    print(f"\n  Best in entire dataset: pIC50={best_lookup:.3f}  {best_lookup_smi[:60]}")
    print(f"  Best in initial train:  pIC50={max(train_targets):.3f}")
    print(f"  Experiment budget:  {args.experiments_per_round}/round x {args.rounds} rounds "
          f"= {args.experiments_per_round * args.rounds} total experiments")

    all_found_reals = []
    total_hits = 0
    total_misses = 0
    for r in round_results:
        m = r["matches"]
        ms = r["misses"]
        total_hits += len(m)
        total_misses += len(ms)
        if m:
            reals = [x["real_pIC50"] for x in m]
            all_found_reals.extend(reals)
            sources = [x.get("source", "?") for x in m]
            print(f"\n  Round {r['round']}: {len(m)} hits / {len(ms)} misses  "
                  f"best={max(reals):.3f}  "
                  f"RMSE={r['oracle_rmse']:.3f}  "
                  f"cumul_best={r['best_found_so_far']:.3f}  "
                  f"sources={sources}")
        else:
            print(f"\n  Round {r['round']}: 0 hits / {len(ms)} misses  "
                  f"RMSE={r['oracle_rmse']:.3f}  "
                  f"cumul_best={r['best_found_so_far']:.3f}")

    print(f"\n  Total: {total_hits} hits / {total_misses} misses "
          f"({total_hits+total_misses} experiments)")

    if all_found_reals:
        overall_best = max(all_found_reals)
        print(f"  Best found: pIC50={overall_best:.3f}  "
              f"(target={best_lookup:.3f}  gap={best_lookup - overall_best:.3f})")
        unique_found = len(set(m["smiles"] for r in round_results for m in r["matches"]))
        print(f"  Unique molecules discovered: {unique_found}")

        good_threshold = best_lookup - 1.0
        good_found = [x for x in all_found_reals if x >= good_threshold]
        if good_found:
            print(f"  'Good' molecules (pIC50 >= {good_threshold:.1f}): {len(good_found)}")

        # Top-5 best discoveries
        all_matches_sorted = sorted(
            [(m["smiles"], m["real_pIC50"], m.get("source", "?"))
             for r in round_results for m in r["matches"]],
            key=lambda x: -x[1]
        )
        print(f"\n  Top-10 discoveries:")
        seen_top = set()
        rank = 0
        for smi, val, src in all_matches_sorted:
            if smi in seen_top:
                continue
            seen_top.add(smi)
            rank += 1
            in_train = " [TRAIN]" if smi in train_set or _canonical(smi) in train_set else ""
            print(f"    {rank}. pIC50={val:.3f}  [{src}]  {smi[:50]}{in_train}")
            if rank >= 10:
                break
    else:
        print("\n  No lookup hits found.")

    # Source breakdown
    hebo_hits = [m for r in round_results for m in r["matches"] if m.get("source") == "hebo_exploit"]
    nn_hits = [m for r in round_results for m in r["matches"] if m.get("source") == "greedy_nn"]
    novelty_hits = [m for r in round_results for m in r["matches"] if m.get("source") == "novelty"]
    struct_hits = [m for r in round_results for m in r["matches"] if m.get("source") == "struct_explore"]
    nn_recent_hits = [m for r in round_results for m in r["matches"] if m.get("source") == "nn_recent"]
    print(f"\n  Source breakdown:")
    print(f"    HEBO exploit:     {len(hebo_hits)} hits")
    print(f"    Greedy NN:        {len(nn_hits)} hits")
    print(f"    Novelty:          {len(novelty_hits)} hits")
    print(f"    NN recent:        {len(nn_recent_hits)} hits")
    print(f"    Struct explore:   {len(struct_hits)} hits")
    if hebo_hits:
        print(f"    HEBO best:        {max(m['real_pIC50'] for m in hebo_hits):.3f}")
    if nn_hits:
        print(f"    Greedy NN best:   {max(m['real_pIC50'] for m in nn_hits):.3f}")
    if novelty_hits:
        print(f"    Novelty best:     {max(m['real_pIC50'] for m in novelty_hits):.3f}")
    if nn_recent_hits:
        print(f"    NN recent best:   {max(m['real_pIC50'] for m in nn_recent_hits):.3f}")
    if struct_hits:
        print(f"    Struct best:      {max(m['real_pIC50'] for m in struct_hits):.3f}")

    # ── Save results ─────────────────────────────────────────────
    results_dir = Path(f"runs/{current_run_id}")

    experiment_output = {
        "run_id": current_run_id,
        "version": "v6d-hebo-only",
        "dataset": str(args.dataset),
        "train_size": len(train_idx),
        "lookup_size": len(lookup_table),
        "experiments_per_round": args.experiments_per_round,
        "total_rounds": args.rounds,
        "hebo_iterations_per_round": args.iterations,
        "fingerprint_bits": args.fp_bits,
        "best_in_dataset": {"smiles": best_lookup_smi, "pIC50": best_lookup},
        "best_in_train": {"pIC50": max(train_targets)},
        "descriptor_dims": len(active_features),
        "rounds": [],
    }
    for r in round_results:
        rd = {
            "round": r["round"],
            "oracle_rmse": r["oracle_rmse"],
            "hits": len(r["matches"]),
            "misses": len(r["misses"]),
            "matches": r["matches"],
            "miss_details": r["misses"],
            "best_found_so_far": r["best_found_so_far"],
            "rounds_since_improvement": r["rounds_since_improvement"],
        }
        if r["matches"]:
            reals = [x["real_pIC50"] for x in r["matches"]]
            rd["best_real"] = max(reals)
        experiment_output["rounds"].append(rd)

    if all_found_reals:
        experiment_output["overall_best_found"] = max(all_found_reals)
        experiment_output["gap_to_best"] = best_lookup - max(all_found_reals)

    exp_json_path = results_dir / "experiment_results.json"
    exp_json_path.write_text(json.dumps(experiment_output, indent=2))
    print(f"\n  Saved: {exp_json_path}")

    all_matches = []
    for r in round_results:
        for m in r["matches"]:
            all_matches.append({
                "round": r["round"],
                "smiles": m["smiles"],
                "predicted_pIC50": m["predicted_target"],
                "knn_pIC50": m.get("knn_pred"),
                "real_pIC50": m["real_pIC50"],
                "error": m["prediction_error"],
                "source": m.get("source", ""),
            })
        for m in r["misses"]:
            all_matches.append({
                "round": r["round"],
                "smiles": m["smiles"],
                "predicted_pIC50": m["predicted_target"],
                "knn_pIC50": m.get("knn_pred"),
                "real_pIC50": None,
                "error": None,
                "source": m.get("source", "miss"),
            })
    if all_matches:
        hits_csv_path = results_dir / "experiment_hits.csv"
        with open(hits_csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_matches[0].keys())
            w.writeheader()
            w.writerows(all_matches)
        print(f"  Saved: {hits_csv_path}")

    print(f"\n  Artifacts: runs/{current_run_id}/")
    return 0


if __name__ == "__main__":
    import sys
    from pathlib import Path as _P
    _root = str(_P(__file__).resolve().parent.parent)
    if _root not in sys.path:
        sys.path.insert(0, _root)
    raise SystemExit(main())
