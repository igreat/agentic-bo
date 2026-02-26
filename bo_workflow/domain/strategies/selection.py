from __future__ import annotations

from .chem import _canonical, _is_drug_like, _tanimoto_fp, _tanimoto_similarity
from .similarity import _select_diverse_seeds


def _cluster_top_hits(
    top_hits: list[tuple[str, float]],
    observed_fps: dict,
    sim_threshold: float = 0.5,
) -> list[list[tuple[str, float]]]:
    clusters: list[list[tuple[str, float]]] = []
    cluster_fps = []

    for smi, pic50 in top_hits:
        fp = observed_fps.get(smi) or _tanimoto_fp(smi)
        if fp is None:
            continue

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
        print(f"  [WITHIN-CLUSTER] cluster={cluster_idx}  score={best_score:.3f}  {best_candidate[:55]}")
    return best_candidate


def select_cross_cluster_bridge(
    all_pool_smiles: list[str],
    observed_smiles: set[str],
    clusters: list[list[tuple[str, float]]],
    observed_fps: dict,
    pair_index: int = 0,
    verbose: bool = False,
) -> str | None:
    if len(clusters) < 2:
        return None

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

        cluster_sims = []
        for _, best_pic50, fps_list in top_clusters:
            max_sim = max(_tanimoto_similarity(fp, cfp) for cfp, _ in fps_list)
            cluster_sims.append((max_sim, best_pic50))

        components = []
        for sim, pic50 in cluster_sims:
            if sim < 0.25:
                break
            components.append(sim * max(0.1, pic50 - 6.0))

        if len(components) < 2:
            continue

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
        print(f"  [BRIDGE] pair={pair_label}  score={best_score:.3f}  sims=[{', '.join(sims_str)}]  {best_candidate[:55]}")
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
    all_obs_fps = list(observed_fps.values())
    if not all_obs_fps:
        return None

    best_candidate = None
    best_min_sim = 999

    for smi in all_pool_smiles:
        can = _canonical(smi)
        if can is None or can in observed_smiles:
            continue
        if not _is_drug_like(smi):
            continue
        fp = _tanimoto_fp(smi)
        if fp is None:
            continue
        max_sim = max(_tanimoto_similarity(fp, ofp) for ofp in all_obs_fps)
        if max_sim < best_min_sim:
            best_min_sim = max_sim
            best_candidate = smi

    if verbose and best_candidate:
        print(f"  [NOVELTY] max_sim_to_observed={best_min_sim:.3f}  {best_candidate[:55]}")
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
    if not observed_results:
        return None

    sorted_hits = sorted(observed_results.items(), key=lambda x: -x[1])
    top_hits = sorted_hits[:top_k_hits]
    clusters = _cluster_top_hits(top_hits, observed_fps, sim_threshold=0.5)

    if not clusters:
        return None

    if verbose:
        print(f"  [STRUCT-EXPLORE] {len(clusters)} clusters in top-{len(top_hits)} hits:")
        for ci, cluster in enumerate(clusters):
            best_in_c = max(cluster, key=lambda x: x[1])
            print(f"    C{ci}: {len(cluster)} hits  best={best_in_c[1]:.3f}  rep={best_in_c[0][:45]}")

    n_modes = 3
    mode = slot_index % n_modes

    if verbose:
        mode_names = ["within-cluster", "cross-bridge", "novelty"]
        print(f"    --> Mode: {mode_names[mode]} (slot={slot_index})")

    candidate = None

    if mode == 0:
        cluster_idx = (slot_index // n_modes) % len(clusters)
        candidate = select_within_cluster(
            all_pool_smiles, observed_smiles, clusters, observed_fps, cluster_idx, verbose=verbose
        )
    elif mode == 1:
        bridge_call_index = slot_index // n_modes
        candidate = select_cross_cluster_bridge(
            all_pool_smiles, observed_smiles, clusters, observed_fps, pair_index=bridge_call_index, verbose=verbose
        )
    elif mode == 2:
        candidate = select_novelty(all_pool_smiles, observed_smiles, observed_fps, verbose=verbose)

    if candidate is None and mode != 0:
        cluster_idx = (slot_index // n_modes) % len(clusters)
        candidate = select_within_cluster(
            all_pool_smiles, observed_smiles, clusters, observed_fps, cluster_idx, verbose=verbose
        )
        if verbose and candidate:
            print(f"    (fallback to within-cluster {cluster_idx})")

    return candidate


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
    if not observed_results:
        return None

    seed_data = _select_diverse_seeds(observed_results, observed_fps, n_seeds=top_k_seeds, pIC50_threshold=7.0)
    if not seed_data:
        return None

    if verbose:
        print(f"  [GREEDY-NN] {len(seed_data)} diverse seeds:")
        for i, (smi, pic50, _) in enumerate(seed_data):
            marker = " <--" if i == (round_num - 1) % len(seed_data) else ""
            print(f"    seed#{i}: pIC50={pic50:.1f}  {smi[:50]}{marker}")

    pool_fps = []
    for smi in all_pool_smiles:
        can = _canonical(smi)
        if can is None or can in observed_smiles:
            continue
        fp = _tanimoto_fp(smi)
        if fp is not None:
            pool_fps.append((smi, can, fp))

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
                print(
                    f"  [GREEDY-NN] PICK: seed#{seed_idx}({seed_smi[:30]}..={seed_pic50:.1f})  "
                    f"sim={best_sim:.3f}  {best_candidate[:55]}"
                )
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
    if not discoveries_order:
        return None

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
            print(f"  [NN-RECENT] seed={seed_can[:35]}..  sim={best_sim:.3f}  {best_candidate[:55]}")
        return best_candidate

    return None
