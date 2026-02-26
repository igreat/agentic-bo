from __future__ import annotations

from .chem import _canonical
from .descriptor import (
    _build_canonical_descriptor_lookup,
    _descriptor_distance_to_observed,
    _pick_uncertainty_exploit_from_pool,
)
from .selection import (
    select_greedy_nn_exploit,
    select_nn_from_recent,
    select_novelty,
    select_structural_neighbor,
)
from .similarity import knn_tanimoto_predict


def prescreen_candidates(
    proposals: list[dict],
    descriptor_lookup: dict[str, dict[str, float]],
    active_features: list[str],
    observed_smiles: set[str],
    observed_results: dict[str, float],
    observed_fps: dict,
    budget: int,
    *,
    all_pool_smiles: list[str] = None,
    discoveries_order: list[str] = None,
    round_num: int = 1,
    best_found_so_far: float = -999,
    rounds_since_improvement: int = 0,
    oracle_rmse: float | None = None,
    ucb_beta: float = 1.0,
    hebo_exploit_slots: int = 1,
    hebo_explore_slots: int = 0,
    hebo_explore_beta: float = 2.0,
    exploit_only: bool = False,
    adaptive_novelty: bool = False,
    novelty_grace_rounds: int = 10,
    stall_rounds_threshold: int = 3,
    verbose: bool = False,
) -> list[dict]:
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

    if exploit_only:
        exploit_selected = []
        for s in scored[:budget]:
            item = dict(s)
            item["source"] = "hebo_exploit"
            exploit_selected.append(item)
        if verbose:
            print(f"\n  [EXPLOIT-ONLY] Selected top-{len(exploit_selected)} by UCB")
        return exploit_selected

    selected: list[dict] = []

    exploit_slots = max(0, int(hebo_exploit_slots))
    explore_slots = max(0, int(hebo_explore_slots))
    if exploit_slots + explore_slots == 0:
        exploit_slots = 1

    total_hebo_slots = min(budget, exploit_slots + explore_slots)
    if total_hebo_slots < exploit_slots:
        exploit_slots = total_hebo_slots
        explore_slots = 0
    elif total_hebo_slots < exploit_slots + explore_slots:
        explore_slots = total_hebo_slots - exploit_slots

    chosen_can = set()

    if scored and exploit_slots > 0:
        ranked_exploit = sorted(scored, key=lambda x: -float(x.get("blended_pred") or 0.0))
        for s in ranked_exploit:
            if len(selected) >= exploit_slots:
                break
            can = s.get("canonical")
            if can in chosen_can:
                continue
            item = dict(s)
            item["source"] = "hebo_exploit"
            selected.append(item)
            chosen_can.add(can)

    if scored and explore_slots > 0:
        ranked_explore = sorted(
            scored,
            key=lambda x: -(
                float(x.get("blended_pred") or 0.0)
                + float(hebo_explore_beta) * float(x.get("uncertainty") or 0.0) * rmse_scale
            ),
        )
        target_total = exploit_slots + explore_slots
        for s in ranked_explore:
            if len(selected) >= target_total:
                break
            can = s.get("canonical")
            if can in chosen_can:
                continue
            item = dict(s)
            item["source"] = "hebo_explore"
            selected.append(item)
            chosen_can.add(can)

    if verbose and selected:
        print(
            f"  [HEBO-SLOTS] selected {len(selected)} "
            f"(exploit={exploit_slots}, explore={explore_slots}, explore_beta={hebo_explore_beta})"
        )

    if not selected and budget >= 1 and all_pool_smiles:
        fallback = _pick_uncertainty_exploit_from_pool(
            all_pool_smiles,
            observed_smiles,
            canonical_desc_lookup,
            active_features,
        )
        if fallback is not None:
            fallback_smi, fallback_dist = fallback
            can = _canonical(fallback_smi)
            knn_pred = knn_tanimoto_predict(fallback_smi, observed_results, observed_fps, k=5)
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

    remaining_budget = budget - len(selected)
    if remaining_budget > 0 and all_pool_smiles:
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
                knn_pred = knn_tanimoto_predict(nn_pick, observed_results, observed_fps, k=5)
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

        if remaining_budget >= 1:
            use_novelty = True
            if (
                adaptive_novelty
                and round_num > novelty_grace_rounds
                and rounds_since_improvement >= stall_rounds_threshold
            ):
                use_novelty = False
                if verbose:
                    print(
                        "  [ADAPTIVE] Stalled run: reallocate novelty slot to exploit "
                        f"(stall={rounds_since_improvement}, threshold={stall_rounds_threshold})"
                    )

            if use_novelty:
                novelty_pick = select_novelty(
                    all_pool_smiles,
                    observed_smiles | {s["canonical"] for s in selected},
                    observed_fps,
                    verbose=verbose,
                )
                if novelty_pick:
                    can = _canonical(novelty_pick)
                    knn_pred = knn_tanimoto_predict(novelty_pick, observed_results, observed_fps, k=5)
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
            else:
                nn_pick_2 = select_greedy_nn_exploit(
                    all_pool_smiles,
                    observed_smiles | {s["canonical"] for s in selected},
                    observed_results,
                    observed_fps,
                    top_k_seeds=8,
                    round_num=round_num + 11,
                    verbose=verbose,
                )
                if nn_pick_2:
                    can = _canonical(nn_pick_2)
                    knn_pred = knn_tanimoto_predict(nn_pick_2, observed_results, observed_fps, k=5)
                    selected.append(
                        {
                            "smiles": nn_pick_2,
                            "canonical": can,
                            "predicted": None,
                            "knn_pred": knn_pred,
                            "blended_pred": knn_pred,
                            "exploit_score": 0.0,
                            "source": "greedy_nn_adaptive",
                        }
                    )
                    remaining_budget -= 1

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
                knn_pred = knn_tanimoto_predict(recent_pick, observed_results, observed_fps, k=5)
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
            pred_str = f"  oracle={s.get('predicted',0):.2f}" if s.get("predicted") is not None else ""
            blend_str = f"  blend={s['blended_pred']:.2f}" if s.get("blended_pred") is not None else ""
            print(f"    {i+1}. [{src}]{pred_str}{knn_str}{blend_str}  {s['smiles'][:50]}")

    return selected[:budget]
