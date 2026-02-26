from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ....engine import BOEngine
from ....molecular.features import compute_descriptors
from ....molecular.types import DescriptorConfig
from ....utils import read_jsonl
from .common import (
    build_egfr_global_argv,
    canonical_smiles,
    json_print,
    load_smiles_inputs,
)


def handle_workflows(args: argparse.Namespace, engine: BOEngine) -> int | None:
    if args.command == "scaffold-edit":
        seed_smiles = None
        if args.seed_smiles:
            p = Path(args.seed_smiles)
            if p.exists():
                seed_smiles = [line.strip() for line in p.read_text().splitlines() if line.strip()]
            else:
                seed_smiles = [s.strip() for s in args.seed_smiles.split(",") if s.strip()]

        constraint_dict = None
        if args.fixed_smarts or args.mutable_smarts:
            constraint_dict = {
                "fixed_smarts": [s.strip() for s in (args.fixed_smarts or "").split(",") if s.strip()],
                "mutable_smarts": [s.strip() for s in (args.mutable_smarts or "").split(",") if s.strip()],
                "fixed_atom_indices": [],
            }

        operations = None
        if args.operations:
            operations = [o.strip() for o in args.operations.split(",") if o.strip()]

        payload = engine.init_scaffold_edit_run(
            dataset_path=args.dataset,
            target_column=args.target,
            objective=args.objective,
            smiles_column=args.smiles_column,
            seed_smiles=seed_smiles,
            top_k_seeds=args.top_k_seeds,
            crem_db_path=args.crem_db,
            mutation_constraint=constraint_dict,
            max_size=args.max_size,
            radius=args.radius,
            max_replacements_per_seed=args.max_replacements,
            operations=operations,
            sa_threshold=args.sa_threshold,
            default_engine=args.engine,
            run_id=args.run_id,
            seed=args.seed,
            fingerprint_bits=args.fingerprint_bits,
            verbose=args.verbose,
        )
        json_print(payload)
        return 0

    if args.command == "augment-energy":
        payload = engine.augment_with_energy(args.run_id, energy_backend=args.backend, verbose=args.verbose)
        json_print(payload)
        return 0

    if args.command == "top-candidates":
        payload = engine.get_top_candidates(args.run_id, top_k=args.top_k, verbose=args.verbose)
        json_print(payload)
        return 0

    if args.command == "validate":
        payload = engine.validate_candidates(args.run_id, method=args.method, top_k=args.top_k, verbose=args.verbose)
        json_print(payload)
        return 0

    if args.command == "screen":
        if args.verbose:
            print("[screen] Initializing SMILES-direct run...", file=sys.stderr)
        state = engine.init_smiles_run(
            dataset_path=args.dataset,
            target_column=args.target,
            objective=args.objective,
            smiles_column=args.smiles_column,
            default_engine=args.engine,
            seed=args.seed,
            default_batch_size=args.batch_size,
            fingerprint_bits=args.fingerprint_bits,
            energy_backend=args.energy_backend,
            discovery=args.discovery,
            verbose=args.verbose,
        )
        run_id = state["run_id"]
        if args.verbose:
            print(f"[screen] run_id={run_id}", file=sys.stderr)

        if args.verbose:
            print("[screen] Training proxy oracle...", file=sys.stderr)
        oracle_result = engine.build_oracle(run_id, verbose=args.verbose)
        if args.verbose:
            print(
                f"[screen] Oracle: {oracle_result['selected_model']} "
                f"(CV RMSE={oracle_result['selected_rmse']:.4f})",
                file=sys.stderr,
            )

        if args.verbose:
            print(f"[screen] Running {args.iterations} BO iterations...", file=sys.stderr)
        engine.run_proxy_optimization(run_id, num_iterations=args.iterations, batch_size=args.batch_size, verbose=args.verbose)

        report = engine.report(run_id, verbose=args.verbose)
        json_print(report)
        return 0

    if args.command == "smiles-discovery":
        input_smiles = load_smiles_inputs(args.smiles, args.smiles_file)

        if args.verbose:
            print(f"[smiles-discovery] {len(input_smiles)} input SMILES loaded", file=sys.stderr)

        state = engine.init_smiles_run(
            dataset_path=args.dataset,
            target_column=args.target,
            objective=args.objective,
            default_engine=args.engine,
            seed=args.seed,
            default_batch_size=args.batch_size,
            fingerprint_bits=args.fingerprint_bits,
            energy_backend=args.energy_backend,
            discovery=True,
            verbose=args.verbose,
        )
        run_id = state["run_id"]

        descriptor_cfg = DescriptorConfig(
            basic=True,
            fingerprint_enabled=True,
            fingerprint_n_bits=args.fingerprint_bits,
            fingerprint_radius=2,
            electronic=True,
            steric=False,
            dft_enabled=args.energy_backend != "none",
            energy_backend=args.energy_backend,
        )
        reference_vectors = [compute_descriptors(smi, config=descriptor_cfg) for smi in input_smiles]

        oracle = engine.build_oracle(run_id, verbose=args.verbose)
        engine.run_proxy_optimization(run_id, num_iterations=args.iterations, batch_size=args.batch_size, verbose=args.verbose)

        run_state = engine._load_state(run_id)
        active_features = list(run_state.get("active_features", []))
        observations = read_jsonl(engine._paths(run_id).observations)

        ranked_by_smiles: dict[str, dict[str, Any]] = {}
        for row in observations:
            matched = row.get("matched_smiles")
            if not isinstance(matched, str) or not matched:
                continue
            candidate_smiles = canonical_smiles(matched)
            if candidate_smiles in input_smiles:
                continue

            x_row = row.get("x", {})
            if not isinstance(x_row, dict):
                x_row = {}

            distance = 0.0
            if active_features and reference_vectors:
                distances = []
                for ref_vec in reference_vectors:
                    d2 = 0.0
                    for f in active_features:
                        xv = float(x_row.get(f, 0.0))
                        rv = float(ref_vec.get(f, 0.0))
                        d2 += (xv - rv) ** 2
                    distances.append(float(np.sqrt(d2)))
                if distances:
                    distance = float(min(distances))

            item = {
                "smiles": candidate_smiles,
                "simulated_score": float(row.get("y", np.nan)),
                "distance_to_input_in_hebo_space": distance,
                "discovery_distance": row.get("discovery_distance"),
                "iteration": row.get("iteration"),
            }

            prev = ranked_by_smiles.get(candidate_smiles)
            if prev is None or item["simulated_score"] > prev["simulated_score"]:
                ranked_by_smiles[candidate_smiles] = item

        ranked = sorted(
            ranked_by_smiles.values(),
            key=lambda r: (-r["simulated_score"], r["distance_to_input_in_hebo_space"]),
        )

        json_print(
            {
                "mode": "smiles_discovery",
                "note": "All scores are proxy-oracle simulations (not real experiments).",
                "run_id": run_id,
                "dataset": str(args.dataset),
                "input_smiles": input_smiles,
                "objective": args.objective,
                "target": args.target,
                "oracle_cv_rmse": oracle.get("selected_rmse"),
                "top_candidates": ranked[: args.top_k],
            }
        )
        return 0

    if args.command == "egfr-ic50-global":
        from ....workflows.egfr_ic50_global import main as egfr_global_main

        delegated_argv = build_egfr_global_argv(args)
        return int(egfr_global_main(delegated_argv))

    return None
