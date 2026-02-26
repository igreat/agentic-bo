from __future__ import annotations

import argparse
import json
import sys
import tempfile
from itertools import product as iter_product
from pathlib import Path

from ....engine import BOEngine
from ....molecular.energy import _estimate_strain_energy, get_energy_features
from ....molecular.feasibility import assess_feasibility
from ....molecular.features import (
    assemble_molecule,
    compute_descriptors,
)
from ....molecular.scaffold import (
    decode_suggestion,
    load_scaffold_spec,
    save_draft_spec,
    smiles_to_draft_spec,
)
from ....molecular.types import DescriptorConfig
from .common import json_print, parse_json_object


def handle_molecular(args: argparse.Namespace, engine: BOEngine) -> int | None:
    if args.command == "init-molecular":
        intent_payload = None
        if args.intent_json is not None:
            intent_payload = parse_json_object(args.intent_json)

        scaffold_spec_path = args.scaffold_spec
        if args.energy_backend is not None:
            with scaffold_spec_path.open("r", encoding="utf-8") as fh:
                spec_raw = json.load(fh)
            desc_cfg = spec_raw.setdefault("descriptor_config", {})
            desc_cfg["energy_backend"] = args.energy_backend
            tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
            json.dump(spec_raw, tmp, indent=2)
            tmp.close()
            scaffold_spec_path = Path(tmp.name)

        payload = engine.init_molecular_run(
            scaffold_spec_path=scaffold_spec_path,
            target_column=args.target,
            objective=args.objective,
            dataset_path=args.dataset,
            default_engine=args.engine,
            run_id=args.run_id,
            seed=args.seed,
            num_initial_random_samples=args.init_random,
            default_batch_size=args.batch_size,
            intent=intent_payload,
            verbose=args.verbose,
        )
        json_print(payload)
        return 0

    if args.command == "check-feasibility":
        smiles = args.smiles
        if smiles is None and args.run_id and args.substituents:
            state = engine._load_state(args.run_id)
            if state.get("mode") != "molecular":
                print("Error: Run is not in molecular mode.", file=sys.stderr)
                return 1
            spec = load_scaffold_spec(state["molecular"]["scaffold_spec_path"])
            subs = json.loads(args.substituents)
            smiles, _choices = decode_suggestion(subs, spec)
        elif smiles is None:
            print("Error: Provide --smiles or both --run-id and --substituents.", file=sys.stderr)
            return 1

        strain_energy = _estimate_strain_energy(smiles)
        result = assess_feasibility(
            smiles,
            sa_threshold=args.sa_threshold,
            strain_energy_kcal=strain_energy,
            strain_threshold_kcal=args.strain_threshold,
        )
        json_print(
            {
                "smiles": smiles,
                "is_feasible": result.is_feasible,
                "sa_score": result.sa_score,
                "num_steps_estimate": result.num_steps_estimate,
                "penalty": result.penalty,
                "strain_energy_kcal": result.strain_energy_kcal,
                "reasons": result.reasons,
            }
        )
        return 0

    if args.command == "compute-descriptors":
        config = None
        if args.config:
            cfg_raw = json.loads(args.config)
            fp_raw = cfg_raw.get("fingerprint", {})
            config = DescriptorConfig(
                basic=cfg_raw.get("basic", True),
                fingerprint_enabled=fp_raw.get("enabled", True) if isinstance(fp_raw, dict) else bool(fp_raw),
                fingerprint_n_bits=fp_raw.get("n_bits", 128) if isinstance(fp_raw, dict) else 128,
                fingerprint_radius=fp_raw.get("radius", 2) if isinstance(fp_raw, dict) else 2,
                electronic=cfg_raw.get("electronic", True),
                steric=cfg_raw.get("steric", True),
            )
        descriptors = compute_descriptors(args.smiles, config=config)

        energy_backend = getattr(args, "energy_backend", "none")
        if energy_backend != "none":
            energy_feats = get_energy_features(args.smiles, cache=None, backend=energy_backend)
            for key, val in energy_feats.items():
                descriptors[f"dft_{key}"] = val

        json_print({"smiles": args.smiles, "descriptors": descriptors})
        return 0

    if args.command == "draft-scaffold-spec":
        positions = None
        if args.positions:
            positions = [p.strip() for p in args.positions.split(",")]

        spec = smiles_to_draft_spec(args.smiles, marked_positions=positions)
        if args.output:
            out_path = save_draft_spec(spec, args.output)
            json_print({"status": "saved", "path": str(out_path)})
        else:
            json_print(spec)
        return 0

    if args.command == "compute-energy":
        features = get_energy_features(args.smiles, cache=None, backend=args.backend, solvent=args.solvent)
        json_print({"smiles": args.smiles, "backend": args.backend, "features": features})
        return 0

    if args.command == "precompute-energies":
        from ....molecular.energy import EnergyCache

        state = engine._load_state(args.run_id)
        if state.get("mode") != "molecular":
            print("Error: Run is not in molecular mode.", file=sys.stderr)
            return 1

        spec = load_scaffold_spec(state["molecular"]["scaffold_spec_path"])
        paths = engine._paths(args.run_id)
        cache = EnergyCache(path=paths.energy_cache, method=args.backend.upper())

        position_names = spec.scaffold.variable_positions
        sub_lists = [spec.libraries[pos].substituents for pos in position_names]
        combos = list(iter_product(*sub_lists))

        if args.verbose:
            print(f"[precompute] {len(combos)} combinations for {len(position_names)} positions", file=sys.stderr)

        computed = 0
        errors = 0
        for combo in combos:
            choices = dict(zip(position_names, combo))
            try:
                full_smiles = assemble_molecule(spec.scaffold, choices)
            except (ValueError, KeyError) as exc:
                if args.verbose:
                    print(f"[precompute] Assembly failed: {exc}", file=sys.stderr)
                errors += 1
                continue

            if full_smiles in cache:
                continue

            try:
                get_energy_features(full_smiles, cache, backend=args.backend, solvent=args.solvent)
                computed += 1
                if args.verbose and computed % 10 == 0:
                    print(f"[precompute] Computed {computed} energies...", file=sys.stderr)
            except Exception as exc:
                if args.verbose:
                    print(f"[precompute] Energy failed: {exc}", file=sys.stderr)
                errors += 1

        cache.save()
        json_print(
            {
                "run_id": args.run_id,
                "backend": args.backend,
                "total_combinations": len(combos),
                "computed": computed,
                "cached": len(cache),
                "errors": errors,
            }
        )
        return 0

    return None
