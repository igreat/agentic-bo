"""CLI subcommands for the BO engine: init, suggest, observe, status, report."""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .engine import BOEngine


def _json_print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _parse_json_object(value: str) -> dict[str, Any]:
    path = Path(value)
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object.")
    return dict(payload)


def _parse_observation_records(value: str) -> list[dict[str, Any]]:
    path = Path(value)
    if path.exists():
        if path.suffix.lower() == ".json":
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, list):
                raise ValueError("JSON observation payload must be a list of objects.")
            return [dict(x) for x in payload]

        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
            if "y" not in frame.columns:
                raise ValueError("CSV observations must include a 'y' column.")
            rows: list[dict[str, Any]] = []
            for _, row in frame.iterrows():
                x = row.drop(labels=["y"]).to_dict()
                rows.append({"x": x, "y": float(row["y"])})
            return rows

    payload = json.loads(value)
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise ValueError("Inline observation payload must be JSON object or list.")
    return [dict(x) for x in payload]


def register_commands(sub: argparse._SubParsersAction) -> None:
    """Register engine subcommands on an existing subparsers group."""
    init_cmd = sub.add_parser("init", help="Initialize run from a dataset")
    init_cmd.add_argument("--dataset", type=Path, required=True)
    init_cmd.add_argument("--target", type=str, required=True)
    init_cmd.add_argument(
        "--objective", type=str, choices=["min", "max"], required=True
    )
    init_cmd.add_argument("--run-id", type=str, default=None)
    init_cmd.add_argument("--seed", type=int, default=7)
    init_cmd.add_argument(
        "--engine",
        type=str,
        choices=["hebo", "bo_lcb", "random"],
        default="hebo",
        help="Default optimizer engine for this run",
    )
    init_cmd.add_argument("--init-random", type=int, default=10)
    init_cmd.add_argument("--batch-size", type=int, default=1)
    init_cmd.add_argument("--max-categories", type=int, default=64)
    init_cmd.add_argument(
        "--intent-json",
        type=str,
        default=None,
        help="Optional JSON object or path to JSON object for original user intent",
    )
    init_cmd.add_argument("--verbose", action="store_true")

    suggest_cmd = sub.add_parser("suggest", help="Suggest next experimental candidates")
    suggest_cmd.add_argument("--run-id", type=str, required=True)
    suggest_cmd.add_argument("--batch-size", type=int, default=None)
    suggest_cmd.add_argument("--verbose", action="store_true")

    observe_cmd = sub.add_parser("observe", help="Record observation(s)")
    observe_cmd.add_argument("--run-id", type=str, required=True)
    observe_cmd.add_argument(
        "--data",
        type=str,
        required=True,
        help="Observations as JSON string/object/list, or path to CSV/JSON file",
    )
    observe_cmd.add_argument("--verbose", action="store_true")

    status_cmd = sub.add_parser("status", help="Show run status")
    status_cmd.add_argument("--run-id", type=str, required=True)

    report_cmd = sub.add_parser("report", help="Generate report and plot")
    report_cmd.add_argument("--run-id", type=str, required=True)
    report_cmd.add_argument("--verbose", action="store_true")

    # --- Molecular subcommands ---

    init_mol_cmd = sub.add_parser(
        "init-molecular",
        help="Initialize a molecular optimization run from a scaffold spec",
    )
    init_mol_cmd.add_argument(
        "--scaffold-spec", type=Path, required=True,
        help="Path to scaffold specification JSON file",
    )
    init_mol_cmd.add_argument("--target", type=str, required=True)
    init_mol_cmd.add_argument(
        "--objective", type=str, choices=["min", "max"], required=True
    )
    init_mol_cmd.add_argument(
        "--dataset", type=Path, default=None,
        help="Optional CSV dataset with existing observations",
    )
    init_mol_cmd.add_argument("--run-id", type=str, default=None)
    init_mol_cmd.add_argument("--seed", type=int, default=7)
    init_mol_cmd.add_argument(
        "--engine", type=str,
        choices=["hebo", "bo_lcb", "random"], default="hebo",
    )
    init_mol_cmd.add_argument("--init-random", type=int, default=10)
    init_mol_cmd.add_argument("--batch-size", type=int, default=1)
    init_mol_cmd.add_argument(
        "--intent-json", type=str, default=None,
        help="Optional JSON object or path for user intent",
    )
    init_mol_cmd.add_argument(
        "--energy-backend", type=str, default=None,
        choices=["none", "xtb", "dft", "ml", "auto"],
        help="Energy computation backend (overrides scaffold spec setting)",
    )
    init_mol_cmd.add_argument("--verbose", action="store_true")

    feas_cmd = sub.add_parser(
        "check-feasibility",
        help="Check synthesis feasibility of a molecule",
    )
    feas_cmd.add_argument(
        "--smiles", type=str, default=None,
        help="SMILES string of the molecule to assess",
    )
    feas_cmd.add_argument(
        "--run-id", type=str, default=None,
        help="Run ID for molecular run (uses scaffold spec for context)",
    )
    feas_cmd.add_argument(
        "--substituents", type=str, default=None,
        help='JSON dict of substituent choices, e.g. \'{"R1": "F", "R2": "methyl"}\'',
    )
    feas_cmd.add_argument(
        "--sa-threshold", type=float, default=6.0,
        help="SA score threshold (default: 6.0)",
    )
    feas_cmd.add_argument(
        "--strain-threshold", type=float, default=33.0,
        help="Strain energy threshold in kcal/mol (default: 33.0). "
             "Recommended: drug-like 25-30, materials 40-50, natural-products 60-80",
    )

    desc_cmd = sub.add_parser(
        "compute-descriptors",
        help="Compute molecular descriptors for a SMILES",
    )
    desc_cmd.add_argument("--smiles", type=str, required=True)
    desc_cmd.add_argument(
        "--config", type=str, default=None,
        help="Optional JSON descriptor config override",
    )
    desc_cmd.add_argument(
        "--energy-backend", type=str, default="none",
        choices=["none", "xtb", "dft", "ml", "auto"],
        help="Energy backend for HOMO/LUMO/strain features (default: none)",
    )

    draft_cmd = sub.add_parser(
        "draft-scaffold-spec",
        help="Generate a draft scaffold spec from a SMILES with R-groups",
    )
    draft_cmd.add_argument(
        "--smiles", type=str, required=True,
        help="SMILES with [*:N] dummy atoms marking variable positions",
    )
    draft_cmd.add_argument(
        "--positions", type=str, default=None,
        help="Comma-separated position names (e.g. R1,R2,R3)",
    )
    draft_cmd.add_argument(
        "--output", type=Path, default=None,
        help="Output path for the draft spec JSON",
    )

    # --- Energy subcommands ---

    energy_cmd = sub.add_parser(
        "compute-energy",
        help="Compute energy features for a single molecule",
    )
    energy_cmd.add_argument("--smiles", type=str, required=True)
    energy_cmd.add_argument(
        "--backend", type=str, default="xtb",
        choices=["xtb", "dft", "ml", "auto"],
        help="Energy computation backend (default: xtb)",
    )
    energy_cmd.add_argument(
        "--solvent", type=str, default=None,
        help="Implicit solvent for GBSA (e.g. water, thf)",
    )

    precompute_cmd = sub.add_parser(
        "precompute-energies",
        help="Precompute energy features for all scaffold+substituent combinations",
    )
    precompute_cmd.add_argument("--run-id", type=str, required=True)
    precompute_cmd.add_argument(
        "--backend", type=str, default="xtb",
        choices=["xtb", "dft", "ml", "auto"],
        help="Energy computation backend (default: xtb)",
    )
    precompute_cmd.add_argument(
        "--solvent", type=str, default=None,
        help="Implicit solvent for GBSA (e.g. water, thf)",
    )
    precompute_cmd.add_argument("--verbose", action="store_true")

    # --- scaffold editing (CReM + HEBO) ---

    se_cmd = sub.add_parser(
        "scaffold-edit",
        help="Initialize CReM scaffold editing + HEBO optimisation run",
    )
    se_cmd.add_argument("--dataset", type=Path, required=True,
                        help="CSV with SMILES + target column")
    se_cmd.add_argument("--target", type=str, required=True,
                        help="Target column name")
    se_cmd.add_argument("--objective", type=str, choices=["min", "max"], required=True)
    se_cmd.add_argument("--smiles-column", type=str, default=None,
                        help="SMILES column name (auto-detected if omitted)")
    se_cmd.add_argument("--seed-smiles", type=str, default=None,
                        help="Comma-separated seed SMILES, or path to file")
    se_cmd.add_argument("--top-k-seeds", type=int, default=10,
                        help="Auto-select top-K diverse seeds (default: 10)")
    se_cmd.add_argument("--crem-db", type=Path, default=None,
                        help="Path to CReM fragment database (.db)")
    se_cmd.add_argument("--fixed-smarts", type=str, default=None,
                        help="SMARTS patterns for fixed regions (comma-separated)")
    se_cmd.add_argument("--mutable-smarts", type=str, default=None,
                        help="SMARTS patterns for mutable regions (comma-separated)")
    se_cmd.add_argument("--max-size", type=int, default=3,
                        help="Max heavy atoms in CReM replacement fragment (default: 3)")
    se_cmd.add_argument("--radius", type=int, default=3,
                        help="CReM context radius (default: 3)")
    se_cmd.add_argument("--max-replacements", type=int, default=100,
                        help="Max mutations per seed molecule (default: 100)")
    se_cmd.add_argument("--operations", type=str, default=None,
                        help="CReM operations: mutate,grow (comma-separated, default: mutate)")
    se_cmd.add_argument("--sa-threshold", type=float, default=6.0,
                        help="SA score threshold (default: 6.0)")
    se_cmd.add_argument("--run-id", type=str, default=None)
    se_cmd.add_argument("--seed", type=int, default=42)
    se_cmd.add_argument("--engine", type=str, default="hebo",
                        choices=["hebo", "bo_lcb", "random"])
    se_cmd.add_argument("--fingerprint-bits", type=int, default=128)
    se_cmd.add_argument("--verbose", action="store_true")

    ae_cmd = sub.add_parser(
        "augment-energy",
        help="Augment scaffold-edit candidates with xTB/DFT energy features",
    )
    ae_cmd.add_argument("--run-id", type=str, required=True)
    ae_cmd.add_argument("--backend", type=str, default="xtb",
                        choices=["xtb", "dft", "ml", "auto"])
    ae_cmd.add_argument("--verbose", action="store_true")

    top_cmd = sub.add_parser(
        "top-candidates",
        help="Show top-K candidates from a completed optimisation run",
    )
    top_cmd.add_argument("--run-id", type=str, required=True)
    top_cmd.add_argument("--top-k", type=int, default=10)
    top_cmd.add_argument("--verbose", action="store_true")

    val_cmd = sub.add_parser(
        "validate",
        help="Validate top-K candidates with feasibility or xTB assessment",
    )
    val_cmd.add_argument("--run-id", type=str, required=True)
    val_cmd.add_argument("--method", type=str, required=True,
                         choices=["feasibility_only", "xtb_plus_feasibility"])
    val_cmd.add_argument("--top-k", type=int, default=10)
    val_cmd.add_argument("--verbose", action="store_true")

    # --- screen: one-command virtual screening ---

    screen_cmd = sub.add_parser(
        "screen",
        help="One-command molecular virtual screening: CSV in, report out",
    )
    screen_cmd.add_argument("--dataset", type=Path, required=True,
                            help="CSV file with SMILES + target column")
    screen_cmd.add_argument("--target", type=str, required=True,
                            help="Target column name (e.g. yield)")
    screen_cmd.add_argument("--objective", type=str, choices=["min", "max"],
                            required=True)
    screen_cmd.add_argument("--smiles-column", type=str, default=None,
                            help="SMILES column name (auto-detected if omitted)")
    screen_cmd.add_argument("--iterations", type=int, default=20,
                            help="Number of BO iterations (default: 20)")
    screen_cmd.add_argument("--seed", type=int, default=42)
    screen_cmd.add_argument("--engine", type=str, default="hebo",
                            choices=["hebo", "bo_lcb", "random"])
    screen_cmd.add_argument("--batch-size", type=int, default=1)
    screen_cmd.add_argument("--fingerprint-bits", type=int, default=128)
    screen_cmd.add_argument("--energy-backend", type=str, default="none",
                            choices=["none", "xtb", "auto"],
                            help="Energy backend for HOMO/LUMO features (default: none)")
    screen_cmd.add_argument("--discovery", action="store_true",
                            help="Discovery mode: optimize in continuous descriptor space "
                                 "and map back to nearest molecule. Enables structure-activity "
                                 "learning beyond existing dataset molecules.")
    screen_cmd.add_argument("--verbose", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    engine = BOEngine(runs_root=args.runs_root)

    if args.command == "init":
        intent_payload = None
        if args.intent_json is not None:
            intent_payload = _parse_json_object(args.intent_json)
        payload = engine.init_run(
            dataset_path=args.dataset,
            target_column=args.target,
            objective=args.objective,
            default_engine=args.engine,
            run_id=args.run_id,
            seed=args.seed,
            num_initial_random_samples=args.init_random,
            default_batch_size=args.batch_size,
            max_categories=args.max_categories,
            intent=intent_payload,
            verbose=args.verbose,
        )
        _json_print(payload)
        return 0

    if args.command == "suggest":
        payload = engine.suggest(
            args.run_id,
            batch_size=args.batch_size,
            verbose=args.verbose,
        )
        _json_print(payload)
        return 0

    if args.command == "observe":
        observations = _parse_observation_records(args.data)
        payload = engine.observe(args.run_id, observations, verbose=args.verbose)
        _json_print(payload)
        return 0

    if args.command == "status":
        payload = engine.status(args.run_id)
        _json_print(payload)
        return 0

    if args.command == "report":
        payload = engine.report(args.run_id, verbose=args.verbose)
        _json_print(payload)
        return 0

    if args.command == "init-molecular":
        intent_payload = None
        if args.intent_json is not None:
            intent_payload = _parse_json_object(args.intent_json)

        # If --energy-backend is provided, inject it into the scaffold spec
        scaffold_spec_path = args.scaffold_spec
        if args.energy_backend is not None:
            # Load spec, set energy_backend, write to temp file
            import tempfile
            with scaffold_spec_path.open("r", encoding="utf-8") as fh:
                spec_raw = json.load(fh)
            desc_cfg = spec_raw.setdefault("descriptor_config", {})
            desc_cfg["energy_backend"] = args.energy_backend
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, encoding="utf-8"
            )
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
        _json_print(payload)
        return 0

    if args.command == "check-feasibility":
        from .molecular.feasibility import assess_feasibility
        from .molecular.scaffold import decode_suggestion, load_scaffold_spec
        from .molecular.features import assemble_molecule

        smiles = args.smiles
        if smiles is None and args.run_id and args.substituents:
            state = engine._load_state(args.run_id)
            if state.get("mode") != "molecular":
                print("Error: Run is not in molecular mode.", file=sys.stderr)
                return 1
            spec = load_scaffold_spec(state["molecular"]["scaffold_spec_path"])
            subs = json.loads(args.substituents)
            smiles, choices = decode_suggestion(subs, spec)
        elif smiles is None:
            print(
                "Error: Provide --smiles or both --run-id and --substituents.",
                file=sys.stderr,
            )
            return 1

        # Always compute strain energy from SSSR heuristic for standalone checks
        from .molecular.energy import _estimate_strain_energy

        strain_energy = _estimate_strain_energy(smiles)

        result = assess_feasibility(
            smiles,
            sa_threshold=args.sa_threshold,
            strain_energy_kcal=strain_energy,
            strain_threshold_kcal=args.strain_threshold,
        )
        _json_print({
            "smiles": smiles,
            "is_feasible": result.is_feasible,
            "sa_score": result.sa_score,
            "num_steps_estimate": result.num_steps_estimate,
            "penalty": result.penalty,
            "strain_energy_kcal": result.strain_energy_kcal,
            "reasons": result.reasons,
        })
        return 0

    if args.command == "compute-descriptors":
        from .molecular.features import compute_descriptors
        from .molecular.types import DescriptorConfig

        config = None
        if args.config:
            cfg_raw = json.loads(args.config)
            fp_raw = cfg_raw.get("fingerprint", {})
            dft_raw = cfg_raw.get("dft", {})
            config = DescriptorConfig(
                basic=cfg_raw.get("basic", True),
                fingerprint_enabled=fp_raw.get("enabled", True) if isinstance(fp_raw, dict) else bool(fp_raw),
                fingerprint_n_bits=fp_raw.get("n_bits", 128) if isinstance(fp_raw, dict) else 128,
                fingerprint_radius=fp_raw.get("radius", 2) if isinstance(fp_raw, dict) else 2,
                electronic=cfg_raw.get("electronic", True),
                steric=cfg_raw.get("steric", True),
            )
        descriptors = compute_descriptors(args.smiles, config=config)

        # Optionally include energy features
        energy_backend = getattr(args, "energy_backend", "none")
        if energy_backend != "none":
            from .molecular.energy import get_energy_features

            energy_feats = get_energy_features(
                args.smiles, cache=None, backend=energy_backend
            )
            # Prefix with "dft_" to match the oracle feature naming convention
            for key, val in energy_feats.items():
                descriptors[f"dft_{key}"] = val

        _json_print({"smiles": args.smiles, "descriptors": descriptors})
        return 0

    if args.command == "draft-scaffold-spec":
        from .molecular.scaffold import smiles_to_draft_spec, save_draft_spec

        positions = None
        if args.positions:
            positions = [p.strip() for p in args.positions.split(",")]

        spec = smiles_to_draft_spec(args.smiles, marked_positions=positions)

        if args.output:
            out_path = save_draft_spec(spec, args.output)
            _json_print({"status": "saved", "path": str(out_path)})
        else:
            _json_print(spec)
        return 0

    if args.command == "compute-energy":
        from .molecular.energy import get_energy_features

        features = get_energy_features(
            args.smiles, cache=None, backend=args.backend, solvent=args.solvent
        )
        _json_print({"smiles": args.smiles, "backend": args.backend, "features": features})
        return 0

    if args.command == "precompute-energies":
        from .molecular.energy import EnergyCache, get_energy_features
        from .molecular.scaffold import load_scaffold_spec
        from itertools import product as iter_product

        state = engine._load_state(args.run_id)
        if state.get("mode") != "molecular":
            print("Error: Run is not in molecular mode.", file=sys.stderr)
            return 1

        spec = load_scaffold_spec(state["molecular"]["scaffold_spec_path"])
        paths = engine._paths(args.run_id)
        cache = EnergyCache(path=paths.energy_cache, method=args.backend.upper())

        # Enumerate all scaffold+substituent combinations
        position_names = spec.scaffold.variable_positions
        sub_lists = [spec.libraries[pos].substituents for pos in position_names]
        combos = list(iter_product(*sub_lists))

        if args.verbose:
            print(
                f"[precompute] {len(combos)} combinations for "
                f"{len(position_names)} positions",
                file=sys.stderr,
            )

        from .molecular.features import assemble_molecule
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
                get_energy_features(
                    full_smiles, cache, backend=args.backend, solvent=args.solvent
                )
                computed += 1
                if args.verbose and computed % 10 == 0:
                    print(
                        f"[precompute] Computed {computed} energies...",
                        file=sys.stderr,
                    )
            except Exception as exc:
                if args.verbose:
                    print(f"[precompute] Energy failed: {exc}", file=sys.stderr)
                errors += 1

        cache.save()
        _json_print({
            "run_id": args.run_id,
            "backend": args.backend,
            "total_combinations": len(combos),
            "computed": computed,
            "cached": len(cache),
            "errors": errors,
        })
        return 0

    if args.command == "scaffold-edit":
        # Parse seed SMILES
        seed_smiles = None
        if args.seed_smiles:
            p = Path(args.seed_smiles)
            if p.exists():
                seed_smiles = [
                    line.strip() for line in p.read_text().splitlines()
                    if line.strip()
                ]
            else:
                seed_smiles = [s.strip() for s in args.seed_smiles.split(",") if s.strip()]

        # Parse mutation constraint
        constraint_dict = None
        if args.fixed_smarts or args.mutable_smarts:
            constraint_dict = {
                "fixed_smarts": [
                    s.strip() for s in (args.fixed_smarts or "").split(",") if s.strip()
                ],
                "mutable_smarts": [
                    s.strip() for s in (args.mutable_smarts or "").split(",") if s.strip()
                ],
                "fixed_atom_indices": [],
            }

        # Parse operations
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
        _json_print(payload)
        return 0

    if args.command == "augment-energy":
        payload = engine.augment_with_energy(
            args.run_id,
            energy_backend=args.backend,
            verbose=args.verbose,
        )
        _json_print(payload)
        return 0

    if args.command == "top-candidates":
        payload = engine.get_top_candidates(
            args.run_id,
            top_k=args.top_k,
            verbose=args.verbose,
        )
        _json_print(payload)
        return 0

    if args.command == "validate":
        payload = engine.validate_candidates(
            args.run_id,
            method=args.method,
            top_k=args.top_k,
            verbose=args.verbose,
        )
        _json_print(payload)
        return 0

    if args.command == "screen":
        # One-command virtual screening: init → build-oracle → run-proxy → report
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
            print(
                f"[screen] Running {args.iterations} BO iterations...",
                file=sys.stderr,
            )
        engine.run_proxy_optimization(
            run_id,
            num_iterations=args.iterations,
            batch_size=args.batch_size,
            verbose=args.verbose,
        )

        report = engine.report(run_id, verbose=args.verbose)
        _json_print(report)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
