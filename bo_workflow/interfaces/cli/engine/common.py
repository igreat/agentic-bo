import json
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem


def json_print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def parse_json_object(value: str) -> dict[str, Any]:
    path = Path(value)
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object.")
    return dict(payload)


def parse_observation_records(value: str) -> list[dict[str, Any]]:
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


def canonical_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def load_smiles_inputs(smiles: str | None, smiles_file: Path | None) -> list[str]:
    if smiles and smiles_file:
        raise ValueError("Use either --smiles or --smiles-file, not both.")

    if smiles:
        return [canonical_smiles(smiles)]

    if smiles_file is None:
        raise ValueError("Provide one of --smiles or --smiles-file.")
    if not smiles_file.exists():
        raise FileNotFoundError(f"SMILES file not found: {smiles_file}")

    ext = smiles_file.suffix.lower()
    raw_smiles: list[str] = []
    if ext == ".csv":
        frame = pd.read_csv(smiles_file)
        if frame.empty:
            raise ValueError("SMILES CSV is empty.")

        candidate_cols = [
            c
            for c in frame.columns
            if c.lower().strip() in {"smiles", "smi", "molecule", "structure"}
        ]
        col = candidate_cols[0] if candidate_cols else str(frame.columns[0])
        raw_smiles = [
            str(v).strip() for v in frame[col].dropna().tolist() if str(v).strip()
        ]
    else:
        for line in smiles_file.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            raw_smiles.append(text.split()[0])

    if not raw_smiles:
        raise ValueError("No SMILES found in input file.")

    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw_smiles:
        can = canonical_smiles(item)
        if can not in seen:
            seen.add(can)
            normalized.append(can)
    return normalized


def build_egfr_global_argv(args: Any) -> list[str]:
    argv: list[str] = [
        "--dataset",
        str(args.dataset),
        "--target-column",
        str(args.target_column),
        "--max-pic50",
        str(args.max_pic50),
        "--split-mode",
        str(args.split_mode),
        "--seed-smiles-column",
        str(args.seed_smiles_column),
        "--seed-count",
        str(args.seed_count),
        "--seed-top-fraction",
        str(args.seed_top_fraction),
        "--seed-quality-mode",
        str(args.seed_quality_mode),
        "--seed-decent-low-q",
        str(args.seed_decent_low_q),
        "--seed-decent-high-q",
        str(args.seed_decent_high_q),
        "--iterations",
        str(args.iterations),
        "--rounds",
        str(args.rounds),
        "--batch-size",
        str(args.batch_size),
        "--experiments-per-round",
        str(args.experiments_per_round),
        "--fp-bits",
        str(args.fp_bits),
        "--energy-backend",
        str(args.energy_backend),
        "--seed",
        str(args.seed),
        "--ucb-beta",
        str(args.ucb_beta),
        "--hebo-exploit-slots",
        str(args.hebo_exploit_slots),
        "--hebo-explore-slots",
        str(args.hebo_explore_slots),
        "--hebo-explore-beta",
        str(args.hebo_explore_beta),
        "--novelty-grace-rounds",
        str(args.novelty_grace_rounds),
        "--stall-rounds-threshold",
        str(args.stall_rounds_threshold),
    ]

    if args.seed_smiles_csv is not None:
        argv.extend(["--seed-smiles-csv", str(args.seed_smiles_csv)])
    if args.fix_tiny_ic50_as_molar:
        argv.append("--fix-tiny-ic50-as-molar")
    if args.no_seed_filter:
        argv.append("--no-seed-filter")
    if args.all_hebo_exploit:
        argv.append("--all-hebo-exploit")
    if args.adaptive_novelty:
        argv.append("--adaptive-novelty")
    else:
        argv.append("--no-adaptive-novelty")
    if args.verbose:
        argv.append("--verbose")

    return argv
