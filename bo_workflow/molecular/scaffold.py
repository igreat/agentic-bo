"""Scaffold decomposition and HEBO design-space generation.

This module handles:
1. Parsing a user-provided scaffold specification (JSON) into a
   ``MolecularDesignSpec``.
2. Converting the spec into HEBO design parameters (categorical for
   substituent positions, numeric/cat for reaction conditions).
3. Decoding HEBO suggestion rows back into full molecules.
4. Generating a draft spec from a raw SMILES with marked R-groups.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rdkit import Chem

from .types import (
    DescriptorConfig,
    FeasibilityConfig,
    MolecularDesignSpec,
    Scaffold,
    Substituent,
    SubstituentLibrary,
)
from .features import assemble_molecule

# ---------------------------------------------------------------------------
# Default substituent library for interactive scaffold generation
# ---------------------------------------------------------------------------

_DEFAULT_SUBSTITUENTS: list[dict[str, str]] = [
    {"name": "H", "smiles": "[H]"},
    {"name": "F", "smiles": "[F]"},
    {"name": "Cl", "smiles": "[Cl]"},
    {"name": "Br", "smiles": "[Br]"},
    {"name": "methyl", "smiles": "C"},
    {"name": "ethyl", "smiles": "CC"},
    {"name": "methoxy", "smiles": "OC"},
    {"name": "OH", "smiles": "O"},
    {"name": "NH2", "smiles": "N"},
    {"name": "NO2", "smiles": "[N+](=O)[O-]"},
    {"name": "CN", "smiles": "C#N"},
    {"name": "CF3", "smiles": "C(F)(F)F"},
]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_scaffold_spec(spec_dict: dict[str, Any]) -> MolecularDesignSpec:
    """Parse a scaffold specification dict into a ``MolecularDesignSpec``.

    The dict is typically loaded from a JSON file.  Expected schema::

        {
            "scaffold_smiles": "c1cc([*:1])c([*:2])cc1...",
            "variable_positions": {
                "R1": {
                    "attachment_atom_idx": 3,       // optional
                    "substituents": [
                        {"name": "H", "smiles": "[H]", "properties": {...}},
                        ...
                    ]
                },
                ...
            },
            "condition_parameters": [               // optional
                {"name": "temperature_C", "type": "num", "lb": 20, "ub": 120},
                {"name": "solvent", "type": "cat", "categories": ["THF", "DMF"]}
            ],
            "descriptor_config": {                  // optional
                "basic": true,
                "fingerprint": {"enabled": true, "n_bits": 128, "radius": 2},
                "electronic": true,
                "steric": true,
                "dft": {"enabled": false, "data_path": null}
            },
            "feasibility": {                        // optional
                "mode": "soft",
                "sa_threshold": 6.0,
                "incompatible_pairs": [
                    {"positions": ["R1","R2"], "pair": ["nitro","amino"],
                     "reason": "Incompatible without protecting group"}
                ]
            }
        }

    Args:
        spec_dict: The raw specification dictionary.

    Returns:
        A fully populated ``MolecularDesignSpec``.

    Raises:
        ValueError: If the spec is missing required fields or has invalid data.
    """
    scaffold_smiles = spec_dict.get("scaffold_smiles")
    if not scaffold_smiles:
        raise ValueError("scaffold_smiles is required in the specification.")

    # Validate scaffold SMILES
    mol = Chem.MolFromSmiles(scaffold_smiles)
    if mol is None:
        raise ValueError(f"Cannot parse scaffold SMILES: {scaffold_smiles}")

    # Parse variable positions
    vp_raw = spec_dict.get("variable_positions")
    if not vp_raw or not isinstance(vp_raw, dict):
        raise ValueError(
            "variable_positions must be a non-empty dict in the specification."
        )

    position_names: list[str] = []
    attachment_atoms: dict[str, int] = {}
    libraries: dict[str, SubstituentLibrary] = {}

    for pos_name, pos_data in vp_raw.items():
        position_names.append(pos_name)

        if "attachment_atom_idx" in pos_data:
            attachment_atoms[pos_name] = int(pos_data["attachment_atom_idx"])

        subs_raw = pos_data.get("substituents", [])
        if not subs_raw:
            raise ValueError(
                f"Position '{pos_name}' must have at least one substituent."
            )

        substituents: list[Substituent] = []
        for s in subs_raw:
            if isinstance(s, str):
                # Simple name — assume it's in default library
                substituents.append(Substituent(name=s, smiles=s))
            elif isinstance(s, dict):
                name = s.get("name", "")
                smiles = s.get("smiles", "")
                props = s.get("properties", {})
                if not name or not smiles:
                    raise ValueError(
                        f"Substituent at position '{pos_name}' missing name/smiles: {s}"
                    )
                substituents.append(
                    Substituent(name=name, smiles=smiles, properties=props)
                )
            else:
                raise ValueError(
                    f"Invalid substituent format at position '{pos_name}': {s}"
                )

        libraries[pos_name] = SubstituentLibrary(
            position=pos_name, substituents=substituents
        )

    scaffold = Scaffold(
        smiles=scaffold_smiles,
        variable_positions=position_names,
        attachment_atoms=attachment_atoms,
    )

    # Parse condition parameters (pass through to HEBO)
    condition_params = list(spec_dict.get("condition_parameters", []))

    # Parse descriptor config
    desc_raw = spec_dict.get("descriptor_config", {})
    fp_raw = desc_raw.get("fingerprint", {})
    dft_raw = desc_raw.get("dft", {})
    energy_backend = desc_raw.get("energy_backend", "none")
    # When an energy backend is selected, automatically enable DFT features
    dft_auto_enabled = energy_backend != "none"
    descriptor_config = DescriptorConfig(
        basic=desc_raw.get("basic", True),
        fingerprint_enabled=fp_raw.get("enabled", True) if isinstance(fp_raw, dict) else bool(fp_raw),
        fingerprint_n_bits=fp_raw.get("n_bits", 128) if isinstance(fp_raw, dict) else 128,
        fingerprint_radius=fp_raw.get("radius", 2) if isinstance(fp_raw, dict) else 2,
        electronic=desc_raw.get("electronic", True),
        steric=desc_raw.get("steric", True),
        dft_enabled=(dft_raw.get("enabled", False) if isinstance(dft_raw, dict) else bool(dft_raw)) or dft_auto_enabled,
        dft_data_path=dft_raw.get("data_path") if isinstance(dft_raw, dict) else None,
        energy_backend=energy_backend,
    )

    # Parse feasibility config
    feas_raw = spec_dict.get("feasibility", {})
    feasibility_config = FeasibilityConfig(
        mode=feas_raw.get("mode", "soft"),
        sa_threshold=float(feas_raw.get("sa_threshold", 6.0)),
        strain_threshold_kcal=float(feas_raw.get("strain_threshold_kcal", 33.0)),
        incompatible_pairs=list(feas_raw.get("incompatible_pairs", [])),
    )

    return MolecularDesignSpec(
        scaffold=scaffold,
        libraries=libraries,
        condition_params=condition_params,
        descriptor_config=descriptor_config,
        feasibility=feasibility_config,
    )


def load_scaffold_spec(path: str | Path) -> MolecularDesignSpec:
    """Load a scaffold specification from a JSON file.

    Args:
        path: Path to the JSON specification file.

    Returns:
        Parsed ``MolecularDesignSpec``.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Scaffold spec not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return parse_scaffold_spec(raw)


# ---------------------------------------------------------------------------
# Design-space conversion
# ---------------------------------------------------------------------------


def spec_to_design_parameters(
    spec: MolecularDesignSpec,
) -> list[dict[str, Any]]:
    """Convert a ``MolecularDesignSpec`` into HEBO design parameters.

    Each variable position becomes a categorical parameter with the
    substituent names as categories.  Condition parameters are passed
    through directly.

    Args:
        spec: The molecular design specification.

    Returns:
        List of HEBO design parameter dicts, ready for ``DesignSpace().parse()``.
    """
    params: list[dict[str, Any]] = []

    # Substituent positions as categorical
    for pos_name in spec.scaffold.variable_positions:
        lib = spec.libraries[pos_name]
        params.append({
            "name": pos_name,
            "type": "cat",
            "categories": lib.names(),
        })

    # Reaction condition parameters
    for cp in spec.condition_params:
        params.append(dict(cp))  # shallow copy

    return params


# ---------------------------------------------------------------------------
# Suggestion decoding
# ---------------------------------------------------------------------------


def decode_suggestion(
    suggestion: dict[str, Any],
    spec: MolecularDesignSpec,
) -> tuple[str, dict[str, Substituent]]:
    """Decode a HEBO suggestion row into a full molecule.

    Args:
        suggestion: A HEBO suggestion dict (e.g. ``{"R1": "F", "R2": "methyl",
            "temperature_C": 80, "solvent": "DMF"}``).
        spec: The molecular design specification.

    Returns:
        A tuple of (canonical SMILES, dict of position → Substituent).

    Raises:
        ValueError: If a position is missing or a substituent name is unknown.
    """
    choices: dict[str, Substituent] = {}
    for pos_name in spec.scaffold.variable_positions:
        sub_name = suggestion.get(pos_name)
        if sub_name is None:
            raise ValueError(
                f"Suggestion missing position '{pos_name}'. "
                f"Keys: {list(suggestion.keys())}"
            )
        lib = spec.libraries[pos_name]
        choices[pos_name] = lib.by_name(str(sub_name))

    full_smiles = assemble_molecule(spec.scaffold, choices)
    return full_smiles, choices


# ---------------------------------------------------------------------------
# Interactive draft spec generation
# ---------------------------------------------------------------------------


def smiles_to_draft_spec(
    smiles: str,
    marked_positions: list[str] | None = None,
) -> dict[str, Any]:
    """Generate a draft scaffold specification from a SMILES with R-groups.

    Given a SMILES string containing ``[*:N]`` dummy atoms (e.g. from a
    user who marked variable positions), produce a JSON-serializable dict
    that can be saved and edited before passing to ``parse_scaffold_spec``.

    If *marked_positions* is provided, it gives the human-readable names
    for each labeled dummy atom (e.g. ``["R1", "R2"]`` for ``[*:1]`` and
    ``[*:2]``).  Otherwise, names default to ``R1``, ``R2``, etc.

    Args:
        smiles: SMILES with ``[*:N]`` dummy atoms.
        marked_positions: Optional list of position names.

    Returns:
        A draft specification dict ready for user review / editing.

    Raises:
        ValueError: If the SMILES has no dummy atoms or cannot be parsed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES: {smiles}")

    # Find dummy atoms and their isotope labels
    dummies: list[tuple[int, int]] = []  # (atom_idx, isotope_label)
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            label = atom.GetIsotope()
            if label == 0:
                label = atom.GetAtomMapNum()
            if label == 0:
                # Auto-assign label based on order
                label = len(dummies) + 1
            dummies.append((atom.GetIdx(), label))

    if not dummies:
        raise ValueError(
            "No dummy atoms ([*:N]) found in SMILES. "
            "Mark variable positions with [*:1], [*:2], etc."
        )

    # Sort by label
    dummies.sort(key=lambda x: x[1])

    # Assign position names
    if marked_positions:
        if len(marked_positions) != len(dummies):
            raise ValueError(
                f"Got {len(marked_positions)} position names but found "
                f"{len(dummies)} dummy atoms in SMILES."
            )
        pos_names = list(marked_positions)
    else:
        pos_names = [f"R{d[1]}" for d in dummies]

    # Build variable_positions with default substituent library
    variable_positions: dict[str, Any] = {}
    for (atom_idx, _label), pos_name in zip(dummies, pos_names):
        # Find neighbor atom for attachment reference
        neighbors = mol.GetAtomWithIdx(atom_idx).GetNeighbors()
        attachment_idx = neighbors[0].GetIdx() if neighbors else -1

        variable_positions[pos_name] = {
            "attachment_atom_idx": attachment_idx,
            "substituents": list(_DEFAULT_SUBSTITUENTS),
        }

    spec: dict[str, Any] = {
        "scaffold_smiles": smiles,
        "variable_positions": variable_positions,
        "condition_parameters": [],
        "descriptor_config": {
            "basic": True,
            "fingerprint": {"enabled": True, "n_bits": 128, "radius": 2},
            "electronic": True,
            "steric": True,
            "dft": {"enabled": False, "data_path": None},
            "energy_backend": "none",
        },
        "feasibility": {
            "mode": "soft",
            "sa_threshold": 6.0,
            "incompatible_pairs": [],
        },
    }

    return spec


def save_draft_spec(spec: dict[str, Any], path: str | Path) -> Path:
    """Save a draft scaffold specification to a JSON file.

    Args:
        spec: The draft specification dict.
        path: Target file path.

    Returns:
        The resolved Path where the file was written.
    """
    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(spec, fh, indent=2, ensure_ascii=False)
    return p
