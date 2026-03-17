"""SMILES → DFT descriptor converter package."""

from .smiles_dft import (
    BH_PRESET,
    ComponentSpec,
    compute_molecule_descriptors,
    detect_smiles_columns,
    encode_dataset_dft,
    encode_reactions_dft,
)

__all__ = [
    "BH_PRESET",
    "ComponentSpec",
    "compute_molecule_descriptors",
    "detect_smiles_columns",
    "encode_dataset_dft",
    "encode_reactions_dft",
]
