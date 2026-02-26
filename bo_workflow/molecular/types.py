"""Shared data types for the molecular optimization workflow.

All modules in the molecular package exchange data through these types.
They are plain dataclasses — no RDKit imports here, so they can be used
in non-molecular contexts without triggering heavy dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Substituent:
    """A chemical substituent that can be placed at a variable position.

    Attributes:
        name: Human-readable label (e.g. "methyl", "phenyl", "F").
        smiles: SMILES fragment for the substituent (e.g. "[CH3]", "c1ccccc1").
        properties: Pre-computed physicochemical properties. Keys may include
            ``hammett_sigma``, ``taft_Es``, ``a_value``, or any user-supplied
            numeric property.  These are merged into the oracle feature vector.
    """

    name: str
    smiles: str
    properties: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SubstituentLibrary:
    """The set of allowed substituents at one variable position.

    Attributes:
        position: Position name matching the scaffold (e.g. "R1").
        substituents: Allowed substituents at this position.
    """

    position: str
    substituents: list[Substituent]

    def names(self) -> list[str]:
        """Return substituent names in order."""
        return [s.name for s in self.substituents]

    def by_name(self, name: str) -> Substituent:
        """Look up a substituent by name.  Raises KeyError if not found."""
        for s in self.substituents:
            if s.name == name:
                return s
        raise KeyError(
            f"Substituent '{name}' not found at position '{self.position}'. "
            f"Available: {self.names()}"
        )


@dataclass(frozen=True)
class Scaffold:
    """Core molecular scaffold with marked variable positions.

    Attributes:
        smiles: SMILES string of the scaffold with dummy atoms marking
            variable positions (e.g. ``c1cc([*:1])c([*:2])cc1``).
            Dummy atoms use the ``[*:N]`` notation where N is the
            isotope-label matching a position name via *variable_positions*.
        variable_positions: Ordered list of position names (e.g. ["R1","R2"]).
        attachment_atoms: Map from position name to the atom index in the
            scaffold molecule where the substituent attaches.  Optional —
            if not provided, attachment is inferred from the dummy atom
            neighbors.
    """

    smiles: str
    variable_positions: list[str]
    attachment_atoms: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class DescriptorConfig:
    """Configuration for which molecular descriptors to compute.

    Attributes:
        basic: Compute basic RDKit descriptors (MW, LogP, TPSA, etc.).
        fingerprint_enabled: Compute Morgan fingerprint bits.
        fingerprint_n_bits: Number of Morgan FP bits.
        fingerprint_radius: Morgan FP radius.
        electronic: Compute electronic descriptors (Gasteiger charges,
            Hammett sigma lookups).
        steric: Compute steric descriptors (Taft Es, A-values).
        dft_enabled: Use pre-computed DFT features.
        dft_data_path: Path to JSON/CSV with DFT properties keyed by SMILES.
        energy_backend: Energy computation backend.
            ``"none"`` disables energy computation (default).
            ``"xtb"`` uses GFN2-xTB via tblite.
            ``"dft"`` uses PySCF B3LYP/6-31G* (offline only).
            ``"ml"`` uses TorchANI ANI-2x on GPU.
            ``"auto"`` tries cache → xTB → ML fallback.
    """

    basic: bool = True
    fingerprint_enabled: bool = True
    fingerprint_n_bits: int = 128
    fingerprint_radius: int = 2
    electronic: bool = True
    steric: bool = True
    dft_enabled: bool = False
    dft_data_path: str | None = None
    energy_backend: str = "none"


@dataclass(frozen=True)
class FeasibilityConfig:
    """Configuration for synthesis feasibility assessment.

    Attributes:
        mode: ``"soft"`` adds a penalty to the objective; ``"hard"`` filters
            out infeasible suggestions entirely.
        sa_threshold: SA score threshold.  Molecules above this threshold
            are penalized (soft) or rejected (hard).  Range: [1, 10].
        strain_threshold_kcal: Strain energy threshold (kcal/mol).  Molecules
            with estimated strain above this value receive a progressive
            penalty ``(strain - threshold) / 25``.  Scene-dependent defaults:
            drug-like (25–30), materials (40–50), natural-product-like (60–80),
            methodology exploration (100+).  Default 33.0 works well for
            typical medicinal chemistry scaffolds.
        incompatible_pairs: List of known incompatible substituent
            combinations.  Each entry is a dict with ``positions``,
            ``pair`` (substituent names), and ``reason``.
    """

    mode: str = "soft"  # "soft" | "hard"
    sa_threshold: float = 6.0
    strain_threshold_kcal: float = 33.0
    incompatible_pairs: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class MolecularDesignSpec:
    """Complete specification for a molecular optimization problem.

    This is the central data structure that bridges the user's chemistry
    intent and the HEBO design space.

    Attributes:
        scaffold: The molecular scaffold.
        libraries: Map from position name to the substituent library.
        condition_params: Additional non-molecular design parameters
            (e.g. temperature, solvent) in HEBO design-parameter format.
        descriptor_config: Which descriptors to compute for the oracle.
        feasibility: Feasibility assessment configuration.
    """

    scaffold: Scaffold
    libraries: dict[str, SubstituentLibrary]
    condition_params: list[dict[str, Any]] = field(default_factory=list)
    descriptor_config: DescriptorConfig = field(default_factory=DescriptorConfig)
    feasibility: FeasibilityConfig = field(default_factory=FeasibilityConfig)


@dataclass
class FeasibilityResult:
    """Result of a synthesis feasibility assessment.

    Attributes:
        is_feasible: Whether the molecule passes the feasibility check.
        sa_score: Synthetic Accessibility score (Ertl 2009).  Range [1, 10],
            lower means easier to synthesize.
        num_steps_estimate: Rough estimate of the number of synthetic steps.
            ``None`` if the estimate is unreliable.
        penalty: Numeric penalty to apply to the objective value (soft mode).
            Zero for feasible molecules.
        reasons: Human-readable explanations of any issues found.
    """

    is_feasible: bool
    sa_score: float
    num_steps_estimate: int | None = None
    penalty: float = 0.0
    reasons: list[str] = field(default_factory=list)
    strain_energy_kcal: float | None = None
