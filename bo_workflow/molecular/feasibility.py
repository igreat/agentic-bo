"""Synthesis feasibility scoring and constraint generation.

This module assesses whether a proposed molecular modification is
synthetically achievable and assigns a feasibility score / penalty.

It provides:
- SA score computation (Ertl 2009) via RDKit
- Reaction compatibility checking (incompatible pairs, steric clashes)
- Rough synthetic step estimation
- Batch feasibility assessment for HEBO suggestion filtering
"""

from __future__ import annotations

import math
from typing import Any

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from .types import (
    FeasibilityResult,
    MolecularDesignSpec,
    Scaffold,
    Substituent,
)

# ---------------------------------------------------------------------------
# SA Score (Synthetic Accessibility)
# ---------------------------------------------------------------------------

# RDKit ships an SA score implementation in Contrib.  We try to use it;
# if it's not available, fall back to a simpler heuristic.
_sa_score_func = None


def _get_sa_score_func():
    """Lazily load the SA score function."""
    global _sa_score_func
    if _sa_score_func is not None:
        return _sa_score_func

    try:
        from rdkit.Chem import RDConfig
        import os
        import sys

        sa_module_path = os.path.join(RDConfig.RDContribDir, "SA_Score")
        if sa_module_path not in sys.path:
            sys.path.insert(0, sa_module_path)
        import sascorer

        _sa_score_func = sascorer.calculateScore
    except (ImportError, OSError):
        _sa_score_func = _heuristic_sa_score
    return _sa_score_func


def _heuristic_sa_score(mol: Chem.Mol) -> float:
    """Heuristic SA score when RDKit Contrib SA_Score is unavailable.

    Returns a value in [1, 10].  Lower = easier to synthesize.
    This is a rough approximation based on molecular complexity.
    """
    n_heavy = mol.GetNumHeavyAtoms()
    n_rings = rdMolDescriptors.CalcNumRings(mol)
    n_stereo = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
    n_bridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    n_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)

    # Simple weighted sum, clamped to [1, 10]
    score = 1.0
    score += n_heavy * 0.05
    score += n_rings * 0.3
    score += n_stereo * 0.5
    score += n_bridgehead * 1.0
    score += n_spiro * 0.8

    return max(1.0, min(10.0, score))


def compute_sa_score(smiles: str) -> float:
    """Compute the Synthetic Accessibility score for a molecule.

    Uses the Ertl (2009) SA score from RDKit Contrib if available,
    otherwise falls back to a simpler heuristic.

    Args:
        smiles: SMILES string of the molecule.

    Returns:
        SA score in [1, 10].  Lower = easier to synthesize.

    Raises:
        ValueError: If SMILES cannot be parsed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES: {smiles}")

    scorer = _get_sa_score_func()
    return float(scorer(mol))


# ---------------------------------------------------------------------------
# Reaction compatibility checking
# ---------------------------------------------------------------------------


def check_reaction_compatibility(
    spec: MolecularDesignSpec,
    choices: dict[str, Substituent],
) -> list[str]:
    """Check for known incompatibilities between chosen substituents.

    Checks:
    1. User-defined incompatible pairs from the spec.
    2. General chemistry heuristics (adjacent bulky groups, etc.).

    Args:
        spec: The molecular design specification.
        choices: Map from position name to chosen Substituent.

    Returns:
        List of human-readable warning strings.  Empty if no issues.
    """
    warnings: list[str] = []

    # 1. Check user-defined incompatible pairs
    for rule in spec.feasibility.incompatible_pairs:
        positions = rule.get("positions", [])
        pair = rule.get("pair", [])
        reason = rule.get("reason", "Incompatible substituent pair")

        if len(positions) == 2 and len(pair) == 2:
            pos_a, pos_b = positions
            name_a, name_b = pair
            choice_a = choices.get(pos_a)
            choice_b = choices.get(pos_b)
            if choice_a and choice_b:
                if (choice_a.name == name_a and choice_b.name == name_b) or (
                    choice_a.name == name_b and choice_b.name == name_a
                ):
                    warnings.append(
                        f"Incompatible pair at {pos_a}={choice_a.name}, "
                        f"{pos_b}={choice_b.name}: {reason}"
                    )

    # 2. Heuristic: multiple bulky groups on adjacent positions
    from .features import _TAFT_ES

    bulky_positions = []
    for pos_name, sub in choices.items():
        es = sub.properties.get("taft_Es", _TAFT_ES.get(sub.name, 0.0))
        if es < -1.0:  # Taft Es < -1.0 means bulky
            bulky_positions.append((pos_name, sub.name, es))

    if len(bulky_positions) >= 2:
        pos_names = [f"{p[0]}={p[1]}" for p in bulky_positions]
        warnings.append(
            f"Multiple bulky substituents detected ({', '.join(pos_names)}). "
            f"Potential steric clash."
        )

    # 3. Heuristic: strongly electron-donating + strongly electron-withdrawing
    # on same ring (may cause unexpected reactivity)
    from .features import _HAMMETT_SIGMA

    strong_edg = []
    strong_ewg = []
    for pos_name, sub in choices.items():
        sigma = sub.properties.get(
            "hammett_sigma", _HAMMETT_SIGMA.get(sub.name, 0.0)
        )
        if sigma < -0.4:
            strong_edg.append((pos_name, sub.name, sigma))
        elif sigma > 0.5:
            strong_ewg.append((pos_name, sub.name, sigma))

    if strong_edg and strong_ewg:
        edg_str = ", ".join(f"{p[0]}={p[1]}" for p in strong_edg)
        ewg_str = ", ".join(f"{p[0]}={p[1]}" for p in strong_ewg)
        warnings.append(
            f"Strong EDG ({edg_str}) and EWG ({ewg_str}) present. "
            f"May affect reaction selectivity or require modified conditions."
        )

    return warnings


# ---------------------------------------------------------------------------
# Synthetic step estimation
# ---------------------------------------------------------------------------


def estimate_synthetic_steps(smiles: str) -> int | None:
    """Rough estimate of synthetic steps required.

    Uses simple heuristics based on molecular complexity.  Returns None
    if the estimate is unreliable (very complex molecule).

    Args:
        smiles: SMILES string of the target molecule.

    Returns:
        Estimated number of synthetic steps, or None.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    n_rings = rdMolDescriptors.CalcNumRings(mol)
    n_stereo = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
    n_heavy = mol.GetNumHeavyAtoms()

    if n_heavy > 50:
        return None  # Too complex for a reliable estimate

    # Base: 1 step per ring, 1 step per stereocenter
    steps = max(1, n_rings) + n_stereo

    # Add steps for heavy atom count
    if n_heavy > 30:
        steps += 2
    elif n_heavy > 20:
        steps += 1

    return steps


# ---------------------------------------------------------------------------
# Main feasibility assessment
# ---------------------------------------------------------------------------


def check_bredt_rule(smiles: str) -> str | None:
    """Check for Bredt's rule violations (double bond at bridgehead).

    This is a topology-based fallback.  When energy-based assessment is
    available, trust the energy penalty over this rule (the energy will
    naturally penalize Bredt violations via high strain).

    Returns a warning string if a violation is detected, else None.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()
    if len(atom_rings) < 2:
        return None

    # Find bridgehead atoms: atoms in 2+ rings
    atom_ring_count: dict[int, int] = {}
    for ring in atom_rings:
        for idx in ring:
            atom_ring_count[idx] = atom_ring_count.get(idx, 0) + 1
    bridgehead_atoms = {idx for idx, cnt in atom_ring_count.items() if cnt >= 2}

    if not bridgehead_atoms:
        return None

    # Check if any bridgehead has a double-bond neighbor
    for idx in bridgehead_atoms:
        atom = mol.GetAtomWithIdx(idx)
        for bond in atom.GetBonds():
            if bond.GetBondTypeAsDouble() == 2.0:
                return (
                    f"Bredt's rule violation: double bond at bridgehead atom {idx}. "
                    f"Anti-Bredt alkenes are generally synthetically inaccessible "
                    f"in small bridged systems."
                )

    return None


def assess_feasibility(
    smiles: str,
    *,
    sa_threshold: float = 6.0,
    spec: MolecularDesignSpec | None = None,
    choices: dict[str, Substituent] | None = None,
    hard_filter: bool = False,
    strain_energy_kcal: float | None = None,
    strain_threshold_kcal: float = 33.0,
) -> FeasibilityResult:
    """Full feasibility assessment for a molecule.

    Combines SA score, reaction compatibility, strain energy, Bredt rule,
    and synthetic step estimation into a single ``FeasibilityResult``.

    Args:
        smiles: SMILES string of the complete molecule.
        sa_threshold: SA score above which molecules are penalized/rejected.
        spec: Optional molecular design spec for compatibility checking.
        choices: Optional substituent choices for compatibility checking.
        hard_filter: If True, molecules above sa_threshold are marked
            infeasible.  If False (default), they get a penalty.
        strain_energy_kcal: Pre-computed strain energy in kcal/mol.  When
            provided, used to add progressive penalty.
        strain_threshold_kcal: Strain energy above this threshold incurs
            a penalty (default 33 kcal/mol).  Set per-scenario via
            ``FeasibilityConfig.strain_threshold_kcal``.

    Returns:
        FeasibilityResult with all assessment details.
    """
    reasons: list[str] = []

    # SA score
    try:
        sa = compute_sa_score(smiles)
    except ValueError:
        sa = 10.0
        reasons.append(f"Cannot parse SMILES for SA score: {smiles}")

    # Synthetic step estimate
    num_steps = estimate_synthetic_steps(smiles)

    # Compatibility checks
    if spec is not None and choices is not None:
        compat_warnings = check_reaction_compatibility(spec, choices)
        reasons.extend(compat_warnings)

    # Bredt rule check (topology-based fallback when no energy available)
    if strain_energy_kcal is None:
        bredt_warning = check_bredt_rule(smiles)
        if bredt_warning is not None:
            reasons.append(bredt_warning)

    # Determine feasibility and penalty
    is_feasible = True
    penalty = 0.0

    if sa > sa_threshold:
        if hard_filter:
            is_feasible = False
            reasons.append(
                f"SA score ({sa:.2f}) exceeds threshold ({sa_threshold})."
            )
        else:
            # Soft penalty: proportional to how much SA exceeds threshold
            penalty = (sa - sa_threshold) * 2.0
            reasons.append(
                f"SA score ({sa:.2f}) exceeds threshold ({sa_threshold}); "
                f"penalty={penalty:.2f}."
            )

    # Strain-energy-based penalty (physics correction for SA Score blind spots)
    if strain_energy_kcal is not None and strain_energy_kcal > strain_threshold_kcal:
        strain_penalty = (strain_energy_kcal - strain_threshold_kcal) / 25.0
        penalty += strain_penalty
        reasons.append(
            f"High strain energy ({strain_energy_kcal:.1f} kcal/mol > "
            f"{strain_threshold_kcal} threshold); penalty={strain_penalty:.2f}."
        )

    # Additional penalty for compatibility warnings
    if reasons and penalty == 0 and sa <= sa_threshold:
        # Mild penalty for compatibility issues even if SA is OK
        n_warnings = len([r for r in reasons if "Incompatible" in r or "clash" in r.lower()])
        if n_warnings > 0:
            penalty = n_warnings * 0.5

    return FeasibilityResult(
        is_feasible=is_feasible,
        sa_score=sa,
        num_steps_estimate=num_steps,
        penalty=penalty,
        reasons=reasons,
        strain_energy_kcal=strain_energy_kcal,
    )


# ---------------------------------------------------------------------------
# Batch assessment (used by suggest() in the engine)
# ---------------------------------------------------------------------------


def assess_feasibility_batch(
    suggestions: list[dict[str, Any]],
    spec: MolecularDesignSpec,
    *,
    sa_threshold: float = 6.0,
    energy_cache: Any | None = None,
) -> list[FeasibilityResult]:
    """Assess feasibility for a batch of HEBO suggestion dicts.

    For each suggestion, assembles the full molecule and runs the
    full feasibility assessment.

    Args:
        suggestions: List of HEBO suggestion dicts (e.g.
            ``[{"R1": "F", "R2": "methyl"}, ...]``).
        spec: The molecular design specification.
        sa_threshold: SA score threshold.
        energy_cache: Optional ``EnergyCache`` instance.  When provided,
            strain energy is looked up and passed to ``assess_feasibility``
            for physics-based penalty.

    Returns:
        List of FeasibilityResult, one per suggestion.
    """
    from .scaffold import decode_suggestion

    results: list[FeasibilityResult] = []
    hard_filter = spec.feasibility.mode == "hard"

    for suggestion in suggestions:
        try:
            full_smiles, choices = decode_suggestion(suggestion, spec)
        except (ValueError, KeyError) as exc:
            results.append(
                FeasibilityResult(
                    is_feasible=False,
                    sa_score=10.0,
                    penalty=10.0,
                    reasons=[f"Failed to assemble molecule: {exc}"],
                )
            )
            continue

        # Look up strain energy from cache if available
        strain_energy: float | None = None
        if energy_cache is not None:
            cached = energy_cache.get(full_smiles)
            if cached is not None:
                strain_val = cached.get("strain_energy_kcal")
                if strain_val is not None and not math.isnan(strain_val):
                    strain_energy = strain_val

        result = assess_feasibility(
            full_smiles,
            sa_threshold=sa_threshold,
            spec=spec,
            choices=choices,
            hard_filter=hard_filter,
            strain_energy_kcal=strain_energy,
            strain_threshold_kcal=spec.feasibility.strain_threshold_kcal,
        )
        results.append(result)

    return results
