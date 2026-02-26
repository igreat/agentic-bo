from __future__ import annotations


def _canonical(smi: str) -> str | None:
    from rdkit import Chem

    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else None


def _tanimoto_fp(smi: str, n_bits: int = 1024):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=n_bits)


def _tanimoto_similarity(fp1, fp2) -> float:
    from rdkit import DataStructs

    if fp1 is None or fp2 is None:
        return 0.0
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def _is_drug_like(smi: str) -> bool:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    m = Chem.MolFromSmiles(smi)
    if m is None:
        return False
    atoms = {a.GetSymbol() for a in m.GetAtoms()}
    if "C" not in atoms:
        return False
    allowed = {"C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "H"}
    if atoms - allowed:
        return False
    mw = Descriptors.MolWt(m)
    if mw < 100 or mw > 900:
        return False
    if rdMolDescriptors.CalcNumRings(m) < 1:
        return False
    return True
