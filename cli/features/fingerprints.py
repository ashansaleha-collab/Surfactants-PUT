from dataclasses import dataclass

import numpy as np
import pandas as pd
import skfp.fingerprints as sfp
from rdkit import Chem
from rdkit.Chem import AllChem, EState
from rdkit.Chem.Crippen import MolLogP
from skfp.fingerprints import MACCSFingerprint

from cli.data import SurfactantsDf

from . import FeatureExtractor


@dataclass
class RdKitFeatures:
    index: int
    fp_all: list[int] | None
    fp_additive: list[int] | None
    mollogp: float | None = None


@dataclass
class _FingerprintExtractor:
    fp_len: int
    # function getting SMILES and returning list[float] of length fp_len
    extractor: callable


def _smiles_to_estate(smi: str | None) -> list[int] | None:
    """Generate EState fingerprint from SMILES string."""
    if not isinstance(smi, str) or smi.strip() == "":
        return None

    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None

        fp = EState.Fingerprinter.FingerprintMol(mol)[1]
        return fp.tolist()
    except Exception:
        return None


def _smiles_to_estate0(smi: str | None) -> list[int] | None:
    """Generate EState fingerprint from SMILES string."""
    if not isinstance(smi, str) or smi.strip() == "":
        return None

    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None

        fp = EState.Fingerprinter.FingerprintMol(mol)[0].astype(np.float32)
        return fp.tolist()
    except Exception:
        return None


def _smiles_to_morgan(smi: str | None) -> list[int] | None:
    """Generate Morgan fingerprint from SMILES string."""
    if not isinstance(smi, str) or smi.strip() == "":
        return None

    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None

        fpgen = AllChem.GetMorganGenerator(radius=2)
        arr = fpgen.GetCountFingerprint(mol, nBits=256)
        return arr.ToBitString().tolist()
    except Exception:
        return None


def _smiles_to_skfp(smi: str | None, fp) -> list[int] | None:
    """Generate fingerprint bytes from SMILES string."""
    if not isinstance(smi, str) or smi.strip() == "":
        return None

    try:
        array = fp.transform([_sanitize_smiles(smi)])
        return array.tolist()[0]
    except TypeError:
        return None


def _get_skfp_extractor(fp):
    return _FingerprintExtractor(
        fp_len=fp.n_features_out,
        extractor=lambda smi: _smiles_to_skfp(smi, fp=fp),
    )


def _fingerprints() -> dict[str, _FingerprintExtractor]:
    # fmt: off
    return {
        "estate": _FingerprintExtractor(
            fp_len=79,
            extractor=_smiles_to_estate,
        ),
        "estate0": _FingerprintExtractor(
            fp_len=79,
            extractor=_smiles_to_estate0,
        ),
        "morgan": _FingerprintExtractor(
            fp_len=256,
            extractor=_smiles_to_morgan,
        ),
        "atompair": _get_skfp_extractor(sfp.AtomPairFingerprint(fp_size=256)),
        "autocorr": _get_skfp_extractor(sfp.AutocorrFingerprint()),
        "avalon": _get_skfp_extractor(sfp.AvalonFingerprint(fp_size=256)),
        "bcut2d": _get_skfp_extractor(sfp.BCUT2DFingerprint()),
        "e3fp": _get_skfp_extractor(sfp.E3FPFingerprint(fp_size=256)),
        "ecfp": _get_skfp_extractor(sfp.ECFPFingerprint(fp_size=256)),
        "electroshape": _get_skfp_extractor(sfp.ElectroShapeFingerprint()),
        "erg": _get_skfp_extractor(sfp.ERGFingerprint()),
        "functionalgroups": _get_skfp_extractor(sfp.FunctionalGroupsFingerprint()),
        "getaway": _get_skfp_extractor(sfp.GETAWAYFingerprint()),
        "ghosecrippen": _get_skfp_extractor(sfp.GhoseCrippenFingerprint()),
        "klekotaroth": _get_skfp_extractor(sfp.KlekotaRothFingerprint()),
        "laggner": _get_skfp_extractor(sfp.LaggnerFingerprint()),
        "layered": _get_skfp_extractor(sfp.LayeredFingerprint(fp_size=256)),
        "lingo": _get_skfp_extractor(sfp.LingoFingerprint(fp_size=256)),
        "maccs": _get_skfp_extractor(sfp.MACCSFingerprint()),
        "map": _get_skfp_extractor(sfp.MAPFingerprint(fp_size=256)),
        "mhfp": _get_skfp_extractor(sfp.MHFPFingerprint(fp_size=256)),
        "mordred": _get_skfp_extractor(sfp.MordredFingerprint()),
        "morse": _get_skfp_extractor(sfp.MORSEFingerprint()),
        "mqns": _get_skfp_extractor(sfp.MQNsFingerprint()),
        "pattern": _get_skfp_extractor(sfp.PatternFingerprint(fp_size=256)),
        "pharmacophore": _get_skfp_extractor(sfp.PharmacophoreFingerprint(fp_size=256)),
        "physiochemicalproperties": _get_skfp_extractor(sfp.PhysiochemicalPropertiesFingerprint(fp_size=256)),
        "pubchem": _get_skfp_extractor(sfp.PubChemFingerprint()),
        "rdf": _get_skfp_extractor(sfp.RDFFingerprint()),
        "rdkit": _get_skfp_extractor(sfp.RDKitFingerprint(fp_size=256)),
        "rdkit2ddescriptors": _get_skfp_extractor(sfp.RDKit2DDescriptorsFingerprint()),
        "secfp": _get_skfp_extractor(sfp.SECFPFingerprint(fp_size=256)),
        "topologicaltorsion": _get_skfp_extractor(sfp.TopologicalTorsionFingerprint(fp_size=256)),
        "usr": _get_skfp_extractor(sfp.USRFingerprint()),
        "usrcat": _get_skfp_extractor(sfp.USRCATFingerprint()),
        "vsa": _get_skfp_extractor(sfp.VSAFingerprint()),
        "whim": _get_skfp_extractor(sfp.WHIMFingerprint()),
    }
    # fmt: on


def _sanitize_smiles(smi: str | None) -> str | None:
    """Perform SMILES sanitization at the string level."""
    if not isinstance(smi, str) or smi.strip() == "":
        return None

    return smi


def _compute_rdkit_features(
    surfactants_db_agg: SurfactantsDf,
    *,
    extractor,
) -> pd.DataFrame:
    rdkit_rows: list[RdKitFeatures] = []

    for idx, row in surfactants_db_agg.iterrows():
        mol = Chem.MolFromSmiles(_sanitize_smiles(row["surfactant_smiles"]))
        if mol is None:
            continue

        # fingerprints
        fp_all = extractor(row["surfactant_smiles"])
        fp_additive = extractor(row["additive_smiles"])
        rdkit_rows.append(
            RdKitFeatures(
                index=idx,
                fp_all=fp_all,
                fp_additive=fp_additive,
                mollogp=MolLogP(mol),
            ),
        )

    return pd.DataFrame(rdkit_rows).set_index("index")


def _postprocess_rdkit(
    surfactants_db_agg_fp: pd.DataFrame,
    *,
    fp_type: str,
    fp_len: int,
):
    # split fingerprint columns into separate columns
    def split_df(basename: str, fp_len: int):
        lst = surfactants_db_agg_fp[basename].tolist()

        print(f"fingerprint {basename} length: {fp_len}")
        column_names = [f"{fp_type}_{basename}_{i}" for i in range(fp_len)]

        # replace [None] with [None,...]
        lst = [x if x is not None else [None] * fp_len for x in lst]
        return pd.DataFrame(
            lst,
            index=surfactants_db_agg_fp.index,
            columns=column_names,
            dtype=float,
        )

    fp_all_df = split_df("fp_all", fp_len)
    fp_additive_df = split_df("fp_additive", fp_len)

    surfactants_db_agg_fp = pd.concat(
        [
            surfactants_db_agg_fp,
            fp_all_df,
            fp_additive_df,
        ],
        axis=1,
    )
    surfactants_db_agg_fp = surfactants_db_agg_fp.drop(
        columns=[
            "fp_all",
            "fp_additive",
        ],
    )

    return surfactants_db_agg_fp


def add_fingerprints(
    surfactants_db_agg: SurfactantsDf,
    *,
    fp_type: str,
) -> pd.DataFrame:
    fingerprint = _fingerprints().get(fp_type)
    if fingerprint is None:
        msg = f"Unknown fingerprint type: {fp_type}"
        raise ValueError(msg)
    rdkit_features_df = _compute_rdkit_features(
        surfactants_db_agg.df,
        extractor=fingerprint.extractor,
    )
    return _postprocess_rdkit(
        rdkit_features_df,
        fp_type=fp_type,
        fp_len=fingerprint.fp_len,
    )


class FingerprintsExtractor(FeatureExtractor):
    def __init__(self, fp_type):
        self.fingerprints_type = fp_type

    def extract(self, surfactants_db_agg: SurfactantsDf) -> pd.DataFrame:
        return add_fingerprints(surfactants_db_agg, fp_type=self.fingerprints_type)
