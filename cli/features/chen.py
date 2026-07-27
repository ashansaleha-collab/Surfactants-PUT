import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED, Descriptors, EState, GraphDescriptors, MolSurf

from cli.data import SurfactantsDf

from . import FeatureExtractor


class ChenExtractor(FeatureExtractor):
    """Extractor that adds 19 "important" descriptors from Chen et al.

    Reference:
    Chen et al. "Prediction of critical micelle concentration (CMC)
    of surfactants based on structural differentiation using machine
    learning" https://doi.org/10.1016/j.colsurfa.2024.135276
    """

    def extract(self, samples: SurfactantsDf) -> pd.DataFrame:
        results = []
        failed_indices = []

        for idx, row in samples.df.iterrows():
            try:
                smiles = row["surfactant_smiles"]
                mol = Chem.MolFromSmiles(smiles)

                if mol is None:
                    failed_indices.append(idx)
                    continue

                # Chen et al.'s 19 descriptors
                descriptors = {
                    "index": idx,
                    "chen_MolLogP": Descriptors.MolLogP(mol),
                    "chen_Chi1v": GraphDescriptors.Chi1v(mol),
                    "chen_VSA_EState7": EState.EState_VSA.EState_VSA7(mol),
                    "chen_VSA_EState8": EState.EState_VSA.EState_VSA8(mol),
                    "chen_qed": QED.qed(mol),
                    "chen_FpDensityMorgan1": Descriptors.FpDensityMorgan1(mol),
                    "chen_SMR_VSA5": MolSurf.SMR_VSA5(mol),
                    "chen_PEOE_VSA6": MolSurf.PEOE_VSA6(mol),
                    "chen_MinPartialCharge": Descriptors.MinPartialCharge(mol),
                    "chen_BertzCT": GraphDescriptors.BertzCT(mol),
                    "chen_VSA_EState1": EState.EState_VSA.EState_VSA1(mol),
                    "chen_MaxAbsEStateIndex": EState.MaxAbsEStateIndex(mol),
                    "chen_Balaban": GraphDescriptors.BalabanJ(mol),
                    "chen_SlogP_VSA2": MolSurf.SlogP_VSA2(mol),
                    "chen_HallKierAlpha": Descriptors.HallKierAlpha(mol),
                    "chen_EState_VSA5": EState.EState_VSA.EState_VSA5(mol),
                    "chen_MaxPartialCharge": Descriptors.MaxPartialCharge(mol),
                    "chen_PEOE_VSA7": MolSurf.PEOE_VSA7(mol),
                    "chen_PEOE_VSA8": MolSurf.PEOE_VSA8(mol),
                }

                results.append(descriptors)

            except Exception as e:
                print(f"⚠️  Failed idx {idx}: {e}")
                failed_indices.append(idx)
                continue

        result = pd.DataFrame(results).set_index("index")

        # chen_MaxPartialCharge may be inf, replace with 0?
        result = result.replace([float("inf"), float("-inf")], 0.0)

        return result
