import pandas as pd

from cli.data import SurfactantsDf
from cli.datasets import Dataset

from . import preprocess


class ExpertDataset(Dataset):
    """Expert Dataset (sources/CMC_surfactants_v2_4.csv)."""

    def samples(self) -> SurfactantsDf:
        surfactants_db = pd.read_csv("sources/CMC_surfactants_v2_4.csv")
        surfactants_db_agg = preprocess.preprocess(surfactants_db)
        surfactants_db_agg = surfactants_db_agg.rename(
            columns={
                "SMILES": "surfactant_smiles",
                "Additives": "additive_smiles",
                "Conc. A.": "additive_concentration",
                "Temp_Celsius": "temperature",
                "pCMC": "pcmc",
            },
        )[
            [
                "surfactant_smiles",
                "temperature",
                "additive_smiles",
                "additive_concentration",
                "pcmc",
            ]
        ]
        return SurfactantsDf(surfactants_db_agg)
