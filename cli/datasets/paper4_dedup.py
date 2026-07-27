import pandas as pd
from cli.data import SurfactantsDf, AnnotatedSurfactantsDf
from . import Dataset


class Paper4DedupDataset(Dataset):
    def __init__(self, *, dedup=False):
        # 1. Load training data to get SMILES to exclude
        try:
            df_train = pd.read_csv("sources/CMC_surfactants_v2_4.csv")
            train_smiles = set(df_train["SMILES"])
        except FileNotFoundError:
            print("Warning: Could not find training data for de-duplication.")
            train_smiles = set()

        # 2. Load the new test data
        df_test = pd.read_csv("sources/Data_paper_4.csv")

        # 3. Filter out training SMILES
        if dedup:
            original_count = len(df_test)
            # Note: Column name is 'String' in this file
            df_filtered = df_test[~df_test["String"].isin(train_smiles)].copy()
            print(
                f"Paper 4: Kept {len(df_filtered)} of {original_count} samples after de-duplication."
            )
        else:
            df_filtered = df_test

        # 4. Normalize columns
        df_proc = pd.DataFrame()
        df_proc["surfactant_smiles"] = df_filtered["String"]
        df_proc["temperature"] = df_filtered["T_C"]
        df_proc["additive_smiles"] = None
        df_proc["additive_concentration"] = None
        df_proc["pcmc"] = df_filtered["log CMC (uM)"]

        # pcmc convertion from uM to -M
        df_proc["pcmc"] = -df_proc["pcmc"] + 6

        df_proc["temperature"] = df_proc["temperature"].astype(float)
        df_proc["additive_concentration"] = df_proc["additive_concentration"].astype(
            float
        )

        self.df = df_proc

    def samples(self) -> AnnotatedSurfactantsDf:
        return AnnotatedSurfactantsDf(self.df)

    def annotated_samples(self) -> AnnotatedSurfactantsDf:
        return AnnotatedSurfactantsDf(self.df)
