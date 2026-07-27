import pandas as pd
from cli.data import SurfactantsDf, AnnotatedSurfactantsDf
from . import Dataset


class Paper1DedupDataset(Dataset):
    def __init__(self, *, dedup=False):
        # 1. Load training data to get SMILES to exclude
        try:
            df_train = pd.read_csv("sources/CMC_surfactants_v2_4.csv")
            train_smiles = set(df_train["SMILES"])
        except FileNotFoundError:
            print("Warning: Could not find training data for de-duplication.")
            train_smiles = set()

        # 2. Load the new test data
        df_test = pd.read_csv("sources/Data_paper_1.csv")

        # 3. Filter out training SMILES
        if dedup:
            original_count = len(df_test)
            df_filtered = df_test[~df_test["smiles"].isin(train_smiles)].copy()
            print(
                f"Paper 1: Kept {len(df_filtered)} of {original_count} samples after de-duplication."
            )
        else:
            df_filtered = df_test

        # 4. Normalize columns to match SurfactantsDf format
        df_proc = pd.DataFrame()
        df_proc["surfactant_smiles"] = df_filtered["smiles"]
        df_proc["temperature"] = df_filtered["T"]
        df_proc["additive_smiles"] = None  # This dataset has no additives
        df_proc["additive_concentration"] = None
        df_proc["pcmc"] = df_filtered["pCMC"]  # The annotation column
        df_proc["temperature"] = df_proc["temperature"].astype(float)
        df_proc["additive_concentration"] = df_proc["additive_concentration"].astype(
            float
        )
        self.df = df_proc

    def samples(self) -> AnnotatedSurfactantsDf:
        return AnnotatedSurfactantsDf(self.df)

    def annotated_samples(self) -> AnnotatedSurfactantsDf:
        # Returns the data with annotation for testing
        return AnnotatedSurfactantsDf(self.df)
