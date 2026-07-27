import pandas as pd

from cli.data import AnnotatedSurfactantsDf, SurfactantsDf

from . import Dataset


class MergedDataset(Dataset):
    def __init__(self, datasets: list[Dataset]):
        samples_list = [dataset.samples().df for dataset in datasets]
        # Flatten into a single DataFrame
        self.df = pd.concat(samples_list, ignore_index=True)
        # Sort by smiles
        self.df = self.df.sort_values(by=["surfactant_smiles", "additive_smiles"])
        # Aggregate duplicates
        self.df = self.df.groupby(
            [
                "surfactant_smiles",
                "temperature",
                "additive_smiles",
                "additive_concentration",
            ],
            dropna=False,
            as_index=False,
        ).agg({"pcmc": "mean"})

    def samples(self) -> SurfactantsDf:
        return SurfactantsDf(self.df)

    def annotated_samples(self) -> AnnotatedSurfactantsDf:
        # Returns the data with annotation for testing
        return AnnotatedSurfactantsDf(self.df)
