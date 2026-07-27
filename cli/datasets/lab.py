import pandas as pd

from cli.data import AnnotatedSurfactantsDf
from . import Dataset


class LabDataset(Dataset):
    """Laboratory validation dataset (sources/lab.csv).

    77 newly measured CMC values for 16 surfactants not present in the
    training data, used for external validation (Fig. 3 in the paper).
    """

    def __init__(self):
        df = pd.read_csv("sources/lab.csv")
        df["temperature"] = df["temperature"].astype(float)
        df["additive_concentration"] = df["additive_concentration"].astype(float)
        self.df = df

    def samples(self) -> AnnotatedSurfactantsDf:
        return AnnotatedSurfactantsDf(self.df)

    def annotated_samples(self) -> AnnotatedSurfactantsDf:
        return AnnotatedSurfactantsDf(self.df)
