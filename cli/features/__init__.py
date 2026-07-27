from abc import ABC, abstractmethod

import pandas as pd

from cli.data import SurfactantsDf


class FeatureExtractor(ABC):
    """A class that generates more features from SurfactantsDf.

    The features may be understood by model classes.

    For example: fingerprints, embeddings,...
    """

    @abstractmethod
    def extract(self, samples: SurfactantsDf) -> pd.DataFrame:
        """Extract features.

        Returns a DataFrame with the same index as samples, but ONLY
        extracted features.
        """
