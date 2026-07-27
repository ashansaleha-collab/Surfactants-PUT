from abc import ABC, abstractmethod

from cli.data import SurfactantsDf


class Dataset(ABC):
    """A dataset of samples."""

    @abstractmethod
    def samples(self) -> SurfactantsDf:
        """Return the samples in the dataset."""
