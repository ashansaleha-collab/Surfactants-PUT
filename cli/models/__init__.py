from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from cli.data import AnnotatedSurfactantsDf, SurfactantsDf


class Model(ABC):
    """A CMC predicting model."""

    def load(self, data_dir: Path) -> None:
        """Load the model from given directory."""
        raise NotImplementedError

    def save(self, data_dir: Path) -> None:
        """Save the model to given directory."""
        raise NotImplementedError

    @abstractmethod
    def train(self, samples: AnnotatedSurfactantsDf, val_samples: AnnotatedSurfactantsDf | None = None) -> dict[str, Any] | None:
        """Train the model on given samples.
        """

    @abstractmethod
    def get_param_grid(self) -> dict[str, list[Any]]:
        """Return hyperparameter grid for tuning.
        """

    @abstractmethod
    def predict(self, samples: SurfactantsDf) -> list[float]:
        """Predict pCMC for given samples."""

    def supports_feature_extractor(self, extractor: str) -> bool:
        """Check if a given feature extractor is supported."""
        return False
