from pathlib import Path
from typing import Any

from sklearn.base import BaseEstimator

from cli.data import AnnotatedSurfactantsDf, SurfactantsDf

from . import Model


class DummyModel(Model):
    """A dummy model that predicts a constant pCMC value.

    Parameters
    ----------
       value (float) - the value to predict

    """

    def __init__(self, params: dict[str, str]):
        self.value = float(params.get("value", 0.0))

    def train(self, _samples: AnnotatedSurfactantsDf) -> None:
        pass

    def predict(self, samples: SurfactantsDf) -> list[float]:
        return [self.value] * len(samples)


class AvgBaselineModel(Model):
    """A model that always predicts the average pCMC from the training data."""

    def __init__(self, params: dict[str, str]):
        self.avg_pcmc = 0.0

    def load(self, data_dir: Path) -> None:
        with data_dir.joinpath("avg_pcmc.txt").open("r") as f:
            self.avg_pcmc = float(f.read().strip())

    def save(self, data_dir: Path) -> None:
        with data_dir.joinpath("avg_pcmc.txt").open("w") as f:
            f.write(str(self.avg_pcmc))

    def train(
        self,
        samples: AnnotatedSurfactantsDf,
        val_samples: AnnotatedSurfactantsDf | None = None,
    ) -> None:
        df = samples.df
        self.avg_pcmc = df["pcmc"].mean()

    def predict(self, samples: SurfactantsDf) -> list[float]:
        return [self.avg_pcmc] * len(samples.df)

    def get_param_grid(self) -> dict[str, list[Any]]:
        return {}


class SklearnModel(Model):
    """A wrapper for sklearn BaseEstimator."""

    def __init__(self, model: BaseEstimator, *, supported_features: set[str] | None, param_grid: dict[str, list[str]] | None = None):
        self.model = model
        self._supported_features = supported_features or set()
        self._param_grid = param_grid or {}

    def train(self, samples: AnnotatedSurfactantsDf, val_samples: AnnotatedSurfactantsDf | None = None) -> None:
        x = samples.df.drop(columns=["pcmc"])
        y = samples.df["pcmc"]
        self.model.fit(x, y)

    def predict(self, samples: SurfactantsDf) -> list[float]:
        return self.model.predict(samples.df).tolist()

    def supports_feature_extractor(self, extractor: str) -> bool:
        return extractor in self._supported_features

    def get_param_grid(self) -> dict[str, list[Any]]:
        return self._param_grid
