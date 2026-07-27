from . import Model
from .dummy import AvgBaselineModel, DummyModel
from .sk import create_knn_baseline_model
from .sk_impls import create_lgbm_model, create_random_forest_model
from .transformer import TransformerModel


def models():
    return {
        "avg": AvgBaselineModel,
        "dummy": DummyModel,
        "knn": create_knn_baseline_model,
        "lgbm": create_lgbm_model,
        "rf": create_random_forest_model,
        "transformer": TransformerModel,
    }


def create_model(model_name: str, params: dict[str, str]) -> Model:
    """Create model instance by name."""
    model_cls = models().get(model_name)
    if model_cls is None:
        msg = f"Unknown model: {model_name}"
        raise ValueError(msg)
    return model_cls(params)
