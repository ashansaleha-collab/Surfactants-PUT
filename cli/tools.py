from dataclasses import dataclass

import pandas as pd
import plotly.express as px
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_validate, StratifiedKFold, train_test_split
import numpy as np
from sklearn.model_selection import ParameterGrid

from cli.models.list import create_model

from .data import AnnotatedSurfactantsDf, SurfactantsDf
from .datasets import Dataset
from .features.list import create_feature_extractor
from .models import Model
from .sample import Sample


@dataclass
class TestResult:
    n_samples: int
    n_annotated: int
    mae: float
    r2: float


def _plot_predicted_vs_actual(
    X: pd.DataFrame,
    y_pred: list[float],
    y_true: list[float],
):
    # X-add expert features
    X = cleanup_and_add_features(SurfactantsDf(X), features=["expert"]).df

    # scatterplot y_pred vs y_true, color = surfactant_type
    fig = px.scatter(
        x=y_true,
        y=y_pred,
        labels={"x": "Actual pCMC", "y": "Predicted pCMC"},
        hover_data={
            "surfactant_smiles": X["surfactant_smiles"],
            "temperature": X["temperature"],
            "additive_smiles": X["additive_smiles"],
            "additive_concentration": X["additive_concentration"],
        },
        # COLOR
        color=X["surfactant_type"],
        title="Predicted vs Actual pCMC",
    )

    # plot y=x line
    fig.add_shape(
        type="line",
        x0=min(y_true),
        y0=min(y_true),
        x1=max(y_true),
        y1=max(y_true),
        line=dict(color="Red", dash="dash"),
    )
    fig.show()


def test_dataset_on_samples(
    samples: AnnotatedSurfactantsDf,
    model: Model,
    *,
    plot: bool = False,
) -> TestResult:
    X, y_true = samples.to_xy()
    y_pred = model.predict(SurfactantsDf(X))

    if plot:
        _plot_predicted_vs_actual(X, y_pred, y_true)

    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return TestResult(
        n_samples=len(samples.df),
        n_annotated=len(y_true),
        mae=mae,
        r2=r2,
    )


def filter_features(model: Model, features: list[str]) -> list[str]:
    """Filter features by keeping only those supported by the model."""
    supported_features = [f for f in features if model.supports_feature_extractor(f)]
    not_supported = set(features) - set(supported_features)
    if not_supported:
        print(f"Warning: skipping features not supported by the model: {not_supported}")
    return supported_features


def test_dataset(dataset: Dataset, model: Model, features: list[str]) -> TestResult:
    features = filter_features(model, features)
    samples = load_dataset(dataset, features)
    return test_dataset_on_samples(samples, model)


def cleanup_and_add_features(
    samples: SurfactantsDf,
    *,
    features: list[str],
    drop_na: bool = True,
):
    is_annotated = "pcmc" in samples.df.columns

    # Select only relevant columns
    samples = SurfactantsDf(
        samples.df[
            [
                "surfactant_smiles",
                "temperature",
                "additive_smiles",
                "additive_concentration",
                *(["pcmc"] if is_annotated else []),
            ]
        ],
    )

    # Filter NAs in annotations
    if is_annotated and drop_na:
        samples_with_features = AnnotatedSurfactantsDf(
            samples.df[samples.df["pcmc"].notna()],
        )
    else:
        samples_with_features = SurfactantsDf(samples.df)

    # Add features
    for feature in features:
        extractor = create_feature_extractor(feature)
        feats = extractor.extract(samples)
        print(f"Feature extractor '{feature}' adds {feats.shape[1]} columns")
        # Merge by index
        samples_with_features = SurfactantsDf(
            samples_with_features.df.merge(
                feats,
                left_index=True,
                right_index=True,
            ),
        )

    return (
        AnnotatedSurfactantsDf(samples_with_features.df)
        if is_annotated and drop_na
        else samples_with_features
    )


def samples_to_df(
    samples: list[Sample],
    *,
    features: list[str],
) -> SurfactantsDf:
    """Convert a list of samples to a dataframe."""
    x_columns = dict[str, list]()

    x_columns["surfactant_smiles"] = pd.Series(
        [s.surfactant_smiles for s in samples],
        dtype=str,
    )
    x_columns["temperature"] = pd.Series(
        [s.temperature for s in samples],
        dtype="float",
    )
    x_columns["additive_smiles"] = pd.Series(
        [s.additive.smiles if s.additive is not None else None for s in samples],
        dtype=str,
    )
    x_columns["additive_concentration"] = pd.Series(
        [s.additive.concentration if s.additive is not None else None for s in samples],
        dtype="float",
    )
    df = pd.DataFrame(x_columns)
    return cleanup_and_add_features(SurfactantsDf(df), features=features)


def load_dataset(
    dataset: Dataset,
    features: list[str],
    *,
    drop_na: bool = True,
) -> AnnotatedSurfactantsDf:
    """Preprocess dataset.

    Filter NA annotations, add features.
    """
    print(f"Loading dataset '{type(dataset)}' with features: {features}")
    return cleanup_and_add_features(
        dataset.samples(),
        features=features,
        drop_na=drop_na,
    )


def dataset_to_x_y(dataset: Dataset, features: list[str]):
    """Convert a given dataset to (X, y) (sklearn format).

    This also drops rows with missing annotations.
    """
    return load_dataset(dataset, features).to_xy()


@dataclass
class CrossValidationFoldResult:
    train_mae: float
    train_r2: float
    test_mae: float
    test_r2: float


def cross_validate_dataset(
    dataset: Dataset,
    model: Model,
    *,
    features: list[str],
    n_folds: int = 5,
    random_state: int = 42,
):
    # our models --> sklearn
    class SklearnWrapper(BaseEstimator):
        def __init__(self, model: Model):
            self.model = model

        def fit(self, X: pd.DataFrame, y: list[float]):
            xy = X.copy()
            xy["pcmc"] = y
            self.model.train(AnnotatedSurfactantsDf(xy))

        def predict(self, X: pd.DataFrame) -> list[float]:
            return self.model.predict(SurfactantsDf(X))

    features = filter_features(model, features)

    X, y = dataset_to_x_y(dataset, features)
    print(f"Using {len(X)} dataset samples")

    sklearn_model = SklearnWrapper(model)
    scores = cross_validate(
        sklearn_model,
        X,
        y,
        cv=KFold(n_splits=n_folds, shuffle=True, random_state=random_state),
        scoring=["neg_mean_absolute_error", "r2"],
        return_train_score=True,
    )

    return [
        CrossValidationFoldResult(
            train_mae=-scores["train_neg_mean_absolute_error"][i],
            train_r2=scores["train_r2"][i],
            test_mae=-scores["test_neg_mean_absolute_error"][i],
            test_r2=scores["test_r2"][i],
        )
        for i in range(n_folds)
    ]

def cross_validate_hyperparam(
    dataset: Dataset,
    model: Model,
    model_name: str,
    original_params: dict[str, str],
    *,
    features: list[str],
    n_folds: int = 5,
    random_state: int = 42,
    val_split: float = 0.2,
    n_bins: int = 4,
) -> list[CrossValidationFoldResult]:
    """K-Fold CV using the model's own early-stopping capable `train` method.

    For each fold a fresh model instance is created via `model.__class__` with
    the same constructor parameters (if available through `_params`). The
    validation fold acts as the early-stopping validation set and the reported
    test metrics.
    """
    features = filter_features(model, features)
    X, y = dataset_to_x_y(dataset, features)
    y = np.array(y)
    print(f"Starting Nested K-Fold ({n_folds} folds) with internal tuning on {len(X)} samples")

    # Create bins for stratification (quartiles by default)
    bins = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")

    outer_cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    splits = outer_cv.split(X, bins)

    results: list[CrossValidationFoldResult] = []

    for fold_idx, (train_val_idx, test_idx) in enumerate(splits, start=1):
        print(f"Processing Fold {fold_idx}/{n_folds}...")

        # Create inner validation holdout from the outer training set
        stratify_vals = bins[train_val_idx]
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_split,
            stratify=stratify_vals,
            random_state=(random_state + fold_idx),
        )

        # Prepare DataFrames for model API
        X_train_val = X.iloc[train_val_idx].copy()
        X_train = X.iloc[train_idx].copy()
        X_val = X.iloc[val_idx].copy()
        X_test = X.iloc[test_idx].copy()

        y_train_val = y[train_val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]
        y_test = y[test_idx]

        df_train = X_train.copy()
        df_train["pcmc"] = y_train
        df_val = X_val.copy()
        df_val["pcmc"] = y_val
        df_train_val = X_train_val.copy()
        df_train_val["pcmc"] = y_train_val
        print(f"Training samples: {len(df_train)}, Validation samples: {len(df_val)}, Test samples: {len(X_test)}")

        # Perform grid search using provided grid or the model's default grid
        param_grid = model.get_param_grid()


        best_mae = float("inf")
        best_params = None
        for candidate in ParameterGrid(param_grid):
            # Instantiate candidate model with params if supported
            model_params = {**original_params, **candidate}
            print(f"Evaluating candidate params: {model_params}")

            candidate_model = create_model(model_name, params=model_params)

            # Train on inner training set and evaluate on validation set
            additional_params = candidate_model.train(AnnotatedSurfactantsDf(df_train), AnnotatedSurfactantsDf(df_val))
            val_preds = candidate_model.predict(SurfactantsDf(X_val))
            val_mae = mean_absolute_error(y_val, val_preds)

            print(f"Candidate params {candidate} achieved val MAE: {val_mae:.4f} (best: {best_mae:.4f})")
            if val_mae < best_mae:
                best_mae = val_mae
                best_params = candidate | (additional_params or {})

        bestest_params = {**original_params, **best_params}

        print(f"Best params for fold {fold_idx}: {bestest_params} with val MAE: {best_mae:.4f}")

        # Instantiate final model with chosen params and train on full outer train
        final_model = create_model(model_name, params=bestest_params)

        final_model.train(AnnotatedSurfactantsDf(df_train_val))

        # Evaluate on training set (full) and test fold
        train_preds = final_model.predict(SurfactantsDf(X_train_val))
        train_mae = mean_absolute_error(y_train_val, train_preds)
        train_r2 = r2_score(y_train_val, train_preds)

        test_preds = final_model.predict(SurfactantsDf(X_test))
        test_mae = mean_absolute_error(y_test, test_preds)
        test_r2 = r2_score(y_test, test_preds)

        print(f"Fold {fold_idx} results: Train MAE={train_mae:.4f}, Train R2={train_r2:.4f}, Test MAE={test_mae:.4f}, Test R2={test_r2:.4f}")

        results.append(
            CrossValidationFoldResult(
                train_mae=train_mae,
                train_r2=train_r2,
                test_mae=test_mae,
                test_r2=test_r2,
            )
        )

    return results
