from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from .dummy import SklearnModel


def create_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
                ["additive_smiles"],
            ),
        ],
        remainder="passthrough",
    )


def create_knn_baseline_model(params):
    k = int(params.get("k", 5))

    preprocessor = create_preprocessor()
    knn_model = KNeighborsRegressor(n_neighbors=k)
    knn_pipeline = Pipeline(
        [
            (
                # surfactant_smiles to molecular descriptors placeholder
                # (use number of C's)
                "smiles_to_features",
                FunctionTransformer(
                    lambda X: X.assign(
                        surfactant_smiles=X["surfactant_smiles"].apply(
                            lambda s: s.count("C"),
                        ),
                    ).rename(columns={"surfactant_smiles": "num_carbons"}),
                ),
            ),
            # use only columns from Sample (num_carbons, temperature, additive_concentration)
            (
                "select_features",
                FunctionTransformer(
                    lambda X: X[
                        [
                            "num_carbons",
                            "temperature",
                            "additive_smiles",
                            "additive_concentration",
                        ]
                    ],
                ),
            ),
            ("preprocessor", preprocessor),
            ("dense", FunctionTransformer(lambda X: X.toarray())),
            ("imputer", KNNImputer(n_neighbors=k)),
            ("model", knn_model),
        ],
    )
    return SklearnModel(
        knn_pipeline,
        supported_features=["expert", "physiochemicalproperties", "chen"],
    )
