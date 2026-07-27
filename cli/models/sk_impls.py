import re

import lightgbm as lgb
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from .sk import SklearnModel


def inplace_one_hot_encode(df: pd.DataFrame, column: str) -> list[str]:
    one_hot = pd.get_dummies(df[column], prefix=column, dummy_na=True)
    df.drop(column, axis=1, inplace=True)
    df[one_hot.columns] = one_hot
    return one_hot.columns.tolist()


def _initial_dataset_setup(surfactants_db_agg_fp: pd.DataFrame):
    feature_cols = surfactants_db_agg_fp.columns.tolist()

    # Drop smiles and additive_smiles (unusable as strings)
    feature_cols.remove("surfactant_smiles")
    feature_cols.remove("additive_smiles")

    # one-hot encode: "surfactant_type"
    if "surfactant_type" in feature_cols:
        feature_cols += inplace_one_hot_encode(surfactants_db_agg_fp, "surfactant_type")
        feature_cols.remove("surfactant_type")

    return surfactants_db_agg_fp[feature_cols]


SKLEARN_SUPPORTED_FEATURES = {
    "chen",
    "estate",
    "expert",
    "morgan",
    "atompair",
    "autocorr",
    "avalon",
    "bcut2d",
    "e3fp",
    "ecfp",
    "electroshape",
    "erg",
    "functionalgroups",
    "getaway",
    "ghosecrippen",
    "klekotaroth",
    "laggner",
    "layered",
    "lingo",
    "maccs",
    "map",
    "mhfp",
    "mordred",
    "morse",
    "mqns",
    "pattern",
    "pharmacophore",
    "physiochemicalproperties",
    "pubchem",
    "rdf",
    "rdkit",
    "rdkit2ddescriptors",
    "secfp",
    "topologicaltorsion",
    "usr",
    "usrcat",
    "vsa",
    "whim",
}


def create_random_forest_model(params):
    max_depth = int(params.get("max_depth", 17))
    n_estimators = int(params.get("n_estimators", 20))

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=42,
        max_depth=max_depth,
    )
    pipeline = Pipeline(
        steps=[
            ("preprocess", FunctionTransformer(_initial_dataset_setup)),
            ("model", model),
        ],
    )
    return SklearnModel(pipeline, supported_features=SKLEARN_SUPPORTED_FEATURES)


def create_lgbm_model(params):
    max_depth = int(params.get("max_depth", -1))
    n_estimators = int(params.get("n_estimators", 100))

    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
    )
    pipeline = Pipeline(
        steps=[
            ("preprocess", FunctionTransformer(_initial_dataset_setup)),
            ("model", model),
        ],
    )
    return SklearnModel(pipeline, supported_features=SKLEARN_SUPPORTED_FEATURES)
