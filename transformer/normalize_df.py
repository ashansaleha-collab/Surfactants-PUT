import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame

# Format: smiles | temp | pCMC [, additive | conc_additives]


def load_normalize_expert_data(path="./sources/CMC_surfactants_database_v2.csv"):
    df = pd.read_csv(path)

    df = df[["SMILES", "Temp_Celsius", "pCMC"]].rename(columns={"SMILES": "smiles", "Temp_Celsius": "temp"})
    df["temp"] = df["temp"].replace("UNK", np.nan).str.replace(",", ".")
    df["temp"] = pd.to_numeric(df["temp"], errors="raise")
    df = df.dropna(subset=["pCMC"])
    df = df.dropna(subset=["temp"])
    return df


def load_normalize_expert_data_additives(path="./sources/CMC_surfactants_database_v2.csv"):
    df = pd.read_csv(path)

    df = df[["SMILES", "Temp_Celsius", "pCMC", "Additives", "Conc. A."]].rename(
        columns={"SMILES": "smiles", "Temp_Celsius": "temp", "Additives": "additive", "Conc. A.": "conc_additives"}
    )

    # Temperature normalization: keep NaNs only where temp truly missing
    df["temp"] = df["temp"].replace("UNK", np.nan).astype(str).str.replace(",", ".", regex=False)
    df["temp"] = pd.to_numeric(df["temp"], errors="coerce")

    # Additive name: normalize blanks to NaN
    if "additive" in df.columns:
        df["additive"] = df["additive"].replace("", np.nan)

    # Additive concentration: convert 'UNK' and blanks to NaN, keep as float
    df["conc_additives"] = (
        df["conc_additives"].replace("UNK", np.nan).replace("", np.nan).astype(str).str.replace(",", ".", regex=False)
    )
    df["conc_additives"] = pd.to_numeric(df["conc_additives"], errors="coerce")

    # Keep only rows with valid target and temperature; allow additive fields to be NaN
    df = df.dropna(subset=["pCMC"])
    df = df.dropna(subset=["temp"])
    return df


def load_normalize_paper_1(path="./sources/Data_paper_1.csv"):
    df = pd.read_csv(path)
    df = df.rename(
        columns={
            "T": "temp",
        }
    )
    df = df[["smiles", "temp", "pCMC"]]
    return df


def load_normalize_paper_4(path="./sources/Data_paper_4.csv"):
    df = pd.read_csv(path)
    df = df.rename(columns={"String": "smiles", "log CMC (uM)": "pCMC", "T_C": "temp"})
    df = df[["smiles", "temp", "pCMC"]]
    return df


def load_concat(
    expert_data=True,
    paper_1=True,
    paper_4=True,
):
    dfs = []
    if expert_data:
        dfs.append(load_normalize_expert_data())
    if paper_1:
        dfs.append(load_normalize_paper_1())
    if paper_4:
        dfs.append(load_normalize_paper_4())
    df = pd.concat(dfs, ignore_index=True)
    # Deduplicate by averaging targets; recompute pCMC from CMC where available
    df = aggregate_targets(df, key_cols=["smiles", "temp"])
    return df


def load_concat_additives(
    expert_data=True,
    paper_1=False,
    paper_4=False,
):
    """
    Concatenate datasets, preserving additive information when available.

    Returns a DataFrame with columns:
      - smiles (str)
      - temp (float)
      - pCMC (float)
      - additive (str or NaN)
      - conc_additives (float or NaN)
    """
    dfs: list[DataFrame] = []

    if expert_data:
        df_expert = load_normalize_expert_data_additives()
        # Ensure required columns exist and types are consistent
        df_expert = df_expert[["smiles", "temp", "pCMC", "additive", "conc_additives"]]
        dfs.append(df_expert)

    def _with_empty_additives(df_in: DataFrame) -> DataFrame:
        df_out = df_in.copy()
        df_out["additive"] = np.nan
        df_out["conc_additives"] = np.nan
        return df_out[["smiles", "temp", "pCMC", "additive", "conc_additives"]]

    if paper_1:
        dfs.append(_with_empty_additives(load_normalize_paper_1()))
    if paper_4:
        dfs.append(_with_empty_additives(load_normalize_paper_4()))

    if not dfs:
        raise ValueError("At least one dataset must be selected to concatenate")

    df = pd.concat(dfs, ignore_index=True)
    # Deduplicate by averaging targets; recompute pCMC from CMC where available
    df = aggregate_targets(df, key_cols=["smiles", "temp", "additive", "conc_additives"])
    return df


def aggregate_targets(df: DataFrame, key_cols: list[str]) -> DataFrame:

    # Build aggregation: first for non-targets, mean for targets
    agg: dict[str, str] = {}
    for col in df.columns:
        if col in key_cols:
            continue
        if col in ("CMC", "pCMC"):
            agg[col] = "mean"
        else:
            agg[col] = "first"

    grouped = df.groupby(key_cols, dropna=False).agg(agg).reset_index()

    # Recalculate pCMC from averaged CMC where available (CMC in mol/L)
    return grouped


if __name__ == "__main__":
    df = load_concat_additives()
    print(df.head())
    print(df.info())
