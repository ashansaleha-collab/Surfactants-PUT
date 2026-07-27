import pandas as pd


def do_type_conversions(surfactants_db: pd.DataFrame):
    surfactants_db = surfactants_db.copy()

    # Drop "pH" and "Conc_A" since they are only very rarely filled
    surfactants_db = surfactants_db.drop(columns=["pH"])

    # Drop rows with NA's
    data_length = len(surfactants_db)
    surfactants_db = surfactants_db.dropna(
        subset=[
            "Tail (C number)",
            "Molecular_Weight",
            "Surfactant_Type",
        ],
    )
    print(f"Dropped {data_length - len(surfactants_db)} rows with NA's")

    # Tail (C number) -> int|None
    surfactants_db["Tail (C number)"] = pd.to_numeric(
        surfactants_db["Tail (C number)"],
    ).astype("Int16")

    # "Surfactant_Type" to categorical
    surfactants_db["Surfactant_Type"] = surfactants_db["Surfactant_Type"].astype(
        "category",
    )

    # Drop counterion from SMILES (separated by ".", keep only first part)
    # FIXME: There are some cases where there is another part of the molecule after
    # the counterion!
    surfactants_db["SMILES_no_counterion"] = (
        surfactants_db["SMILES"].str.split(".").str[0]
    )

    # CMC, pCMC, Temp_Celsius, to float|None
    surfactants_db["CMC"] = pd.to_numeric(
        surfactants_db["CMC"],
        errors="coerce",
    )
    surfactants_db["pCMC"] = pd.to_numeric(
        surfactants_db["pCMC"],
        errors="coerce",
    )
    surfactants_db["Conc. A."] = pd.to_numeric(
        surfactants_db["Conc. A."].replace("UNK", pd.NA),
        errors="coerce",
    )
    surfactants_db["Temp_Celsius"] = pd.to_numeric(
        surfactants_db["Temp_Celsius"].replace("UNK", pd.NA),
        errors="coerce",
    )

    # Use pH average for missing values, similarly for temperature
    # avg_ph = surfactants_db["pH"].mean()
    # surfactants_db["pH"] = surfactants_db["pH"].fillna(avg_ph)
    avg_temp = surfactants_db["Temp_Celsius"].mean()
    surfactants_db["Temp_Celsius"] = surfactants_db["Temp_Celsius"].fillna(avg_temp)

    # Drop null Conc.A if Additives is not null
    surfactants_db_additives_nonnull_but_concentration_null = surfactants_db[
        surfactants_db["Additives"].notna() & ~surfactants_db["Conc. A."].notna()
    ]
    n_dropped = len(surfactants_db_additives_nonnull_but_concentration_null)
    surfactants_db = surfactants_db.drop(
        surfactants_db_additives_nonnull_but_concentration_null.index,
    )
    print(f"Dropped {n_dropped} rows with null Conc.A but non-null Additives")
    return surfactants_db


def aggregate(surfactants_db: pd.DataFrame):
    # group by SMILES, aggregate numerical values by mean, take first
    # of the rest (these are derived from the SMILES anyway)
    dfa = (
        surfactants_db.groupby(
            [
                "SMILES",
                "Counterion",
                "Additives",
                "Conc. A.",
                # "pH",
                "Temp_Celsius",
            ],
            observed=True,
            dropna=False,
        )
        .agg(
            {
                # TODO: Include tails/heads here?
                "SMILES_no_counterion": "first",
                "Tail (C number)": "first",
                "Surfactant_Type": "first",
                "Molecular_Weight": "first",
                "CMC": ["mean", "std"],
                "pCMC": ["mean", "std"],
            },
        )
        .reset_index()
    )
    # flatten multiindex columns
    dfa.columns = ["_".join(col).strip("_") for col in dfa.columns.values]
    # drop "_first" suffix
    dfa = dfa.rename(
        columns={
            col: col.replace("_first", "")
            for col in dfa.columns
            if col.endswith("_first")
        },
    )
    dfa = dfa.rename(
        columns={
            "CMC_mean": "CMC",
            "pCMC_mean": "pCMC",
        },
    )
    # replace std NaN's with 0 (only one sample)
    dfa["CMC_std"] = dfa["CMC_std"].fillna(0)
    dfa["pCMC_std"] = dfa["pCMC_std"].fillna(0)
    return dfa


def preprocess(surfactants_db: pd.DataFrame) -> pd.DataFrame:
    surfactants_db = do_type_conversions(surfactants_db)
    surfactants_db = aggregate(surfactants_db)
    print(f"Dataset size after aggregation: {len(surfactants_db)}")
    return surfactants_db
