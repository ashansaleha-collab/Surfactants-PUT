from typedframe import TypedDataFrame


class SurfactantsDf(TypedDataFrame):
    schema = {  # noqa: RUF012
        "surfactant_smiles": str,
        "temperature": float,  # in Celsius
        "additive_smiles": str,  # (or None)
        "additive_concentration": float,  # (or None if additive_smiles is None)
    }

    def __init__(self, df):
        super().__init__(df)

        if df["surfactant_smiles"].isnull().any():
            msg = "SurfactantsDf cannot have null values in 'surfactant_smiles' column."
            raise ValueError(msg)
        if df["temperature"].isnull().any():
            msg = "SurfactantsDf cannot have null values in 'temperature' column."
            raise ValueError(msg)
        df_additive_nonnull_but_concentration_null = df[
            (df["additive_smiles"].notnull()) & (df["additive_concentration"].isnull())
        ]
        if not df_additive_nonnull_but_concentration_null.empty:
            msg = (
                "SurfactantsDf cannot have null values in 'additive_concentration' "
                "when 'additive_smiles' is not null."
            )
            raise ValueError(msg)

    def annotated(self) -> "AnnotatedSurfactantsDf":
        """Return the annotated subset of this df."""
        return AnnotatedSurfactantsDf(self.df[self.df["pcmc"].notna()])


class AnnotatedSurfactantsDf(SurfactantsDf):
    schema = {  # noqa: RUF012
        "pcmc": float,
    }

    def __init__(self, df):
        super().__init__(df)

        # Check for nulls in pCMC
        if self.df["pcmc"].isnull().any():
            msg = "AnnotatedSurfactantsDf cannot have null values in 'pcmc' column."
            raise ValueError(msg)

    def to_xy(self):
        """Return this df split to X, y (for sklearn)."""
        X = self.df.drop(columns=["pcmc"])
        y = self.df["pcmc"].tolist()
        return X, y
