# cli/commands/pca.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from cli import tools
from cli.datasets.list import create_dataset


ColorBy = Literal["pcmc", "temperature", "additive_concentration", "none"]


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--train",
        type=str,
        required=True,
        help="Dataset name to load (e.g. expert, paper4, everything)",
    )
    parser.add_argument(
        "--features",
        "-f",
        type=str,
        required=False,
        help="Comma-separated list of feature extractors (e.g. expert,chen,maccs)",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=10,
        help="Number of PCA components to compute (default: 10)",
    )
    parser.add_argument(
        "--color-by",
        type=str,
        default="pcmc",
        choices=["pcmc", "temperature", "additive_concentration", "none"],
        help="Column used to color the 2D scatter (default: pcmc)",
    )
    parser.add_argument(
        "--save-prefix",
        type=str,
        default=None,
        help="If set, save figures and CSVs with this path prefix (e.g. out/pca_run)",
    )
    parser.add_argument(
        "--no-standardize",
        action="store_true",
        help="Disable StandardScaler (by default, features are standardized).",
    )



def _parse_features(s: Optional[str]) -> list[str]:
    return [t.strip() for t in s.split(",") if t.strip()] if s else []


def _select_numeric_matrix(df: pd.DataFrame, *, drop_cols: list[str]) -> pd.DataFrame:
    keep = [c for c in df.columns if c not in drop_cols]
    X = df[keep]
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    return X[num_cols]


def _plot_explained_variance(pca: PCA, *, title_suffix: str = "", show: bool = True):
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    idx = np.arange(1, len(evr) + 1)

    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.bar(idx, evr)
    ax1.plot(idx, cum, marker="o")
    ax1.set_xlabel("Principal component")
    ax1.set_ylabel("Explained variance ratio / cumulative")
    ax1.set_title(f"PCA explained variance{title_suffix}")
    ax1.grid(True)
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def _plot_scatter(pc_df: pd.DataFrame, color_by: ColorBy, show: bool = True):
    if not {"PC1", "PC2"}.issubset(pc_df.columns):
        return None

    fig, ax = plt.subplots(figsize=(6, 6))
    if color_by == "none" or color_by not in pc_df.columns:
        ax.scatter(pc_df["PC1"], pc_df["PC2"], alpha=0.7)
    else:
        cvals = pc_df[color_by]
        sc = ax.scatter(pc_df["PC1"], pc_df["PC2"], c=cvals, alpha=0.8)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(color_by)

    ax.axhline(0, lw=0.5, ls="--")
    ax.axvline(0, lw=0.5, ls="--")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"PCA scatter (colored by: {color_by})")
    ax.grid(True)
    fig.tight_layout()
    if show:
        plt.show()
    return fig



def run(args: argparse.Namespace) -> None:
    features = _parse_features(args.features)
    ds = create_dataset(args.train)
    samples = tools.load_dataset(ds, features=features)  
    df = samples.df.copy()

    drop_cols = ["surfactant_smiles", "additive_smiles"]
    X = _select_numeric_matrix(df, drop_cols=drop_cols)

    if X.empty:
        print("No numeric columns available for PCA. Add feature extractors (e.g. -f expert,chen,maccs).")
        return

    X = X.replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy="mean")
    X_imp = pd.DataFrame(imputer.fit_transform(X), index=X.index, columns=X.columns)

    if args.no_standardize:
        X_std = X_imp.to_numpy(dtype=float)
    else:
        X_std = StandardScaler().fit_transform(X_imp)

    n_comp = min(args.n_components, X_std.shape[1])
    pca = PCA(n_components=n_comp, svd_solver="auto", random_state=42)
    Z = pca.fit_transform(X_std)

    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    print(f"PCA components: computed {n_comp} on {X_std.shape[0]} samples, {X_std.shape[1]} numeric features")
    for i, (e, c) in enumerate(zip(evr, cum), start=1):
        print(f"  PC{i:02d}: {e:6.3%}  | cumulative: {c:6.3%}")

    pc_cols = {f"PC{i+1}": Z[:, i] for i in range(n_comp)}
    pc_df = pd.DataFrame(pc_cols, index=X_imp.index)

    for col in ["pcmc", "temperature", "additive_concentration"]:
        if col in df.columns:
            pc_df[col] = df.loc[X_imp.index, col]

    ev_fig = _plot_explained_variance(pca, title_suffix=f"  (n={n_comp})", show=True)

    color_by: ColorBy = args.color_by
    scatter_fig = None
    if n_comp >= 2:
        scatter_fig = _plot_scatter(pc_df, color_by=color_by, show=True)

    if args.save_prefix:
        prefix = Path(args.save_prefix)
        prefix_parent = prefix.parent if prefix.suffix else prefix.parent
        prefix_parent.mkdir(parents=True, exist_ok=True)

        pc_df.to_csv(f"{prefix}_scores.csv", index=False)
        loadings = pd.DataFrame(
            pca.components_.T,
            index=X_imp.columns,
            columns=[f"PC{i+1}" for i in range(n_comp)],
        )
        loadings.to_csv(f"{prefix}_loadings.csv")

        if ev_fig is not None:
            ev_fig.savefig(f"{prefix}_variance.png", dpi=150, bbox_inches="tight")
        if scatter_fig is not None:
            scatter_fig.savefig(f"{prefix}_scatter_{color_by}.png", dpi=150, bbox_inches="tight")

        def top_feats(pc_name: str, k: int = 12):
            vec = loadings[pc_name].abs().sort_values(ascending=False).head(k)
            print(f"\nTop |{pc_name}| features:")
            for f, v in vec.items():
                print(f"  {f:40s} {v:8.4f}")

        if n_comp >= 1:
            top_feats("PC1")
        if n_comp >= 2:
            top_feats("PC2")

        print(
            f"\nSaved: {prefix}_scores.csv, {prefix}_loadings.csv"
            f"{', ' + str(prefix) + '_variance.png' if ev_fig is not None else ''}"
            f"{', ' + str(prefix) + f'_scatter_{color_by}.png' if scatter_fig is not None else ''}"
        )
