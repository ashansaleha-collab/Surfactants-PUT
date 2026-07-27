import argparse

import pandas as pd

from cli import tools
from cli.data import SurfactantsDf
from cli.datasets.list import create_dataset
from cli.sample import Additive, Sample

from . import add_model_args, load_or_train_model


def add_args(parser: argparse.ArgumentParser):
    add_model_args(parser)

    parser.add_argument(
        "--train",
        type=str,
        help="Train dataset. If not given, the model will be loaded if supported.",
        required=False,
    )

    # Subparsers:
    # - single -s smiles -t temp [-a additive -c conc_a] - sample from cmdline
    # - missing [(-f dataset.csv)|(-d dataset)] - predict missing values from a given dataset

    sp = parser.add_subparsers(dest="subcommand", required=True)

    sp_single = sp.add_parser("single", help="Predict single sample from command line")
    sp_single.add_argument(
        "--surfactant",
        "-s",
        type=str,
        help="Surfactant SMILES string. Required if --dataset is not given.",
        required=False,
    )
    sp_single.add_argument(
        "--tempc",
        "-t",
        type=float,
        help="Temperature in Celsius. Required if --dataset is not given.",
        required=False,
    )
    sp_single.add_argument(
        "--additive",
        "-a",
        type=str,
        help="Additive SMILES string.",
    )
    sp_single.add_argument(
        "--additive-conc",
        "-c",
        type=float,
        help="Additive concentration.",
    )

    sp_missing = sp.add_parser(
        "missing",
        help="Predict missing pCMC values from a given dataset",
    )
    sp_missing.add_argument(
        "--file",
        "-f",
        type=str,
        help="Dataset file (CSV)",
    )
    sp_missing.add_argument(
        "--dataset",
        "-d",
        type=str,
        help="Dataset name",
    )


def _load_samples_from_csv(filepath: str, features: list[str]) -> SurfactantsDf:
    samples = pd.read_csv(filepath)
    return tools.load_dataset(SurfactantsDf(samples), features)


def _get_only_missing(samples: SurfactantsDf):
    filtered_df = (
        samples.df[samples.df["pcmc"].isna()]
        .reset_index(drop=True)
        .drop(columns=["pcmc"])
    )
    return SurfactantsDf(filtered_df)


def run(args: argparse.Namespace) -> None:
    try:
        model, features = load_or_train_model(args)
    except ValueError as e:
        print(f"Error: {e}")
        return

    match args.subcommand:
        case "single":
            samples = tools.samples_to_df(
                [
                    Sample(
                        surfactant_smiles=args.surfactant,
                        temperature=args.tempc,
                        additive=Additive(
                            smiles=args.additive,
                            concentration=args.additive_conc or 0.0,
                        )
                        if args.additive
                        else None,
                    ),
                ],
                features=features,
            )
        case "missing":
            if args.file:
                samples = _load_samples_from_csv(args.file, features)
            elif args.dataset:
                samples = tools.load_dataset(
                    create_dataset(args.dataset),
                    features=features,
                    drop_na=False,
                )
            else:
                print("Error: Either --file or --dataset must be provided.")
                return

            samples = _get_only_missing(samples)
            print(f"Found {len(samples.df)} samples with missing pCMC values.")

    prediction = model.predict(samples)

    # can print up to 10 predictions
    out_csv_file = "predictions.csv"
    if len(prediction) <= 10:
        for i, pred in enumerate(prediction):
            print(f"Sample {i + 1}: Predicted pCMC = {pred:.4f}")
        print(f"Results saved to {out_csv_file}.")
    else:
        print(
            f"Predicted pCMC for {len(prediction)} samples. Results saved to {out_csv_file}.",
        )

    # create output df (AnnotatedSurfactantDb)
    output_df = samples.df.copy()
    output_df["predicted_pcmc"] = prediction

    # Keep only relevant columns
    output_df = output_df[
        [
            "surfactant_smiles",
            "temperature",
            "additive_smiles",
            "additive_concentration",
            "predicted_pcmc",
        ]
    ]

    # write csv
    output_df.to_csv(out_csv_file, index=False)
