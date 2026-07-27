import argparse

import matplotlib.pyplot as plt
import numpy as np

from cli import tools
from cli.sample import Additive, Sample

from . import add_model_args, load_or_train_model


def add_args(parser: argparse.ArgumentParser):
    add_model_args(parser)

    parser.add_argument(
        "--train",
        type=str,
        help="Train dataset. If not given, the model will be loaded if supported",
        required=False,
    )
    parser.add_argument(
        "--surfactant",
        "-s",
        type=str,
        help="Surfactant SMILES string.",
        required=True,
    )
    parser.add_argument(
        "--additive",
        "-a",
        type=str,
        help="Additive SMILES string.",
    )
    parser.add_argument(
        "--additive-conc",
        "-c",
        type=float,
        help="Additive concentration.",
    )


def _plot_ceteris_paribus(
    smiles: str,
    feature: str,
    features_val: np.array,
    predictions: np.array,
):
    plt.figure(figsize=(8, 6))
    plt.plot(features_val, predictions, marker="o")
    plt.title(f"Ceteris Paribus Profile for {smiles}\nFeature: {feature}")
    plt.xlabel(f"{feature}")
    plt.ylabel("Predicted pCMC")
    plt.grid()
    plt.show()


def run(args: argparse.Namespace) -> None:
    try:
        model, features = load_or_train_model(args)
    except ValueError as e:
        print(f"Error: {e}")
        return

    # TODO: Implement additive_conc, and additive (Will need a list of additives?)

    temperatures = np.linspace(0, 100, num=20)

    samples_list = []
    for tempc in temperatures:
        sample = Sample(
            surfactant_smiles=args.surfactant,
            temperature=tempc,
            additive=Additive(
                smiles=args.additive,
                concentration=args.additive_conc or 0.0,
            )
            if args.additive
            else None,
        )
        samples_list.append(sample)

    samples = tools.samples_to_df(samples_list, features=features)
    prediction = model.predict(samples)

    _plot_ceteris_paribus(
        smiles=args.surfactant,
        feature="Temperature (C)",
        features_val=temperatures,
        predictions=prediction,
    )
