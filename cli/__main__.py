import argparse

from rdkit import rdBase

from cli.commands import cross_validate, predict, profile, train_test, pca


def main() -> None:
    rdBase.DisableLog("rdApp.error")

    parser = argparse.ArgumentParser(description="CLI for CMC prediction models.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    sp_cross_validate = subparsers.add_parser(
        "cross_validate",
        help="Perform cross-validation on a given dataset.",
    )
    cross_validate.add_args(sp_cross_validate)

    sp_predict = subparsers.add_parser(
        "predict",
        help="Predict pCMC for a given surfactant sample.",
    )
    predict.add_args(sp_predict)

    sp_profile = subparsers.add_parser(
        "profile",
        help="Show a ceteris paribus model profile for a given surfactant. Only temperature supported for now.",
    )
    profile.add_args(sp_profile)

    sp_train_test = subparsers.add_parser(
        "train_test",
        help="Train & test model on a given dataset.",
    )
    train_test.add_args(sp_train_test)

    sp_pca = subparsers.add_parser(
        "pca",
        help="Run PCA on a dataset.",
    )
    pca.add_args(sp_pca)

    args = parser.parse_args()

    match args.command:
        case "cross_validate":
            cross_validate.run(args)
        case "predict":
            predict.run(args)
        case "profile":
            profile.run(args)
        case "train_test":
            train_test.run(args)
        case "pca":
            pca.run(args)


main()
