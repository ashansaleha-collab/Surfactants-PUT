import argparse

from cli import storage, tools
from cli.datasets.list import create_dataset
from cli.models import Model
from cli.models.list import create_model


def parse_parameters(param_list: list[str] | None) -> dict[str, str]:
    params = dict[str, str]()
    if param_list is None:
        return params
    for param in param_list:
        if "=" not in param:
            msg = f"Invalid parameter: {param}, expected key=value format"
            raise ValueError(msg)
        key, value = param.split("=", 1)
        params[key] = value
    return params


def parse_comma_separated_list(list_str: str | None) -> list[str]:
    return [f.strip() for f in list_str.split(",") if f.strip()] if list_str else []


def create_model_from_args(args: argparse.Namespace) -> Model:
    model_params = parse_parameters(args.param)
    return create_model(args.model, model_params)


def add_model_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--model",
        "-M",
        type=str,
        required=True,
        help="Model to use for prediction.",
    )

    parser.add_argument(
        "--param",
        "-P",
        action="append",
        help="Model parameter in key=value format.",
    )

    parser.add_argument(
        "--features",
        "-f",
        type=str,
        help="Comma-separated list of feature extractors to use.",
        required=False,
    )


def print_test_result(result: tools.TestResult):
    print(
        f"Tested {result.n_samples} samples ({result.n_annotated} annotated).",
    )
    print(f"MAE: {result.mae:.2f}, R²: {result.r2:.2f}")


def print_table_row(
    *,
    features: list[str],
    model: str,
    model_params: dict[str, str],
    train_dataset: str,
    train_mae: float,
    train_r2: float,
    test_dataset: str,
    test_mae: float,
    test_r2: float,
):
    params_str = ",".join(f"{k}={v}" for k, v in model_params.items())
    if params_str != "":
        params_str = f" ({params_str})"
    features_str = ",".join(features)
    print(
        f"| {features_str:30} | {model}{params_str} | {train_dataset:20} | "
        f"{train_mae:7.2f} | {train_r2:5.2f} | {test_dataset:20} | {test_mae:7.2f} | {test_r2:5.2f} |",
    )


def load_or_train_model(args):
    """Load or train the model from command line args.

    Expected args:
    - args.features: comma-separated list of features
    - args.model: model name
    - args.param: model parameters
    - args.train: train dataset (optional)

    Returns:
    - model: the loaded or trained model
    - features: list of features used

    """
    try:
        model = create_model_from_args(args)
    except ValueError as e:
        msg = f"Error constructing model: {e}"
        raise ValueError(msg) from e

    features = tools.filter_features(model, parse_comma_separated_list(args.features))

    if args.train:
        print(f"Loading train dataset '{args.train}'...")
        train_ds = create_dataset(args.train)
        print(f"Training model '{args.model}'")
        train_samples = tools.load_dataset(
            train_ds,
            features=features,
        )
        model.train(train_samples)
        print(f"Using model '{args.model}' trained on '{args.train}' for prediction.")
    else:
        try:
            model_path = storage.model_path(args.model)
            model.load(model_path)
        except NotImplementedError as e:
            # FIXME: Train+predict?
            msg = f"Model '{args.model}' does not support loading. Please provide a training dataset using --train."
            raise ValueError(msg) from e
        except Exception as e:
            msg = f"Error loading model '{args.model}': {e}"
            raise ValueError(msg) from e

        print(f"Using model '{args.model}' from '{model_path}' for prediction.")

    return model, features
