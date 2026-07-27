import argparse

from cli import storage, tools
from cli.data import SurfactantsDf, AnnotatedSurfactantsDf
from cli.datasets.list import create_dataset

import joblib

from . import (
    add_model_args,
    create_model_from_args,
    parse_comma_separated_list,
    parse_parameters,
    print_table_row,
    print_test_result,
)


def add_args(parser: argparse.ArgumentParser):
    add_model_args(parser)

    parser.add_argument(
        "--train",
        type=str,
        help="Train dataset. If not given, the model will be loaded and evaluated (needs --train first)",
        required=False,
    )

    parser.add_argument(
        "--test",
        type=str,
        help="Test dataset. If not given, but train is given, the model will only be trained.",
        required=False,
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot predicted vs actual (test only)",
        required=False,
    )


def _train_test_split(
    samples: AnnotatedSurfactantsDf,
    *,
    test_frac: float,
) -> tuple[AnnotatedSurfactantsDf, AnnotatedSurfactantsDf]:
    n_total = len(samples.df)
    n_test = int(n_total * test_frac)
    shuffled = samples.df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_samples = shuffled.iloc[:n_test]
    train_samples = shuffled.iloc[n_test:]
    return AnnotatedSurfactantsDf(train_samples), AnnotatedSurfactantsDf(test_samples)


def _load_samples(
    train: str | None,
    test: str | None,
    *,
    features: list[str],
) -> tuple[AnnotatedSurfactantsDf | None, AnnotatedSurfactantsDf | None]:
    if not train and not test:
        msg = "At least one of train or test dataset must be specified."
        raise ValueError(msg)

    if train and not test:
        print(f"load_samples: Using train dataset '{train}' for training only.")
        return tools.load_dataset(create_dataset(train), features=features), None

    if not train and test:
        print(f"load_samples: Using test dataset '{test}' for testing only.")
        return None, tools.load_dataset(create_dataset(test), features=features)

    assert train is not None  # noqa: S101
    assert test is not None  # noqa: S101

    # train == test -> load train, split into 20% test
    if train == test:
        frac = 0.2
        print(
            f"load_samples: Splitting dataset '{train}' into train/test ({frac * 100:.0f}% test).",
        )
        full_ds = tools.load_dataset(create_dataset(train), features=features)
        train_ds, test_ds = _train_test_split(full_ds, test_frac=frac)
        return train_ds, test_ds

    # train != test -> load both
    print(f"load_samples: Loading train dataset '{train}' and test dataset '{test}'.")
    train_ds = tools.load_dataset(create_dataset(train), features=features)
    test_ds = tools.load_dataset(create_dataset(test), features=features)
    return train_ds, test_ds


class ModelWrapper:
    def __init__(self, model, features):
        self.model = model
        self.features = features

    def predict(self, df):
        dff = tools.cleanup_and_add_features(
            SurfactantsDf(df),
            features=self.features,
            drop_na=False,
        )
        return self.model.predict(dff)


def run(args: argparse.Namespace) -> None:
    try:
        model = create_model_from_args(args)
    except ValueError as e:
        print(f"Error constructing model: {e}")
        return

    features = parse_comma_separated_list(args.features)
    features = tools.filter_features(model, features)
    try:
        train_samples, test_samples = _load_samples(
            args.train,
            args.test,
            features=features,
        )
    except ValueError as e:
        print(f"Error: {e}")
        return

    print(f"Train length: {len(train_samples.df) if train_samples else 0} samples")
    print(f"Test length: {len(test_samples.df) if test_samples else 0} samples")

    if train_samples:
        print(f"Training model '{args.model}'")
        model.train(train_samples)
        model_path = storage.ensure_model_path(args.model)
        print(f"Saving model '{args.model}' to '{model_path}'")
        try:
            joblib.dump(ModelWrapper(model, features), model_path / f"{args.model}.pkl")
        except NotImplementedError:
            print(
                f"Warning: Model '{args.model}' does not support saving. Skipping save step.",
            )

    if test_samples:
        if not train_samples:
            # !train & test - load model from file
            try:
                p = storage.model_path(args.model) / f"{args.model}.pkl"
                print(f"Loading model '{args.model}' from '{p}'")
                try:
                    model.load(p)
                except NotImplementedError:
                    print(
                        f"Error: Model '{args.model}' does not support loading. Cannot proceed with testing.",
                    )
                    return
            except Exception as e:
                print(f"Error loading model '{args.model}': {e}")
                return
        else:
            # train & test - can use local instance
            pass

        train_result = (
            tools.test_dataset_on_samples(train_samples, model)
            if train_samples
            else None
        )
        test_result = tools.test_dataset_on_samples(test_samples, model, plot=args.plot)
        print_test_result(test_result)

        train_ds_name = args.train if args.train else "N/A"
        test_ds_name = args.test if args.test else "N/A"
        if args.train == args.test:
            train_ds_name = train_ds_name + " (80%)"
            test_ds_name = test_ds_name + " (20%)"

        print_table_row(
            features=features,
            model=args.model,
            model_params=parse_parameters(args.param),
            train_dataset=train_ds_name,
            train_mae=train_result.mae if train_result is not None else float("nan"),
            train_r2=train_result.r2 if train_result is not None else float("nan"),
            test_dataset=test_ds_name,
            test_mae=test_result.mae,
            test_r2=test_result.r2,
        )
