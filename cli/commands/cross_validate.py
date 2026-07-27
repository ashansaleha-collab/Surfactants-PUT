import argparse

from cli import tools
from cli.datasets.list import create_dataset

from . import (
    add_model_args,
    create_model_from_args,
    parse_comma_separated_list,
    parse_parameters,
    print_table_row,
)


def add_args(parser: argparse.ArgumentParser):
    add_model_args(parser)

    parser.add_argument(
        "--train",
        type=str,
        help="Dataset for cross-validation.",
        required=True,
    )


def run(args: argparse.Namespace) -> None:
    try:
        model = create_model_from_args(args)
    except ValueError as e:
        print(f"Error constructing model: {e}")
        return

    train_ds = create_dataset(args.train)

    features = parse_comma_separated_list(args.features)

    n_folds = 5
    cv_results = tools.cross_validate_hyperparam(
        train_ds,
        model,
        features=features,
        n_folds=n_folds,
        model_name=args.model,
        original_params=parse_parameters(args.param),
    )

    # Average results
    def avg_std(values: list[float]) -> tuple[float, float]:
        avg = sum(values) / len(values)
        variance = sum((v - avg) ** 2 for v in values) / len(values)
        stddev = variance**0.5
        return avg, stddev

    avg_train_mae, std_train_mae = avg_std([r.train_mae for r in cv_results])
    avg_train_r2, std_train_r2 = avg_std([r.train_r2 for r in cv_results])
    avg_test_mae, std_test_mae = avg_std([r.test_mae for r in cv_results])
    avg_test_r2, std_test_r2 = avg_std([r.test_r2 for r in cv_results])

    print(f"{n_folds}-fold cross validation results:")
    print(
        f"Train  MAE: {avg_train_mae:.2f} ± {std_train_mae:.2f}, R²: {avg_train_r2:.2f} ± {std_train_r2:.2f}",
    )
    print(
        f"Test   MAE: {avg_test_mae:.2f} ± {std_test_mae:.2f}, R²: {avg_test_r2:.2f} ± {std_test_r2:.2f}",
    )

    print_table_row(
        features=features,
        model=args.model,
        model_params=parse_parameters(args.param),
        train_dataset=args.train,
        train_mae=avg_train_mae,
        train_r2=avg_train_r2,
        test_dataset=f"({n_folds}-fold CV)",
        test_mae=avg_test_mae,
        test_r2=avg_test_r2,
    )
