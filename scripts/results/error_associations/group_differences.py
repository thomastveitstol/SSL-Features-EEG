import dataclasses
import enum
import os
import re
from pathlib import Path
from typing import Optional, Dict

import numpy
import pandas
import yaml
from progressbar import progressbar

from elecssl.data.paths import get_results_dir
from elecssl.data.results_analysis import get_successful_runs, get_input_and_target_freq_bands, higher_is_better, \
    is_better

# ---------------
# Convenient classes
# ---------------
@dataclasses.dataclass(frozen=True)
class _Model:
    path: Path
    val_performance: float
    error_association: Dict[str, Dict[str, float]]


class ErrorType(enum.Enum):
    DIFF = "difference"
    RATIO = "ratio"


# ---------------
# Convenient functions
# ---------------
def _select_relevant_runs(results_dir, *, input_freq_band, target_freq_band):
    """Return only the runs with requested input and target frequency bands"""
    # Get successful only
    _successful_runs = get_successful_runs(results_dir)

    # Get the runs with requested input and target frequency bands
    runs_to_use = []
    for run in _successful_runs:
        # Checking config file
        with open(results_dir / run / "config.yml") as file:
            config = yaml.safe_load(file)
        in_band, target_band = get_input_and_target_freq_bands(config)

        # Maybe add the run for analysis
        if in_band == input_freq_band and target_band == target_freq_band:
            runs_to_use.append(run)

    return tuple(runs_to_use)


def _get_val_score_and_epoch(path, main_metric, balance_validation_performance):
    # -------------
    # Input check
    # -------------
    if not isinstance(balance_validation_performance, bool):
        raise TypeError(f"Expected argument 'balance_validation_performance' to be boolean, but found "
                        f"{type(balance_validation_performance)}")

    # -------------
    # Get the best epoch, as evaluated on the validation set
    # -------------
    if balance_validation_performance:
        val_df_path = path / "sub_groups_plots" / "dataset_name" / main_metric / f"val_{main_metric}.csv"
        val_df = pandas.read_csv(val_df_path)

        val_performances = numpy.mean(val_df.values, axis=-1)

        # Get the best performance and its epoch
        if higher_is_better(metric=main_metric):
            val_idx = numpy.argmax(val_performances)
        else:
            val_idx = numpy.argmin(val_performances)

        # Currently, we only actually need the 'main_metric'
        val_metric = val_performances[val_idx]

    else:
        # Load the dataframe of the validation performances
        val_df = pandas.read_csv(path / "val_history_metrics.csv")

        # Get the best performance and its epoch
        if higher_is_better(metric=main_metric):
            val_idx = numpy.argmax(val_df[main_metric])
        else:
            val_idx = numpy.argmin(val_df[main_metric])
        val_metric = val_df[main_metric][val_idx]

    # Return the val score and epoch
    return val_metric, val_idx


def _has_test_results(fold_path, error_type: ErrorType, variable):
    # Create path
    path = fold_path / "error_associations" / error_type.value / variable

    # Check if test results are available
    return any(file[:4] == "test" for file in os.listdir(path))


def _get_group_error_associations(fold_path, *, error_type, variable, dataset, epoch):
    # Create path
    path = fold_path / "error_associations" / error_type.value / variable

    # The file names are supposed to be 'test_{metric}_{group}' (although 'group' can be a lot)
    errors: Dict[str, Dict[str, float]] = {}
    csv_file_names = (file_name for file_name in os.listdir(path)
                      if file_name[:4] == "test" and file_name.endswith(".csv"))
    for file_name in csv_file_names:
        # Read file
        df = pandas.read_csv(path / file_name)

        # Get metric and group name
        metric = "_".join(file_name.split(sep="_")[1:-1])
        group = re.split(r"[_.]", file_name)[-2]

        # Add association for the requested dataset
        if metric not in errors:
            errors[metric] = {}
        errors[metric][group] = df[dataset][epoch]

    return errors


# ---------------
# Main functions to use
# ---------------
def _get_errors_associations(results_dir, *, input_freq_band, target_freq_band, variable, error_type, main_metric,
                             dataset, balance_validation_performance):
    # Select runs
    runs = _select_relevant_runs(results_dir, input_freq_band=input_freq_band, target_freq_band=target_freq_band)

    # -------------
    # Loop through runs
    # -------------
    best_model: Optional[_Model] = None
    for run in progressbar(runs, prefix="Run ", redirect_stdout=True):
        run_path = results_dir / run

        folds = (path for path in os.listdir(run_path) if path[:5] == "Fold_")
        for fold in folds:
            fold_path = run_path / fold

            # As per now, I'm restricting the estimates to be on test data only. I'm also assuming a
            # leave-one-dataset-out experiment, so todo: change here if it the assumption becomes wrong
            if not _has_test_results(fold_path=fold_path, error_type=error_type, variable=variable):
                continue

            # Getting validation score and epoch
            val_score, epoch = _get_val_score_and_epoch(path=fold_path, main_metric=main_metric,
                                                        balance_validation_performance=balance_validation_performance)

            # Maybe update the error associations of interest
            if best_model is None or is_better(metric=main_metric, old_performance=best_model.val_performance,
                                               new_performance=val_score):
                group_error_associations = _get_group_error_associations(
                    fold_path=fold_path, error_type=error_type, variable=variable, epoch=epoch, dataset=dataset
                )

                best_model = _Model(
                    path=fold_path, val_performance=val_score, error_association=group_error_associations
                )

    # -------------
    # Print results
    # -------------
    print(f"{' Error associations ':=^30}")
    print(f"Path: {best_model.path}")
    for metric, groups in best_model.error_association.items():
        for group, association in groups.items():
            print(f"{metric} ({_GROUP_NAMES[variable][group]}): {association}")


_GROUP_NAMES = {"clinical_status": {"0": "AD", "1": "FTD", "2": "CG",
                                    "0-1": "AD-FTD", "0-2": "AD-CG", "1-2": "FTD-CG"},
                "sex": {"0": "Male", "1": "Female",
                        "0-1": "Male-Female"}}


def main():
    # --------------
    # Hyperparameters
    # --------------
    input_freq_band = "alpha"
    target_freq_band = "theta"
    variable = "clinical_status"
    error_type = ErrorType.RATIO
    dataset = "All"
    main_metric = "mae"
    balance_validation_performance = False

    # --------------
    # Do analysis
    # --------------
    _get_errors_associations(
        results_dir=get_results_dir(), input_freq_band=input_freq_band, target_freq_band=target_freq_band,
        variable=variable, error_type=error_type, dataset=dataset, main_metric=main_metric,
        balance_validation_performance=balance_validation_performance
    )


if __name__ == "__main__":
    main()
