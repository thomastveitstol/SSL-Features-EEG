import dataclasses
import os
from pathlib import Path
from typing import Dict, Optional

import numpy
import pandas
import yaml
from progressbar import progressbar

from elecssl.data.paths import get_results_dir
from elecssl.data.results_analysis import higher_is_better, get_test_dataset_name, get_input_and_target_freq_bands, \
    is_better, get_successful_runs


@dataclasses.dataclass(frozen=True)
class _Model:
    path: Path  # Path to the results
    val_performance: float
    test_performance: Dict[str, float]  # The metrics table (e.g., {"mse": 10.7, "mae": 3.3})



def _get_val_test_metrics(path, main_metric, balance_validation_performance):
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
        val_df = pandas.read_csv(os.path.join(path, "val_history_metrics.csv"))

        # Get the best performance and its epoch
        if higher_is_better(metric=main_metric):
            val_idx = numpy.argmax(val_df[main_metric])
        else:
            val_idx = numpy.argmin(val_df[main_metric])
        val_metric = val_df[main_metric][val_idx]

    # -------------
    # Get test performance (all metrics)
    # -------------
    test_df = pandas.read_csv(path / "test_history_metrics.csv")

    test_metrics = {metric: test_df[metric][val_idx] for metric in test_df.columns}

    # Return
    return val_metric, test_metrics, get_test_dataset_name(path)


def _get_best_performances(*, results_dir, main_metric, balance_validation_performance):
    best_models: Dict[str, Dict[str, Dict[str, Optional[_Model]]]] = {}

    for run in progressbar(get_successful_runs(results_dir), prefix="Run ", redirect_stdout=True):
        run_path = results_dir / run

        # Loop through the folds
        folds = (path for path in os.listdir(run_path) if path[:5] == "Fold_")
        for fold in folds:
            # I need to get the validation performance (for making model selection), test performance (in case the
            # current run and fold is best for this dataset), and the dataset name (to know which dataset the
            # metrics are for). I also need the frequency band for input and target data
            val_metric, test_metrics, test_dataset = _get_val_test_metrics(
                path=run_path / fold, main_metric=main_metric,
                balance_validation_performance=balance_validation_performance
            )

            # -------------
            # Get the frequency band of input and target
            # -------------
            # Load the config file
            with open(run_path / "config.yml", "r") as file:
                config = yaml.safe_load(file)

            input_freq_band, target_freq_band = get_input_and_target_freq_bands(config)

            # -------------
            # (Maybe) update best model
            # -------------
            if test_dataset not in best_models:
                best_models[test_dataset] = {}
            if input_freq_band not in best_models[test_dataset]:
                best_models[test_dataset][input_freq_band] = {}
            if target_freq_band not in best_models[test_dataset][input_freq_band]:
                best_models[test_dataset][input_freq_band][target_freq_band] = None

            old_model = best_models[test_dataset][input_freq_band][target_freq_band]
            if old_model is None or is_better(metric=main_metric,
                                              old_performance=old_model.val_performance,  # type: ignore
                                              new_performance=val_metric):
                best_models[test_dataset][input_freq_band][target_freq_band] = _Model(
                    path=run_path / fold, val_performance=val_metric, test_performance=test_metrics
                )

    # -------------
    # Print results
    # -------------
    print(f"{' Results ':=^30}")
    for dataset, input_freq_band_results in best_models.items():
        print(f"\n{f' {dataset} ':-^40}")
        for input_band, target_freq_band_results in input_freq_band_results.items():
            for target_band, model in target_freq_band_results.items():
                if model is None:
                    continue
                print(f"{input_band} -> {target_band}: {model.test_performance[main_metric]:.2f}")

def main():
    main_metric = "pearson_r"
    results_dir = get_results_dir()
    balance_validation_performance = False

    _get_best_performances(results_dir=results_dir, main_metric=main_metric,
                           balance_validation_performance=balance_validation_performance)


if __name__ == "__main__":
    main()
