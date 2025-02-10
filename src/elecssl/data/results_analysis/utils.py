import os
from pathlib import Path
from types import MappingProxyType
from typing import NamedTuple

import optuna
import pandas

from elecssl.data.datasets.dataset_base import OcularState


# ----------------
# Globals
# ----------------
PRETTY_NAME = MappingProxyType({
    "delta": "Delta",
    "theta": "Theta",
    "alpha": "Alpha",
    "beta": "Beta",
    "gamma": "Gamma"
})
FREQ_BAND_ORDER = ("Delta", "Theta", "Alpha", "Beta", "Gamma")


# ----------------
# Smaller convenient classes
# ----------------
class InOutOcularStates(NamedTuple):
    input_data: OcularState
    target: OcularState


# ----------------
# Smaller convenient functions
# ----------------
def higher_is_better(metric):
    if metric in ("pearson_r", "spearman_rho", "r2_score"):
        return True
    elif metric in ("mae", "mse", "mape"):
        return False
    else:
        raise ValueError(f"Metric {metric} not recognised")


def is_better(metric, *, old_performance, new_performance):
    # Input checks
    assert isinstance(metric, str), f"Expected metric to be string, but found {type(metric)}"
    assert isinstance(old_performance, float), f"Expected 'old_metrics' to be float, but found {type(old_performance)}"
    assert isinstance(new_performance, float), f"Expected 'new_metrics' to be float, but found {type(new_performance)}"

    # Return
    if higher_is_better(metric=metric):
        return new_performance > old_performance
    else:
        return new_performance < old_performance


# ----------------
# Functions for getting stuff
# ----------------
def load_hpo_study(name: str, path: Path):
    storage_url = f"sqlite:///{path / name}.db"
    return optuna.load_study(study_name=name, storage=storage_url)


def get_successful_regression_performance_runs(results_dir):
    # Get all folder names
    all_runs = os.listdir(results_dir)

    # Filter to get the successful ones only
    return tuple(run for run in all_runs if os.path.isfile(results_dir / run / "finished_successfully.txt")
                 and "regression_performance" in run)


def get_successful_runs(results_dir):
    # Get all folder names
    all_runs = os.listdir(results_dir)

    # Filter to get the successful ones only
    return tuple(run for run in all_runs if os.path.isfile(results_dir / run / "finished_successfully.txt"))


def get_input_and_target_freq_bands(config):
    # -----------
    # Get frequency band of target data
    # -----------
    target_freq_band = config["Training"]["target"].split("_")[-2]

    # -----------
    # Input frequency band
    # -----------
    input_freq_band = set()
    for dataset_details in config["Datasets"].values():
        # Get the target from the preprocessed version
        _pattern = "band_pass-"
        version = dataset_details["pre_processed_version"]
        i0 = version.find(_pattern) + len(_pattern)
        i1 = version.find("--", i0)

        # Add it to frequency bands found
        input_freq_band.add(version[i0:i1])

    # Should only be one
    if len(input_freq_band) != 1:
        raise ValueError(f"Expected a single frequency band for the inputs but found (N={len(input_freq_band)}): "
                         f"{input_freq_band}")

    # Return tuple
    return tuple(input_freq_band)[0], target_freq_band


def get_input_and_target_freq_ocular_states(config):
    # -----------
    # Ocular state of target data
    # -----------
    target_ocular_state = OcularState(config["Training"]["target"].split("_")[-1].upper())

    # -----------
    # Input ocular state
    # -----------
    input_ocular_state = set()
    for dataset_details in config["Datasets"].values():
        # Get the ocular state from the preprocessed version
        version = dataset_details["pre_processed_version"]

        # Add it to frequency bands found
        input_ocular_state.add(OcularState(version.split(sep="/")[0].split(sep="_")[-1].upper()))

    # Should only be one
    if len(input_ocular_state) != 1:
        raise ValueError(f"Expected a single ocular state for the inputs but found (N={len(input_ocular_state)}): "
                         f"{input_ocular_state}")

    # Return tuple
    return InOutOcularStates(input_data=tuple(input_ocular_state)[0], target=target_ocular_state)


def get_test_dataset_name(path):
    """Function for getting the name of the test dataset. A test is also made to ensure that the test set only contains
    one dataset"""
    # Load the test predictions
    test_df = pandas.read_csv(path / "test_history_predictions.csv")

    # Check the number of datasets in the test set
    if len(set(test_df["dataset"])) != 1:
        raise ValueError(f"Expected only one dataset to be present in the test set predictions, but that was not "
                         f"the case for the path {path}. Found {set(test_df['dataset'])}")

    # Return the dataset name
    dataset_name: str = test_df["dataset"][0]
    return dataset_name
