import os
from typing import Any, Dict, List

import yaml

from elecssl.data.data_preparation.band_pass_filter import BandPass
from elecssl.data.datasets.dataset_base import DatasetInfo, OcularState
from elecssl.data.datasets.getter import get_dataset


def _single_ocular_state(config):
    # ---------------
    # Prepare info
    # ---------------
    # Get datasets and info
    datasets: List[DatasetInfo] = []
    for dataset_name, info in config["Datasets"].items():
        # Get the dataset
        dataset = get_dataset(dataset_name)

        # Make dataset info object
        subjects = dataset.get_subject_ids()
        if info["num_subjects"] != "all":
            subjects = subjects[:info["num_subjects"]]

        # Add dataset info
        datasets.append(
            DatasetInfo(dataset=dataset, subjects=subjects,
                        kwargs={"ocular_state": OcularState(config["OcularState"]), **info["kwargs"]})
        )

    # ---------------
    # Save the transformed input data
    # ---------------
    BandPass().save_data(datasets=tuple(datasets), config=config, save_data=True, plot_data=False)


def main():
    # ---------------
    # Read config file
    # ---------------
    # todo: unnecessary with a new script just for a new config file...
    with open(os.path.join(os.path.dirname(__file__), "config_files", "band_pass_add_dataset.yml")) as file:
        config: Dict[str, Any] = yaml.safe_load(file)

    # ---------------
    # Loop through the ocular states
    # ---------------
    for ocular_state, dataset_config in config["OcularStates"].items():
        # Fix config file for current ocular state
        ocular_state_config = config.copy()
        ocular_state_config["OcularState"] = ocular_state
        ocular_state_config["Datasets"] = dataset_config["Datasets"]
        del ocular_state_config["OcularStates"]

        # Create features for current ocular state
        _single_ocular_state(ocular_state_config)


if __name__ == "__main__":
    main()
