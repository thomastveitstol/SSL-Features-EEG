import os
from typing import Any, Dict, List

import yaml

from elecssl.data.data_preparation.band_pass_filter import BandPass
from elecssl.data.datasets.dataset_base import DatasetInfo, OcularState
from elecssl.data.datasets.getter import get_dataset


def main():
    # ---------------
    # Read config file
    # ---------------
    with open(os.path.join(os.path.dirname(__file__), "config_files", "band_pass.yml")) as file:
        config: Dict[str, Any] = yaml.safe_load(file)

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


if __name__ == "__main__":
    main()
