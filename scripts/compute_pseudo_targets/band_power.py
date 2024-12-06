import os
from datetime import date, datetime
from typing import List, Any, Dict

import yaml
from python_utils.time import epoch

from elecssl.data.datasets.dataset_base import OcularState, DatasetInfo
from elecssl.data.datasets.getter import get_dataset
from elecssl.data.feature_computations.band_power import compute_band_powers
from elecssl.data.paths import get_eeg_features_storage_path


def _single_ocular_state(config):
    # ---------------
    # Get all info
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
                        kwargs={"ocular_state": OcularState(config["OcularState"]),**info["kwargs"]})
        )

    # ---------------
    # Compute band powers
    # ---------------
    band_powers = compute_band_powers(
        datasets=tuple(datasets), frequency_bands=config["FrequencyBands"], verbose=config["verbose"],
        aggregation_method=config["AggregationMethod"], average_reference=config["AverageReference"],
        autoreject=config["Autoreject"], epochs=config["Epochs"], crop=config["Crop"]
    )

    # ---------------
    # Saving
    # ---------------
    folder_root_path = get_eeg_features_storage_path()
    for feature in band_powers.columns:
        if feature in ("Dataset", "Subject-ID"):
            # These are not features, so skipping them
            continue

        # Make directory
        feature_name = f"band_power_{feature}_{config["OcularState"].lower()}"
        folder = folder_root_path / feature_name
        try:
            os.mkdir(folder)
        except FileExistsError as e:
            if not config["IgnoreExistingFolder"]:
                raise e

        # Save config file (but remove info about the other frequency bands)
        freq_band_config = config.copy()
        del freq_band_config["FrequencyBands"]
        freq_band_config["FrequencyBand"] = {feature: config["FrequencyBands"][feature]}

        with open(folder / f"config_{date.today()}_{datetime.now().strftime('%H%M%S')}.yml", "w") as file:
            yaml.safe_dump(freq_band_config, file)

        # Save the feature matrix
        band_powers[["Dataset", "Subject-ID", feature]].to_csv((folder / feature_name).with_suffix(".csv"), index=False)


def main():
    # ---------------
    # Read config file
    # ---------------
    with open(os.path.join(os.path.dirname(__file__), "config_files", "band_power.yml")) as file:
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
