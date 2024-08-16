import os
from typing import List, Any, Dict

import yaml

from elecssl.data.datasets.dataset_base import OcularState
from elecssl.data.datasets.getter import get_dataset
from elecssl.data.feature_computations.band_power import compute_band_powers, DatasetInfo
from elecssl.data.paths import get_eeg_features_storage_path


def main():
    # ---------------
    # Get all info
    # ---------------
    # Read config file
    with open(os.path.join(os.path.dirname(__file__), "config_files", "band_power.yml")) as file:
        config: Dict[str, Any] = yaml.safe_load(file)

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
    band_powers = compute_band_powers(datasets=tuple(datasets), frequency_bands=config["FrequencyBands"],
                                      aggregation_method=config["AggregationMethod"])

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
        os.mkdir(folder)

        # Save config file
        with open(folder / "config.yml", "w") as file:
            yaml.safe_dump(config, file)

        # Save the feature matrix
        band_powers[["Dataset", "Subject-ID", feature]].to_csv((folder / feature_name).with_suffix(".csv"), index=False)



if __name__ == "__main__":
    main()
