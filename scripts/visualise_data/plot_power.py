"""
Script for plotting the power distributions which are supposed to be used as pseudo-targets

From the plot, log-transforming the pseudo targets looks the best
"""
import numpy
import pandas
import seaborn
from matplotlib import pyplot

from elecssl.data.paths import get_eeg_features_storage_path


def _get_power_distributions(datasets, targets, log_transform):
    path = get_eeg_features_storage_path()
    _folder_paths = (path / target for target in targets)
    powers = {"Dataset": [], "Freq band": [], "Value": []}
    for target in targets:
        # Load the .csv file
        df = pandas.read_csv((path / target / target).with_suffix(".csv"))

        # Get the frequency band name
        band_name = target.split("_")[-2]

        # Add the pseudo target values for all subjects in the datasets of interest
        for dataset, target_value in zip(df["Dataset"], df[band_name]):
            if dataset not in datasets:
                continue

            powers["Dataset"].append(dataset)
            powers["Freq band"].append(band_name)
            powers["Value"].append(numpy.log10(target_value) if log_transform else target_value)

    return powers


def _get_above_threshold(datasets, targets, log_transform, thresholds):
    path = get_eeg_features_storage_path()
    _folder_paths = (path / target for target in targets)
    powers = {"Dataset": [], "Subject-ID": [], "Freq band": [], "Value": []}
    for target in targets:
        # Load the .csv file
        df = pandas.read_csv((path / target / target).with_suffix(".csv"))

        # Get the frequency band name
        band_name = target.split("_")[-2]

        # Add the pseudo target values for all subjects in the datasets of interest
        for dataset, subject_id, target_value in zip(df["Dataset"], df["Subject-ID"], df[band_name]):
            if dataset not in datasets:
                continue

            # Compute value
            value = numpy.log10(target_value) if log_transform else target_value

            # Maybe add the info
            if value > thresholds[target]:

                powers["Dataset"].append(dataset)
                powers["Subject-ID"].append(subject_id)
                powers["Freq band"].append(band_name)
                powers["Value"].append(numpy.log10(target_value) if log_transform else target_value)

    return powers


def main():
    # --------------
    # Make selections
    # --------------
    log_transform = True
    datasets = ("Wang", "LEMON", "DortmundVital")
    ocular_state = "eo"
    thresholds = {f"band_power_delta_{ocular_state}": 2, f"band_power_theta_{ocular_state}": 1.5,
                  f"band_power_alpha_{ocular_state}": 2, f"band_power_beta_{ocular_state}": 1.7,
                  f"band_power_gamma_{ocular_state}": 1.5}
    targets = tuple(thresholds)

    # --------------
    # Get the powers
    # --------------
    power = _get_power_distributions(datasets, targets, log_transform=log_transform)
    above_thresholds = _get_above_threshold(datasets, targets, log_transform, thresholds)
    for dataset_name, subject_id, freq_band in zip(above_thresholds["Dataset"], above_thresholds["Subject-ID"],
                                                   above_thresholds["Freq band"]):
        print(f"{dataset_name} ({freq_band}): {subject_id}")

    # --------------
    # Plot
    # --------------
    seaborn.violinplot(power, x="Freq band", y="Value", hue="Dataset")

    pyplot.show()


if __name__ == "__main__":
    main()
