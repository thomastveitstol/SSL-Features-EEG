"""
Script for plotting the power distributions which are supposed to be used as pseudo-targets

From the plot, log-transforming the pseudo targets looks the best
"""
import itertools
import os.path

import numpy
import pandas
import seaborn

from elecssl.data.paths import get_eeg_features_storage_path
from elecssl.data.results_analysis import PRETTY_NAME, FREQ_BAND_ORDER


def _get_power_distributions(datasets, targets, log_transform):
    path = get_eeg_features_storage_path()
    _folder_paths = (path / target for target in targets)
    powers = {"Dataset": [], "Freq band": [], "Subject-ID": [], "Value": []}
    for target in targets:
        # Load the .csv file
        df = pandas.read_csv((path / target / target).with_suffix(".csv"))

        # Get the frequency band name
        band_name = target.split("_")[-2]

        # Add the pseudo target values for all subjects in the datasets of interest
        for dataset, subject_id, target_value in zip(df["Dataset"], df["Subject-ID"], df[band_name]):
            if dataset not in datasets:
                continue

            powers["Dataset"].append(dataset)
            powers["Freq band"].append(PRETTY_NAME[band_name])
            powers["Subject-ID"].append(subject_id)
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


def _create_plots(*, datasets, diag_kind, dpi, height, log_transform, lower_kind, normalise, ocular_state, targets,
                  thresholds, upper_kind, print_above_thresholds):
    # --------------
    # Get the powers
    # --------------
    power = _get_power_distributions(datasets, targets, log_transform=log_transform)
    if print_above_thresholds:
        above_thresholds = _get_above_threshold(datasets, targets, log_transform, thresholds)
        for dataset_name, subject_id, freq_band, value in zip(above_thresholds["Dataset"], above_thresholds["Subject-ID"],
                                                              above_thresholds["Freq band"], above_thresholds["Value"]):
            print(f"{dataset_name} ({freq_band} = {value:.2f}): {subject_id}")

    # Convert to dataframe
    df = pandas.DataFrame(power)
    df = df.pivot(index=["Subject-ID", "Dataset"], columns="Freq band", values="Value")
    df.columns.name = None  # Remove the name "Freq band" from columns
    df.reset_index(inplace=True)
    df.sort_values(by="Dataset", inplace=True)

    # Maybe normalise
    if normalise:
        pseudo_targets = [col for col in df.columns if col not in ("Dataset", "Subject-ID")]
        df[pseudo_targets] = df[pseudo_targets].div(df[pseudo_targets].sum(axis=1), axis=0)
        print(df)

    # --------------
    # Plot
    # --------------
    pairplot_fig = seaborn.PairGrid(df, vars=FREQ_BAND_ORDER, hue="Dataset", height=height, hue_order=datasets)

    pairplot_fig.map_lower(**lower_kind)
    pairplot_fig.map_upper(**upper_kind)
    pairplot_fig.map_diag(**diag_kind)
    pairplot_fig.add_legend()

    fig_name = (f"power_distribution_{ocular_state}_log_{str(log_transform).lower()}_normalised_"
                f"{str(normalise).lower()}.png")
    save_path = os.path.join(os.path.dirname(__file__), fig_name)
    pairplot_fig.savefig(save_path, dpi=dpi)


def main():
    # --------------
    # Make selections
    # --------------
    datasets = ("DortmundVital", "LEMON", "Wang")
    ocular_state = "eo"
    thresholds = {f"band_power_delta_{ocular_state}": 4.5, f"band_power_theta_{ocular_state}": 4.5,
                  f"band_power_alpha_{ocular_state}": 4.5, f"band_power_beta_{ocular_state}": 4.5,
                  f"band_power_gamma_{ocular_state}": 4.5}
    targets = tuple(thresholds)

    print_above_thresholds = False
    lower_kind = {"func": seaborn.scatterplot, "hue_order": datasets, "s": 30, "alpha": 1}
    upper_kind = {"func": seaborn.kdeplot, "alpha": 1, "levels": 5}
    diag_kind = {"func": seaborn.kdeplot, "fill": True}
    height = 2.5
    dpi = 70

    # --------------
    # Create plots
    # --------------
    for log_transform, normalise in itertools.product((True, False), repeat=2):
        _create_plots(
            datasets=datasets, diag_kind=diag_kind, dpi=dpi, height=height, log_transform=log_transform,
            lower_kind=lower_kind, normalise=normalise, ocular_state=ocular_state, targets=targets, thresholds=thresholds,
            upper_kind=upper_kind, print_above_thresholds=print_above_thresholds
        )


if __name__ == "__main__":
    main()
