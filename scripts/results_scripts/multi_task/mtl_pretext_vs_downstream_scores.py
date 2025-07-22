"""
Script for plotting all pareto optimal solutions for all trials
"""
import os
from pathlib import Path

import optuna
import pandas
import seaborn
from matplotlib import pyplot

from elecssl.data.paths import get_results_dir
from elecssl.models.experiments.hpo_experiment import ExperimentType

_SELECTION_METRIC = {"pretext": "r2_score", "downstream": "r2_score"}  # Unelegant, but this was indeed used


def main():
    experiment_time = "2025-06-21_192244"
    experiment_name = f"experiments_{experiment_time}"
    experiments_path = Path(get_results_dir() / experiment_name)
    metrics_to_plot = {"pretext": "r2_score", "downstream": "r2_score"}

    study_type = ExperimentType.MULTI_TASK
    study_name = study_type.value  # type: ignore

    # --------------
    # Get performance scores for all trials
    # --------------
    study_path = experiments_path / study_name / f"{study_name}-study.db"
    study_storage = f"sqlite:///{study_path}"

    study = optuna.load_study(study_name=f"{study_name}-study", storage=study_storage)

    # Loop through trials
    results = {"Downstream": [], "Pretext": [], "Pseudo-target": []}
    for trial in study.trials:
        assert isinstance(trial, optuna.trial.FrozenTrial)
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue

        trial_path = experiments_path / study_name / f"hpo_trial_{trial.number}" / "split_0"

        # Loop through all test performance scores
        downstream_score_file_names = (
            file_name for file_name in os.listdir(trial_path)
            if file_name.startswith("test_epoch") and file_name.endswith("downstream_history_metrics.csv")
        )
        validation_scores = pandas.read_csv(
            trial_path / "val_pretext_history_metrics.csv",
            usecols=[metrics_to_plot["pretext"]])[metrics_to_plot["pretext"]]
        pseudo_target = trial.params["out_freq_band"].capitalize()
        for downstream_test_file_name in downstream_score_file_names:
            epoch = int(downstream_test_file_name.split("_")[2])

            # ---------------
            # Get downstream and pretext performance scores
            # ---------------
            # Downstream (from test set)
            downstream_score = pandas.read_csv(
                trial_path / downstream_test_file_name,
                usecols=[metrics_to_plot["downstream"]])[metrics_to_plot["downstream"]].item()
            results["Downstream"].append(downstream_score)

            # Pretext (from validation set)
            results["Pretext"].append(validation_scores[epoch].item())

            # Add the pseudo target
            results["Pseudo-target"].append(pseudo_target)
    df = pandas.DataFrame(results)
    print(df)

    # --------------
    # Plotting
    # --------------
    seaborn.scatterplot(df, x="Pretext", y="Downstream", hue="Pseudo-target")
    pyplot.show()


if __name__ == "__main__":
    main()
