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


_PRETTY_NAME = {"pearson_r": "Pearson's r", "spearman_rho": "Spearman's rho", "r2_score": r"$R^2$",
                "conc_cc": "Conc. CC", "explained_variance": "Explained variance"}
_FREQ_BAND_ORDER = ("Delta", "Theta", "Alpha", "Beta", "Gamma")


def main():
    experiment_time = "2025-06-21_192244"
    experiment_name = f"experiments_{experiment_time}"
    experiments_path = Path(get_results_dir() / experiment_name)
    metrics_to_plot = {"pretext": "r2_score", "downstream": "r2_score"}

    performance_thresholds = {"r2_score": (-0.3, 1), "explained_variance": (-0.3, 1), "spearman_rho": (-0.2, 1),
                              "pearson_r": (-0.2, 1), "conc_cc": (-0.2, 1)}
    threshold_d = performance_thresholds[metrics_to_plot["downstream"]]
    threshold_p = performance_thresholds[metrics_to_plot["pretext"]]

    study_type = ExperimentType.MULTI_TASK
    study_name = study_type.value  # type: ignore

    # --------------
    # Get performance scores for all trials
    # --------------
    study_path = experiments_path / study_name / f"{study_name}-study.db"
    study_storage = f"sqlite:///{study_path}"

    study = optuna.load_study(study_name=f"{study_name}-study", storage=study_storage)
    trials = study.trials
    pareto_optimal_trials = study.best_trials
    study = None

    # Loop through trials
    results = {"Downstream": [], "Pretext": [], "Pseudo-target": [], "Pareto-optimal": [], "Downstream truncated": [],
               "Pretext truncated": []}
    for trial in trials:
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
            # Get information such as downstream and pretext performance scores
            # ---------------
            # Downstream (from test set)
            downstream_score = pandas.read_csv(
                trial_path / downstream_test_file_name,
                usecols=[metrics_to_plot["downstream"]])[metrics_to_plot["downstream"]].item()
            if threshold_d[0] < downstream_score < threshold_d[1]:
                results["Downstream"].append(downstream_score)
                results["Downstream truncated"].append(False)
            else:
                results["Downstream truncated"].append(True)
                if downstream_score < threshold_d[0]:
                    results["Downstream"].append(threshold_d[0])
                elif downstream_score > threshold_d[1]:
                    results["Downstream"].append(threshold_d[1])
                else:
                    raise RuntimeError("This should never happen...")

            # Pretext (from validation set)
            validation_score = validation_scores[epoch].item()
            if threshold_p[0] < validation_score < threshold_p[1]:
                results["Pretext"].append(validation_score)
                results["Pretext truncated"].append(False)
            else:
                results["Pretext truncated"].append(True)
                if validation_score < threshold_p[0]:
                    results["Pretext"].append(threshold_p[0])
                elif validation_score > threshold_p[1]:
                    results["Pretext"].append(threshold_p[1])
                else:
                    raise RuntimeError("This should never happen...")

            # Add the pseudo target and if the trial was pareto optimal
            results["Pseudo-target"].append(pseudo_target)
            results["Pareto-optimal"].append(trial in pareto_optimal_trials)

    df = pandas.DataFrame(results)

    # --------------
    # Plotting
    # --------------
    fontsize = 12
    title_fontsize = fontsize + 3
    pyplot.figure(figsize=(7, 5))
    pyplot.grid()

    x = "Pretext"
    y = "Downstream"
    seaborn.scatterplot(df, x=x, y=y, hue="Pseudo-target", alpha=0.2, s=20, hue_order=_FREQ_BAND_ORDER)
    seaborn.scatterplot(df[df["Pareto-optimal"]], x=x, y=y, hue="Pseudo-target", s=100, marker="*",
                        edgecolor="black", legend=False, hue_order=_FREQ_BAND_ORDER)

    # Cosmetics
    pyplot.title("Downstream performance vs. pretext performance", fontsize=title_fontsize)
    pyplot.xticks(fontsize=fontsize)
    pyplot.yticks(fontsize=fontsize)
    pyplot.xlabel(f"{x} ({_PRETTY_NAME[metrics_to_plot[x.lower()]]})", fontsize=fontsize)
    pyplot.ylabel(f"{y} ({_PRETTY_NAME[metrics_to_plot[y.lower()]]})", fontsize=fontsize)

    x0 = threshold_p[0] - (threshold_p[1] - threshold_p[0]) * 0.02
    x1 = threshold_p[1] + (threshold_p[1] - threshold_p[0]) * 0.02
    y0 = threshold_d[0] - (threshold_d[1] - threshold_d[0]) * 0.02
    y1 = threshold_d[1] + (threshold_d[1] - threshold_d[0]) * 0.02

    pyplot.xlim((x0, x1))
    pyplot.ylim((y0, y1))

    pyplot.show()


if __name__ == "__main__":
    main()
