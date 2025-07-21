"""
Script for comparing the different experiments
"""
import os
from pathlib import Path
from typing import Dict, List, Union, Sequence

import optuna
import pandas
import seaborn
from matplotlib import pyplot

from elecssl.data.paths import get_results_dir
from elecssl.data.results_analysis.utils import is_better
from elecssl.models.metrics import higher_is_better


_PRETTY_NAME = {
    "prediction_models": "Prediction\nmodels", "pretraining": "Pretraining", "simple_elecssl": "S. Elecssl",
    "multivariable_elecssl": "M. Elecssl", "multi_task": "Multi Task"
}


def _find_best_trial(trials: Sequence[optuna.Trial], eval_method):
    # assumes higher the better
    return max(trials, key=eval_method)


def _min_max(trial: optuna.Trial):
    return min(trial.values)


def _get_epoch(trial_path: Path, metric):
    """Hard-coded evaluation method"""
    candidate_epochs = []
    test_file_names = (name for name in os.listdir(trial_path) if name.startswith("test_epoch")
                       and name.endswith("pretext_history_predictions.csv"))
    for file_name in test_file_names:
        candidate_epochs.append(int(file_name.split("_")[2]))

    # Load the validation scores
    downstream_val_scores = pandas.read_csv(trial_path / "val_history_metrics.csv", usecols=[metric])
    pretext_val_scores = pandas.read_csv(trial_path / "val_pretext_history_metrics.csv", usecols=[metric])

    best_epoch = -1
    best_score = None
    for epoch in candidate_epochs:
        if higher_is_better(metric):
            score = min(downstream_val_scores[metric][epoch], pretext_val_scores[metric][epoch])
        else:
            score = max(downstream_val_scores[metric][epoch], pretext_val_scores[metric][epoch])

        if best_score is None or is_better(metric=metric, new_performance=score, old_performance=best_score):
            best_epoch = epoch
            best_score = score

    return best_epoch


def main():
    experiment_time = "2025-06-21_192244"
    study_names = ("prediction_models", "multi_task",)

    experiment_name = f"experiments_{experiment_time}"
    experiments_path = Path(get_results_dir() / experiment_name)
    metric = "r2_score"

    # -------------
    # Aggregate results in a dataframe. Currently, using validation score
    # -------------
    results: Dict[str, List[Union[str, float]]] = {"Performance": [], "Experiment": []}

    # Add features + linear regression
    linreg_df = pandas.read_csv(experiments_path / "ml_features" / "test_scores.csv")
    assert linreg_df[metric].shape[0] == 1

    results["Performance"].append(linreg_df[metric][0])
    results["Experiment"].append("Band power\n+ Lin. Reg.")

    # Add everything else
    for study_name in study_names:
        study_path = (experiments_path / study_name / f"{study_name}-study.db")
        study_storage = f"sqlite:///{study_path}"

        study = optuna.load_study(study_name=f"{study_name}-study", storage=study_storage)

        for trial in study.trials:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue

            # Get the test performance
            trial_path = experiments_path / study_name / f"hpo_trial_{trial.number}" / "split_0"

            if study_name != "multi_task":
                test_score = pandas.read_csv(trial_path / "test_history_metrics.csv", usecols=[metric])[metric].item()
            else:
                best_epoch = _get_epoch(trial_path=trial_path, metric=metric)
                test_score = pandas.read_csv(trial_path / f"test_epoch_{best_epoch}_downstream_history_metrics.csv",
                                             usecols=[metric])[metric].item()

            # Add test performance
            results["Performance"].append(test_score)
            results["Experiment"].append(_PRETTY_NAME[study_name])

    # Create df
    df = pandas.DataFrame(results)

    # -------------
    # Plotting
    # -------------
    fontsize = 12
    title_fontsize = fontsize + 3
    pyplot.figure(figsize=(7, 5))

    x = "Performance"
    y = "Experiment"
    seaborn.boxplot(df, x=x, y=y, linewidth=1.2, showfliers=False, fill=True)
    seaborn.stripplot(data=df, x=x, y=y, jitter=True, size=3, marker='o', alpha=0.8)

    # Cosmetics
    pyplot.xlim((-1, 1))

    pyplot.title("Comparison of experiments", fontsize=title_fontsize)
    pyplot.xticks(fontsize=fontsize)
    pyplot.yticks(rotation=30, fontsize=fontsize)

    pyplot.xlabel(x, fontsize=fontsize)
    pyplot.ylabel(y, fontsize=fontsize)

    # Theme (shading with grey)
    for i, _ in enumerate(set(df[y])):
        if i % 2 == 0:  # Shade alternate categories
            pyplot.axhspan(i - 0.5, i + 0.5, color="lightgrey", alpha=0.5)
    pyplot.grid()
    pyplot.tight_layout()

    pyplot.show()


if __name__ == "__main__":
    main()
