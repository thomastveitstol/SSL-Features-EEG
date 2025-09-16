"""
Script for comparing the different experiments
"""
import os
from pathlib import Path
from typing import Dict, List, Union, Sequence, Literal

import optuna
import pandas
import seaborn
from matplotlib import pyplot
from matplotlib.axes import Axes

from elecssl.data.paths import get_results_dir
from elecssl.data.results_analysis.utils import is_better
from elecssl.models.metrics import higher_is_better

_PRETTY_NAME = {
    "prediction_models": "Prediction\nmodels", "pretraining": "Pre-training", "simple_elecssl": "S. supervised\nHPO",
    "multivariable_elecssl": "M. supervised\nHPO", "multi_task": "MTL", "r2_score": r"$R^2$",
    "explained_variance": "Explained variance", "conc_cc": "Conc. CC", "pearson_r": "Pearson's r",
    "spearman_rho": "Spearman's rho",
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

    assert best_epoch != -1

    return best_epoch


_SELECTION_METRIC = "r2_score"  # it makes no sense to change this because this was used during HPO


def main():
    experiment_time = "2025-06-23_104856"
    study_names = ("prediction_models", "pretraining", "simple_elecssl", "multivariable_elecssl", "multi_task")

    experiment_name = f"experiments_{experiment_time}"
    experiments_path = Path(get_results_dir() / experiment_name)
    metrics: Sequence[Literal["explained_variance", "r2_score", "conc_cc", "pearson_r", "spearman_rho"]] = \
        ("spearman_rho",)
    decimals_to_print = 2
    dpi = 300

    # -------------
    # Do analysis
    # -------------
    results: Dict[str, List[Union[str, float, bool, int]]] = {
        **{_PRETTY_NAME[metric]: [] for metric in metrics}, "Experiment": [], "Selected": [], "Trial": []}

    # Add features + linear regression
    linreg_df = pandas.read_csv(experiments_path / "ml_features" / "test_scores.csv")
    assert all(linreg_df[metric].shape[0] == 1 for metric in metrics)

    for metric in metrics:
        results[_PRETTY_NAME[metric]].append(linreg_df[metric][0])
    results["Experiment"].append("Band power\n+ Lin. Reg.")
    results["Selected"].append(True)
    results["Trial"].append(0)

    # --------------
    # Get performance scores for all trials
    # --------------
    for study_name in study_names:
        study_path = (experiments_path / study_name / f"{study_name}-study.db")
        study_storage = f"sqlite:///{study_path}"

        study = optuna.load_study(study_name=f"{study_name}-study", storage=study_storage)

        # --------------
        # Get the trial after model selection
        # --------------
        if study_name == "multi_task":
            pareto_optimal_trials = study.best_trials
            best_trial = _find_best_trial(pareto_optimal_trials, _min_max)
        else:
            best_trial = study.best_trial

        # --------------
        # And all other trials
        # --------------
        for trial in study.trials:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue

            # Get the test performance
            trial_path = experiments_path / study_name / f"hpo_trial_{trial.number}"

            if study_name == "multi_task":
                best_epoch = _get_epoch(trial_path=trial_path / "split_0", metric=_SELECTION_METRIC)
                test_score = pandas.read_csv(
                    trial_path / "split_0" / f"test_epoch_{best_epoch}_downstream_history_metrics.csv",
                    usecols=list(metrics))
            elif study_name in ("simple_elecssl", "multivariable_elecssl"):
                test_score = pandas.read_csv(trial_path / "test_scores.csv", usecols=list(metrics))
            else:
                test_score = pandas.read_csv(trial_path / "split_0" / "test_history_metrics.csv",
                                             usecols=list(metrics))

            # Add test performance
            for metric in metrics:
                results[_PRETTY_NAME[metric]].append(test_score[metric].item())
            results["Experiment"].append(_PRETTY_NAME[study_name])
            results["Selected"].append(trial == best_trial)
            results["Trial"].append(trial.number)

    # Create df
    df = pandas.DataFrame(results)

    # -------------
    # Plotting
    # -------------
    fontsize = 12
    title_fontsize = fontsize + 3
    fig, axes = pyplot.subplots(1, 2, figsize=(11, 5))  # a little hard-coded...

    print(df[df["Selected"]].round({_PRETTY_NAME[metric]: decimals_to_print for metric in metrics}))
    for i, (ax, metric) in enumerate(zip(axes, metrics)):
        x = _PRETTY_NAME[metric]
        y = "Experiment"
        seaborn.boxplot(df, x=x, y=y, linewidth=1.2, showfliers=False, fill=True, ax=ax)
        seaborn.stripplot(data=df, x=x, y=y, jitter=True, size=2, marker='o', alpha=0.5, ax=ax)
        seaborn.stripplot(data=df[df["Selected"]], x=x, y=y, dodge=True, size=9, alpha=1, marker="*", color="black",
                          ax=ax)

        # Cosmetics
        ax.set_xlim((-1, 1))

        ax.tick_params(axis="x", labelsize=fontsize)
        ax.tick_params(axis="y", rotation=30, labelsize=fontsize)

        ax.set_xlabel(x, fontsize=fontsize)
        ax.set_ylabel(y, fontsize=fontsize)

        if i != 0:  # Remove y-labels and ticks
            ax.set_yticks([])
            ax.set_ylabel("")

        # Theme (shading with grey)
        for j, _ in enumerate(set(df[y])):
            if j % 2 == 0:  # Shade alternate categories
                ax.axhspan(j - 0.5, j + 0.5, color="lightgrey", alpha=0.5)
        ax.grid()

    fig.suptitle("Comparison of experiments", fontsize=title_fontsize)
    pyplot.tight_layout()

    pyplot.savefig(Path(os.path.dirname(__file__)) / f"experiments_comparisons_{'_'.join(metrics)}.png", dpi=dpi)
    pyplot.show()


if __name__ == "__main__":
    main()
