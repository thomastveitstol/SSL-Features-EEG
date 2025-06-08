"""
Script for plotting training or validation curves
"""
import enum
import os
from pathlib import Path

import pandas
import seaborn
import yaml
from matplotlib import pyplot

from elecssl.data.paths import get_results_dir


class _CurveMode(enum.Enum):
    TRAIN = "train"
    VAL = "val"


def _get_strategy(trial_path: Path):
    with open(trial_path / "hpc_config.yml") as file:
        config = yaml.safe_load(file)
    return config["Strategy"]["name"]


def main():
    curve_mode = _CurveMode.VAL
    experiment_time = "2025-06-08_100404"
    # "2025-06-07_210521"  # "2025-06-08_010835"  # "2025-06-08_033700"  # "2025-06-08_064801"
    pretext_metric = "pearson_r"
    downstream_metric = pretext_metric
    y_lim = (-1, 1)

    experiment_name = f"experiments_{experiment_time}"
    experiments_path = Path(get_results_dir() / experiment_name / "multi_task")


    # Loop through all trials
    _folders = (folder for folder in os.listdir(experiments_path) if os.path.isdir(experiments_path / folder)
                and folder.startswith("hpo_trial"))
    for trial_folder in _folders:
        # Hard-coding single split
        trial_path = experiments_path / trial_folder / "split_0"

        # Get curves
        try:
            pretext_scores = pandas.read_csv(trial_path / f"{curve_mode.value}_pretext_history_metrics.csv",
                                             usecols=[pretext_metric])[pretext_metric].tolist()
            downstream_scores = pandas.read_csv(trial_path / f"{curve_mode.value}_history_metrics.csv",
                                                usecols=[downstream_metric])[downstream_metric].tolist()
        except FileNotFoundError:
            continue

        scores = pandas.DataFrame(
            {"Score": pretext_scores + downstream_scores,
             f"Metric": [f"Pretext {pretext_metric}"] * len(pretext_scores)
                         + [f"Downstream {downstream_metric}"] * len(pretext_scores),
             f"Epoch": list(range(len(pretext_scores))) + list(range(len(downstream_scores)))})

        # Make plot
        pyplot.figure()
        seaborn.lineplot(data=scores, x="Epoch", y="Score", hue="Metric")
        pyplot.title(_get_strategy(Path(os.path.dirname(trial_path))))
        pyplot.ylim(y_lim)
        pyplot.grid()

    pyplot.show()


if __name__ == "__main__":
    main()
