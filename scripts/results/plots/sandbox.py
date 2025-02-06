import seaborn
from matplotlib import pyplot

from elecssl.data.paths import get_results_dir
from elecssl.data.results_analysis import load_hpo_study


def main():
    study = load_hpo_study(get_results_dir() / "hpo_experiment_2025-01-31_163041")

    scores = tuple(trial.value for trial in study.trials)
    trial_number = tuple(trial.number for trial in study.trials)

    seaborn.scatterplot(x=trial_number, y=scores)
    pyplot.show()


if __name__ == "__main__":
    main()