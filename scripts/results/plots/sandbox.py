import seaborn
from matplotlib import pyplot

from elecssl.data.paths import get_results_dir
from elecssl.data.results_analysis import load_hpo_study


def main():
    results_dir = get_results_dir()
    study_paths = (results_dir /  "elecssl" / "elecssl_hpo_experiment_2025-02-06_132313",
                   results_dir /  "pretraining" / "pretraining_hpo_experiment_2025-02-06_133604",
                   results_dir /  "prediction_models" / "prediction_models_hpo_experiment_2025-02-06_134542")

    for study_path in study_paths:
        study = load_hpo_study(study_path)

        scores = tuple(trial.value for trial in study.trials)
        trial_number = tuple(trial.number for trial in study.trials)

        seaborn.scatterplot(x=trial_number, y=scores)
    pyplot.show()


if __name__ == "__main__":
    main()