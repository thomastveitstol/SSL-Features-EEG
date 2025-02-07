import seaborn
from matplotlib import pyplot

from elecssl.data.paths import get_results_dir
from elecssl.data.results_analysis import load_hpo_study
from elecssl.models.experiments.hpo_experiment import PredictionModelsHPO, PretrainHPO


def plot_studies():
    results_dir = get_results_dir()
    study_paths = (results_dir /  "elecssl" / "elecssl_hpo_experiment_2025-02-06_154222",
                   results_dir /  "pretraining" / "pretraining_hpo_experiment_2025-02-07_030752",
                   results_dir /  "prediction_models" / "prediction_models_hpo_experiment_2025-02-07_064413")

    for study_path in study_paths:
        study = load_hpo_study(name=f"{study_path.parts[-2]}-study", path=study_path)

        scores = tuple(trial.value for trial in study.trials)
        trial_number = tuple(trial.number for trial in study.trials)

        seaborn.scatterplot(x=trial_number, y=scores)
    pyplot.show()


def get_df():
    results_dir = get_results_dir()
    selection_metric = "r2_score"
    target_metrics = ("pearson_r", "spearman_rho", "r2_score", "mae", "mse", "mape")
    method = "mean"

    # Generate dataframes
    prediction_models_df = PredictionModelsHPO.generate_test_scores_df(
        path=results_dir / "prediction_models" / "prediction_models_hpo_experiment_2025-02-07_064413",
        selection_metric=selection_metric, target_metrics=target_metrics, method=method
    )
    print(prediction_models_df[["run", "test_r2_score"]])

    pretraining_df = PretrainHPO.generate_test_scores_df(
        path=results_dir / "pretraining" / "pretraining_hpo_experiment_2025-02-07_030752",
        selection_metric=selection_metric, target_metrics=target_metrics, method=method
    )
    print(pretraining_df[["run", "test_r2_score"]])


def main():
    get_df()


if __name__ == "__main__":
    main()