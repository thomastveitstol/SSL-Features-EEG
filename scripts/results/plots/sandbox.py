import pandas
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


def make_boxplots():
    # ---------------
    # Some design choices
    # ---------------
    # How to compute and aggregate
    selection_metric = "r2_score"
    target_metrics = ("pearson_r", "spearman_rho", "r2_score", "mae", "mse", "mape")
    method = "mean"

    # Plot details
    x = "test_pearson_r"
    y = "experiment_name"

    # Other stuff
    results_dir = get_results_dir()

    # ---------------
    # Analysis
    # ---------------
    # Generate dataframe
    prediction_models_df = PredictionModelsHPO.generate_test_scores_df(
        path=results_dir / "prediction_models" / "prediction_models_hpo_experiment_2025-02-07_064413",
        selection_metric=selection_metric, target_metrics=target_metrics, method=method, include_experiment_name=True
    )
    pretraining_df = PretrainHPO.generate_test_scores_df(
        path=results_dir / "pretraining" / "pretraining_hpo_experiment_2025-02-07_030752",
        selection_metric=selection_metric, target_metrics=target_metrics, method=method, include_experiment_name=True
    )
    df = pandas.concat((prediction_models_df, pretraining_df), axis="rows")
    df.reset_index(inplace=True)

    # Plotting
    seaborn.boxplot(data=df, x=x, y=y, linewidth=1.2, dodge=True, showfliers=False, fill=False)
    seaborn.stripplot(data=df, x=x, y=y, jitter=True, dodge=True, size=3, alpha=0.5, marker='o')

    pyplot.show()


def main():
    make_boxplots()


if __name__ == "__main__":
    main()