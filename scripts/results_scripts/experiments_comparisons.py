"""
Script for comparing the different experiments
"""
from pathlib import Path
from typing import Dict, List, Union

import optuna
import pandas
import seaborn
from matplotlib import pyplot

from elecssl.data.paths import get_results_dir


_PRETTY_NAME = {
    "prediction_models": "Prediction\nmodels", "pretraining": "Pretraining", "simple_elecssl": "S. Elecssl",
    "multivariable_elecssl": "M. Elecssl"
}


def main():
    experiment_time = "2025-04-01_171150"
    study_names = ("prediction_models", "pretraining", "simple_elecssl", "multivariable_elecssl")

    experiment_name = f"experiments_{experiment_time}"
    experiments_path = Path(get_results_dir() / experiment_name)

    # -------------
    # Aggregate results in a dataframe. Currently, using validation score
    # -------------
    results: Dict[str, List[Union[str, float]]] = {"Performance": [], "Experiment": []}

    # Add features + linear regression
    linreg_df = pandas.read_csv(experiments_path / "ml_features" / "val_score.csv")
    assert linreg_df["value"].shape[0] == 1

    results["Performance"].append(linreg_df["value"][0])
    results["Experiment"].append("Band power\n+ Lin. Reg.")

    # Add everything else
    for study_name in study_names:
        study_path = (experiments_path / study_name / f"{study_name}-study.db")
        study_storage = f"sqlite:///{study_path}"

        experiment_df = optuna.load_study(study_name=f"{study_name}-study", storage=study_storage).trials_dataframe()
        experiment_df = experiment_df[~experiment_df["value"].isnull()]

        # Using validation performance
        results["Performance"] .extend(experiment_df["value"].tolist())
        results["Experiment"].extend([_PRETTY_NAME[study_name]] * experiment_df.shape[0])

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
