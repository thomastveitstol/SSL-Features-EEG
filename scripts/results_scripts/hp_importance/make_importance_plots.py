import os
from pathlib import Path
from typing import Literal

import pandas
import seaborn
from matplotlib import pyplot


_STUDIES = Literal["prediction_models", "pretraining", "simple_elecssl", "multivariable_elecssl"]


def main():
    effect: Literal["marginals", "interactions"] = "marginals"
    study_name: _STUDIES = "pretraining"
    percentile = 90
    num_hps = 5  # Plotting only the top 'num_hps' HP effects

    experiment_time = "2025-04-02_173237"

    experiment_name = f"experiments_{experiment_time}"
    file_name = f"{effect}_{study_name}_percentile_{percentile}"

    # Fix dataframe
    df = pandas.read_csv(Path(os.path.dirname(__file__)) / "importance_scores" / experiment_name / file_name)
    if effect == "interactions" and num_hps is not None:
        df = df[df["Rank"] < num_hps]

    id_vars = ("Rank", "HP", "Mean", "Std") if effect == "interactions" else ("HP", "Mean", "Std")
    df = df.melt(id_vars=id_vars, var_name="Tree", value_name="Importance")
    df.sort_values(by=["Mean"], inplace=True, ascending=False)
    if effect == "marginals" and num_hps is not None:
        top_hps = df["HP"].drop_duplicates(inplace=False)[:num_hps]
        print(top_hps)
        df = df[df["HP"].isin(top_hps)]
        print(df)

    # Plot
    seaborn.boxplot(df, y="Importance",  x="HP")

    pyplot.show()


if __name__ == "__main__":
    main()
