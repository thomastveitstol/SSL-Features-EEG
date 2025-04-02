import os
from pathlib import Path

import pandas


def main():
    effect = "interactions"
    study_name = "multivariable_elecssl"
    percentile = 75

    experiment_time = "2025-04-01_171150"

    experiment_name = f"experiments_{experiment_time}"
    file_name = f"{effect}_{study_name}_percentile_{percentile}"
    df = pandas.read_csv(Path(os.path.dirname(__file__)) / "importance_scores" / experiment_name / file_name)
    print(df)


if __name__ == "__main__":
    main()
