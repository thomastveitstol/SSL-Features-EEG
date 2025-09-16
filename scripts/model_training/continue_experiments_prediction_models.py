"""
Script for continuing with prediction models, in case it gets killed or something
"""
import os
from datetime import datetime
from pathlib import Path

from elecssl.data.paths import get_results_dir
from elecssl.models.experiments.hpo_experiment import AllHPOExperiments, ExperimentType


def _extract_datetime_func(*, prefix):
    def _extract_datetime(s):
        return datetime.strptime(s[len(prefix):], "%Y-%m-%d_%H%M%S")
    return _extract_datetime


def _get_newest_experiment_name(results_dir: Path, *, prefix: str):
    return max(os.listdir(results_dir), key=_extract_datetime_func(prefix=prefix))


def main():
    # -------------
    # Get the path of the newest experiment
    # -------------
    results_dir = get_results_dir()

    experiment_name = _get_newest_experiment_name(results_dir=results_dir, prefix="experiments_")

    with AllHPOExperiments.load_previous(results_dir / experiment_name) as experiments:
        experiments.resume_experiments(include=(ExperimentType.PREDICTION_MODELS,))


if __name__ == "__main__":
    main()
