"""
Script for continuing with multitask learning experiments.
"""
import os
import time
from datetime import datetime
from pathlib import Path

from elecssl.data.paths import get_results_dir
from elecssl.models.experiments.hpo_experiment import AllHPOExperiments, ExperimentType


def _extract_datetime_func(*, prefix):
    def _extract_datetime(s):
        return datetime.strptime(s[len(prefix):], "%Y-%m-%d_%H%M%S")
    return _extract_datetime


def _get_newest_experiment_name(results_dir: Path, *, prefix: str):
    experiments = os.listdir(results_dir)
    if not experiments:
        # The first script (which must be submitted first) has not yet started, try again in 60 seconds
        time.sleep(60)
    return max(experiments, key=_extract_datetime_func(prefix=prefix))


def main():
    # -------------
    # Get the path of the newest experiment
    # -------------
    results_dir = get_results_dir()

    experiment_name = _get_newest_experiment_name(results_dir=results_dir, prefix="experiments_")

    with AllHPOExperiments.load_previous(results_dir / experiment_name) as experiments:
        experiments.resume_experiments(include=(ExperimentType.MULTI_TASK,))


if __name__ == "__main__":
    main()
