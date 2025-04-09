"""
Script which can be used to resume experiments. Hopefully it won't be used though...
"""
import os
from datetime import datetime
from pathlib import Path

from elecssl.data.paths import get_results_dir
from elecssl.models.experiments.hpo_experiment import AllHPOExperiments


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
        experiments.resume_experiments()


if __name__ == "__main__":
    main()
