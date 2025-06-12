"""
Main script for running experiments
"""
import os
from pathlib import Path

from elecssl.data.paths import get_results_dir
from elecssl.models.experiments.hpo_experiment import AllHPOExperiments


def main():
    exclude_experiments = ("prediction_models", "pretraining", "simple_elecssl", "multivariable_elecssl")

    # Paths
    results_dir = get_results_dir()
    config_path = Path(os.path.dirname(__file__)) / "config_files"

    # Run experiments
    with AllHPOExperiments(results_dir=results_dir, config_path=config_path, is_continuation=False) as experiments:
        experiments.run_experiments(exclude=exclude_experiments)


if __name__ == "__main__":
    main()
