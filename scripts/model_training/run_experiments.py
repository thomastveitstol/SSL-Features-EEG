"""
Main script for running prediction models experiments. To run effectively on TSD, I split the running into
(1) ml_features and prediction_models, (2) pretraining, simple_elecssl, and multivariable_elecssl, and (3) multi_task.

This script needs to be run first to successfully create subject splits, folders, and the yaml files
"""
import os
from pathlib import Path

import torch

from elecssl.data.paths import get_results_dir
from elecssl.models.experiments.hpo_experiment import AllHPOExperiments


def main():
    exclude_experiments = ("pretraining", "simple_elecssl", "multivariable_elecssl", "multi_task")

    # Paths
    results_dir = get_results_dir()
    config_path = Path(os.path.dirname(__file__)) / "config_files"

    print('CUDA available:', torch.cuda.is_available())
    print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')

    # Run experiments
    with AllHPOExperiments(results_dir=results_dir, config_path=config_path, is_continuation=False) as experiments:
        experiments.run_experiments(exclude=exclude_experiments)


if __name__ == "__main__":
    main()
