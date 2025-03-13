"""
Main script for running experiments
"""
import os
from pathlib import Path

from elecssl.data.paths import get_results_dir
from elecssl.models.experiments.hpo_experiment import AllHPOExperiments


def main():
    # Paths
    results_dir = get_results_dir()

    config_folder = Path(os.path.dirname(__file__)) / "config_files"
    static_folder = config_folder / "static_configurations"  # Non-HPO related configurations
    hpd_folder = config_folder / "hyperparameter_distributions"  # HPO related configurations

    # Run experiments
    with AllHPOExperiments(results_dir=results_dir, hpd_configs_path=hpd_folder,
                           experiments_configs_path=static_folder) as experiments:
        experiments.run_experiments()


if __name__ == "__main__":
    main()
