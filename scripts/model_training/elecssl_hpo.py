import os
from pathlib import Path

from elecssl.data.paths import get_results_dir
from elecssl.models.experiments.hpo_experiment import HPOExperiment


def main():
    # ----------------
    # Some configurations
    # ----------------
    hp_config_path = Path(os.path.dirname(__file__)) / "config_files" / "hp_distributions"
    experiments_config_path = Path(os.path.dirname(__file__)) / "config_files" / "configurations"
    results_dir = get_results_dir()

    # ----------------
    # Run experiments
    # ----------------
    HPOExperiment(
        hp_config_path=hp_config_path, experiments_config_path=experiments_config_path, results_dir=results_dir
    )


if __name__ == "__main__":
    main()
