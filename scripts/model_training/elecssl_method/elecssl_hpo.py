import os
from datetime import date, datetime
from pathlib import Path

from elecssl.data.paths import get_results_dir
from elecssl.models.experiments.hpo_experiment import HPOExperiment


def main():
    # ----------------
    # Some configurations
    # ----------------
    hp_config_path = Path(os.path.dirname(__file__)) / "config_files" / "hp_distributions"
    experiments_config_path = Path(os.path.dirname(__file__)) / "config_files" / "configurations"
    results_dir = get_results_dir() / f"hpo_experiment_{date.today()}_{datetime.now().strftime('%H%M%S')}"

    # ----------------
    # Run experiments
    # ----------------
    HPOExperiment(
        hp_config_path=hp_config_path, experiments_config_path=experiments_config_path, results_dir=results_dir
    ).run_hyperparameter_optimisation()


if __name__ == "__main__":
    main()
