import os
from datetime import date, datetime
from pathlib import Path

import yaml

from elecssl.data.paths import get_results_dir
from elecssl.models.experiments.generate_config_file import generate_config_file
from elecssl.models.experiments.single_experiment import SSLExperiment
from elecssl.models.sampling_distributions import get_yaml_loader
from elecssl.models.utils import add_yaml_constructors


def _run_experiments(config_name):
    # ---------------
    # Load config file with sampling
    # ---------------
    # Get loader for the sampling distributions
    loader = get_yaml_loader()

    # Add additional formatting
    loader = add_yaml_constructors(loader)

    # Create path to config file
    config_path = (Path(os.path.dirname(__file__)) / "config_files" / config_name).with_suffix(".yml")
    with open(config_path) as file:
        config = yaml.load(file, Loader=loader)

    # ---------------
    # Fix/prepare config file
    # ---------------
    config = generate_config_file(config)

    # ---------------
    # Run experiment
    # ---------------
    results_path = get_results_dir() / f"debug_{config_name}_{date.today()}_{datetime.now().strftime('%H%M%S')}"
    with SSLExperiment(config=config, pre_processing_config={"general": {"resample": 90}},
                       results_path=results_path) as experiment:
        experiment.run_experiment()


def main():
    # ----------------
    # Select config file to use
    # ----------------
    experiments_config_name = "regression_performance"

    # ----------------
    # Run experiments
    # ----------------
    _run_experiments(config_name=experiments_config_name)


if __name__ == "__main__":
    main()
