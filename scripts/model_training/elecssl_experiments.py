import os
from datetime import date, datetime

import yaml

from elecssl.data.paths import get_results_dir
from elecssl.models.experiments.generate_config_file import generate_config_file
from elecssl.models.experiments.single_experiment import SSLExperiment
from elecssl.models.sampling_distributions import get_yaml_loader
from elecssl.models.utils import add_yaml_constructors


def main():
    # ---------------
    # Load config file with sampling
    # ---------------
    # Get loader for the sampling distributions
    loader = get_yaml_loader()

    # Add additional formatting
    loader = add_yaml_constructors(loader)
    with open(os.path.join(os.path.dirname(__file__), "config_files", "experiments_config.yml")) as file:
        config = yaml.load(file, Loader=loader)

    # ---------------
    # Fix/prepare config file
    # ---------------
    config = generate_config_file(config)

    # ---------------
    # Run experiment
    # ---------------
    results_path = get_results_dir() / f"debug_{date.today()}_{datetime.now().strftime('%H%M%S')}"
    with SSLExperiment(config=config, pre_processing_config={"general": {"resample": 90}},
                       results_path=results_path) as experiment:
        experiment.run_experiment()


if __name__ == "__main__":
    main()
