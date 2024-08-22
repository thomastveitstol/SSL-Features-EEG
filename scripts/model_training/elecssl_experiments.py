import os
from datetime import date, datetime

import yaml

from elecssl.data.paths import get_results_dir
from elecssl.models.experiments.single_experiment import SSLExperiment


def main():
    # ---------------
    # Load config file (this will be changed later)
    # ---------------
    with open(os.path.join(os.path.dirname(__file__), "config_files", "test_config.yml")) as file:
        config = yaml.safe_load(file)

    # ---------------
    # Run experiment
    # ---------------
    results_path = get_results_dir() / f"debug_{date.today()}_{datetime.now().strftime('%H%M%S')}"
    with SSLExperiment(config=config, pre_processing_config={"general": {"resample": 90}},
                       results_path=results_path) as experiment:
        experiment.run_experiment()


if __name__ == "__main__":
    main()
