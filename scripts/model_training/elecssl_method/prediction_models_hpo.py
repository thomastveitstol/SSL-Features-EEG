import os
from pathlib import Path

from elecssl.data.paths import get_results_dir
from elecssl.models.experiments.hpo_experiment import PredictionModelsHPO


def main():
    # ----------------
    # Some paths
    # ----------------
    config_folder = Path(os.path.dirname(__file__)) / "config_files"

    # Non-HPO related configurations
    static_conf_folder = config_folder / "static_configurations"
    shared_static_path = static_conf_folder / "shared_conf.yml"
    prediction_models_static_path = static_conf_folder / "prediction_models_conf.yml"

    # HPO related configurations
    hpd_folder = config_folder / "hyperparameter_distributions"
    shared_hpd_path = hpd_folder / "shared_hpds.yml"
    prediction_models_hpd_path = hpd_folder / "prediction_models_hpds.yml"

    results_dir = get_results_dir()

    # ----------------
    # Run experiment
    # ----------------
    # Non-pretrained prediction models
    prediction_models_experiment = PredictionModelsHPO(
        hp_config_paths=(shared_hpd_path, prediction_models_hpd_path),
        experiments_config_paths=(shared_static_path, prediction_models_static_path),
        results_dir=results_dir
    )
    with prediction_models_experiment as experiment:
        experiment.run_hyperparameter_optimisation()


if __name__ == "__main__":
    main()
