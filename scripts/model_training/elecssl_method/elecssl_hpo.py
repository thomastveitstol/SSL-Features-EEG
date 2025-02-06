import os
from pathlib import Path

from elecssl.data.paths import get_results_dir
from elecssl.models.experiments.hpo_experiment import ElecsslHPO, PredictionModelsHPO, PretrainHPO


def main():
    # ----------------
    # Some paths
    # ----------------
    config_folder = Path(os.path.dirname(__file__)) / "config_files"

    # Non-HPO related configurations
    static_conf_folder = config_folder / "static_configurations"

    shared_static_path = static_conf_folder / "shared_conf.yml"
    elecssl_static_path = static_conf_folder / "elecssl_conf.yml"
    pretraining_shared_static_path = static_conf_folder / "pretraining_shared_conf.yml"
    pretraining_downstream_static_path = static_conf_folder / "pretraining_downstream_conf.yml"
    pretraining_pretext_static_path = static_conf_folder / "pretraining_pretext_conf.yml"
    prediction_models_static_path = static_conf_folder / "prediction_models_conf.yml"

    # HPO related configurations
    hpd_folder = config_folder / "hyperparameter_distributions"

    shared_hpd_path = hpd_folder / "shared_hpds.yml"
    elecssl_hpd_path = hpd_folder / "elecssl_hpds.yml"
    pretraining_downstream_hpd_path = hpd_folder / "pretraining_downstream_hpds.yml"
    pretraining_pretext_hpd_path = hpd_folder / "pretraining_pretext_hpds.yml"
    prediction_models_hpd_path = hpd_folder / "prediction_models_hpds.yml"

    results_dir = get_results_dir()

    # ----------------
    # Run experiments
    # ----------------
    # Elecssl
    elecssl_experiment = ElecsslHPO(
        hp_config_paths=(shared_hpd_path, elecssl_hpd_path),
        experiments_config_paths=(shared_static_path, elecssl_static_path),
        results_dir=results_dir
    )
    with elecssl_experiment as experiment:
        experiment.run_hyperparameter_optimisation()

    # Pre-training
    pretrain_experiment = PretrainHPO(
        hp_config_paths=(shared_hpd_path,),
        experiments_config_paths=(shared_static_path, pretraining_shared_static_path),
        results_dir=results_dir,
        pretext_hp_config_paths=(shared_hpd_path, pretraining_pretext_hpd_path,),
        pretext_experiments_config_paths=(shared_static_path, pretraining_shared_static_path,
                                          pretraining_pretext_static_path,),
        downstream_experiments_config_paths=(shared_static_path, pretraining_shared_static_path,
                                             pretraining_downstream_static_path,),
        downstream_hp_config_paths=(shared_hpd_path, pretraining_downstream_hpd_path,)
    )
    with pretrain_experiment as experiment:
        experiment.run_hyperparameter_optimisation()

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
