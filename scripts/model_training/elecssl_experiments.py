import itertools
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime
from pathlib import Path
from typing import Dict

import optuna
import pandas
import yaml
from sklearn.tree import DecisionTreeRegressor

from elecssl.data.paths import get_results_dir
from elecssl.models.experiments.single_experiment import SSLExperiment
from elecssl.models.hp_suggesting import suggest_hyperparameters
from elecssl.models.sampling_distributions import get_yaml_loader
from elecssl.models.utils import add_yaml_constructors


def _run_single_job(experiments_config, sampling_config, trial: optuna.Trial, in_ocular_state, out_ocular_state,
                    in_freq_band, out_freq_band, results_dir):
    feature_extractor_name = f"{in_ocular_state}{out_ocular_state}{in_freq_band}{out_freq_band}"

    # ---------------
    # Suggest / sample hyperparameters
    # ---------------
    suggested_hyperparameters = suggest_hyperparameters(
        name=feature_extractor_name, config=sampling_config, trial=trial
    )

    # ---------------
    # Learn expectation values and extract biomarkers
    # ---------------
    results_path = results_dir / (f"hpo_{trial.number}_{feature_extractor_name}_{date.today()}_"
                                  f"{datetime.now().strftime('%H%M%S')}")
    with SSLExperiment(hp_config=suggested_hyperparameters, experiments_config=experiments_config,
                       pre_processing_config={}, results_path=results_path) as experiment:
        ...


def create_objective(experiments_config, sampling_config):
    # todo: wouldn't it be better to make this a class instead?
    # ---------------
    # Create objective
    # ---------------
    def objective(trial: optuna.Trial):
        # ---------------
        # Create configurations for all feature extractors
        # ---------------
        spaces = (experiments_config["OcularStates"], experiments_config["FrequencyBands"],
                  experiments_config["FrequencyBands"])

        # Make directory for current iteration
        results_dir = get_results_dir() / f"debug_hpo_{trial.number}_{date.today()}_{datetime.now().strftime('%H%M%S')}"
        os.mkdir(results_dir)

        # ---------------
        # Using multiprocessing  # todo: should turn this off if using GPU?
        # ---------------
        feature_extractors_biomarkers: Dict[str, pandas.DataFrame] = {}
        with ProcessPoolExecutor(max_workers=experiments_config["MultiProcessing"]["max_workers"]) as executor:
            print("Multiprocessing")
            for (in_ocular_state, out_ocular_state), in_freq_band, out_freq_band in itertools.product(*spaces):
                executor.submit(
                    _run_single_job, experiments_config=experiments_config, sampling_config=sampling_config,
                    trial=trial, in_ocular_state=in_ocular_state, out_ocular_state=out_ocular_state,
                    in_freq_band=in_freq_band, out_freq_band=out_freq_band, results_dir=results_dir
                )

                # todo: Can call it .learn_biomarkers or something
                #    biomarkers = executor.submit(experiment.run_experiment)
                #    feature_extractors_biomarkers[feature_extractor_name] = biomarkers

        # ---------------
        # Use the biomarkers
        # ---------------
        # Combine all biomarkers to a single dataframe

        # Create ML  todo: make a package with ML models
        ml_model = DecisionTreeRegressor()

        # todo: Random splits? cross validation? I actually need to HPO the ML model too...
        #score = ml_model.fit(feature_extractors_biomarkers)

        return 0.4

    return objective


def _run_experiments(hp_config_name, experiments_config_name, experiment_name):
    # ---------------
    # Load HP distributions files
    # ---------------
    # Get loader for the sampling distributions
    loader = get_yaml_loader()

    # Add additional formatting
    loader = add_yaml_constructors(loader)

    # Create path to config file
    hp_config_path = (Path(os.path.dirname(__file__)) / "config_files" / hp_config_name).with_suffix(".yml")
    with open(hp_config_path) as file:
        hp_config = yaml.load(file, Loader=loader)

    # ---------------
    # Load HP distributions files
    # ---------------
    # Create path to config file
    experiments_config_path = (Path(os.path.dirname(__file__)) / "config_files" /
                               experiments_config_name).with_suffix(".yml")
    with open(experiments_config_path) as file:
        experiments_config = yaml.load(file, Loader=loader)

    # ---------------
    # HPO with optuna
    # ---------------
    # Create study
    study = optuna.create_study(**experiments_config["HPOStudy"])

    # Optimise
    study.optimize(
        create_objective(experiments_config=experiments_config, sampling_config=hp_config),
        n_trials=3
    )

    # ---------------
    # Fix/prepare config file
    # ---------------
    #config = generate_config_file(config)

    # ---------------
    # Run experiment
    # ---------------
    #results_path = get_results_dir() / f"debug_{config_name}_{date.today()}_{datetime.now().strftime('%H%M%S')}"
    #with SSLExperiment(config=config, pre_processing_config={"general": {"resample": 90}},
    #                   results_path=results_path) as experiment:
    #    experiment.run_experiment()


def main():
    # ----------------
    # Select config file to use
    # ----------------
    experiment_name = "hpo"
    experiments_config_name = "configurations"
    hp_distributions = "hp_distributions"

    # ----------------
    # Run experiments
    # ----------------
    _run_experiments(experiments_config_name=experiments_config_name, hp_config_name=hp_distributions,
                     experiment_name=experiment_name)


if __name__ == "__main__":
    main()
