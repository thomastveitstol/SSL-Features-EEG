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

from elecssl.models.experiments.single_experiment import SSLExperiment
from elecssl.models.hp_suggesting import suggest_hyperparameters
from elecssl.models.sampling_distributions import get_yaml_loader
from elecssl.models.utils import add_yaml_constructors, add_yaml_representers


class HPOExperiment:
    """
    Class for running hyperparameter optimisation
    """

    __slots__ = ("_experiments_config", "_sampling_config", "_results_dir")

    def __init__(self, hp_config_path, experiments_config_path, results_dir):
        # ---------------
        # Load HP distributions files
        # ---------------
        # Get loader for the sampling distributions
        loader = get_yaml_loader()

        # Add additional formatting
        loader = add_yaml_constructors(loader)

        # Create path to config file
        hp_config_path = Path(hp_config_path).with_suffix(".yml")
        with open(hp_config_path) as file:
            hp_config = yaml.load(file, Loader=loader)

        # ---------------
        # Load the other configurations
        # ---------------
        # Create path to config file
        experiments_config_path = Path(experiments_config_path).with_suffix(".yml")
        with open(experiments_config_path) as file:
            experiments_config = yaml.load(file, Loader=loader)

        # ---------------
        # Set attributes
        # ---------------
        self._experiments_config = experiments_config
        self._sampling_config = hp_config
        self._results_dir = results_dir

        # Store the experiments config file
        safe_dumper = add_yaml_representers(yaml.SafeDumper)
        with open(results_dir / "experiments_config.yml", "w") as file:
            yaml.dump(experiments_config, file, Dumper=safe_dumper)

    def run_hyperparameter_optimisation(self):
        """Run HPO with optuna"""
        # Create study
        study = optuna.create_study(**self.hpo_study_config["HPOStudy"])

        # Optimise
        study.optimize(self._create_objective(), n_trials=self.hpo_study_config["NTrials"])

    def _create_objective(self):

        def _objective(trial):
            # ---------------
            # Create configurations for all feature extractors
            # ---------------
            spaces = (self._experiments_config["OcularStates"], self._experiments_config["FrequencyBands"],
                      self._experiments_config["FrequencyBands"])

            # Make directory for current iteration
            results_dir = self._results_dir / f"debug_hpo_{trial.number}_{date.today()}_{datetime.now().strftime('%H%M%S')}"
            os.mkdir(results_dir)

            # ---------------
            # Using multiprocessing  # todo: should turn this off if using GPU?
            # ---------------
            feature_extractors_biomarkers: Dict[str, pandas.DataFrame] = {}
            with ProcessPoolExecutor(max_workers=self._experiments_config["MultiProcessing"]["max_workers"]) as executor:
                print("Multiprocessing")
                for (in_ocular_state, out_ocular_state), in_freq_band, out_freq_band in itertools.product(*spaces):
                    executor.submit(
                        self._run_single_job, experiments_config=self._experiments_config,
                        sampling_config=self._sampling_config, trial=trial, in_ocular_state=in_ocular_state,
                        out_ocular_state=out_ocular_state, in_freq_band=in_freq_band, out_freq_band=out_freq_band,
                        results_dir=results_dir
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
            # score = ml_model.fit(feature_extractors_biomarkers)

            return 0.4

        return _objective

    @staticmethod
    def _run_single_job(experiments_config, sampling_config, trial: optuna.Trial, in_ocular_state, out_ocular_state,
                    in_freq_band, out_freq_band, results_dir):
        """Method for running a single SSL experiments"""
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

    # --------------
    # Properties
    # --------------
    @property
    def hpo_study_config(self):
        return self._experiments_config["HPO"]
