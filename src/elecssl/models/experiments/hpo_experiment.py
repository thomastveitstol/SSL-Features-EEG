import abc
import copy
import itertools
import os
import random
import traceback
import warnings
from datetime import date, datetime
from functools import reduce
from pathlib import Path
from typing import Dict, Any, Callable, Tuple, List, Literal, Set, Union, Optional, NamedTuple, Type

import numpy
import optuna
import pandas
import yaml  # type: ignore[import-untyped]
from optuna.trial import FrozenTrial
from progressbar import progressbar
from scipy.stats import NearConstantInputWarning, ConstantInputWarning

from elecssl.data.datasets.getter import get_dataset
from elecssl.data.paths import get_numpy_data_storage_path
from elecssl.data.results_analysis.hyperparameters import to_hyperparameter
from elecssl.data.results_analysis.utils import load_hpo_study
from elecssl.data.subject_split import Subject, subjects_tuple_to_dict, get_data_split, simple_random_split
from elecssl.models.experiments.single_experiment import SingleExperiment
from elecssl.models.hp_suggesting import make_trial_suggestion, suggest_spatial_dimension_mismatch, suggest_loss, \
    suggest_dl_architecture, get_optuna_sampler
from elecssl.models.metrics import PlotNotSavedWarning, higher_is_better
from elecssl.models.ml_models.ml_model_base import MLModel
from elecssl.models.utils import add_yaml_constructors, verify_type, verified_performance_score, \
    merge_dicts_strict, remove_prefix, remove_prefix_from_keys


# --------------
# Small convenient classes
# --------------
class HPORun(NamedTuple):
    experiment: Type['HPOExperiment']
    path: Path


# --------------
# HPO baseclasses
# --------------
class MainExperiment(abc.ABC):
    """
    Base class for experiments to be run
    """

    __slots__ = ("_experiments_config", "_sampling_config", "_results_path")
    _name: str

    def __init__(self, hp_config, experiments_config, results_dir, is_continuation):
        # ---------------
        # Set attributes
        # ---------------
        if is_continuation:
            self._results_path = results_dir / self._name

            # Input check
            self.verify_results_dir_exists(self._results_path)

            # Load the configurations files
            with open(self._results_path / "experiments_config.yml") as file:
                experiments_config = yaml.safe_load(file)
            with open(self._results_path / "hpd_config.yml") as file:
                sampling_config = yaml.safe_load(file)

            self._experiments_config: Dict[str, Any] = experiments_config
            self._sampling_config: Dict[str, Any] = sampling_config
            return

        self._experiments_config = experiments_config
        self._sampling_config = hp_config
        self._results_path = results_dir / self._name

        # ---------------
        # Save files
        # ---------------
        # Make directory
        os.mkdir(self._results_path)

        # Save the config files
        _save_yaml_file(results_path=self._results_path, config_file_name="experiments_config.yml",
                        config=self._experiments_config, make_read_only=True)
        _save_yaml_file(results_path=self._results_path, config_file_name="hpd_config.yml",
                        config=self._sampling_config, make_read_only=True)

    # -------------
    # Dunder methods for context manager (using the 'with' statement). See this video from mCoding for more information
    # on context managers https://www.youtube.com/watch?v=LBJlGwJ899Y&t=640s
    # -------------
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """This will execute when exiting the with statement. It will NOT execute if the run was killed by the operating
        system, which can happen if too much data is loaded into memory"""
        # If everything was as it should, create an open file that indicates no errors
        if exc_val is None:
            with open(self._results_path / f"finished_successfully_{date.today()}_"
                                           f"{datetime.now().strftime('%H%M%S')}.txt", "w"):
                pass
            return

        # Otherwise, document the error received in a text file
        file_name = (self._results_path / f"{exc_type.__name__}_{date.today()}_"
                                          f"{datetime.now().strftime('%H%M%S')}").with_suffix(".txt")
        with open(file_name, "w") as file:
            file.write("Traceback (most recent call last):\n")
            traceback.print_tb(exc_tb, file=file)  # type: ignore[arg-type]
            file.write(f"{exc_type.__name__}: {exc_val}")

    # --------------
    # Methods for analysis
    # --------------
    @abc.abstractmethod
    def generate_test_scores_df(self, *args, **kwargs) -> pandas.DataFrame:
        """Method for generating a dataframe which summarises the performance scores such that it can be analysed"""

    # --------------
    # Methods for checking if results were as expected.
    # Mostly using methods with @classmethod so that post-hoc checks can be done as well with only the path provided
    # --------------
    @classmethod
    @abc.abstractmethod
    def get_test_subjects(cls, path) -> Set[Subject]:
        """Get the test subjects"""

    @classmethod
    @abc.abstractmethod
    def verify_test_set_integrity(cls, path):
        """
        Method for checking the test set integrity after HPO has been run

        Parameters
        ----------
        path : pathlib.Path
            The path to where all results of the HPO are

        Returns
        -------
        None
        """

    def integrity_check_test_set(self):
        self.verify_test_set_integrity(path=self.results_path)

    @staticmethod
    def verify_equal_test_sets(hpo_runs: Tuple[HPORun, ...]):
        # Get all test sets
        test_sets: List[Set[Subject]] = []
        for run in hpo_runs:
            test_set = run.experiment.get_test_subjects(path=run.path)
            assert isinstance(test_set, set), f"Expected test set to be a set, but found {type(test_set)}"
            assert all(isinstance(subject, Subject) for subject in test_set), \
                f"Expected subjects om test set to be of type 'Subject', but found {set(type(s) for s in test_set)}"
            test_sets.append(test_set)

        # Check if they are all equal
        if not all(test_set == test_sets[0] for test_set in test_sets):
            raise DissimilarTestSetsError

    # --------------
    # Input checks
    # --------------
    @classmethod
    def verify_results_dir_exists(cls, results_dir):
        """This should be used to verify if a results dir exists. Should only be used when it is supposed to exist"""
        if not os.path.isdir(results_dir):
            raise ExperimentNotFoundError(f"The results path {results_dir} of the attempted continued study does not "
                                          f"exist. This is likely due to (1) the path was incorrect, or (2) the HPO "
                                          f"experiment ({cls}) should not have been initialised as a continuation")

    @classmethod
    def is_existing_dir(cls, results_dir):
        """Check if the experiment directory exists is the provided directory"""
        try:
            cls.verify_results_dir_exists(results_dir / cls.get_name())
            return True
        except ExperimentNotFoundError:
            return False

    # --------------
    # Properties
    # --------------
    # Not a property, but pretty close (chaining classmethod and property was removed in Python3.13)
    @classmethod
    def get_name(cls):
        return cls._name

    @property
    def results_path(self):
        return self._results_path


class HPOExperiment(MainExperiment):
    """
    Base class for running hyperparameter optimisation
    """

    __slots__ = ()
    _test_predictions_file_name: str  # The name of the csv file which contains the test predictions
    _optimisation_predictions_file_name: Tuple[str, ...]  # In these csv files, test subjects should NOT be present
    # (will be used for checking test set integrity)

    @classmethod
    def load_previous(cls, path):
        """Method for loading a previous study"""
        return cls(hp_config=None, experiments_config=None, results_dir=path, is_continuation=True)

    # --------------
    # Methods for running HPO
    # --------------
    def continue_hyperparameter_optimisation(self, num_trials: Optional[int]):
        """Method for continuing on an HPO study. May be used, e.g., if the initial run was killed."""
        # Create sampler
        sampler = get_optuna_sampler(self.hpo_study_config["HPOStudy"]["sampler"],
                                     **self.hpo_study_config["HPOStudy"]["sampler_kwargs"])

        # Load study
        study = self.load_study(sampler=sampler)

        # Optimise
        if num_trials is None:
            num_trials = self.hpo_study_config["num_trials"] - len(study.trials)
        with warnings.catch_warnings():
            for warning in self._experiments_config["Warnings"]["ignore"]:
                warnings.filterwarnings(action="ignore", category=_get_warning(warning))
            study.optimize(self._create_objective(), n_trials=num_trials)

    def run_hyperparameter_optimisation(self):
        """Run HPO with optuna"""
        # Create study
        study = self._create_study()

        # Optimise
        with warnings.catch_warnings():
            for warning in self._experiments_config["Warnings"]["ignore"]:
                warnings.filterwarnings(action="ignore", category=_get_warning(warning))
            study.optimize(self._create_objective(), n_trials=self.hpo_study_config["num_trials"])

    def _create_study(self):
        """Creates and returns the study object"""
        # Create sampler
        sampler = get_optuna_sampler(self.hpo_study_config["HPOStudy"]["sampler"],
                                     **self.hpo_study_config["HPOStudy"]["sampler_kwargs"])

        # Create study
        study_name, storage_path = self._get_study_name_and_storage_path(results_path=self._results_path)
        return optuna.create_study(study_name=study_name, storage=storage_path, sampler=sampler,
                                   direction=self.hpo_study_config["HPOStudy"]["direction"],
                                   load_if_exists=False)

    def load_study(self, sampler):
        """Returns the study object"""
        study_name, storage_path = self._get_study_name_and_storage_path(results_path=self._results_path)
        return optuna.load_study(study_name=study_name, storage=storage_path, sampler=sampler)

    @abc.abstractmethod
    def _create_objective(self) -> Callable[[optuna.Trial], float]:
        """Method which returns the function to study.optimise. It needs to take a trial argument of type optuna.Trial
        and return a score"""

    # --------------
    # Methods for HPO sampling
    # --------------
    def _suggest_common_hyperparameters(self, trial, name, in_freq_band, preprocessed_config_path, skip_training,
                                        skip_loss):
        suggested_hps: Dict[str, Any] = {"Preprocessing": {}}
        name_prefix = "" if name is None else f"{name}_"

        # Preprocessing
        for param_name, (distribution, distribution_kwargs) in self._sampling_config["Preprocessing"].items():
            suggested_hps["Preprocessing"][param_name] = make_trial_suggestion(
                trial=trial, name=f"{name_prefix}{param_name}", method=distribution, kwargs=distribution_kwargs
            )

        # Training
        if not skip_training:
            suggested_hps["Training"] = self._suggest_training_hpcs(trial=trial, name=name,
                                                                    hpd_config=self._sampling_config)

        # Normalisation
        normalisation = trial.suggest_categorical(f"{name_prefix}normalisation",
                                                  **self._sampling_config["normalisation"])

        # Convolutional monge mapping normalisation
        if self._experiments_config["enable_cmmn"]:
            raise NotImplementedError("Hyperparameter sampling with CMMN has not been implemented yet...")
        else:
            cmmn = {"use_cmmn_layer": False, "kwargs": {}}

            # DL architecture
            suggested_hps["DLArchitecture"] = suggest_dl_architecture(
                name=name, trial=trial, config=self._sampling_config["DLArchitectures"],
                suggested_preprocessing_steps=suggested_hps["Preprocessing"], freq_band=in_freq_band,
                preprocessed_config_path=preprocessed_config_path
            )

        # Loss
        if not skip_loss:
            suggested_hps["Loss"] = suggest_loss(name=name, trial=trial, config=self._sampling_config["Loss"])

        # Handling varied numbers of electrodes
        suggested_hps["SpatialDimensionMismatch"] = suggest_spatial_dimension_mismatch(
            name=name, trial=trial, config=self._sampling_config, normalisation=normalisation, cmmn=cmmn
        )

        # Maybe add normalisation to DL architecture
        if suggested_hps["SpatialDimensionMismatch"]["name"] == "Interpolation":
            suggested_hps["DLArchitecture"]["normalise"] = normalisation
            suggested_hps["DLArchitecture"]["CMMN"] = cmmn

        # Domain discriminator
        if self._experiments_config["enable_domain_discriminator"]:
            raise NotImplementedError(
                "Hyperparameter sampling with domain discriminator has not been implemented yet...")
        else:
            # todo: not a fan of having to specify training method, especially now that the name is somewhat misleading
            if not skip_training:
                suggested_hps["Training"]["method"] = "downstream_training"
            suggested_hps["DomainDiscriminator"] = None

        return suggested_hps

    @staticmethod
    def _suggest_training_hpcs(trial, name, hpd_config):
        name_prefix = "" if name is None else f"{name}_"

        # Training
        suggested_train_hpcs = dict()
        for param_name, (distribution, distribution_kwargs) in hpd_config["Training"].items():
            suggested_train_hpcs[param_name] = make_trial_suggestion(
                trial=trial, name=f"{name_prefix}{param_name}", method=distribution, kwargs=distribution_kwargs
            )
        return suggested_train_hpcs

    # --------------
    # Methods for analysis
    # --------------
    @classmethod
    def generate_test_scores_df(cls, path, *, target_metrics, selection_metric, method: Literal["mean", "median"],
                                include_experiment_name=True):
        """
        Method for generate a dataframe which summarises the performance scores of an HPO run, such that it can be
        analysed

        todo: How to aggregate results when model selection gives multiple models? Current implementation aggregates
            performance scores, but could aggregate predictions too...

        Parameters
        ----------
        path : Path
            The path to the HPO run
        target_metrics : tuple[str, ...]
        selection_metric : str
            Metric used for model selection
        method : {"mean", "median"}
            To aggregate predictions/scores by mean or median
        include_experiment_name : bool
            If the name of the experiment (e.g., 'prediction_models') should be added in a column

        Returns
        -------
        pandas.DataFrame
        """
        # Check if the study object is found
        if f"{cls._name}-study.db" not in os.listdir(path):
            raise FileNotFoundError(f"Could not find the study object at the path provided ('{cls._name}') {path}. "
                                    f"Remember to use the correct class for the study object. File names in the "
                                    f"provided path: {os.listdir(path)}")

        # --------------
        # Loop through all iterations in the HPO
        # --------------
        # Initialisation  todo: must implement _refit metrics
        scores: Dict[str, List[float]] = {
            "run": [], "trial_number": [], **{f"val_{metric}": [] for metric in target_metrics},
            **{f"test_{metric}": [] for metric in target_metrics}
        }

        hpo_iterations = tuple(folder for folder in os.listdir(path) if os.path.isdir(path / folder)
                               and (folder.startswith("hpo_")))
        for hpo_iteration in progressbar(hpo_iterations, redirect_stdout=True, prefix="Trial "):
            trial_path = path / hpo_iteration

            # Initialise dictionaries with all scores for the trial
            trial_val_scores: Dict[str, List[float]] = {metric: [] for metric in target_metrics}
            trial_test_scores: Dict[str, List[float]] = {metric: [] for metric in target_metrics}

            # Get the performance for each fold
            folds = (fold for fold in os.listdir(trial_path) if os.path.isdir(trial_path / fold)
                     and fold.lower().startswith("split_"))
            for fold in folds:
                # Get fold scores, but accept that some trials may have been pruned
                try:
                    fold_val_scores, fold_test_scores = cls._get_performance_scores(
                        trial_path / fold, selection_metric=selection_metric, target_metrics=target_metrics
                    )
                except FileNotFoundError:
                    continue  # todo: should I skip the fold or the entire trial?

                # Add them to trial scores
                for metric, val_score in fold_val_scores.items():
                    trial_val_scores[metric].append(val_score)
                for metric, test_score in fold_test_scores.items():
                    trial_test_scores[metric].append(test_score)

            # Aggregate fold scores
            if method == "mean":
                agg_folds_method = numpy.mean
            elif method == "median":
                agg_folds_method = numpy.median  # type: ignore[assignment]
            else:
                raise ValueError(f"Unexpected method to aggregate scores across folds: '{method}'")
            for metric, val_scores in trial_val_scores.items():
                scores[f"val_{metric}"].append(agg_folds_method(val_scores))  # type: ignore[arg-type]
            for metric, test_scores in trial_test_scores.items():
                scores[f"test_{metric}"].append(agg_folds_method(test_scores))  # type: ignore[arg-type]

            # Add the rest of the info
            scores["trial_number"].append(int(hpo_iteration.split("_")[-1]))
            scores["run"].append(hpo_iteration)

        # Convert to dataframe
        df = pandas.DataFrame(scores)
        df.sort_values(by="trial_number", inplace=True)
        df.reset_index(inplace=True)

        # Maybe add experiment name indicator
        if include_experiment_name:
            df["experiment_name"] = [cls._name] * df.shape[0]

        return df

    @classmethod
    def _get_performance_scores(cls, path, *, selection_metric, target_metrics):
        # --------------
        # Get validation performance and optimal epoch
        # --------------
        val_scores, epoch = cls._get_validation_scores_and_epoch(path, selection_metric=selection_metric,
                                                                 target_metrics=target_metrics)

        # --------------
        # Get test performance
        # --------------
        test_df = pandas.read_csv(os.path.join(path, "test_history_metrics.csv"))
        test_scores = {target_metric: test_df[target_metric].iat[epoch] for target_metric in target_metrics}

        # Todo: get refit performance scores and add to results
        return val_scores, test_scores

    @staticmethod
    def _get_validation_scores_and_epoch(path, *, selection_metric, target_metrics):
        # Load the dataframe of the validation performances
        val_df = pandas.read_csv(os.path.join(path, "val_history_metrics.csv"))

        # Get the best performance and its epoch. If all scores are nan values (which happens for correlation
        # coefficients constant with constant prediction arrays), the epoch is set to -1, which is acceptable
        if higher_is_better(metric=selection_metric):
            epoch = numpy.argmax(val_df[selection_metric])
        else:
            epoch = numpy.argmin(val_df[selection_metric])

        # Performance
        val_scores = {target_metric: val_df[target_metric].iat[epoch] for target_metric in target_metrics}
        return val_scores, epoch

    # --------------
    # Methods for checking if results were as expected
    # --------------
    @classmethod
    def verify_test_set_integrity(cls, path):
        # Check if the test set always contain the same set of subject
        test_subjects = cls._verify_test_set_consistency(path=path)

        # Check if any of the test subjects were in training or validation
        cls._verify_test_set_exclusivity(path=path, test_subjects=test_subjects)

    def integrity_check_test_set(self):
        self.verify_test_set_integrity(path=self.results_path)

    @classmethod
    def _verify_test_set_consistency(cls, path):
        """
        Verify that the test set is consistent across trials and folds. If not, an 'InconsistentTestSetError' is raised

        Parameters
        ----------
        path : Path

        Returns
        -------
        Set[Subject]
            The test set subjects
        """
        expected_subjects: Set[Subject] = set()

        # Loop through all trials
        trial_folders = tuple(file_name for file_name in os.listdir(path)
                              if file_name.startswith("hpo_") and os.path.isdir(path / file_name))
        for trial_folder in progressbar(trial_folders, redirect_stdout=True, prefix="Trial test set "):
            trial_path = path / trial_folder

            # Loop through all folds within the trial
            fold_folders = (name for name in os.listdir(trial_path)
                            if name.lower().startswith("fold_") and os.path.isdir(trial_path / name))
            for fold_folder in fold_folders:
                fold_path = trial_path / fold_folder

                # Load the subjects from the test predictions, but also accept that some trials may have been pruned
                try:
                    test_history_subjects = pandas.read_csv(
                        (fold_path / cls._test_predictions_file_name).with_suffix(".csv"),
                        usecols=("dataset", "sub_id"))
                except FileNotFoundError:
                    continue

                # Convert to set of 'Subject'
                subjects = set(Subject(dataset_name=row.dataset, subject_id=row.sub_id)  # type: ignore[attr-defined]
                               for row in test_history_subjects.itertuples(index=False))

                # Check with expected subjects (or make it expected subjects, if it is the first)
                if expected_subjects:
                    if subjects != expected_subjects:
                        raise InconsistentTestSetError
                else:
                    expected_subjects = subjects

        return expected_subjects

    @classmethod
    def _verify_test_set_exclusivity(cls, path, test_subjects):
        """
        Method which should be used to verify that no test subject were in the validation or train set for any of the
        trials and folds. If there is overlap, a 'NonExclusiveTestSetError' is raised

        Parameters
        ----------
        path : Path
        test_subjects : Set[Subject]

        Returns
        -------
        None
        """
        # Loop through all trials
        trial_folders = tuple(file_name for file_name in os.listdir(path)
                              if file_name.startswith("hpo_") and os.path.isdir(path / file_name))
        for trial_folder in progressbar(trial_folders, redirect_stdout=True, prefix="Trial non-test set "):
            trial_path = path / trial_folder

            # Loop through all folds within the trial
            fold_folders = tuple(name for name in os.listdir(trial_path)
                                 if name.lower().startswith("split_") and os.path.isdir(trial_path / name))
            for fold_folder in fold_folders:
                fold_path = trial_path / fold_folder

                # Load the subjects from the predictions that were used for optimisation (typically train and
                # validation), but also accept that some trials may have been pruned
                for predictions in cls._optimisation_predictions_file_name:
                    try:
                        subjects_df = pandas.read_csv((fold_path / predictions).with_suffix(".csv"),
                                                      usecols=("dataset", "sub_id"))
                    except FileNotFoundError:
                        continue

                    # Convert to set of 'Subject'
                    subjects = set(
                        Subject(dataset_name=row.dataset, subject_id=row.sub_id)  # type: ignore[attr-defined]
                        for row in subjects_df.itertuples(index=False))

                    # Check if there is overlap
                    if not subjects.isdisjoint(test_subjects):
                        overlap = subjects & test_subjects
                        raise NonExclusiveTestSetError(
                            f"Test subjects were found in the optimisation set {predictions} for trial {trial_folder}, "
                            f"fold {fold_folder}. These subjects are (N={len(overlap)})): {overlap}"
                        )

    @classmethod
    def get_test_subjects(cls, path):
        """Get the test subjects, while also checking consistency"""
        return cls._verify_test_set_consistency(path=path)

    # --------------
    # Methods for HP analysis
    # --------------
    @classmethod
    def get_study_name(cls):
        return f"{cls._name}-study"

    @classmethod
    def add_hp_configurations_to_dataframe(cls, df, hps, results_dir, skip_if_exists=True):
        """
        Method for adding hyperparameter configurations to a dataframe.

        Parameters
        ----------
        df : pandas.DataFrame
            The input dataframe needs the 'run' column to know which run to extract the HPC from
        hps : str | tuple[str, ...]
            Hyperparameter configurations to add to the dataframe
        results_dir : pathlib.Path
        skip_if_exists : bool
            To skip the HP if it already exists in the dataframe

        Returns
        -------
        pandas.DataFrame
        """
        hps = (hps,) if isinstance(hps, str) else hps

        # (Maybe) skipping HPs which are already in the dataframe
        if skip_if_exists:
            hps = tuple(hp for hp in hps if hp not in df.columns)

        # Infer the HPCs from the study object
        study = load_hpo_study(name=cls.get_study_name(), path=results_dir)
        hpcs = {trial.number: trial.params for trial in study.get_trials()}

        # Loop through all unique runs to make dataframe
        trial_numbers = set(df["trial_number"])
        configs: Dict[str, Any] = {"trial_number": [], **{hp_name: [] for hp_name in hps}}
        for trial_number in trial_numbers:
            configs["trial_number"].append(trial_number)
            for hp_name in hps:
                configs[hp_name].append(hpcs[trial_number][f"{cls._name}_{hp_name}"])

        # Add to existing df
        return df.merge(pandas.DataFrame(configs), on="trial_number")

    def create_unconditional_hp_config_space(self, name, hpd_config):
        # todo: havent tested this, not sure if I completed it
        preprocessing_config_space: Dict[str, Any] = dict()

        # Preprocessing
        for param_name, (distribution, distribution_kwargs) in hpd_config["Preprocessing"].items():
            preprocessing_config_space[param_name] = to_hyperparameter(
                name=f"{name}_{param_name}", method=distribution, **distribution_kwargs
            )

        # Training
        train_config_space = self._get_training_config_space(name=name, hpd_config=hpd_config)

        # Normalisation
        config_space: Dict[str, Any] = dict()
        config_space["normalisation"] = to_hyperparameter(name=f"{name}_normalisation", method="categorical",
                                                          **hpd_config["normalisation"])

        # Domain adaptation
        domain_adaptation = []
        if self._experiments_config["enable_cmmn"]:
            domain_adaptation.append("CMMN")
        if self._experiments_config["enable_domain_discriminator"]:
            domain_adaptation.append("DD")
        if self._experiments_config["enable_cmmn"] and self._experiments_config["enable_domain_discriminator"]:
            domain_adaptation.append("CMMN + DD")
        if domain_adaptation:
            domain_adaptation.append("Nothing")
            config_space["Domain adaptation"] = to_hyperparameter(name=f"{name}_domain_adaptation",
                                                                  method="categorical", choices=domain_adaptation)
        # Loss
        config_space["Loss"] = to_hyperparameter(name=f"{name}_loss", **hpd_config["Loss"]["loss"])

        # Handling varied numbers of electrodes
        config_space["Spatial method"] = to_hyperparameter(name=f"{name}_spatial_method", method="categorical",
                                                           **hpd_config["SpatialDimensionMismatch"])

        return merge_dicts_strict(train_config_space, config_space, preprocessing_config_space)

    @staticmethod
    def _get_training_config_space(name, hpd_config):
        # Training
        configs = dict()
        for param_name, (distribution, distribution_kwargs) in hpd_config["Training"].items():
            configs[param_name] = to_hyperparameter(
                name=f"{name}_{param_name}", method=distribution, kwargs=distribution_kwargs
            )
        return configs

    # --------------
    # Small convenient methods
    # --------------
    def _get_hpo_folder_path(self, trial_number: int):
        return self._results_path / f"hpo_trial_{trial_number}"

    @classmethod
    def _get_study_name_and_storage_path(cls, results_path):
        study_name = f"{cls._name}-study"
        path = (results_path / study_name).with_suffix(".db")
        return study_name, f"sqlite:///{path}"

    # --------------
    # Properties
    # --------------
    @property
    def hpo_study_config(self):
        return self._experiments_config["HPO"]


# --------------
# Single HPO experiments
# --------------
class MLFeatureExtraction(MainExperiment):
    """
    Class for 'normal' machine learning using normal feature extraction. It does not really use HPO
    """

    _name = "ml_features"

    def __init__(self, *, experiments_config, hp_config, results_dir):
        super().__init__(experiments_config=experiments_config, hp_config=hp_config, is_continuation=False,
                         results_dir=results_dir)

    # --------------
    # Overriding abstract methods
    # --------------
    def generate_test_scores_df(self, *args, **kwargs) -> pandas.DataFrame:
        raise NotImplementedError

    @classmethod
    def get_test_subjects(cls, path) -> Set[Subject]:
        # Verify test set integrity and return the resulting set of subjects
        return cls._verify_test_set_integrity(path=path)

    @classmethod
    def _verify_test_set_integrity(cls, path) -> Set[Subject]:
        # Load test predictions
        test_predictions_df = pandas.read_csv(path / "test_predictions.csv", usecols=("dataset", "sub_id"))

        # Convert to correct class, make a sanity check, and return
        subjects = tuple(Subject(dataset_name=row.dataset, subject_id=row.sub_id)  # type: ignore[attr-defined]
                         for row in test_predictions_df.itertuples(index=False))

        # Subjects should not be duplicated
        subject_set = set(subjects)
        assert len(subject_set) == len(subjects), (f"Duplicates were found in prediction history. Number of "
                                                   f"predictions: {len(subjects)}. Number of unique subjects: "
                                                   f"{len(subject_set)}")
        return subject_set

    @classmethod
    def verify_test_set_integrity(cls, path):
        # Verify without returning anything
        cls._verify_test_set_integrity(path=path)

    # -------------
    # Methods for using ML on extracted features
    # -------------
    def _build_features_matrix(self):
        """
        Function for building a data matrix

        Returns
        -------
        pandas.DataFrame
        """
        features = tuple(self._experiments_config["features"])

        # Initialise data matrix
        data_matrix: Dict[str, Any] = {"subject": [], "clinical_target": [], **{feature: [] for feature in features}}
        all_subjects = []

        # Loop though all datasets to use
        for dataset_name, dataset_kwargs in self._experiments_config["Datasets"].items():
            # Get the dataset
            dataset = get_dataset(dataset_name=dataset_name)

            # Get the subject IDs
            sample_size = dataset_kwargs["num_subjects"]
            if sample_size == "all":
                subjects = dataset.get_subject_ids()
            else:
                subjects = dataset.get_subject_ids()[:sample_size]

            # Update data matrix
            dataset_subjects = [Subject(subject_id=subject, dataset_name=dataset_name) for subject in subjects]
            all_subjects.extend(dataset_subjects)
            for feature in features:
                feature_array = dataset.load_targets(target=feature)
                data_matrix[feature].extend(feature_array)
            target_array = dataset.load_targets(target=self._experiments_config["downstream_target"])
            data_matrix["clinical_target"].extend(target_array)

        # Make dataframe, fix index, and return
        data_matrix["subject"] = all_subjects
        df = pandas.DataFrame(data_matrix)
        df.set_index("subject", inplace=True)

        df["sub_id"] = [sub.subject_id for sub in df.index]
        df["dataset"] = [sub.dataset_name for sub in df.index]
        df.dropna(inplace=True)

        return df

    def evaluate(self):
        # -------------
        # Get features
        # -------------
        df = self._build_features_matrix()

        # Save it
        df.to_csv(self.results_path / "dataframe.csv", index=False)

        # -------------
        # Compute predictive value
        # -------------
        # todo: I should store the validation score too...
        _compute_biomarker_predictive_value(
            df=df, test_split_config=self._experiments_config["TestSplit"],
            subject_split_config=self._experiments_config["MLModelSubjectSplit"],
            ml_model_hp_config=self._sampling_config["MLModel"],
            ml_model_settings_config=self._experiments_config["MLModelSettings"],
            save_test_predictions=self._experiments_config["save_test_predictions"], results_dir=self.results_path,
            verbose=True
        )

    # --------------
    # Properties
    # --------------
    @property
    def results_path(self):
        return self._results_path


class PredictionModelsHPO(HPOExperiment):
    """
    Class for the prediction models
    """

    __slots__ = ()
    _name = "prediction_models"
    _test_predictions_file_name = "test_history_predictions"
    _optimisation_predictions_file_name = ("train_history_predictions", "val_history_predictions")

    def _create_objective(self):
        def _objective(trial: optuna.Trial):
            # ---------------
            # Suggest / sample hyperparameters
            # ---------------
            in_freq_band = "all"  # todo: hard coded :(

            suggested_hyperparameters = self.suggest_hyperparameters(
                name=None, trial=trial, in_freq_band=in_freq_band
            )

            # ---------------
            # Just a little bit of adding stuff where it needs
            # ---------------
            in_ocular_state = suggested_hyperparameters["ocular_state"]
            experiments_config, preprocessing_config_file = _get_prepared_experiments_config(
                experiments_config=self._experiments_config.copy(), in_freq_band=in_freq_band,
                in_ocular_state=in_ocular_state, suggested_hyperparameters=suggested_hyperparameters
            )

            # ---------------
            # Train prediction model
            # ---------------
            # Make directory for current iteration
            results_dir = self._get_hpo_folder_path(trial.number)
            with SingleExperiment(hp_config=suggested_hyperparameters, pre_processing_config=preprocessing_config_file,
                                  experiments_config=experiments_config, results_path=results_dir,
                                  fine_tuning=None, experiment_name=None) as experiment:
                experiment.run_experiment(combined_datasets=None)

            # ---------------
            # Get the performance
            # ---------------
            return _get_aggregated_val_score(trial_results_dir=results_dir, metric=self.train_config["main_metric"],
                                             aggregation_method=self._experiments_config["val_scores_aggregation"])

        return _objective

    def suggest_hyperparameters(self, trial, name, in_freq_band):
        name_prefix = "" if name is None else f"{name}_"
        in_ocular_state = trial.suggest_categorical(f"{name_prefix}ocular_state",
                                                    **self._sampling_config["OcularStates"])
        preprocessing_config_path = _get_preprocessing_config_path(ocular_state=in_ocular_state)
        suggested_hps = self._suggest_common_hyperparameters(trial, name, in_freq_band=in_freq_band,
                                                             preprocessed_config_path=preprocessing_config_path,
                                                             skip_loss=False, skip_training=False)
        suggested_hps["ocular_state"] = in_ocular_state
        return suggested_hps

    # --------------
    # Properties
    # --------------
    @property
    def train_config(self):
        return {**self._sampling_config["Training"], **self._experiments_config["Training"]}


class PretrainHPO(HPOExperiment):
    """
    Class for using the pretext task for pretraining
    """

    __slots__ = ("_pretext_experiments_config", "_pretext_sampling_config", "_downstream_experiments_config",
                 "_downstream_sampling_config")
    _name = "pretraining"
    _test_predictions_file_name = "test_history_predictions"
    _optimisation_predictions_file_name = ("train_history_predictions", "val_history_predictions",
                                           "pretext_train_history_predictions", "pretext_val_history_predictions")

    @classmethod
    def load_previous(cls, path):
        return cls(results_dir=path, is_continuation=True, downstream_hp_config=None,
                   downstream_experiments_config=None, pretext_hp_config=None, pretext_experiments_config=None,
                   experiments_config=None, hp_config=None)

    def __init__(self, *, experiments_config, hp_config, downstream_hp_config: Optional[Dict[str, Any]],
                 downstream_experiments_config: Optional[Dict[str, Any]],
                 pretext_hp_config: Optional[Dict[str, Any]],
                 pretext_experiments_config: Optional[Dict[str, Any]],
                 results_dir: Path, is_continuation: bool):

        super().__init__(experiments_config=experiments_config, hp_config=hp_config, results_dir=results_dir,
                         is_continuation=is_continuation)

        # ---------------
        # Set attributes
        # ---------------
        if is_continuation:
            # Load the configurations files
            with open(self._results_path / "downstream_experiments_config.yml") as file:
                loaded_downstream_experiments_config = yaml.safe_load(file)
            with open(self._results_path / "downstream_hpd_config.yml") as file:
                loaded_downstream_sampling_config = yaml.safe_load(file)
            with open(self._results_path / "pretext_experiments_config.yml") as file:
                loaded_pretext_experiments_config = yaml.safe_load(file)
            with open(self._results_path / "pretext_hpd_config.yml") as file:
                loaded_pretext_sampling_config = yaml.safe_load(file)

            self._downstream_experiments_config: Dict[str, Any] = loaded_downstream_experiments_config
            self._downstream_sampling_config: Dict[str, Any] = loaded_downstream_sampling_config
            self._pretext_experiments_config: Dict[str, Any] = loaded_pretext_experiments_config
            self._pretext_sampling_config: Dict[str, Any] = loaded_pretext_sampling_config
            return

        # Type checks (and to make mypy stop complaining)
        assert downstream_experiments_config is not None
        assert downstream_hp_config is not None
        assert pretext_experiments_config is not None
        assert pretext_hp_config is not None

        self._downstream_experiments_config = downstream_experiments_config
        self._downstream_sampling_config = downstream_hp_config
        self._pretext_experiments_config = pretext_experiments_config
        self._pretext_sampling_config = pretext_hp_config

        # ---------------
        # Save the config files
        # ---------------
        # Downstream config files
        _save_yaml_file(results_path=self._results_path, config_file_name="downstream_experiments_config.yml",
                        config=self._downstream_experiments_config, make_read_only=True)
        _save_yaml_file(results_path=self._results_path, config_file_name="downstream_hpd_config.yml",
                        config=self._downstream_sampling_config, make_read_only=True)

        # Pretext config files
        _save_yaml_file(results_path=self._results_path, config_file_name="pretext_experiments_config.yml",
                        config=self._pretext_experiments_config, make_read_only=True)
        _save_yaml_file(results_path=self._results_path, config_file_name="pretext_hpd_config.yml",
                        config=self._pretext_sampling_config, make_read_only=True)

    def _create_objective(self):
        def _objective(trial: optuna.Trial):
            # todo: should I also add these as HPs?
            in_ocular_state = self._experiments_config["in_ocular_state"]
            in_freq_band = self._experiments_config["in_freq_band"]
            out_ocular_state = self._experiments_config["out_ocular_state"]

            # ---------------
            # Suggest / sample hyperparameters
            # ---------------
            # These HPCs are shared between pretext task and downstream task. Such as the DL architecture
            suggested_shared_hyperparameters = self._suggest_shared_hyperparameters(
                name=None, trial=trial, in_freq_band=in_freq_band
            )

            # These HPCs are specific to the pretext task
            pretext_specific_hpcs, datasets_to_use = self._suggest_pretext_specific_hyperparameters(name="pretext",
                                                                                                    trial=trial)

            # These HPCs are specific to the downstream task
            downstream_specific_hpcs = self._suggest_downstream_specific_hyperparameters(name="downstream", trial=trial)

            # Combine the shared and specific HPCs
            downstream_hpcs = {**suggested_shared_hyperparameters, **downstream_specific_hpcs}

            suggested_shared_hyperparameters = suggested_shared_hyperparameters.copy()
            pretext_hpcs = {**suggested_shared_hyperparameters.copy(), **pretext_specific_hpcs}

            # ---------------
            # Some additions to the experiment configs
            # ---------------
            # For the pretext task
            incomplete_pretext_experiments_config = merge_dicts_strict(
                self._experiments_config, self._pretext_experiments_config
            )

            # Add the selected datasets to pretext task (including subgroups for performance tracking). The ones in the
            # experiments config file are only the available ones, not the ones we will always use
            incomplete_pretext_experiments_config["Datasets"] = dict()
            for dataset_name, dataset_info in datasets_to_use.items():
                incomplete_pretext_experiments_config["Datasets"][dataset_name] = dataset_info
            incomplete_pretext_experiments_config["SubGroups"]["sub_groups"]["dataset_name"] = tuple(
                dataset_name for dataset_name in datasets_to_use)

            pretext_experiments_config, preprocessing_config_file = _get_prepared_experiments_config(
                experiments_config=incomplete_pretext_experiments_config, in_freq_band=in_freq_band,
                in_ocular_state=in_ocular_state, suggested_hyperparameters=pretext_hpcs
            )

            # Must set saving model of pretext task to true
            pretext_experiments_config["Saving"]["save_model"] = True

            # Adding target  TODO: verify that band power is not already log-transformed!!!
            pseudo_target = f"band_power_{pretext_specific_hpcs['out_freq_band']}_{out_ocular_state}"
            if self._pretext_experiments_config["Training"]["log_transform_targets"] is not None:
                pseudo_target = (f"{self._pretext_experiments_config['Training']['log_transform_targets']}_"
                                 f"{pseudo_target}")
            pretext_experiments_config["Training"]["target"] = pseudo_target

            # Downstream task
            downstream_experiments_config, _ = _get_prepared_experiments_config(
                experiments_config=merge_dicts_strict(self._experiments_config, self._downstream_experiments_config),
                in_freq_band=in_freq_band, in_ocular_state=in_ocular_state, suggested_hyperparameters=downstream_hpcs
            )

            # ---------------
            # Train on pretext task
            # ---------------
            # Make directory for current iteration
            results_dir = self._get_hpo_folder_path(trial.number)

            # Execute pre-training
            with SingleExperiment(hp_config=pretext_hpcs, pre_processing_config=preprocessing_config_file,
                                  experiments_config=pretext_experiments_config, results_path=results_dir,
                                  fine_tuning=None, experiment_name="pretext") as experiment:
                combined_datasets = experiment.run_experiment(combined_datasets=None)

            # ---------------
            # Train on downstream task
            # ---------------
            # Remove pretext task datasets
            combined_datasets.remove_datasets(
                to_remove=tuple(dataset for dataset in self._pretext_sampling_config["Datasets"]
                                if dataset not in self._downstream_sampling_config["Datasets"])
            )

            # Run experiment
            fine_tuning = "pretext"
            with SingleExperiment(hp_config=downstream_hpcs, pre_processing_config=preprocessing_config_file,
                                  experiments_config=downstream_experiments_config, results_path=results_dir,
                                  fine_tuning=fine_tuning, experiment_name=None) as experiment:
                experiment.run_experiment(combined_datasets=combined_datasets)

            # ---------------
            # Compute the performance
            # ---------------
            score = _get_aggregated_val_score(
                trial_results_dir=results_dir, metric=self._downstream_experiments_config["Training"]["main_metric"],
                aggregation_method=self._downstream_experiments_config["val_scores_aggregation"]
            )

            # ---------------
            # (Maybe) compute SSL biomarkers for future use
            # ---------------
            if self._experiments_config["save_ssl_biomarkers"]:
                # Create and save dataframe with target and residuals
                out_freq_band = pretext_specific_hpcs['out_freq_band']
                residual_feature_name = f"{in_ocular_state}{out_ocular_state}{in_freq_band}{out_freq_band}"

                df = _make_single_residuals_df(
                    results_dir=results_dir / "split_0", pseudo_target=pseudo_target, feature_name=residual_feature_name,
                    downstream_target=self._downstream_experiments_config["Training"]["target"],
                    deviation_method=self._experiments_config["elecssl_deviation_method"],
                    in_ocular_state=in_ocular_state, experiment_name="pretext",
                    pretext_main_metric=self._pretext_experiments_config["Training"]["main_metric"],
                    include_pseudo_targets=self._experiments_config["elecssl_include_pseudo_targets"],
                    continuous_testing=self._pretext_experiments_config["Training"]["continuous_testing"]
                )
                df.to_csv(results_dir / "ssl_biomarkers.csv", index=False)

            # Return the score
            return score

        return _objective

    def _suggest_pretext_specific_hyperparameters(self, trial, name):
        suggested_hps = dict()
        name_prefix = "" if name is None else f"{name}_"

        # Suggest e.g. alpha or beta band power
        suggested_hps["out_freq_band"] = trial.suggest_categorical(
            name=f"{name_prefix}out_freq_band", **self._pretext_sampling_config["out_freq_band"]
        )

        # -------------
        # Pick the datasets to be used for pre-training
        # -------------
        # The pre-training excluded
        datasets_to_use = dict()
        if "left_out_dataset" in self._pretext_experiments_config["SubjectSplit"]["kwargs"]:
            pretrain_excluded = self._pretext_experiments_config["SubjectSplit"]["kwargs"]["left_out_dataset"]
            datasets_to_use[pretrain_excluded] = self._pretext_experiments_config["Datasets"][pretrain_excluded]
        else:
            pretrain_excluded = "NO_DATASET"  # convenient that the variable still exists

        # The datasets for pretraining
        datasets_for_pretraining = tuple(dataset for dataset in self._pretext_experiments_config["Datasets"]
                                         if dataset != pretrain_excluded)
        possible_pretrain_combinations = _generate_dataset_combinations(datasets_for_pretraining)
        pretrain_combinations = trial.suggest_categorical(f"{name_prefix}datasets",
                                                          choices=possible_pretrain_combinations)

        for dataset_name in _datasets_str_to_tuple(pretrain_combinations):
            datasets_to_use[dataset_name] = self._pretext_experiments_config["Datasets"][dataset_name]

        # Training
        hpd_config = merge_dicts_strict(self._sampling_config, self._pretext_sampling_config)
        suggested_hps["Training"] = self._suggest_training_hpcs(trial=trial, name=name, hpd_config=hpd_config)

        # Loss
        suggested_hps["Loss"] = suggest_loss(name=name, trial=trial, config=hpd_config["Loss"])

        # Domain discriminator
        if self._experiments_config["enable_domain_discriminator"]:
            raise NotImplementedError(
                "Hyperparameter sampling with domain discriminator has not been implemented yet...")
        else:
            # todo: not a fan of having to specify training method, especially now that the name is somewhat misleading
            suggested_hps["Training"]["method"] = "downstream_training"
            suggested_hps["DomainDiscriminator"] = None

        return suggested_hps, datasets_to_use

    def _suggest_downstream_specific_hyperparameters(self, name, trial):
        suggested_hps = {"Loss": suggest_loss(name=name, trial=trial, config=self._sampling_config["Loss"]),
                         "Training": self._suggest_training_hpcs(trial=trial, name=name,
                                                                 hpd_config=self._sampling_config)}

        # Domain discriminator
        if self._experiments_config["enable_domain_discriminator"]:
            raise NotImplementedError(
                "Hyperparameter sampling with domain discriminator has not been implemented yet...")
        else:
            # todo: not a fan of having to specify training method, especially now that the name is somewhat misleading
            suggested_hps["Training"]["method"] = "downstream_training"
            suggested_hps["DomainDiscriminator"] = None

        return suggested_hps

    def _suggest_shared_hyperparameters(self, trial, name, in_freq_band):
        preprocessing_config_path = _get_preprocessing_config_path(
            ocular_state=self._experiments_config["in_ocular_state"]
        )
        suggested_hps = self._suggest_common_hyperparameters(trial, name, in_freq_band=in_freq_band,
                                                             preprocessed_config_path=preprocessing_config_path,
                                                             skip_loss=True, skip_training=True)

        # Need to remove some HPCs because they are not shared
        del suggested_hps["DomainDiscriminator"]
        return suggested_hps

    # --------------
    # Methods for assisting re-usage of runs
    # --------------
    def get_trials_and_folders(self, pretrained_only, complete_only):
        # Load study
        study = self.load_study(sampler=None)

        # Get the trials and corresponding folder names
        trials_and_folders: List[Tuple[optuna.trial.FrozenTrial, Path]] = []
        for trial in study.trials:
            # Folder name
            folder_name = self._get_hpo_folder_path(trial.number)

            # Check if it exists
            if not os.path.isdir(folder_name):
                raise FileNotFoundError(f"For trial number {trial.number}, the corresponding folder {folder_name} was "
                                        f"not found")

            # (Maybe) skip
            if verify_type(complete_only, bool) and trial.state != optuna.trial.TrialState.COMPLETE:
                continue
            if verify_type(pretrained_only, bool) and not os.path.isfile(folder_name / "ssl_biomarkers.csv"):
                continue

            # Append
            trials_and_folders.append((trial, folder_name))

        return tuple(trials_and_folders)

    # --------------
    # Properties
    # --------------
    @property
    def train_config(self):
        return {**self._sampling_config["Training"], **self._experiments_config["Training"]}


class SimpleElecsslHPO(HPOExperiment):
    """
    Class for learning expectation values and using the most predictive one as input to an ML model. This class uses
    only a single such residual for predictive modelling
    """

    __slots__ = ()
    _name = "simple_elecssl"
    _test_predictions_file_name = "test_predictions"
    _optimisation_predictions_file_name = ("pretext_train_history_predictions", "pretext_val_history_predictions")

    def suggest_hyperparameters(self, trial, name):
        name_prefix = "" if name is None else f"{name}_"

        # Sample frequency bands
        out_freq_band = trial.suggest_categorical(f"{name_prefix}out_freq_band",
                                                  **self._sampling_config["out_freq_band"])
        _in_freq_band = trial.suggest_categorical(f"{name_prefix}in_freq_band", **self._sampling_config["in_freq_band"])
        in_freq_band = out_freq_band if _in_freq_band == "same" else _in_freq_band

        # Input ocular state should be fixed
        in_ocular_state = self._experiments_config["in_ocular_state"]
        preprocessing_config_path = _get_preprocessing_config_path(ocular_state=in_ocular_state)

        # -------------
        # Pick the datasets to be used for pre-training
        # -------------
        # The pre-training excluded
        datasets_to_use = dict()
        if "left_out_dataset" in self._experiments_config["SubjectSplit"]["kwargs"]:
            pretrain_excluded = self._experiments_config["SubjectSplit"]["kwargs"]["left_out_dataset"]
            datasets_to_use[pretrain_excluded] = self._experiments_config["Datasets"][pretrain_excluded]
        else:
            pretrain_excluded = "NO_DATASET"  # convenient that the variable still exists

        # The datasets for pretraining
        datasets_for_pretraining = tuple(dataset for dataset in self._experiments_config["Datasets"]
                                         if dataset != pretrain_excluded)
        possible_pretrain_combinations = _generate_dataset_combinations(datasets_for_pretraining)
        pretrain_combinations = trial.suggest_categorical(f"{name_prefix}datasets",
                                                          choices=possible_pretrain_combinations)

        for dataset_name in _datasets_str_to_tuple(pretrain_combinations):
            datasets_to_use[dataset_name] = self._experiments_config["Datasets"][dataset_name]

        # All other HPs
        suggested_hps = self._suggest_common_hyperparameters(trial, name, in_freq_band=in_freq_band,
                                                             preprocessed_config_path=preprocessing_config_path,
                                                             skip_training=False, skip_loss=False)
        suggested_hps["MLModel"] = self._sampling_config["MLModel"]

        return in_freq_band, out_freq_band, preprocessing_config_path, suggested_hps, datasets_to_use

    def _create_objective(self) -> Callable[[optuna.Trial], float]:

        def _objective(trial: optuna.Trial):
            # Make directory for current iteration
            results_dir = self._get_hpo_folder_path(trial.number)
            os.mkdir(results_dir)

            # ---------------
            # Suggest / sample hyperparameters
            # ---------------
            experiment_name = None
            (in_freq_band, out_freq_band, preprocessing_config_path,
             suggested_hyperparameters, datasets_to_use) = self.suggest_hyperparameters(name=experiment_name,
                                                                                        trial=trial)

            # ---------------
            # Just a bit of preparation...
            # ---------------
            # Ocular states are currently fixed
            in_ocular_state = self._experiments_config["in_ocular_state"]
            out_ocular_state = self._experiments_config["out_ocular_state"]

            # Add the target to config file
            experiment_config_file = self._experiments_config.copy()
            pseudo_target = f"band_power_{out_freq_band}_{out_ocular_state}"  # OC fixed
            if experiment_config_file["Training"]["log_transform_targets"] is not None:
                pseudo_target = f"{experiment_config_file['Training']['log_transform_targets']}_{pseudo_target}"
            experiment_config_file["Training"]["target"] = pseudo_target

            # Add the selected datasets to pretext task (including subgroups for performance tracking). The ones in the
            # experiments config file are only the available ones, not the ones we will always use
            experiment_config_file["Datasets"] = dict()
            for dataset_name, dataset_info in datasets_to_use.items():
                experiment_config_file["Datasets"][dataset_name] = dataset_info
            experiment_config_file["SubGroups"]["sub_groups"]["dataset_name"] = tuple(
                dataset_name for dataset_name in datasets_to_use)

            # ---------------
            # Run pretext task
            # ---------------
            experiments_config, preprocessing_config_file = _get_prepared_experiments_config(
                experiments_config=experiment_config_file, in_freq_band=in_freq_band,
                in_ocular_state=in_ocular_state, suggested_hyperparameters=suggested_hyperparameters
            )
            residual_feature_name = f"{in_ocular_state}{out_ocular_state}{in_freq_band}{out_freq_band}"

            # Convenient to make folder structure the same as MultivariableElecssl
            results_path = results_dir / f"hpo_{trial.number}_{residual_feature_name}"
            with SingleExperiment(hp_config=suggested_hyperparameters, experiments_config=experiments_config,
                                  pre_processing_config=preprocessing_config_file, results_path=results_path,
                                  fine_tuning=None, experiment_name=experiment_name) as experiment:
                experiment.run_experiment(combined_datasets=None)

            # ---------------
            # Extract expectation values and biomarkers
            # ---------------
            df = _make_single_residuals_df(
                results_dir=results_path / "split_0", pseudo_target=pseudo_target, feature_name=residual_feature_name,
                downstream_target=self._experiments_config["clinical_target"], in_ocular_state=in_ocular_state,
                deviation_method=self._experiments_config["deviation_method"], experiment_name=experiment_name,
                pretext_main_metric=self._experiments_config["Training"]["main_metric"],
                include_pseudo_targets=self._experiments_config["include_pseudo_targets"],
                continuous_testing=self._experiments_config["Training"]["continuous_testing"]
            )
            df.to_csv(results_dir / "ssl_biomarkers.csv", index=False)

            # ---------------
            # Use the biomarkers
            # ---------------
            score = _compute_biomarker_predictive_value(
                df, test_split_config=self._experiments_config["TestSplit"],
                subject_split_config=self._experiments_config["MLModelSubjectSplit"],
                ml_model_hp_config=self.ml_model_hp_config, ml_model_settings_config=self.ml_model_settings_config,
                save_test_predictions=self._experiments_config["save_test_predictions"], results_dir=results_dir,
                verbose=True
            )
            return score

        return _objective

    # --------------
    # Methods for running HPO
    # --------------
    def reuse_pretrained_runs(self, pretrain_hpo: PretrainHPO):
        # Load (or create) elecssl study
        try:
            study = self.load_study(sampler=None)
        except KeyError:
            study = self._create_study()
        num_previous_studies = len(study.trials)

        # Re-use the trials
        reused_trials: List[FrozenTrial] = []
        for i, (trial, trial_path) in enumerate(pretrain_hpo.get_trials_and_folders(complete_only=True,
                                                                                    pretrained_only=True)):
            trial_number = num_previous_studies + i
            folder_path = self._get_hpo_folder_path(trial_number)

            # Copy
            os.mkdir(folder_path)

            # ---------------
            # Use the biomarkers to get performance
            # ---------------
            # Load the biomarkers
            df = pandas.read_csv(trial_path / "ssl_biomarkers.csv")
            df.to_csv(folder_path / "ssl_biomarkers.csv", index=False)  # nice to have here too

            # Compute score
            score = _compute_biomarker_predictive_value(
                df, subject_split_config=self._experiments_config["MLModelSubjectSplit"], results_dir=folder_path,
                test_split_config=self._experiments_config["TestSplit"], ml_model_hp_config=self.ml_model_hp_config,
                ml_model_settings_config=self.ml_model_settings_config, verbose=True,
                save_test_predictions=self._experiments_config["save_test_predictions"]
            )

            # Add trial with info
            if trial.state != optuna.trial.TrialState.COMPLETE:
                raise RuntimeError(f"Expected pretrained trial to be complete, but received {trial.state} "
                                   f"({trial_path})")
            hpcs, distributions = self._get_pretrain_hpcs_and_distributions(pretrain_trial=trial)
            reused_trials.append(optuna.trial.create_trial(
                state=trial.state, system_attrs=dict(), intermediate_values=dict(), value=score,
                params=hpcs, distributions=distributions, user_attrs={"Pretraining re-used": trial.number}
            ))

        # Add the trials
        study.add_trials(reused_trials)

    @staticmethod
    def _get_pretrain_hpcs_and_distributions(pretrain_trial):
        """Method for extracting the HPCs and HPDs of a trial run in PretrainHPO, such that the trial can be re-used for
        SimpleElecssl"""
        # -------------
        # HPCs
        # -------------
        # Get the non-downstream HPCs. Also, remove 'pretext_' prefix
        hpcs = remove_prefix_from_keys(
            {name: value for name, value in pretrain_trial.params.items() if not name.startswith("downstream")},
            prefix="pretext_"
        )

        # -------------
        # HPDs
        # -------------
        # Get the non-downstream HPDs. Also, remove 'pretext_' prefix
        distributions = remove_prefix_from_keys(
            {name: value for name, value in pretrain_trial.distributions.items() if not name.startswith("downstream")},
            prefix="pretext_"
        )

        return hpcs, distributions

    # --------------
    # Methods for assisting re-usage of runs
    # --------------
    def get_trials_and_folders(self, completed_only):
        """
        Method for getting the trials and folder paths of all HPO trials conducted. It also includes the re-used ones,
        if any

        Parameters
        ----------
        completed_only : bool
        """
        # Load study
        study = self.load_study(sampler=None)

        trials_and_folders: Dict[str, List[Tuple[FrozenTrial, Path]]] = dict()
        for trial in study.trials:
            # Get expected folder name
            folder_name = self._get_hpo_folder_path(trial.number)

            # Check if it exists
            if not os.path.isdir(folder_name):
                raise FileNotFoundError(f"For trial number {trial.number}, the corresponding folder {folder_name} was "
                                        f"not found")

            # (Maybe) skip
            if verify_type(completed_only, bool) and trial.state != optuna.trial.TrialState.COMPLETE:
                continue

            # Get the feature name
            feature_name = self._get_feature_name(path=folder_name)

            # Add it
            if feature_name not in trials_and_folders:
                trials_and_folders[feature_name] = []
            trials_and_folders[feature_name].append((trial, folder_name))

        return {name: tuple(hpo_trials) for name, hpo_trials in trials_and_folders.items()}

    @staticmethod
    def _get_feature_name(path: Path):
        """ Get the expected feature name"""
        # Load the biomarkers csv file column names and remove some of the expected unrelated ones
        feature_names = pandas.read_csv(path / "ssl_biomarkers.csv", nrows=0).drop(
            labels=["dataset", "sub_id", "clinical_target"], axis="columns"
        ).columns.tolist()

        # Check that there is only one or two columns left, as expected
        assert len(feature_names) in (1, 2), (f"Expected 1 or 2 columns, but got {len(feature_names)} columns: "
                                              f"{feature_names}")

        # If there are two columns, identify the actual feature by removing the pseudo-target
        if len(feature_names) == 2:
            feature_names = [name for name in feature_names if not name.startswith("pt_")]

        # Return the remaining feature name
        assert len(feature_names) == 1, (f"After removing pseudo-targets, expected exactly 1 remaining feature column, "
                                         f"but got {len(feature_names)}: {feature_names}")

        return feature_names[0]

    # --------------
    # Methods for checking if results were as expected
    # --------------
    @classmethod
    def _verify_test_set_consistency(cls, path):
        """This class stores test predictions differently, so need to override it"""
        # todo: copied from MultivariateElecsslHPO
        expected_subjects: Set[Subject] = set()

        # Loop through all trials
        trial_folders = tuple(file_name for file_name in os.listdir(path)
                              if file_name.startswith("hpo_") and os.path.isdir(path / file_name))
        for trial_folder in progressbar(trial_folders, redirect_stdout=True, prefix="Trial test set "):
            trial_path = path / trial_folder

            # Load the subjects from the test predictions, but also accept that some trials may have been pruned
            try:
                test_subjects_df = pandas.read_csv((trial_path / cls._test_predictions_file_name).with_suffix(".csv"),
                                                   usecols=("dataset", "sub_id"))
            except FileNotFoundError:
                continue

            # Convert to set of 'Subject'
            subjects = set(Subject(dataset_name=row.dataset, subject_id=row.sub_id)  # type: ignore[attr-defined]
                           for row in test_subjects_df.itertuples(index=False))

            # Check with expected subjects (or make it expected subjects, if it is the first)
            if expected_subjects:
                if subjects != expected_subjects:
                    raise InconsistentTestSetError
            else:
                expected_subjects = subjects

        return expected_subjects

    @classmethod
    def _verify_test_set_exclusivity(cls, path, test_subjects):
        """As this class does not store the model train and validation predictions, we will rather check the pretext
        tasks."""  # todo: copied from MultivariateElecsslHPO
        # Loop through all trials
        trial_folders = tuple(file_name for file_name in os.listdir(path)
                              if file_name.startswith("hpo_") and os.path.isdir(path / file_name))
        for trial_folder in progressbar(trial_folders, redirect_stdout=True, prefix="Trial non-test set "):
            trial_path = path / trial_folder

            # Loop through all pretext tasks within the trial
            pretext_folders = (name for name in os.listdir(trial_path)
                               if name.startswith("hpo_") and os.path.isdir(trial_path / name))
            for pretext_folder in pretext_folders:
                pretext_path = trial_path / pretext_folder

                # Loop through all folds (should be one, but better to just live with this code now)
                fold_folders = (name for name in os.listdir(trial_path)
                                if name.lower().startswith("split_") and os.path.isdir(pretext_path / name))
                for fold_folder in fold_folders:
                    fold_path = pretext_path / fold_folder
                    # Load the subjects from the predictions that were used for optimisation
                    for predictions in cls._optimisation_predictions_file_name:
                        try:
                            subjects_df = pandas.read_csv((fold_path / predictions).with_suffix(".csv"),
                                                          usecols=("dataset", "sub_id"))
                        except FileNotFoundError:
                            continue

                        # Convert to set of 'Subject'
                        subjects = set(
                            Subject(dataset_name=row.dataset, subject_id=row.sub_id)  # type: ignore[attr-defined]
                            for row in subjects_df.itertuples(index=False))

                        # Check if there is overlap
                        if not subjects.isdisjoint(test_subjects):
                            overlap = subjects & test_subjects
                            raise NonExclusiveTestSetError(
                                f"Test subjects were found in the optimisation set {predictions} for trial "
                                f"{trial_folder}, fold {fold_folder}. These subjects are (N={len(overlap)})): {overlap}"
                            )

    # --------------
    # Properties
    # --------------
    @property
    def ml_model_hp_config(self):
        return self._sampling_config["MLModel"]

    @property
    def ml_model_settings_config(self):
        return self._experiments_config["MLModelSettings"]


class MultivariableElecsslHPO(HPOExperiment):
    """
    Class for using the learned residuals as input to an ML model
    """

    __slots__ = ()
    _name = "multivariable_elecssl"
    _test_predictions_file_name = "test_predictions"
    _optimisation_predictions_file_name = ("pretext_train_history_predictions", "pretext_val_history_predictions")

    def suggest_hyperparameters(self, trial, name, in_freq_band, preprocessing_config_path):
        suggested_hps = self._suggest_common_hyperparameters(trial, name, in_freq_band=in_freq_band,
                                                             preprocessed_config_path=preprocessing_config_path,
                                                             skip_training=False, skip_loss=False)
        suggested_hps["MLModel"] = self._sampling_config["MLModel"]

        # -------------
        # Pick the datasets to be used for pre-training
        # -------------
        name_prefix = "" if name is None else f"{name}_"

        # The pre-training excluded
        datasets_to_use = dict()
        if "left_out_dataset" in self._experiments_config["SubjectSplit"]["kwargs"]:
            pretrain_excluded = self._experiments_config["SubjectSplit"]["kwargs"]["left_out_dataset"]
            datasets_to_use[pretrain_excluded] = self._experiments_config["Datasets"][pretrain_excluded]
        else:
            pretrain_excluded = "NO_DATASET"  # convenient that the variable still exists

        # The datasets for pretraining
        datasets_for_pretraining = tuple(dataset for dataset in self._experiments_config["Datasets"]
                                         if dataset != pretrain_excluded)
        possible_pretrain_combinations = _generate_dataset_combinations(datasets_for_pretraining)
        pretrain_combinations = trial.suggest_categorical(f"{name_prefix}datasets",
                                                          choices=possible_pretrain_combinations)

        for dataset_name in _datasets_str_to_tuple(pretrain_combinations):
            datasets_to_use[dataset_name] = self._experiments_config["Datasets"][dataset_name]

        return suggested_hps, datasets_to_use

    def _create_objective(self):

        def _objective(trial: optuna.Trial):
            # ---------------
            # Create configurations for all feature extractors
            # ---------------
            spaces = self._experiments_config["IOSpaces"]

            # Make directory for current iteration
            results_dir = self._get_hpo_folder_path(trial.number)
            os.mkdir(results_dir)

            # ---------------
            # Train/extract all residuals
            # ---------------
            biomarkers: Dict[Subject, Dict[str, float]] = dict()
            for (in_ocular_state, out_ocular_state), (in_freq_band, out_freq_band) in spaces:
                feature_extractor_name = f"{in_ocular_state}{out_ocular_state}{in_freq_band}{out_freq_band}"

                # ---------------
                # Just a bit of preparation...
                # ---------------
                # Get the number of EEG epochs per experiment  todo: a little hard-coded?
                preprocessing_config_path = _get_preprocessing_config_path(ocular_state=in_ocular_state)
                with open(preprocessing_config_path) as file:
                    preprocessing_config = yaml.safe_load(file)
                    num_epochs = preprocessing_config["Details"]["num_epochs"]

                # Add the target to config file
                experiment_config_file = self._experiments_config.copy()
                target_name = f"band_power_{out_freq_band}_{out_ocular_state}"
                if experiment_config_file["Training"]["log_transform_targets"] is not None:
                    target_name = f"{experiment_config_file['Training']['log_transform_targets']}_{target_name}"
                experiment_config_file["Training"]["target"] = target_name

                # ---------------
                # Suggest / sample hyperparameters
                # ---------------
                suggested_hyperparameters, datasets_to_use = self.suggest_hyperparameters(
                    name=feature_extractor_name, trial=trial, in_freq_band=in_freq_band,
                    preprocessing_config_path=preprocessing_config_path
                )

                # Add the selected datasets to pretext task (including subgroups for performance tracking). The ones in
                # the experiments config file are only the available ones, not the ones we will always use
                experiment_config_file["Datasets"] = dict()
                for dataset_name, dataset_info in datasets_to_use.items():
                    experiment_config_file["Datasets"][dataset_name] = dataset_info
                experiment_config_file["SubGroups"]["sub_groups"]["dataset_name"] = tuple(
                    dataset_name for dataset_name in datasets_to_use)

                # ---------------
                # Initiate experiment
                # ---------------
                include_pseudo_targets = self._experiments_config["include_pseudo_targets"]
                feature_extractor_name, *outputs = self._run_single_job(
                    experiments_config=experiment_config_file, trial_number=trial.number,
                    suggested_hyperparameters=suggested_hyperparameters, in_ocular_state=in_ocular_state,
                    in_freq_band=in_freq_band, results_dir=results_dir,
                    downstream_target=self._experiments_config["clinical_target"],
                    deviation_method=self._experiments_config["deviation_method"],
                    num_eeg_epochs=num_epochs, feature_extractor_name=feature_extractor_name,
                    pretext_main_metric=self._experiments_config["Training"]["main_metric"],
                    include_pseudo_targets=self._experiments_config["include_pseudo_targets"],
                    continuous_testing=self._experiments_config["Training"]["continuous_testing"]
                )

                if include_pseudo_targets:
                    subjects, deviations, clinical_targets, pseudo_targets = outputs
                else:
                    subjects, deviations, clinical_targets = outputs
                    pseudo_targets = itertools.cycle((None,))

                # Collect the resulting 'biomarkers'
                for subject, deviation, target, pseudo_target in zip(subjects, deviations, clinical_targets,
                                                                     pseudo_targets):
                    # Maybe add the target (not optimal code...)
                    if subject not in biomarkers:
                        biomarkers[subject] = {"clinical_target": target}

                    # Add the deviation
                    biomarkers[subject][feature_extractor_name] = deviation

                    # Maybe add pseudo target
                    if include_pseudo_targets:
                        biomarkers[subject][f"pt_{target_name}"] = pseudo_target

            # Make it a dataframe and save it
            df = pandas.DataFrame.from_dict(biomarkers, orient="index")
            df["dataset"] = [idx.dataset_name for idx in df.index]
            df["sub_id"] = [idx.subject_id for idx in df.index]
            df.to_csv(results_dir / "ssl_biomarkers.csv", index=False)

            # ---------------
            # Use the biomarkers
            # ---------------
            score = _compute_biomarker_predictive_value(
                df, subject_split_config=self._experiments_config["MLModelSubjectSplit"],
                test_split_config=self._experiments_config["TestSplit"], ml_model_hp_config=self.ml_model_hp_config,
                ml_model_settings_config=self.ml_model_settings_config, results_dir=results_dir,
                save_test_predictions=self._experiments_config["save_test_predictions"], verbose=True
            )

            return score

        return _objective

    @staticmethod
    def _run_single_job(experiments_config, suggested_hyperparameters, trial_number, in_ocular_state, in_freq_band,
                        results_dir, downstream_target, deviation_method, num_eeg_epochs, pretext_main_metric,
                        feature_extractor_name, include_pseudo_targets, continuous_testing) -> Tuple[Any, ...]:
        """Method for running a single SSL experiments"""
        experiments_config, preprocessing_config_file = _get_prepared_experiments_config(
            experiments_config=experiments_config, in_freq_band=in_freq_band, in_ocular_state=in_ocular_state,
            suggested_hyperparameters=suggested_hyperparameters
        )

        # ---------------
        # Learn on the pretext regression task
        # ---------------
        experiment_name = "pretext"
        results_path = results_dir / f"hpo_{trial_number}_{feature_extractor_name}"
        with SingleExperiment(hp_config=suggested_hyperparameters, experiments_config=experiments_config,
                              pre_processing_config=preprocessing_config_file, results_path=results_path,
                              fine_tuning=None, experiment_name=experiment_name) as experiment:
            experiment.run_experiment(combined_datasets=None)

        # ---------------
        # Extract expectation values and biomarkers
        # ---------------
        outputs = _get_delta_and_variable(
            path=results_path / "split_0", target=experiments_config["Training"]["target"],
            downstream_target=downstream_target, deviation_method=deviation_method, num_eeg_epochs=num_eeg_epochs,
            pretext_main_metric=pretext_main_metric, experiment_name=experiment_name,
            include_pseudo_targets=include_pseudo_targets, continuous_testing=continuous_testing
        )

        return feature_extractor_name, *outputs

    # --------------
    # Methods for running HPO
    # --------------
    def reuse_simple_elecssl_runs(self, simple_elecssl_hpo: SimpleElecsslHPO):
        # Load or create study
        try:
            study = self.load_study(sampler=None)
        except KeyError:
            study = self._create_study()

        # -----------------
        # Re-use trials
        # -----------------
        # Random sampling
        self._random_simple_elecssl_reuse(study=study, simple_elecssl_hpo=simple_elecssl_hpo)

        # Best individual biomarkers  todo: keep it up, don't give up now!
        # Mixed integer linear programming solution
        # Score-based sampling

    def _random_simple_elecssl_reuse(self, study: optuna.Study, simple_elecssl_hpo: SimpleElecsslHPO):
        for _ in range(self._experiments_config["num_random_reuse"]):
            self._single_random_simple_elecssl_reuse(study=study, simple_elecssl_hpo=simple_elecssl_hpo)

    def _single_random_simple_elecssl_reuse(self, study: optuna.Study, simple_elecssl_hpo: SimpleElecsslHPO):
        # --------------
        # Make biomarkers dataframe
        # --------------
        biomarkers_dfs: List[pandas.DataFrame] = []
        hyperparameters: Dict[str, Any] = dict()
        hp_distributions: Dict[str, Any] = dict()
        user_attrs: Dict[str, Any] = dict()
        for feature_name, trials_and_folder_paths in simple_elecssl_hpo.get_trials_and_folders(
                completed_only=True).items():
            # Randomly select a trial and also get the corresponding path
            trial, folder_path = random.choice(trials_and_folder_paths)

            # Load the biomarker df
            biomarkers_dfs.append(pandas.read_csv(folder_path / "ssl_biomarkers.csv"))

            # Add the HPCs, HPDs, and which SimpleElecssl trial it was taken from
            hpcs, distributions = self._get_simple_elecssl_hpcs_and_distributions(trial)

            hyperparameters.update({f"{feature_name}_{hp_name}": hp_value for hp_name, hp_value in hpcs.items()})
            hp_distributions.update({f"{feature_name}_{hp_name}": hp_dist
                                     for hp_name, hp_dist in distributions.items()})
            user_attrs[f"Simple elecssl re-used ({feature_name})"] = trial.number

        # Merge to single df
        df = _merge_ssl_biomarkers_dataframes(biomarkers_dfs)

        # --------------
        # Use the biomarkers
        # --------------
        # Save it first because it is convenient
        multielecssl_trial_number = len(study.trials)
        multielecssl_folder_path = self._get_hpo_folder_path(trial_number=multielecssl_trial_number)
        os.mkdir(multielecssl_folder_path)
        df.to_csv(multielecssl_folder_path / "ssl_biomarkers.csv", index=False)

        # Compute score
        score = _compute_biomarker_predictive_value(
            df, subject_split_config=self._experiments_config["MLModelSubjectSplit"],
            test_split_config=self._experiments_config["TestSplit"], ml_model_hp_config=self.ml_model_hp_config,
            ml_model_settings_config=self.ml_model_settings_config, verbose=True,
            save_test_predictions=self._experiments_config["save_test_predictions"],
            results_dir=multielecssl_folder_path
        )

        reused_trial = optuna.trial.create_trial(
            state=optuna.trial.TrialState.COMPLETE, params=hyperparameters, distributions=hp_distributions,
            user_attrs=user_attrs, system_attrs=dict(), intermediate_values=dict(), value=score
        )
        study.add_trial(reused_trial)

    @staticmethod
    def _get_simple_elecssl_hpcs_and_distributions(simple_elecssl_trial):
        """Method for extracting the HPCs and HPDs of a trial run in SimpleElecssl, such that the trial can be re-used
        for MultivariableElecssl"""
        hpcs = {name: value for name, value in simple_elecssl_trial.params.items()}
        distributions = {name: value for name, value in simple_elecssl_trial.distributions.items()}
        return hpcs, distributions

    # --------------
    # Methods for analysis
    # --------------
    @classmethod
    def generate_test_scores_df(cls, path, *, target_metrics, selection_metric, method: Literal["mean", "median"],
                                include_experiment_name=True):
        raise NotImplementedError

    # --------------
    # Methods for checking if results were as expected
    # --------------
    @classmethod
    def _verify_test_set_consistency(cls, path):
        """This class stores test predictions differently, so need to override it"""
        expected_subjects: Set[Subject] = set()

        # Loop through all trials
        trial_folders = tuple(file_name for file_name in os.listdir(path)
                              if file_name.startswith("hpo_") and os.path.isdir(path / file_name))
        for trial_folder in progressbar(trial_folders, redirect_stdout=True, prefix="Trial test set "):
            trial_path = path / trial_folder

            # Load the subjects from the test predictions, but also accept that some trials may have been pruned
            try:
                test_subjects_df = pandas.read_csv((trial_path / cls._test_predictions_file_name).with_suffix(".csv"),
                                                   usecols=("dataset", "sub_id"))
            except FileNotFoundError:
                continue

            # Convert to set of 'Subject'
            subjects = set(Subject(dataset_name=row.dataset, subject_id=row.sub_id)  # type: ignore[attr-defined]
                           for row in test_subjects_df.itertuples(index=False))

            # Check with expected subjects (or make it expected subjects, if it is the first)
            if expected_subjects:
                if subjects != expected_subjects:
                    raise InconsistentTestSetError
            else:
                expected_subjects = subjects

        return expected_subjects

    @classmethod
    def _verify_test_set_exclusivity(cls, path, test_subjects):
        """As this class does not store the model train and validation predictions, we will rather check the pretext
        tasks."""
        # Loop through all trials
        trial_folders = tuple(file_name for file_name in os.listdir(path)
                              if file_name.startswith("hpo_") and os.path.isdir(path / file_name))
        for trial_folder in progressbar(trial_folders, redirect_stdout=True, prefix="Trial non-test set "):
            trial_path = path / trial_folder

            # Loop through all pretext tasks within the trial
            pretext_folders = (name for name in os.listdir(trial_path)
                               if name.startswith("hpo_") and os.path.isdir(trial_path / name))
            for pretext_folder in pretext_folders:
                pretext_path = trial_path / pretext_folder

                # Loop through all splits (should be one, but better to just live with this code now)
                fold_folders = (name for name in os.listdir(trial_path)
                                if name.lower().startswith("split_") and os.path.isdir(pretext_path / name))
                for fold_folder in fold_folders:
                    fold_path = pretext_path / fold_folder
                    # Load the subjects from the predictions that were used for optimisation
                    for predictions in cls._optimisation_predictions_file_name:
                        try:
                            subjects_df = pandas.read_csv((fold_path / predictions).with_suffix(".csv"),
                                                          usecols=("dataset", "sub_id"))
                        except FileNotFoundError:
                            continue

                        # Convert to set of 'Subject'
                        subjects = set(
                            Subject(dataset_name=row.dataset, subject_id=row.sub_id)  # type: ignore[attr-defined]
                            for row in subjects_df.itertuples(index=False))

                        # Check if there is overlap
                        if not subjects.isdisjoint(test_subjects):
                            overlap = subjects & test_subjects
                            raise NonExclusiveTestSetError(
                                f"Test subjects were found in the optimisation set {predictions} for trial "
                                f"{trial_folder}, split {fold_folder}. These subjects are (N={len(overlap)})): {overlap}"
                            )

    # --------------
    # Properties
    # --------------
    @property
    def ml_model_hp_config(self):
        return self._sampling_config["MLModel"]

    @property
    def ml_model_settings_config(self):
        return self._experiments_config["MLModelSettings"]


# --------------
# Combining all HPO experiments (including baselines)
# --------------
class AllHPOExperiments:
    """
    Class which combines all the HPO experiments
    """

    # __slots__ = ("_hpd_configs_path", "_experiments_configs_path", "_results_path")
    __slots__ = ("_defaults_config", "_downstream_experiments_config", "_pretext_experiments_config",
                 "_shared_hpds", "_specific_hpds", "_results_path", "_specific_experiments_config")

    # -------------
    # Dunder methods for context manager (using the 'with' statement). See this video from mCoding for more information
    # on context managers https://www.youtube.com/watch?v=LBJlGwJ899Y&t=640s
    # -------------
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """This will execute when exiting the with statement. It will NOT execute if the run was killed by the operating
        system, which can happen if too much data is loaded into memory"""
        # If everything was as it should, create an open file that indicates no errors
        if exc_val is None:
            return

        # Otherwise, document the error received in a text file
        file_name = (self._results_path / f"{exc_type.__name__}_{date.today()}_"
                                          f"{datetime.now().strftime('%H%M%S')}").with_suffix(".txt")
        with open(file_name, "w") as file:
            file.write("Traceback (most recent call last):\n")
            traceback.print_tb(exc_tb, file=file)  # type: ignore[arg-type]
            file.write(f"{exc_type.__name__}: {exc_val}")

    def __init__(self, *, results_dir: Path, config_path: Optional[Path], is_continuation):
        if is_continuation:
            # Input check
            if not os.path.isdir(results_dir):
                raise ExperimentNotFoundError(f"The provided results path {results_dir} does not exist. Can therefore "
                                              f"not load the experiment object")
            self._results_path = results_dir

            # When continuing studies, the preferred way is to use configurations that are already in the HPO
            # experiment  subfolders. However, we will still load it in case a new study needs to be initiated (say,
            # e.g., that we ran out of time on TSD before an HPO run was even initiated)
            # Experiments config files
            self._downstream_experiments_config = self._load_yaml_file(
                results_path=self._results_path, config_file_name="downstream_experiments_config.yml")
            self._pretext_experiments_config = self._load_yaml_file(
                results_path=self._results_path, config_file_name="pretext_experiments_config.yml")
            self._specific_experiments_config = self._load_yaml_file(
                results_path=self._results_path, config_file_name="specific_experiments_config.yml")

            # HP distributions
            self._shared_hpds = self._load_yaml_file(
                results_path=self._results_path, config_file_name="shared_hpds.yml")
            self._specific_hpds = self._load_yaml_file(
                results_path=self._results_path, config_file_name="specific_hpds.yml")

            # Defaults
            self._defaults_config = self._load_yaml_file(
                results_path=self._results_path, config_file_name="defaults_config.yml")
            return

        # ---------------
        # Load configuration files
        # ---------------
        # Input check
        if config_path is None:
            raise TypeError("Expected a path to config files, but received None. It can only be None if the object is "
                            "initialised with 'is_continuation=True', but that was not case.")

        # Get loader
        loader = yaml.SafeLoader
        loader = add_yaml_constructors(loader)
        with open(config_path / "default_settings.yml") as file:
            defaults_config = yaml.load(file, Loader=loader)
        with open(config_path / "specific_settings.yml") as file:
            specific_experiments_config = yaml.load(file, Loader=loader)
        with open(config_path / "experiments_config.yml") as file:
            experiments_config = yaml.load(file, Loader=loader)
        with open(config_path / "shared_hyperparameters.yml") as file:
            shared_hpds = yaml.load(file, Loader=loader)
        with open(config_path / "specific_hyperparameters.yml") as file:
            specific_hpds = yaml.load(file, Loader=loader)

        # Add some details
        downstream_main_metric = experiments_config["DownstreamTraining"]["main_metric"]
        study_direction = "maximize" if higher_is_better(downstream_main_metric) else "minimize"
        defaults_config["HPO"]["HPOStudy"]["direction"] = study_direction

        # ---------------
        # Set attributes
        # ---------------
        # Experiments config
        downstream_prefix = "downstream"
        pretext_prefix = "pretext"
        self._downstream_experiments_config = {
            remove_prefix(key, prefix=downstream_prefix, case_sensitive=False): value
            for key, value in experiments_config.items() if key.lower().startswith(downstream_prefix)}
        self._pretext_experiments_config = {
            remove_prefix(key, prefix=pretext_prefix, case_sensitive=False): value
            for key, value in experiments_config.items()if key.lower().startswith(pretext_prefix)}
        self._specific_experiments_config = specific_experiments_config

        # Hyperparameter distributions
        self._shared_hpds = shared_hpds
        self._specific_hpds = specific_hpds

        # Defaults
        self._defaults_config = defaults_config

        # Path where the results will be stored
        self._results_path = results_dir / f"experiments_{date.today()}_{datetime.now().strftime('%H%M%S')}"

        # Make directory
        os.mkdir(self._results_path)

        # ---------------
        # Store config files such that everything can be loaded later
        # (and scientific reproducibility :))
        # ---------------
        # Experiments config files
        _save_yaml_file(results_path=self._results_path, config_file_name="downstream_experiments_config.yml",
                        config=self._downstream_experiments_config, make_read_only=True)
        _save_yaml_file(results_path=self._results_path, config_file_name="pretext_experiments_config.yml",
                        config=self._pretext_experiments_config, make_read_only=True)
        _save_yaml_file(results_path=self._results_path, config_file_name="specific_experiments_config.yml",
                        config=self._specific_experiments_config, make_read_only=True)

        # HP distributions
        _save_yaml_file(results_path=self._results_path, config_file_name="shared_hpds.yml",
                        config=self._shared_hpds, make_read_only=True)
        _save_yaml_file(results_path=self._results_path, config_file_name="specific_hpds.yml",
                        config=self._specific_hpds, make_read_only=True)

        # Defaults
        _save_yaml_file(results_path=self._results_path, config_file_name="defaults_config.yml",
                        config=self._defaults_config, make_read_only=True)

    # --------------
    # Loading of study .yml files
    # --------------
    @staticmethod
    def _load_yaml_file(*, results_path: Path, config_file_name: str):
        """Method for loading a config file"""
        with open((results_path / config_file_name).with_suffix(".yml")) as file:
            config = yaml.safe_load(file)
        return config

    # --------------
    # Main HPO experiments
    # --------------
    def run_experiments(self):
        # --------------
        # HPO experiments
        # --------------
        # Feature extraction + ML
        ml_features = self.run_ml_features()

        # Prediction models
        # prediction_models = self.run_prediction_models_hpo()

        # Pretraining
        pretrain = self.run_pretraining_hpo()

        # Simple Elecssl
        simple_elecssl = self.run_simple_elecssl_hpo(pretrain)

        # Multivariable Elecssl
        multivariable_elecssl = self.run_multivariable_elecssl_hpo(simple_elecssl)

        # --------------
        # Test set integrity tests
        # --------------
        self.verify_test_set_integrity((ml_features, pretrain, simple_elecssl,
                                        multivariable_elecssl))

        # --------------
        # Dataframe creation
        # --------------

        # --------------
        # Create file indicating successful finalisation
        # --------------
        with open(self._results_path / f"finished_successfully_{date.today()}_"
                                       f"{datetime.now().strftime('%H%M%S')}.txt", "w"):
            pass

    # --------------
    # Single HPO experiments
    # --------------
    def run_ml_features(self):
        # Create the config file
        keys = ("TestSplit", "MLModelSubjectSplit", "MLModelSettings")
        _config = {key: self.pretext_experiments_config[key] for key in keys}
        _config["save_test_predictions"] = self.defaults_config["save_test_predictions"]
        _config["Datasets"] = self.downstream_experiments_config["Datasets"]
        _config["downstream_target"] = self.downstream_experiments_config["Training"]["target"]
        config = merge_dicts_strict(_config, self.specific_experiments_config["MLFeatureExtraction"])

        # Run the experiment
        with MLFeatureExtraction(experiments_config=config, hp_config=self.specific_hpds["MLFeatureExtraction"],
                                 results_dir=self._results_path) as experiment:
            experiment.evaluate()

        return experiment

    def run_prediction_models_hpo(self):
        # Create merged config files
        hp_config = merge_dicts_strict(self.shared_hpds, self.specific_hpds["PredictionModelsHPO"])
        experiments_config = merge_dicts_strict(self.defaults_config, self.downstream_experiments_config,
                                                self.specific_experiments_config["PredictionModelsHPO"])

        # Run experiments
        with PredictionModelsHPO(experiments_config=experiments_config, hp_config=hp_config,
                                 results_dir=self._results_path, is_continuation=False) as experiment:
            experiment.run_hyperparameter_optimisation()

        return experiment

    def run_pretraining_hpo(self):
        # Create merged config files
        experiments_config = merge_dicts_strict(self.defaults_config, self.specific_experiments_config["PretrainHPO"])

        # Add some elecssl details
        deviation_method = self.specific_experiments_config["SimpleElecsslHPO"]["deviation_method"]
        include_pseudo_targets = self.specific_experiments_config["SimpleElecsslHPO"]["include_pseudo_targets"]

        experiments_config["elecssl_deviation_method"] = deviation_method
        experiments_config["elecssl_include_pseudo_targets"] = include_pseudo_targets

        # Model with pre-training
        with PretrainHPO(experiments_config=experiments_config, hp_config=self.shared_hpds,
                         downstream_experiments_config=self.downstream_experiments_config,
                         pretext_experiments_config=self.pretext_experiments_config,
                         pretext_hp_config=self.specific_hpds["PretrainHPO"]["pretext"],
                         downstream_hp_config=self.specific_hpds["PretrainHPO"]["downstream"],
                         results_dir=self._results_path, is_continuation=False) as experiment:
            experiment.run_hyperparameter_optimisation()

        return experiment

    def run_simple_elecssl_hpo(self, pretrain_experiment):
        # Create merged config files
        hp_config = merge_dicts_strict(self.shared_hpds, self.specific_hpds["SimpleElecsslHPO"])
        experiments_config = merge_dicts_strict(self.defaults_config, self.pretext_experiments_config,
                                                self.specific_experiments_config["SimpleElecsslHPO"])
        experiments_config["clinical_target"] = self.downstream_experiments_config["Training"]["target"]

        # Elecssl with one independent variable/residual
        with SimpleElecsslHPO(hp_config=hp_config, experiments_config=experiments_config,
                              results_dir=self._results_path, is_continuation=False) as experiment:
            experiment.reuse_pretrained_runs(pretrain_experiment)
            experiment.continue_hyperparameter_optimisation(num_trials=experiments_config["num_additional_trials"])

        return experiment

    def run_multivariable_elecssl_hpo(self, simple_elecssl_experiment):
        # Create merged config files
        hp_config = merge_dicts_strict(self.shared_hpds, self.specific_hpds["MultivariableElecsslHPO"])
        experiments_config = merge_dicts_strict(self.defaults_config, self.pretext_experiments_config,
                                                self.specific_experiments_config["MultivariableElecsslHPO"])
        experiments_config["clinical_target"] = self.downstream_experiments_config["Training"]["target"]

        # Elecssl with multiple independent variables/residuals
        with MultivariableElecsslHPO(hp_config=hp_config, experiments_config=experiments_config,
                                     results_dir=self._results_path, is_continuation=False) as experiment:
            experiment.reuse_simple_elecssl_runs(simple_elecssl_experiment)
            experiment.continue_hyperparameter_optimisation(experiments_config["num_additional_trials"])

        return experiment

    # --------------
    # Methods for continuing studies
    # (Particularly) convenient is something goes wrong on TSD
    # --------------
    @classmethod
    def load_previous(cls, path: Path):
        """Method for loading a previous study"""
        return cls(results_dir=path, config_path=None, is_continuation=True)

    def continue_prediction_models_hpo(self, num_trials: Optional[int]):
        experiment_class = PredictionModelsHPO
        if experiment_class.is_existing_dir(self._results_path):
            with experiment_class.load_previous(self._results_path) as experiment:
                experiment.continue_hyperparameter_optimisation(num_trials)
        else:
            self.run_prediction_models_hpo()

    def continue_pretraining_hpo(self, num_trials: Optional[int]):
        experiment_class = PretrainHPO
        if experiment_class.is_existing_dir(self._results_path):
            with experiment_class.load_previous(self._results_path) as experiment:
                experiment.continue_hyperparameter_optimisation(num_trials)
        else:
            self.run_pretraining_hpo()

    def continue_simple_elecssl_hpo(self, num_trials: Optional[int]):
        experiment_class = SimpleElecsslHPO
        if experiment_class.is_existing_dir(self._results_path):
            with experiment_class.load_previous(self._results_path) as experiment:
                experiment.continue_hyperparameter_optimisation(num_trials)
        else:
            # We restrict this method to only allowing SimpleElecsslHPO to be used AFTER a PretrainHPO has been executed
            self.run_simple_elecssl_hpo(pretrain_experiment=PretrainHPO.load_previous(path=self._results_path))

    def continue_multivariable_elecssl_hpo(self, num_trials: Optional[int]):
        if MultivariableElecsslHPO.is_existing_dir(self._results_path):
            with MultivariableElecsslHPO.load_previous(self._results_path) as experiment:
                experiment.continue_hyperparameter_optimisation(num_trials)
        else:
            # We restrict this method to only allowing SimpleElecsslHPO to be used AFTER a PretrainHPO has been executed
            self.run_multivariable_elecssl_hpo(
                simple_elecssl_experiment=SimpleElecsslHPO.load_previous(path=self._results_path))

    # --------------
    # Test set integrity
    # --------------
    @staticmethod
    def verify_test_set_integrity(experiments: Tuple[MainExperiment, ...]):
        # Individual checks
        for experiment in experiments:
            experiment.integrity_check_test_set()

        # Check across the experiments
        check_equal_test_sets(experiments)

    # --------------
    # Properties
    # --------------
    @property
    def defaults_config(self):
        return copy.deepcopy(self._defaults_config)

    @property
    def specific_experiments_config(self):
        return copy.deepcopy(self._specific_experiments_config)

    @property
    def downstream_experiments_config(self):
        return copy.deepcopy(self._downstream_experiments_config)

    @property
    def pretext_experiments_config(self):
        return copy.deepcopy(self._pretext_experiments_config)

    @property
    def shared_hpds(self):
        return copy.deepcopy(self._shared_hpds)

    @property
    def specific_hpds(self):
        return copy.deepcopy(self._specific_hpds)


# --------------
# Exceptions
# --------------
class InconsistentTestSetError(Exception):
    """This error should be raised if the test set should be consistent across different trials/folds, but is not"""


class NonExclusiveTestSetError(Exception):
    """This error should be raised if there are subject in the test set which are also in train/validation for any
    trial/fold"""


class DissimilarTestSetsError(Exception):
    """This error should be raised if the test set across different experiments are not the same"""


class ExperimentNotFoundError(Exception):
    """This error should be raised if an experiment was attempted loaded, but it wasn't found at the expected path"""


# --------------
# Functions
# --------------
def _save_yaml_file(*, results_path: Path, config_file_name: str, config: Dict[str, Any], make_read_only: bool):
    """Method for saving a config file"""
    # Save the config file
    file_path = (results_path / config_file_name).with_suffix(".yml")
    with open(file_path, "w") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    # (Maybe) make it read only
    if make_read_only:
        os.chmod(file_path, 0o444)


def _generate_dataset_combinations(datasets):
    """
    Generate all possible combinations of datasets as tuples of strings.

    Parameters
    ----------
    datasets : tuple of str
        A tuple of dataset names (e.g., ('D1', 'D2', 'D3')).

    Returns
    -------
    list of tuple of str
        A list of tuples, where each tuple contains a combination of datasets as strings
        (e.g., ('D1',), ('D1', 'D2'), ('D1+D2',)).

    Examples
    --------
    >>> _generate_dataset_combinations(('D1', 'D2'))
    ('D1', 'D2', 'D1+D2')
    >>> _generate_dataset_combinations(('D1', 'D2', 'D3'))
    ('D1', 'D2', 'D3', 'D1+D2', 'D1+D3', 'D2+D3', 'D1+D2+D3')
    """
    combinations = []

    # Generate combinations for each non-empty subset of datasets
    for r in range(1, len(datasets) + 1):
        for combo in itertools.combinations(datasets, r):
            # Join dataset names together to form combinations like ('D1+D2')
            combo_str = '+'.join(combo)
            combinations.append(combo_str)

    return tuple(combinations)


def _datasets_str_to_tuple(datasets) -> Tuple[str, ...]:
    """
    Convert from dataset names which are merged by '+' to a tuple of strings

    Parameters
    ----------
    datasets : str

    Returns
    -------
    tuple[str, ...]

    Examples
    --------
    >>> _datasets_str_to_tuple("D1")
    ('D1',)
    >>> _datasets_str_to_tuple("D1+D3")
    ('D1', 'D3')
    >>> _datasets_str_to_tuple("D1+D4+D8+D10")
    ('D1', 'D4', 'D8', 'D10')
    >>> _datasets_str_to_tuple("D2")
    ('D2',)
    >>> _datasets_str_to_tuple("D3+D3+D3")
    ('D3', 'D3', 'D3')
    """
    return tuple(datasets.split("+"))


def verify_equal_test_sets(hpo_runs: Tuple[HPORun, ...]):
    # Get all test sets
    test_sets: List[Set[Subject]] = []
    for run in hpo_runs:
        test_set = run.experiment.get_test_subjects(path=run.path)
        assert isinstance(test_set, set), f"Expected test set to be a set, but found {type(test_set)}"
        assert all(isinstance(subject, Subject) for subject in test_set), \
            f"Expected subjects om test set to be of type 'Subject', but found {set(type(s) for s in test_set)}"
        test_sets.append(test_set)

    # Check if they are all equal
    if not all(test_set == test_sets[0] for test_set in test_sets):
        raise DissimilarTestSetsError


def check_equal_test_sets(hpo_experiments):
    hpo_runs = tuple(HPORun(experiment=experiment.__class__, path=experiment.results_path)
                     for experiment in hpo_experiments)
    verify_equal_test_sets(hpo_runs)


def _merge_ssl_biomarkers_dataframes(dfs):
    """
    Merges a list of DataFrames on the 'dataset' and 'sub_id' columns,
    and drops redundant 'clinical_target' columns if they exist. Assumes
    that the 'clinical_target' column exists and has identical values in all DataFrames.

    Parameters
    ----------
    dfs : list[pandas.DataFrame]
        A list of DataFrames to be merged. Each DataFrame should contain the columns
        'dataset', 'sub_id', and 'clinical_target'. The 'clinical_target' column
        is expected to have identical values across all DataFrames.

    Returns
    -------
    pandas.DataFrame
        A DataFrame resulting from the merge of the input DataFrames on 'dataset'
        and 'sub_id', with all redundant 'clinical_target' columns dropped, leaving only one.

    Raises
     ------
     ValueError
        If the 'dataset' and 'sub_id' identifiers do not match exactly across all DataFrames.

    Examples
    --------
    >>> my_df1 = pandas.DataFrame({"dataset": ["A", "B", "C"], "sub_id": ["1", "2", "3"],
    ...                            "clinical_target": [100, 200, 300], "col1": [10, 20, 30]})
    >>> my_df2 = pandas.DataFrame({"dataset": ["C", "B", "A"], "sub_id": ["3", "2", "1"],
    ...                            "clinical_target": [100, 200, 300], "col2": [100, 200, 300]})
    >>> my_df3 = pandas.DataFrame({"dataset": ["B", "A", "C"],"sub_id": ["2", "1", "3"],
    ...                            "clinical_target": [100, 200, 300], "col3": [5, 6, 7]})
    >>> my_dfs = [my_df1, my_df2, my_df3]
    >>> my_merged_df = _merge_ssl_biomarkers_dataframes(my_dfs)
    >>> my_merged_df
      dataset sub_id  clinical_target  col1  col2  col3
    0       A      1              100    10   300     6
    1       B      2              200    20   200     5
    2       C      3              300    30   100     7
    >>> my_df1 = pandas.DataFrame({"dataset": ["A", "B", "C", "D"], "sub_id": ["1", "2", "3", "4"],
    ...                            "clinical_target": [100, 200, 300, 400], "col1": [10, 20, 30, 40]})
    >>> my_df2 = pandas.DataFrame({"dataset": ["C", "B", "E"], "sub_id": ["3", "2", "5"],
    ...                            "clinical_target": [300, 200, 500], "col2": [100, 200, 300]})
    >>> _merge_ssl_biomarkers_dataframes([my_df1, my_df2])
    Traceback (most recent call last):
    ...
    ValueError: Identifiers in the 'dataset' and 'sub_id' columns do not match exactly between DataFrames.
    """
    # Ensure all DataFrames have the same identifiers
    base_identifiers = set(dfs[0][["dataset", "sub_id"]].apply(tuple, axis=1))
    for df in dfs[1:]:
        current_identifiers = set(df[["dataset", "sub_id"]].apply(tuple, axis=1))
        if base_identifiers != current_identifiers:
            raise ValueError("Identifiers in the 'dataset' and 'sub_id' columns do not match exactly between "
                             "DataFrames.")

    # Make copies of DataFrames and drop duplicate 'clinical_target' columns
    dfs = [dfs[0]] + [df.drop(columns=["clinical_target"], errors="ignore", inplace=False) for df in dfs[1:]]

    # Merge
    return reduce(lambda left, right: pandas.merge(left, right, on=["dataset", "sub_id"], how="inner"), dfs)


def _make_single_residuals_df(*, results_dir, feature_name, pseudo_target, downstream_target, deviation_method,
                              in_ocular_state, pretext_main_metric, experiment_name, include_pseudo_targets,
                              continuous_testing):
    outputs = _get_delta_and_variable(
        path=results_dir, target=pseudo_target, downstream_target=downstream_target, deviation_method=deviation_method,
        num_eeg_epochs=_get_num_eeg_epochs(ocular_state=in_ocular_state), pretext_main_metric=pretext_main_metric,
        experiment_name=experiment_name, include_pseudo_targets=verify_type(include_pseudo_targets, bool),
        continuous_testing=continuous_testing
    )

    biomarkers: Dict[str, List[Union[str, float]]] = {"dataset": [], "sub_id": [], "clinical_target": [],
                                                      feature_name: []}
    if include_pseudo_targets:
        subject_ids, deviations, clinical_targets, pseudo_targets = outputs
        biomarkers[f"pt_{pseudo_target}"] = []
    else:
        subject_ids, deviations, clinical_targets = outputs
        pseudo_targets = itertools.cycle((None,))

    for subject, deviation, target, pseudo_target_value in zip(subject_ids, deviations, clinical_targets,
                                                               pseudo_targets):
        # Add everything
        biomarkers["dataset"].append(subject.dataset_name)
        biomarkers["sub_id"].append(subject.subject_id)
        biomarkers["clinical_target"].append(target)
        biomarkers[feature_name].append(deviation)
        if include_pseudo_targets:
            biomarkers[f"pt_{pseudo_target}"].append(pseudo_target_value)

    # Make it a dataframe
    return pandas.DataFrame(biomarkers)


def _compute_biomarker_predictive_value(df, *, subject_split_config, test_split_config, ml_model_hp_config,
                                        ml_model_settings_config, save_test_predictions, results_dir, verbose):
    # Create the subject splitting
    non_test_subjects, test_subjects = simple_random_split(
        subjects=tuple(Subject(subject_id=row.sub_id, dataset_name=row.dataset)  # type: ignore[attr-defined]
                       for row in df.itertuples(index=False)),
        split_percent=test_split_config["split_percentage"], seed=test_split_config["seed"], require_seeding=True,
        sort_first=True
    )

    split_kwargs = {"dataset_subjects": subjects_tuple_to_dict(non_test_subjects), **subject_split_config["kwargs"]}
    biomarker_evaluation_splits = get_data_split(split=subject_split_config["name"], **split_kwargs).splits

    # Create ML model
    ml_model = MLModel(
        model=ml_model_hp_config["model"], model_kwargs=ml_model_hp_config["kwargs"],
        splits=biomarker_evaluation_splits,
        evaluation_metric=ml_model_settings_config["evaluation_metric"],
        aggregation_method=ml_model_settings_config["aggregation_method"]
    )

    # Set index because it is convenient
    df = df.copy()
    df["subject"] = [Subject(dataset_name=row.dataset, subject_id=row.sub_id) for row in df.itertuples(index=False)]
    df = df.set_index("subject")

    # Do evaluation (used as feedback to HPO algorithm)  todo: must implement splitting test
    score = ml_model.evaluate_features(non_test_df=df.loc[list(non_test_subjects)])
    if verbose:
        print(f"Training done! Obtained {ml_model_settings_config['aggregation_method']} "
              f"{ml_model_settings_config['evaluation_metric']} = {score}")

    # I will save the test results as well for convenience
    if verify_type(save_test_predictions, bool):
        test_predictions, test_scores = ml_model.predict_and_score(
            df=df.loc[list(test_subjects)], metrics=ml_model_settings_config["metrics"],
            aggregation_method=ml_model_settings_config["test_prediction_aggregation"]
        )

        # Convert to more convenient format
        test_predictions_dict: Dict[str, List[Union[str, float]]] = {"dataset": [], "sub_id": [], "pred": []}
        for subject, prediction in zip(test_subjects, test_predictions):
            test_predictions_dict["dataset"].append(subject.dataset_name)
            test_predictions_dict["sub_id"].append(subject.subject_id)
            test_predictions_dict["pred"].append(prediction)

        # Save predictions and scores on test set (the score provided to the HPO algorithm should be stored by
        # optuna)
        test_predictions_df = pandas.DataFrame(test_predictions_dict)
        test_scores_df = pandas.DataFrame([test_scores])

        test_predictions_df = test_predictions_df.round(
            decimals=ml_model_settings_config["test_predictions_decimals"]
        )
        test_scores_df = test_scores_df.round(decimals=ml_model_settings_config["test_scores_decimals"])

        test_predictions_df.to_csv(results_dir / "test_predictions.csv", index=False)
        test_scores_df.to_csv(results_dir / "test_scores.csv", index=False)

    return score


def _excluded_dataset_only(*, dataset_config, subject_split_config):
    """
    Check if the only dataset in the dataset config is a hold-out dataset in the subject split config

    Parameters
    ----------
    dataset_config : dict[str, Any]
    subject_split_config : dict[str, Any]

    Returns
    -------
    bool

    Examples
    --------
    >>> _excluded_dataset_only(dataset_config={"L": {"num_subjects": 20}},
    ...                        subject_split_config={"kwargs": {"left_out_dataset": "L"}})
    True
    >>> _excluded_dataset_only(dataset_config={"L": {"num_subjects": 20}},
    ...                        subject_split_config={"kwargs": {"left_out_dataset": "B"}})
    False
    >>> _excluded_dataset_only(dataset_config={"L": {"num_subjects": 20}, "B": {"num_subjects": 30}},
    ...                        subject_split_config={"kwargs": {"left_out_dataset": "L"}})
    False
    """
    # todo: quite hard-coding tbh...
    # Check if more than one datasets
    datasets = tuple(dataset_config.keys())
    if len(datasets) > 1:
        return False

    # Check if there even is a hold-out dataset
    if "left_out_dataset" not in subject_split_config["kwargs"]:
        return False

    # Make check
    assert len(datasets) == 1
    dataset = datasets[0]

    return dataset == subject_split_config["kwargs"]["left_out_dataset"]


def _get_delta_and_variable(path, *, target, downstream_target, deviation_method, num_eeg_epochs, pretext_main_metric,
                            experiment_name, include_pseudo_targets, continuous_testing):
    # todo: make test
    # ----------------
    # Select epoch
    # ----------------
    # todo: not really 'fold' anymore...
    if verify_type(continuous_testing, bool):
        epoch = _get_best_val_epoch(path=path, pretext_main_metric=pretext_main_metric, experiment_name=experiment_name)
    else:
        epoch = 0  # The selection has already been made

    # ----------------
    # Get the biomarkers and the (clinical) variable
    # ----------------
    # todo: no need to read the entire csv file...
    prefix_name = "" if experiment_name is None else f"{experiment_name}_"
    test_predictions = pandas.read_csv(os.path.join(path, f"{prefix_name}test_history_predictions.csv"))
    subject_ids = test_predictions["sub_id"]

    # Check the number of datasets in the test set
    datasets = set(test_predictions["dataset"])
    if len(datasets) != 1:
        raise NotImplementedError(f"This implementation only works when a single dataset in the test set predictions "
                                  f"was used. This was not the case for {path}. Found {datasets}")
    dataset_name = tuple(datasets)[0]

    # Average the predictions per EEG epoch
    i0 = num_eeg_epochs * epoch + 2
    i1 = i0 + num_eeg_epochs
    predictions = test_predictions.iloc[:, i0:i1].mean(axis=1)

    # Get pseudo-targets
    pseudo_targets = get_dataset(dataset_name).load_targets(target=target, subject_ids=subject_ids)

    # Get the clinical variable
    var = get_dataset(dataset_name).load_targets(target=downstream_target, subject_ids=subject_ids)

    # Remove nan values  todo: should not be necessary...
    mask = ~numpy.isnan(var).copy()

    pseudo_targets = pseudo_targets[mask]  # type: ignore
    predictions = predictions[mask]
    var = var[mask]  # type: ignore
    subject_ids = subject_ids[mask]

    # Get the deviation
    if deviation_method in ("delta", "gap", "diff", "difference"):
        delta = predictions - pseudo_targets
    elif deviation_method == "ratio":
        delta = predictions / pseudo_targets
    else:
        raise ValueError(f"Unrecognised method: {deviation_method}")

    subjects = tuple(Subject(subject_id=sub_id, dataset_name=dataset_name) for sub_id in subject_ids)

    # Return
    if verify_type(include_pseudo_targets, bool):
        return subjects, delta, var, pseudo_targets

    return subjects, delta, var


def _get_aggregated_val_score(*, trial_results_dir, aggregation_method, metric):
    """Get the validation score of a trial"""
    eval_method = max if higher_is_better(metric=metric) else min

    # Get scores from all folds. Using best scores
    scores: List[float] = []
    for fold in os.listdir(trial_results_dir):
        if not os.path.isdir(trial_results_dir / fold):
            # All folders are assumed to be folds, for allowing possible changes in the future
            continue
        df = pandas.read_csv(trial_results_dir / fold / "val_history_metrics.csv")
        score = eval_method(df[metric])
        scores.append(verified_performance_score(score=score, metric=metric))

    # Aggregate and return
    if aggregation_method == "mean":
        return numpy.mean(scores)
    elif aggregation_method == "median":
        return numpy.median(scores)
    raise ValueError(f"Method for aggregating the validation scores across folds was not recognised: "
                     f"{aggregation_method}")


def _get_preprocessing_config_path(ocular_state):
    # Get file names
    preprocessing_path = get_numpy_data_storage_path() / f"preprocessed_band_pass_{ocular_state}"
    config_files = tuple(file_name for file_name in os.listdir(preprocessing_path) if file_name.startswith("config"))

    # Make sure there is only one and return it
    assert len(config_files) == 1, f"Expected only one config file, but found {len(config_files)}: {config_files}"
    return preprocessing_path / config_files[0]


def _get_num_eeg_epochs(ocular_state):
    # Get file names
    preprocessing_path = get_numpy_data_storage_path() / f"preprocessed_band_pass_{ocular_state}"
    config_files = tuple(file_name for file_name in os.listdir(preprocessing_path) if file_name.startswith("config"))

    # Make sure there is only one and return it
    assert len(config_files) == 1, f"Expected only one config file, but found {len(config_files)}: {config_files}"
    with open(preprocessing_path / config_files[0]) as file:
        preprocessing_config = yaml.safe_load(file)

    return preprocessing_config["Details"]["num_epochs"]


def _get_best_val_epoch(path, experiment_name, *, pretext_main_metric):
    # Load the .csv file with the metrics
    prefix_name = "" if experiment_name is None else f"{experiment_name}_"
    val_df = pandas.read_csv(os.path.join(path, f"{prefix_name}val_history_metrics.csv"))

    # Get the epoch which maximises the performance
    if higher_is_better(metric=pretext_main_metric):
        return numpy.argmax(val_df[pretext_main_metric])
    else:
        return numpy.argmin(val_df[pretext_main_metric])


def _get_warning(warning):
    if warning == "NearConstantInputWarning":
        return NearConstantInputWarning
    elif warning == "ConstantInputWarning":
        return ConstantInputWarning
    elif warning == "UserWarning":
        return UserWarning
    elif warning == "PlotNotSavedWarning":
        return PlotNotSavedWarning
    elif warning == "FutureWarning":
        return FutureWarning
    else:
        raise ValueError(f"Warning {warning} not understood")


def _get_prepared_experiments_config(experiments_config, in_freq_band, in_ocular_state, suggested_hyperparameters):
    # Load the preprocessing file and add some necessary info
    with open(_get_preprocessing_config_path(ocular_state=in_ocular_state)) as file:
        f_max = yaml.safe_load(file)["FrequencyBands"][in_freq_band][-1]
    preprocessing_config_file = suggested_hyperparameters["Preprocessing"].copy()
    _resample = f_max * preprocessing_config_file['sfreq_multiple'] * preprocessing_config_file['input_length']
    preprocessing_config_file["resample"] = _resample

    # Add the number of input time steps if required for the DL model
    if "num_time_steps" in suggested_hyperparameters["DLArchitecture"]["kwargs"] \
            and suggested_hyperparameters["DLArchitecture"]["kwargs"]["num_time_steps"] == "UNAVAILABLE":
        _resample = preprocessing_config_file["resample"]
        suggested_hyperparameters["DLArchitecture"]["kwargs"]["num_time_steps"] = _resample

    # Add the sampling frequency if required for the DL model
    if "sampling_freq" in suggested_hyperparameters["DLArchitecture"]["kwargs"] \
            and suggested_hyperparameters["DLArchitecture"]["kwargs"]["sampling_freq"] == "UNAVAILABLE":
        sampling_freq = preprocessing_config_file['sfreq_multiple'] * f_max
        suggested_hyperparameters["DLArchitecture"]["kwargs"]["sampling_freq"] = sampling_freq

    # Add the preprocessed version to all datasets
    preprocessed_version = (f"preprocessed_band_pass_{in_ocular_state}/data--band_pass-{in_freq_band}--"
                            f"input_length-{preprocessing_config_file['input_length']}s--"
                            f"autoreject-{preprocessing_config_file['autoreject']}--"
                            f"sfreq-{preprocessing_config_file['sfreq_multiple']}fmax")

    experiments_config = experiments_config.copy()
    for dataset_info in experiments_config["Datasets"].values():
        dataset_info["pre_processed_version"] = preprocessed_version
    return experiments_config, preprocessing_config_file
