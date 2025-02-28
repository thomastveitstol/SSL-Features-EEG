import abc
import concurrent
import os
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Any, Callable, Tuple, List, Iterable, Literal, Set, Union

import numpy
import optuna
import pandas
import yaml  # type: ignore[import-untyped]
from progressbar import progressbar
from scipy.stats import NearConstantInputWarning, ConstantInputWarning

from elecssl.data.datasets.getter import get_dataset
from elecssl.data.paths import get_numpy_data_storage_path
from elecssl.data.results_analysis.hyperparameters import to_hyperparameter
from elecssl.data.results_analysis.utils import load_hpo_study
from elecssl.data.subject_split import Subject, subjects_tuple_to_dict, get_data_split, simple_random_split
from elecssl.models.experiments.single_experiment import SingleExperiment
from elecssl.models.hp_suggesting import make_trial_suggestion, \
    suggest_spatial_dimension_mismatch, suggest_loss, suggest_dl_architecture, get_optuna_sampler
from elecssl.models.metrics import PlotNotSavedWarning, higher_is_better
from elecssl.models.ml_models.ml_model_base import MLModel
from elecssl.models.sampling_distributions import get_yaml_loader
from elecssl.models.utils import add_yaml_constructors, verify_type, merge_dicts, \
    verified_performance_score, merge_dicts_strict


class HPOExperiment(abc.ABC):
    """
    Base class for running hyperparameter optimisation
    """

    __slots__ = ("_experiments_config", "_sampling_config", "_results_path")
    _name: str

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
            with open(self._results_path / "finished_successfully.txt", "w"):
                pass
            return

        # Otherwise, document the error received in a text file
        with open((self._results_path / exc_type.__name__).with_suffix(".txt"), "w") as file:
            file.write("Traceback (most recent call last):\n")
            traceback.print_tb(exc_tb, file=file)
            file.write(f"{exc_type.__name__}: {exc_val}")

    def __init__(self, hp_config_paths: Tuple[Path, ...], experiments_config_paths: Tuple[Path, ...],
                 results_dir: Path):
        # ---------------
        # Load HP distributions files
        # ---------------
        # Merge all HP distribution files. Typically (most likely always) a shared HPD yml file and a specific one
        hp_config = _merge_config_files_from_paths(hp_config_paths)

        # ---------------
        # Load the other configurations
        # ---------------
        # Merge all configuration setting files.
        experiments_config = _merge_config_files_from_paths(experiments_config_paths)

        # ---------------
        # Set attributes
        # ---------------
        self._experiments_config: Dict[str, Any] = experiments_config
        self._sampling_config: Dict[str, Any] = hp_config
        _debug_mode = "debug_" if verify_type(self._experiments_config["debugging"], bool) else ""
        self._results_path = results_dir / self._name / (f"{_debug_mode}{self._name}_hpo_experiment_{date.today()}_"
                                                         f"{datetime.now().strftime('%H%M%S')}")

        # Make directory
        os.mkdir(self._results_path)

        # Save the config files
        with open(self._results_path / "experiments_config.yml", "w") as file:
            yaml.safe_dump(experiments_config, file)
        with open(self._results_path / "hpd_config.yml", "w") as file:
            yaml.safe_dump(hp_config, file)

    def _get_hpo_folder_path(self, trial: optuna.Trial):
        return self._results_path / f"hpo_{trial.number}_{date.today()}_{datetime.now().strftime('%H%M%S')}"

    def run_hyperparameter_optimisation(self):
        """Run HPO with optuna"""
        # Create study
        study_name = f"{self._name}-study"
        storage_path = (self._results_path / study_name).with_suffix(".db")
        sampler = get_optuna_sampler(self.hpo_study_config["HPOStudy"]["sampler"],
                                     **self.hpo_study_config["HPOStudy"]["sampler_kwargs"])
        study = optuna.create_study(study_name=study_name, storage=f"sqlite:///{storage_path}", sampler=sampler,
                                    direction=self.hpo_study_config["HPOStudy"]["direction"])

        # Optimise
        with warnings.catch_warnings():
            for warning in self._experiments_config["Warnings"]["ignore"]:
                warnings.filterwarnings(action="ignore", category=_get_warning(warning))
            study.optimize(self._create_objective(), n_trials=self.hpo_study_config["num_trials"])

    @abc.abstractmethod
    def _create_objective(self) -> Callable[[optuna.Trial], float]:
        """Method which returns the function to study.optimise. It needs to take a trial argument of type optuna.Trial
        and return a score"""

    def _suggest_common_hyperparameters(self, trial, name, in_freq_band, preprocessed_config_path):
        suggested_hps: Dict[str, Any] = {"Preprocessing": {}}

        # Preprocessing
        for param_name, (distribution, distribution_kwargs) in self._sampling_config["Preprocessing"].items():
            suggested_hps["Preprocessing"][param_name] = make_trial_suggestion(
                trial=trial, name=f"{name}_{param_name}", method=distribution, kwargs=distribution_kwargs
            )

        # Training
        suggested_hps["Training"] = self._suggest_training_hpcs(trial=trial, name=name,
                                                                hpd_config=self._sampling_config)

        # Normalisation
        normalisation = trial.suggest_categorical(f"{name}_normalisation", **self._sampling_config["normalisation"])

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
            suggested_hps["Training"]["method"] = "downstream_training"
            suggested_hps["DomainDiscriminator"] = None

        return suggested_hps

    @staticmethod
    def _suggest_training_hpcs(trial, name, hpd_config):
        # Training
        suggested_train_hpcs = dict()
        for param_name, (distribution, distribution_kwargs) in hpd_config["Training"].items():
            suggested_train_hpcs[param_name] = make_trial_suggestion(
                trial=trial, name=f"{name}_{param_name}", method=distribution, kwargs=distribution_kwargs
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

        # todo: the 'debug_' prefix should be removed
        hpo_iterations = tuple(folder for folder in os.listdir(path) if os.path.isdir(path / folder)
                               and (folder.startswith(cls._name) or folder.startswith(f"debug_{cls._name}")))
        for hpo_iteration in progressbar(hpo_iterations, redirect_stdout=True, prefix="Trial "):
            trial_path = path / hpo_iteration

            # Initialise dictionaries with all scores for the trial
            trial_val_scores: Dict[str, List[float]] = {metric: [] for metric in target_metrics}
            trial_test_scores: Dict[str, List[float]] = {metric: [] for metric in target_metrics}

            # Get the performance for each fold
            folds = (fold for fold in os.listdir(trial_path) if os.path.isdir(trial_path / fold)
                     and fold.startswith("Fold_"))
            for fold in folds:
                # Get fold scores
                fold_val_scores, fold_test_scores = cls._get_performance_scores(
                    trial_path / fold, selection_metric=selection_metric, target_metrics=target_metrics
                )

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
            _indicator = "hpo_"
            start_idx = hpo_iteration.find(_indicator) + len(_indicator)
            end_idx = hpo_iteration.find('_', start_idx)

            scores["trial_number"].append(int(hpo_iteration[start_idx:end_idx]))
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
        # Check if the test set always contain the same set of subject
        test_subjects = cls._verify_test_set_consistency(path=path)

        # Check if any of the test subjects were in training or validation
        cls._verify_test_set_exclusivity(path=path, test_subjects=test_subjects)

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
        trial_folders = (file_name for file_name in os.listdir(path)
                         if file_name.startswith(cls._name) and os.path.isdir(file_name))
        for trial_folder in trial_folders:
            trial_path = path / trial_folder

            # Loop through all folds within the trial
            fold_folders = (name for name in os.listdir(trial_path)
                            if name.lower().startswith("fold_") and os.path.isdir(name))
            for fold_folder in fold_folders:
                fold_path = path / fold_folder

                # Load the subjects from the test predictions, but also accept that some trials may have been pruned
                try:
                    test_history_subjects = pandas.read_csv(fold_path / "test_history_predictions.csv",
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
        trial_folders = (file_name for file_name in os.listdir(path)
                         if file_name.startswith(cls._name) and os.path.isdir(file_name))
        for trial_folder in trial_folders:
            trial_path = path / trial_folder

            # Loop through all folds within the trial
            fold_folders = (name for name in os.listdir(trial_path)
                            if name.lower().startswith("fold_") and os.path.isdir(name))
            for fold_folder in fold_folders:
                fold_path = path / fold_folder

                # Load the subjects from the train and validation predictions, but also accept that some trials may have
                # been pruned
                try:
                    train_subjects_df = pandas.read_csv(fold_path / "train_history_predictions.csv",
                                                        usecols=("dataset", "sub_id"))
                    val_subjects_df = pandas.read_csv(fold_path / "val_history_predictions.csv",
                                                      usecols=("dataset", "sub_id"))
                except FileNotFoundError:
                    continue

                # Convert to set of 'Subject'
                train_subjects = set(
                    Subject(dataset_name=row.dataset, subject_id=row.sub_id)  # type: ignore[attr-defined]
                    for row in train_subjects_df.itertuples(index=False))
                val_subjects = set(
                    Subject(dataset_name=row.dataset, subject_id=row.sub_id)  # type: ignore[attr-defined]
                    for row in val_subjects_df.itertuples(index=False))

                # Check if there is overlap
                if not train_subjects.isdisjoint(test_subjects):
                    overlap = train_subjects & test_subjects
                    raise NonExclusiveTestSetError(
                        f"Test subjects were found in the train set for trial {trial_folder}, fold {fold_folder}. "
                        f"These subjects are (N={len(overlap)})): {overlap}"
                    )
                if not val_subjects.isdisjoint(test_subjects):
                    overlap = val_subjects & test_subjects
                    raise NonExclusiveTestSetError(
                        f"Test subjects were found in the validation set for trial {trial_folder}, fold {fold_folder}. "
                        f"These subjects are (N={len(overlap)})): {overlap}"
                    )

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
    # Properties
    # --------------
    @property
    def hpo_study_config(self):
        return self._experiments_config["HPO"]


class PredictionModelsHPO(HPOExperiment):
    """
    Class for the prediction models
    """

    __slots__ = ()
    _name = "prediction_models"

    def _create_objective(self):
        def _objective(trial: optuna.Trial):
            # ---------------
            # Suggest / sample hyperparameters
            # ---------------
            in_freq_band = "all"  # todo: hard coded :(

            suggested_hyperparameters = self.suggest_hyperparameters(
                name=self._name, trial=trial, in_freq_band=in_freq_band
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
            results_dir = self._get_hpo_folder_path(trial)
            with SingleExperiment(hp_config=suggested_hyperparameters, pre_processing_config=preprocessing_config_file,
                                  experiments_config=experiments_config, results_path=results_dir,
                                  fine_tuning=None, experiment_name=None) as experiment:
                experiment.run_experiment()

            # ---------------
            # Get the performance
            # ---------------
            return _get_aggregated_val_score(trial_results_dir=results_dir, metric=self.train_config["main_metric"],
                                             aggregation_method=self._experiments_config["val_scores_aggregation"])

        return _objective

    def suggest_hyperparameters(self, trial, name, in_freq_band):
        in_ocular_state = trial.suggest_categorical(f"{name}_ocular_state", **self._sampling_config["OcularStates"])
        preprocessing_config_path = _get_preprocessing_config_path(ocular_state=in_ocular_state)
        suggested_hps = self._suggest_common_hyperparameters(trial, name, in_freq_band=in_freq_band,
                                                             preprocessed_config_path=preprocessing_config_path)
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

    def __init__(self, hp_config_paths: Tuple[Path, ...], experiments_config_paths: Tuple[Path, ...],
                 downstream_hp_config_paths: Tuple[Path, ...], downstream_experiments_config_paths: Tuple[Path, ...],
                 pretext_hp_config_paths: Tuple[Path, ...], pretext_experiments_config_paths: Tuple[Path, ...],
                 results_dir: Path):
        """The 'hp_config_paths' and 'experiments_config_paths' are shared configurations for the pretext and downstream
        task, such as if using no datasets for pre-training should lead to a pruning of the trial"""
        # ---------------
        # Initialise and create everything for the shared stuff
        # ---------------
        super().__init__(hp_config_paths, experiments_config_paths, results_dir)

        # ---------------
        # Create the pretext-specific configurations
        # ---------------
        self._downstream_experiments_config = _merge_config_files_from_paths(downstream_experiments_config_paths)
        self._downstream_sampling_config = _merge_config_files_from_paths(downstream_hp_config_paths)

        self._pretext_experiments_config = _merge_config_files_from_paths(pretext_experiments_config_paths)
        self._pretext_sampling_config = _merge_config_files_from_paths(pretext_hp_config_paths)

        # ---------------
        # Save the config files
        # ---------------
        # Downstream config files
        with open(self._results_path / "downstream_experiments_config.yml", "w") as file:
            yaml.safe_dump(self._downstream_experiments_config, file)
        with open(self._results_path / "downstream_hpd_config.yml", "w") as file:
            yaml.safe_dump(self._downstream_sampling_config, file)

        # Pretext config files
        with open(self._results_path / "pretext_experiments_config.yml", "w") as file:
            yaml.safe_dump(self._pretext_experiments_config, file)
        with open(self._results_path / "pretext_hpd_config.yml", "w") as file:
            yaml.safe_dump(self._pretext_sampling_config, file)

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
                name=self._name, trial=trial, in_freq_band=in_freq_band
            )

            # These HPCs are specific to the pretext task
            pretext_specific_hpcs, datasets_to_use = self._suggest_pretext_specific_hyperparameters(name="pretext",
                                                                                                    trial=trial)

            # These HPCs are specific to the downstream task
            downstream_specific_hpcs = self._suggest_downstream_specific_hyperparameters(name="downstream", trial=trial)

            # Combine the shared and specific HPCs
            downstream_hpcs = {**suggested_shared_hyperparameters, **downstream_specific_hpcs}

            suggested_shared_hyperparameters = suggested_shared_hyperparameters.copy()  # Remove train and loss HPCs
            del suggested_shared_hyperparameters["Training"]
            del suggested_shared_hyperparameters["Loss"]
            del suggested_shared_hyperparameters["DomainDiscriminator"]
            pretext_hpcs = {**suggested_shared_hyperparameters.copy(), **pretext_specific_hpcs}

            # ---------------
            # Some additions to the experiment configs
            # ---------------
            # For the pretext task
            incomplete_pretext_experiments_config = self._pretext_experiments_config.copy()

            # Add the selected datasets to pretext task
            assert "Datasets" not in incomplete_pretext_experiments_config
            incomplete_pretext_experiments_config["Datasets"] = dict()
            for dataset_name, dataset_info in datasets_to_use.items():
                incomplete_pretext_experiments_config["Datasets"][dataset_name] = dataset_info

            pretext_experiments_config, preprocessing_config_file = _get_prepared_experiments_config(
                experiments_config=incomplete_pretext_experiments_config, in_freq_band=in_freq_band,
                in_ocular_state=in_ocular_state, suggested_hyperparameters=pretext_hpcs
            )

            # Must set saving model of pretext task to true
            pretext_experiments_config["Saving"]["save_model"] = True

            # Adding target
            target_name = f"band_power_{pretext_specific_hpcs['out_freq_band']}_{out_ocular_state}"
            pretext_experiments_config["Training"]["target"] = target_name

            # Downstream task
            downstream_experiments_config, _ = _get_prepared_experiments_config(
                experiments_config=self._downstream_experiments_config.copy(), in_freq_band=in_freq_band,
                in_ocular_state=in_ocular_state, suggested_hyperparameters=downstream_hpcs
            )

            # ---------------
            # Train on pretext task
            # ---------------
            # Make directory for current iteration
            results_dir = self._get_hpo_folder_path(trial)

            # Only pre-train if we have datasets to pre-train on. Trial pruning is handled elsewhere
            if pretext_experiments_config["Datasets"]:
                with SingleExperiment(hp_config=pretext_hpcs, pre_processing_config=preprocessing_config_file,
                                      experiments_config=pretext_experiments_config, results_path=results_dir,
                                      fine_tuning=None, experiment_name="pretext") as experiment:
                    experiment.run_experiment()

            # ---------------
            # Train on downstream task
            # ---------------
            fine_tuning = "pretext" if pretext_experiments_config["Datasets"] else None
            with SingleExperiment(hp_config=downstream_hpcs, pre_processing_config=preprocessing_config_file,
                                  experiments_config=downstream_experiments_config, results_path=results_dir,
                                  fine_tuning=fine_tuning, experiment_name=None) as experiment:
                experiment.run_experiment()

            # ---------------
            # Return the performance
            # ---------------
            return _get_aggregated_val_score(
                trial_results_dir=results_dir, metric=self._downstream_experiments_config["Training"]["main_metric"],
                aggregation_method=self._downstream_experiments_config["val_scores_aggregation"]
            )

        return _objective

    def _suggest_pretext_specific_hyperparameters(self, trial, name):
        suggested_hps = dict()

        # Suggest e.g. alpha or beta band power
        suggested_hps["out_freq_band"] = trial.suggest_categorical(
            name=f"{name}_out_freq_band", **self._pretext_sampling_config["out_freq_band"]
        )
        suggested_hps["pretext_main_metric"] = trial.suggest_categorical(
            name=f"{name}_selection_metric", **self._pretext_sampling_config["selection_metric"]
        )

        # Pick the datasets to be used for pre-training
        datasets_to_use = dict()
        for dataset_name, dataset_info in self._pretext_sampling_config["Datasets"].items():
            to_use = trial.suggest_categorical(name=f"{name}_{dataset_name}", choices={True, False})
            if to_use:
                datasets_to_use[dataset_name] = dataset_info

        # (Maybe) prune the trial
        if not datasets_to_use and self._experiments_config["force_pretraining"]:
            raise optuna.TrialPruned

        # Training
        suggested_hps["Training"] = self._suggest_training_hpcs(trial=trial, name=name,
                                                                hpd_config=self._pretext_sampling_config)

        # Loss
        suggested_hps["Loss"] = suggest_loss(name=name, trial=trial, config=self._pretext_sampling_config["Loss"])

        # Domain discriminator
        if self._experiments_config["enable_domain_discriminator"]:
            raise NotImplementedError(
                "Hyperparameter sampling with domain discriminator has not been implemented yet...")
        else:
            # todo: not a fan of having to specify training method, especially now that the name is somewhat misleading
            suggested_hps["Training"]["method"] = "downstream_training"
            suggested_hps["DomainDiscriminator"] = None

        return suggested_hps, datasets_to_use

    @staticmethod
    def _suggest_downstream_specific_hyperparameters(name, trial):
        return dict()  # todo: should sample Adam HPCs as it makes no sense to use the same as in the pretext task

    def _suggest_shared_hyperparameters(self, trial, name, in_freq_band):
        preprocessing_config_path = _get_preprocessing_config_path(
            ocular_state=self._experiments_config["in_ocular_state"]
        )
        suggested_hps = self._suggest_common_hyperparameters(trial, name, in_freq_band=in_freq_band,
                                                             preprocessed_config_path=preprocessing_config_path)
        return suggested_hps

    # --------------
    # Properties
    # --------------
    @property
    def train_config(self):
        return {**self._sampling_config["Training"], **self._experiments_config["Training"]}


class ElecsslHPO(HPOExperiment):
    """
    Class for using the learned residuals as input to an ML model
    """

    __slots__ = ()
    _name = "elecssl"

    def suggest_hyperparameters(self, trial, name, in_freq_band, preprocessing_config_path):
        suggested_hps = self._suggest_common_hyperparameters(trial, name, in_freq_band=in_freq_band,
                                                             preprocessed_config_path=preprocessing_config_path)
        suggested_hps["MLModel"] = self._sampling_config["MLModel"]
        return suggested_hps

    def _create_objective(self):

        def _objective(trial: optuna.Trial):
            # ---------------
            # Create configurations for all feature extractors
            # ---------------
            spaces = self._experiments_config["IOSpaces"]

            # Make directory for current iteration
            results_dir = self._get_hpo_folder_path(trial)
            os.mkdir(results_dir)

            # ---------------
            # Using multiprocessing  # todo: should turn this off if using GPU?
            # ---------------
            with ProcessPoolExecutor(max_workers=self._experiments_config["MultiProcessing"]["max_workers"]) as ex:
                print("Multiprocessing")
                experiments = []
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
                    suggested_hyperparameters = self.suggest_hyperparameters(
                        name=feature_extractor_name, trial=trial, in_freq_band=in_freq_band,
                        preprocessing_config_path=preprocessing_config_path
                    )

                    # ---------------
                    # Initiate experiment
                    # ---------------
                    experiments.append(
                        ex.submit(
                            self._run_single_job, experiments_config=experiment_config_file, trial_number=trial.number,
                            suggested_hyperparameters=suggested_hyperparameters, in_ocular_state=in_ocular_state,
                            out_ocular_state=out_ocular_state, in_freq_band=in_freq_band, out_freq_band=out_freq_band,
                            results_dir=results_dir, clinical_target=self._experiments_config["clinical_target"],
                            deviation_method=self._experiments_config["deviation_method"],
                            log_transform_clinical_target=self._experiments_config["log_transform_clinical_target"],
                            num_eeg_epochs=num_epochs, feature_extractor_name=feature_extractor_name,
                            pretext_main_metric=self._experiments_config["pretext_main_metric"]
                        )
                    )

                # Collect the resulting 'biomarkers'
                biomarkers: Dict[Subject, Dict[str, float]] = dict()
                for process in concurrent.futures.as_completed(experiments):
                    subjects, deviations, clinical_targets, feature_extractor_name = process.result()
                    for subject, deviation, target in zip(subjects, deviations, clinical_targets):
                        # Maybe add the target (not optimal code...)
                        if subject not in biomarkers:
                            biomarkers[subject] = {"clinical_target": target}

                        # Add the deviation
                        biomarkers[subject][feature_extractor_name] = deviation

                # Make it a dataframe and save it
                df = pandas.DataFrame.from_dict(biomarkers, orient="index")
                df.to_csv(results_dir / "ssl_biomarkers.csv")

            # ---------------
            # Use the biomarkers
            # ---------------
            # Create the subject splitting
            non_test_subjects, test_subjects = simple_random_split(
                subjects=biomarkers.keys(), split_percent=self._experiments_config["TestSplit"]["split_percentage"],
                seed=self._experiments_config["TestSplit"]["seed"], require_seeding=True
            )

            split_kwargs = {"dataset_subjects": subjects_tuple_to_dict(non_test_subjects),
                            **self._experiments_config["MLModelSubjectSplit"]["kwargs"]}
            biomarker_evaluation_splits = get_data_split(
                split=self._experiments_config["MLModelSubjectSplit"]["name"], **split_kwargs
            ).splits

            # Create ML model
            ml_model = MLModel(
                model=self.ml_model_hp_config["model"], model_kwargs=self.ml_model_hp_config["kwargs"],
                splits=biomarker_evaluation_splits,
                evaluation_metric=self.ml_model_settings_config["evaluation_metric"],
                aggregation_method=self.ml_model_settings_config["aggregation_method"]
            )

            # Do evaluation (used as feedback to HPO algorithm)  todo: must implement splitting test
            score = ml_model.evaluate_features(non_test_df=df.loc[list(non_test_subjects)])
            print(f"Training done! Obtained {self.ml_model_settings_config['aggregation_method']} "
                  f"{self.ml_model_settings_config['evaluation_metric']} = {score}")

            # I will save the test results as well. Although it is conventional to not even open the test set before the
            # HPO, keep in mind that as long the feedback to the HPO is not related to the test performance (and test
            # set), then it will not bias the experiments. The model selection is also purely based on non-test set
            # evaluated performance. Evaluating on the test set for every iteration may be interesting from an ML
            # developers perspective, as one can e.g. find out if there are indications on the HPO leading to
            # overfitting on the non-test set (although it would not be valid performance estimation to go back after
            # checking the test set performance). Maybe I should call it "optimisation excluded set"?
            if verify_type(self._experiments_config["save_test_predictions"], bool):
                test_predictions, test_scores = ml_model.predict_and_score(
                    df=df.loc[list(test_subjects)], metrics=self.ml_model_settings_config["metrics"],
                    aggregation_method=self.ml_model_settings_config["test_prediction_aggregation"]
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
                    decimals=self.ml_model_settings_config["test_predictions_decimals"]
                )
                test_scores_df = test_scores_df.round(decimals=self.ml_model_settings_config["test_scores_decimals"])

                test_predictions_df.to_csv(results_dir / "test_predictions.csv", index=False)
                test_scores_df.to_csv(results_dir / "test_scores.csv", index=False)

            return score

        return _objective

    @staticmethod
    def _run_single_job(experiments_config, suggested_hyperparameters, trial_number, in_ocular_state, out_ocular_state,
                        in_freq_band, out_freq_band, results_dir, clinical_target, deviation_method, num_eeg_epochs,
                        log_transform_clinical_target, pretext_main_metric, feature_extractor_name):
        """Method for running a single SSL experiments"""
        experiments_config, preprocessing_config_file = _get_prepared_experiments_config(
            experiments_config=experiments_config, in_freq_band=in_freq_band, in_ocular_state=in_ocular_state,
            suggested_hyperparameters=suggested_hyperparameters
        )

        # ---------------
        # Learn on the pretext regression task
        # ---------------
        experiment_name = "pretext"
        results_path = results_dir / (f"hpo_{trial_number}_{feature_extractor_name}_{date.today()}_"
                                      f"{datetime.now().strftime('%H%M%S')}")
        with SingleExperiment(hp_config=suggested_hyperparameters, experiments_config=experiments_config,
                              pre_processing_config=preprocessing_config_file, results_path=results_path,
                              fine_tuning=None, experiment_name=experiment_name) as experiment:
            experiment.run_experiment()

        # ---------------
        # Extract expectation values and biomarkers
        # ---------------
        # todo: a better solution could be to make the 'run_experiment' return the features...
        # todo: slightly hard-coded target
        subject_ids, deviation, clinical_target = _get_delta_and_variable(
            path=results_path / "Fold_0", target=f"band_power_{out_freq_band}_{out_ocular_state}",
            variable=clinical_target, deviation_method=deviation_method, log_var=log_transform_clinical_target,
            num_eeg_epochs=num_eeg_epochs, pretext_main_metric=pretext_main_metric, experiment_name=experiment_name
        )

        return subject_ids, deviation, clinical_target, feature_extractor_name

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
    def verify_test_set_integrity(cls, path):
        """This class (1) stores test predictions differently and (2) does not currently save train and validation
        predictions for the ML model. Hence, only checking for test set consistency"""
        expected_subjects: Set[Subject] = set()

        # Loop through all trials
        trial_folders = (file_name for file_name in os.listdir(path)
                         if file_name.startswith("hpo_") and os.path.isdir(file_name))
        for trial_folder in trial_folders:
            trial_path = path / trial_folder

            # Load the subjects from the test predictions, but also accept that some trials may have been pruned
            try:
                test_subjects_df = pandas.read_csv(trial_path / "test_predictions.csv", usecols=("dataset", "sub_id"))
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
# Exceptions
# --------------
class InconsistentTestSetError(Exception):
    """This error should be raised if the test set should be consistent across different trials/folds, but is not"""


class NonExclusiveTestSetError(Exception):
    """This error should be raised if there are subject in the test set which are also in train/validation for any
    trial/fold"""


# --------------
# Functions
# --------------
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


def _merge_config_files_from_paths(paths: Iterable[Path]):
    # Get loader for the sampling distributions
    loader = get_yaml_loader()

    # Add additional formatting
    loader = add_yaml_constructors(loader)

    # Load the config files
    configs: List[Dict[str, Any]] = []
    for config_path in paths:
        if config_path.suffix not in (".yml", "yaml"):
            raise ValueError(f"Tried to open as a .yml file, but the suffix was not recognised: "
                             f"{config_path.suffix}")
        with open(config_path) as file:
            config_file = yaml.load(file, Loader=loader)

            # If the config file is empty, we interpret this as if the dict should be empty
            configs.append(dict() if config_file is None else config_file)

    # Merge and return
    return merge_dicts(*configs)


def _get_preprocessing_config_path(ocular_state):
    # Get file names
    preprocessing_path = get_numpy_data_storage_path() / f"preprocessed_band_pass_{ocular_state}"
    config_files = tuple(file_name for file_name in os.listdir(preprocessing_path) if file_name.startswith("config"))

    # Make sure there is only one and return it
    assert len(config_files) == 1, f"Expected only one config file, but found {len(config_files)}: {config_files}"
    return preprocessing_path / config_files[0]


def _get_best_val_epoch(path, experiment_name, *, pretext_main_metric):
    # Load the .csv file with the metrics
    prefix_name = "" if experiment_name is None else f"{experiment_name}_"
    val_df = pandas.read_csv(os.path.join(path, f"{prefix_name}val_history_metrics.csv"))

    # Get the epoch which maximises the performance
    if higher_is_better(metric=pretext_main_metric):
        return numpy.argmax(val_df[pretext_main_metric])
    else:
        return numpy.argmin(val_df[pretext_main_metric])


def _get_delta_and_variable(path, *, target, variable, deviation_method, log_var, num_eeg_epochs, pretext_main_metric,
                            experiment_name):
    # ----------------
    # Select epoch
    # ----------------
    # todo: not really 'fold' anymore...
    epoch = _get_best_val_epoch(path=path, pretext_main_metric=pretext_main_metric, experiment_name=experiment_name)

    # ----------------
    # Get the biomarkers and the (clinical) variable
    # ----------------
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

    # Get targets
    ground_truth = get_dataset(dataset_name).load_targets(target=target, subject_ids=subject_ids)

    # Get the variable
    var = get_dataset(dataset_name).load_targets(target=variable, subject_ids=subject_ids)

    # Remove nan values
    mask = ~numpy.isnan(var).copy()

    ground_truth = ground_truth[mask]  # type: ignore
    predictions = predictions[mask]
    var = var[mask]  # type: ignore
    subject_ids = subject_ids[mask]

    # Get the deviation
    if deviation_method in ("delta", "gap", "diff", "difference"):
        delta = predictions - ground_truth
    elif deviation_method == "ratio":
        delta = predictions / ground_truth  # todo: should it be the inverse?
    else:
        raise ValueError(f"Unrecognised method: {deviation_method}")

    # Return
    subjects = tuple(Subject(subject_id=sub_id, dataset_name=dataset_name) for sub_id in subject_ids)
    assert isinstance(log_var, bool), f"Expected 'log_var' to be boolean, but found type {type(log_var)}"
    if log_var:
        return subjects, delta, numpy.log10(var)
    else:
        return subjects, delta, var


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
