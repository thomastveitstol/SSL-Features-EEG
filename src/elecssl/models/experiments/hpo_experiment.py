import abc
import concurrent
import os
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Any, Callable, Tuple, List

import numpy
import optuna
import pandas
import yaml  # type: ignore[import-untyped]
from scipy.stats import NearConstantInputWarning, ConstantInputWarning

from elecssl.data.datasets.getter import get_dataset
from elecssl.data.paths import get_numpy_data_storage_path
from elecssl.data.results_analysis import higher_is_better
from elecssl.data.subject_split import Subject, subjects_tuple_to_dict, get_data_split, simple_random_split
from elecssl.models.experiments.single_experiment import SingleExperiment
from elecssl.models.hp_suggesting import make_trial_suggestion, \
    suggest_spatial_dimension_mismatch, suggest_loss, suggest_dl_architecture
from elecssl.models.metrics import PlotNotSavedWarning
from elecssl.models.ml_models.ml_model_base import MLModel
from elecssl.models.sampling_distributions import get_yaml_loader
from elecssl.models.utils import add_yaml_constructors, add_yaml_representers, verify_type, merge_dicts, \
    verified_performance_score


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
        # If everything was as it should, just exit
        if exc_val is None:
            return None

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
        # Get loader for the sampling distributions
        loader = get_yaml_loader()

        # Add additional formatting
        loader = add_yaml_constructors(loader)

        # Merge all HP distribution files. Typically (most likely always) a shared HPD yml file and a specific one
        hp_configs: List[Dict[str, Any]] = []
        for hp_config_path in hp_config_paths:
            if hp_config_path.suffix not in (".yml", "yaml"):
                raise ValueError(f"Tried to open as a .yml file, but the suffix was not recognised: "
                                 f"{hp_config_path.suffix}")
            with open(hp_config_path) as file:
                config_file = yaml.load(file, Loader=loader)

                # If the config file is empty, we interpret this as if the dict should be empty
                hp_configs.append(dict() if config_file is None else config_file)
        hp_config = merge_dicts(*hp_configs)

        # ---------------
        # Load the other configurations
        # ---------------
        # Merge all configuration setting files. Typically (most likely always) a shared settings yml file and a
        # specific one
        experiments_configs: List[Dict[str, Any]] = []
        for experiments_config_path in experiments_config_paths:
            if experiments_config_path.suffix not in (".yml", "yaml"):
                raise ValueError(f"Tried to open as a .yml file, but the suffix was not recognised: "
                                 f"{experiments_config_path.suffix}")
            with open(experiments_config_path) as file:
                config_file = yaml.load(file, Loader=loader)

                # If the config file is empty, we interpret this as if the dict should be empty
                experiments_configs.append(dict() if config_file is None else config_file)
        experiments_config = merge_dicts(*experiments_configs)

        # ---------------
        # Set attributes
        # ---------------
        self._experiments_config: Dict[str, Any] = experiments_config
        self._sampling_config: Dict[str, Any] = hp_config
        self._results_path = results_dir / (f"{self._name}_hpo_experiment_{date.today()}_"
                                            f"{datetime.now().strftime('%H%M%S')}")

        # Make directory
        os.mkdir(self._results_path)

        # Save the config files
        safe_dumper = add_yaml_representers(yaml.SafeDumper)
        with open(self._results_path / "experiments_config.yml", "w") as file:
            yaml.dump(experiments_config, file, Dumper=safe_dumper)
        with open(self._results_path / "hpd_config.yml", "w") as file:
            yaml.dump(hp_config, file, Dumper=safe_dumper)

    def _get_hpo_folder_path(self, trial: optuna.Trial):
        _debug_mode = "debug_" if verify_type(self._experiments_config["debugging"], bool) else ""
        return self._results_path / (f"{_debug_mode}{self._name}_hpo_{trial.number}_{date.today()}_"
                                     f"{datetime.now().strftime('%H%M%S')}")

    def run_hyperparameter_optimisation(self):
        """Run HPO with optuna"""
        # Create study
        study_name = "optuna-study"
        storage_path = (self._results_path / study_name).with_suffix(".db")
        study = optuna.create_study(
            study_name=study_name, storage=f"sqlite:///{storage_path}", **self.hpo_study_config["HPOStudy"]
        )

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
        suggested_hps: Dict[str, Any] = {"Preprocessing": {}, "Training": {}}

        # Preprocessing
        for param_name, (distribution, distribution_kwargs) in self._sampling_config["Preprocessing"].items():
            suggested_hps["Preprocessing"][param_name] = make_trial_suggestion(
                trial=trial, name=f"{name}_{param_name}", method=distribution, kwargs=distribution_kwargs
            )

        # Training
        for param_name, (distribution, distribution_kwargs) in self._sampling_config["Training"].items():
            suggested_hps["Training"][param_name] = make_trial_suggestion(
                trial=trial, name=f"{name}_{param_name}", method=distribution, kwargs=distribution_kwargs
            )

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

    # --------------
    # Properties
    # --------------
    @property
    def hpo_study_config(self):
        return self._experiments_config["HPO"]


class PretrainHPO(HPOExperiment):
    """
    Class for using the pretext task for pretraining
    """

    __slots__ = ()
    _name = "pretraining"

    def _create_objective(self):
        raise NotImplementedError


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
                                  experiments_config=experiments_config, results_path=results_dir) as experiment:
                experiment.run_experiment()

            # ---------------
            # Get the performance
            # ---------------
            return self._get_aggregated_val_score(results_dir)

        return _objective

    def suggest_hyperparameters(self, trial, name, in_freq_band):
        in_ocular_state = trial.suggest_categorical(f"{name}_ocular_state", **self._sampling_config["OcularStates"])
        preprocessing_config_path = _get_preprocessing_config_path(ocular_state=in_ocular_state)
        suggested_hps = self._suggest_common_hyperparameters(trial, name, in_freq_band=in_freq_band,
                                                             preprocessed_config_path=preprocessing_config_path)
        suggested_hps["ocular_state"] = in_ocular_state
        return suggested_hps

    def _get_aggregated_val_score(self, trial_results_dir):
        """Get the validation score of a trial"""
        metric = self.train_config["main_metric"]
        eval_method = max if higher_is_better(metric=metric) else min

        # Get scores from all folds. Using best scores
        scores: List[float] = []
        for fold in os.listdir(trial_results_dir):
            if not os.path.isdir(trial_results_dir / fold):
                # All folders are assumed to be folds, for allowing possible changes in the future
                continue
            df = pandas.read_csv(trial_results_dir / fold / "val_history_metrics.csv")
            score = eval_method(df[metric])
            print(score)
            scores.append(verified_performance_score(score=score, metric=metric))

        # Aggregate and return
        aggregation_method = self._experiments_config["val_scores_aggregation"]
        if aggregation_method == "mean":
            return numpy.mean(scores)
        elif aggregation_method == "median":
            return numpy.median(scores)
        raise ValueError(f"Method for aggregating the validation scores across folds was not recognised: "
                         f"{aggregation_method}")

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
            _debug_mode = "debug_" if verify_type(self._experiments_config["debugging"], bool) else ""
            results_dir = self._results_path / (f"{_debug_mode}hpo_{trial.number}_{date.today()}_"
                                                f"{datetime.now().strftime('%H%M%S')}")
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

                # Collect the resulting biomarkers
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

                # Save predictions and scores on test set (the score provided to the HPO algorithm should be stored by
                # optuna)
                test_predictions_df = pandas.DataFrame.from_dict({"sub_id": test_subjects, "pred": test_predictions})
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
        results_path = results_dir / (f"hpo_{trial_number}_{feature_extractor_name}_{date.today()}_"
                                      f"{datetime.now().strftime('%H%M%S')}")
        with SingleExperiment(hp_config=suggested_hyperparameters, experiments_config=experiments_config,
                              pre_processing_config=preprocessing_config_file, results_path=results_path) as experiment:
            experiment.run_experiment()

        # ---------------
        # Extract expectation values and biomarkers
        # ---------------
        # todo: a better solution could be to make the 'run_experiment' return the features...
        # todo: slightly hard-coded target
        subject_ids, deviation, clinical_target = _get_delta_and_variable(
            path=results_path / "Fold_0", target=f"band_power_{out_freq_band}_{out_ocular_state}",
            variable=clinical_target, deviation_method=deviation_method, log_var=log_transform_clinical_target,
            num_eeg_epochs=num_eeg_epochs, pretext_main_metric=pretext_main_metric
        )

        return subject_ids, deviation, clinical_target, feature_extractor_name

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
# Functions
# --------------
def _get_preprocessing_config_path(ocular_state):
    # Get file names
    preprocessing_path = get_numpy_data_storage_path() / f"preprocessed_band_pass_{ocular_state}"
    config_files = tuple(file_name for file_name in os.listdir(preprocessing_path) if file_name.startswith("config"))

    # Make sure there is only one and return it
    assert len(config_files) == 1, f"Expected only one config file, but found {len(config_files)}: {config_files}"
    return preprocessing_path / config_files[0]


def _get_best_val_epoch(path, *, pretext_main_metric):
    # Load the .csv file with the metrics
    val_df = pandas.read_csv(os.path.join(path, "val_history_metrics.csv"))

    # Get the epoch which maximises the performance
    if higher_is_better(metric=pretext_main_metric):
        return numpy.argmax(val_df[pretext_main_metric])
    else:
        return numpy.argmin(val_df[pretext_main_metric])


def _get_delta_and_variable(path, *, target, variable, deviation_method, log_var, num_eeg_epochs, pretext_main_metric):
    # ----------------
    # Select epoch
    # ----------------
    # todo: not really 'fold' anymore...
    epoch = _get_best_val_epoch(path=path, pretext_main_metric=pretext_main_metric)

    # ----------------
    # Get the biomarkers and the (clinical) variable
    # ----------------
    test_predictions = pandas.read_csv(os.path.join(path, "test_history_predictions.csv"))
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

    # Add the preprocessed version to all datasets
    preprocessed_version = (f"preprocessed_band_pass_{in_ocular_state}/data--band_pass-{in_freq_band}--"
                            f"input_length-{preprocessing_config_file['input_length']}s--"
                            f"autoreject-{preprocessing_config_file['autoreject']}--"
                            f"sfreq-{preprocessing_config_file['sfreq_multiple']}fmax")

    experiments_config = experiments_config.copy()
    for dataset_info in experiments_config["Datasets"].values():
        dataset_info["pre_processed_version"] = preprocessed_version
    return experiments_config, preprocessing_config_file
