import concurrent
import itertools
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Any

import numpy
import optuna
import pandas
import yaml  # type: ignore[import-untyped]
from scipy.stats import NearConstantInputWarning, ConstantInputWarning

from elecssl.data.datasets.getter import get_dataset
from elecssl.data.paths import get_numpy_data_storage_path
from elecssl.data.results_analysis import higher_is_better
from elecssl.data.subject_split import Subject, subjects_tuple_to_dict, get_data_split, simple_random_split
from elecssl.models.experiments.single_experiment import SSLExperiment
from elecssl.models.hp_suggesting import suggest_hyperparameters
from elecssl.models.ml_models.ml_model_base import MLModel
from elecssl.models.sampling_distributions import get_yaml_loader
from elecssl.models.utils import add_yaml_constructors, add_yaml_representers


class HPOExperiment:
    """
    Class for running hyperparameter optimisation

    todo: make this a context manager
    """

    __slots__ = ("_experiments_config", "_sampling_config", "_results_dir")

    def __init__(self, hp_config_path: Path, experiments_config_path: Path, results_dir: Path):
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
        self._experiments_config: Dict[str, Any] = experiments_config
        self._sampling_config: Dict[str, Any] = hp_config
        self._results_dir = results_dir

        # Make directory
        os.mkdir(results_dir)

        # Store the experiments config file
        safe_dumper = add_yaml_representers(yaml.SafeDumper)
        with open(results_dir / "experiments_config.yml", "w") as file:
            yaml.dump(experiments_config, file, Dumper=safe_dumper)

    def run_hyperparameter_optimisation(self):
        """Run HPO with optuna"""
        # Create study
        study = optuna.create_study(
            study_name="optuna_study", storage=self._results_dir, **self.hpo_study_config["HPOStudy"]
        )

        # Optimise
        with warnings.catch_warnings():
            for warning in self._experiments_config["Warnings"]["ignore"]:
                warnings.filterwarnings(action="ignore", category=_get_warning(warning))
            study.optimize(self._create_objective(), n_trials=self.hpo_study_config["num_trials"])

        # Get the best parameters
        print(f"Best HPC: {study.best_params}")

    def _create_objective(self):

        def _objective(trial: optuna.Trial):
            # ---------------
            # Create configurations for all feature extractors
            # ---------------
            spaces = (self._experiments_config["OcularStates"], self._experiments_config["FrequencyBands"],
                      self._experiments_config["FrequencyBands"])

            # Make directory for current iteration
            results_dir = self._results_dir / (f"debug_hpo_{trial.number}_{date.today()}_"
                                               f"{datetime.now().strftime('%H%M%S')}")
            os.mkdir(results_dir)

            # ---------------
            # Using multiprocessing  # todo: should turn this off if using GPU?
            # ---------------
            print(self._experiments_config)
            with ProcessPoolExecutor(max_workers=self.hpo_study_config["MultiProcessing"]["max_workers"]) as executor:
                print("Multiprocessing")
                experiments = []
                for (in_ocular_state, out_ocular_state), in_freq_band, out_freq_band in itertools.product(*spaces):
                    feature_extractor_name = f"{in_ocular_state}{out_ocular_state}{in_freq_band}{out_freq_band}"

                    # ---------------
                    # Just a bit of preparation...
                    # ---------------
                    # Get the number of EEG epochs per experiment  todo: a little hard-coded?
                    preprocessing_path = (get_numpy_data_storage_path() / f"preprocessed_band_pass_{in_ocular_state}" /
                                          "config.yml")
                    with open(preprocessing_path) as file:
                        preprocessing_config = yaml.safe_load(file)
                        num_epochs = preprocessing_config["Details"]["num_epochs"]

                    # Add the target to config file
                    experiment_config_file = self._experiments_config.copy()
                    experiment_config_file["Training"]["target"] = f"band_power_{out_freq_band}_{out_ocular_state}"

                    # ---------------
                    # Suggest / sample hyperparameters
                    # ---------------
                    suggested_hyperparameters = suggest_hyperparameters(
                        name=feature_extractor_name, config=self._sampling_config, trial=trial,
                        experiments_config=experiment_config_file
                    )

                    # ---------------
                    # Initiate experiment
                    # ---------------
                    experiments.append(
                        executor.submit(
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
            print("Training the ML model on biomarkers...")

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

            # todo: Save the model?
            print(f"Trial params: {trial.params}")
            return score

        return _objective

    @staticmethod
    def _run_single_job(experiments_config, suggested_hyperparameters, trial_number, in_ocular_state, out_ocular_state,
                        in_freq_band, out_freq_band, results_dir, clinical_target, deviation_method, num_eeg_epochs,
                        log_transform_clinical_target, pretext_main_metric, feature_extractor_name):
        """Method for running a single SSL experiments"""
        # Load the preprocessing file and add some necessary info
        with open(get_numpy_data_storage_path() / f"preprocessed_band_pass_{in_ocular_state}" / "config.yml") as file:
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

        # ---------------
        # Learn on the pretext regression task
        # ---------------
        results_path = results_dir / (f"hpo_{trial_number}_{feature_extractor_name}_{date.today()}_"
                                      f"{datetime.now().strftime('%H%M%S')}")
        with SSLExperiment(hp_config=suggested_hyperparameters, experiments_config=experiments_config,
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
    def hpo_study_config(self):
        return self._experiments_config["HPO"]

    @property
    def ml_model_hp_config(self):
        return self._sampling_config["MLModel"]

    @property
    def ml_model_settings_config(self):
        return self._experiments_config["MLModelSettings"]


# --------------
# Functions
# --------------
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
    else:
        raise ValueError(f"Warning {warning} not understood")
