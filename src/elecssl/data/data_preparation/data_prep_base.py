import abc
import json
from pathlib import Path
from typing import Any, Dict, Tuple, List, Optional

import autoreject
import pandas
import yaml  # type: ignore[import-untyped]
from progressbar import progressbar

from elecssl.data.datasets.dataset_base import MNELoadingError, DatasetInfo, OcularState
from elecssl.data.datasets.getter import get_dataset
from elecssl.data.subject_split import Subject


class InsufficientNumEpochsError(Exception):
    """Used when the number of epochs is less that accepted """


class TransformationBase(abc.ABC):

    _augmentation_name: str

    # -----------------
    # Methods for saving
    # -----------------
    @abc.abstractmethod
    def _apply_and_save_single_data(self, raw, subject, config, preprocessing, plot_data, save_data, save_to,
                                    return_rejected_epochs) -> Optional[Dict[int, Tuple[int, ...]]]:
        """
        Method for applying transformation and saving a single epochs object

        Parameters
        ----------
        raw : mne.io.BaseRaw
        subject : elecssl.data.subject_split.Subject
        config : dict[str, typing.Any]
        plot_data : bool
        save_data : bool
        save_to : Path
        return_rejected_epochs : bool

        Returns
        -------
        None | dict[int, tuple[int, ...]]
        """

    def prepare_data_for_experiments(self, *, config_path: Path, save_to: Path):
        # ---------------
        # Read config file
        # ---------------
        with open(config_path) as file:
            config: Dict[str, Any] = yaml.safe_load(file)

        # ---------------
        # Loop through the ocular states
        # ---------------
        for ocular_state, dataset_config in config["OcularStates"].items():
            # Fix config file for current ocular state
            ocular_state_config = config.copy()
            ocular_state_config["OcularState"] = ocular_state
            ocular_state_config["Datasets"] = dataset_config["Datasets"]
            del ocular_state_config["OcularStates"]

            # Create features for current ocular state
            path = self._get_path(ocular_state=ocular_state, input_data_path=save_to)
            self._prepare_single_ocular_state_for_experiments(ocular_state_config, save_to=path)

    def _prepare_single_ocular_state_for_experiments(self, ocular_state_config, save_to):
        # ---------------
        # Prepare info
        # ---------------
        # Get datasets and info
        datasets: List[DatasetInfo] = []
        for dataset_name, info in ocular_state_config["Datasets"].items():
            # Get the dataset
            dataset = get_dataset(dataset_name)

            # Make dataset info object
            subjects = dataset.get_subject_ids()
            if info["num_subjects"] != "all":
                subjects = subjects[:info["num_subjects"]]

            # Add dataset info
            datasets.append(
                DatasetInfo(dataset=dataset, subjects=subjects,
                            kwargs={"ocular_state": OcularState(ocular_state_config["OcularState"]), **info["kwargs"]})
            )

        # ---------------
        # Save the transformed input data
        # ---------------
        self._save_data(datasets=tuple(datasets), config=ocular_state_config, save_to=save_to, save_data=True,
                        plot_data=False)

    def _save_data(self, datasets: Tuple[DatasetInfo, ...], config: Dict[str, Any], plot_data, save_data, save_to):
        """Method for preparing and saving input data"""
        # --------------
        # Create folders
        # --------------
        if save_data:
            print("Creating folders...")
            self._create_folders(config, save_to=save_to)

        # --------------
        # Pre-process, transform, and save (or plot, or both)
        # --------------
        num_datasets = len(datasets)
        print("Processing data...")

        # Loop through all datasets
        loading_fails: Dict[str, List[str]] = {"dataset": [], "sub_id": []}
        insufficient_data: Dict[str, List[str]] = {"dataset": [], "sub_id": []}
        rejected_epochs: Dict[int, Dict[str, Tuple[int, ...]]] = {  # {5 seconds: {"LEMON|sub-003": (3, 4, 9)}}
            input_length: dict() for input_length in config["Details"]["input_length"]}
        for i, info in enumerate(datasets):
            print(f"\t({i + 1}/{num_datasets}) {type(info.dataset).__name__}")

            # Loop through all provided subjects for the dataset
            for subject in progressbar(info.subjects, prefix=f"{type(info.dataset).__name__} ", redirect_stdout=True):
                # Load the EEG
                try:
                    eeg = info.dataset.load_single_mne_object(subject_id=subject, **info.kwargs)
                except MNELoadingError:
                    loading_fails["dataset"].append(info.dataset.name)
                    loading_fails["sub_id"].append(subject)
                    continue

                # Save the data
                try:
                    rejects = self._apply_and_save_single_data(
                        eeg, subject=Subject(subject_id=subject, dataset_name=type(info.dataset).__name__),
                        config=config, plot_data=plot_data, save_data=save_data, save_to=save_to,
                        preprocessing=config["InitialPreprocessing"][type(info.dataset).__name__],
                        return_rejected_epochs=True
                    )

                    assert rejects and isinstance(rejects, dict), rejects
                    for epoch_duration, in_length_rejected in rejects.items():
                        rejected_epochs[epoch_duration][f"{info.dataset.name}|{subject}"] = in_length_rejected

                except InsufficientNumEpochsError:
                    insufficient_data["dataset"].append(info.dataset.name)
                    insufficient_data["sub_id"].append(subject)
                    continue

        # --------------
        # Document failures and rejected epochs
        # --------------
        pandas.DataFrame(loading_fails).to_csv(save_to / "loading_fails.csv", index=False)
        pandas.DataFrame(insufficient_data).to_csv(save_to / "insufficient_data.csv", index=False)
        with open(save_to / "dropped_epochs.json", "w") as file:
            json.dump(rejected_epochs, file, indent=4)  # type: ignore[arg-type]

    # -----------------
    # Path methods
    # -----------------
    def _get_path(self, ocular_state, input_data_path: Path):
        return input_data_path / f"preprocessed_{self._augmentation_name}_{ocular_state.lower()}"

    @abc.abstractmethod
    def _get_folder_name(self, *args, **kwargs):
        """
        Method for getting the expected folder name, given specified configurations

        Parameters
        ----------
        args
        kwargs

        Returns
        -------
        pathlib.Path
        """

    @abc.abstractmethod
    def _create_folders(self, config, save_to):
        """Method for creating all expected folders"""


# ------------
# Functions
# ------------
def run_autoreject(epochs, *, autoreject_resample, seed, default_num_splits):
    """Function for running autoreject. Operates in-place"""
    if autoreject_resample is not None:
        epochs.resample(autoreject_resample, verbose=False)
    reject = autoreject.AutoReject(verbose=False, random_state=seed, cv=min(default_num_splits, len(epochs)))
    return reject.fit_transform(epochs, return_log=True)
