import abc
from typing import Any, Dict, Tuple

from progressbar import progressbar

from elecssl.data.datasets.dataset_base import MNELoadingError, DatasetInfo
from elecssl.data.paths import get_numpy_data_storage_path
from elecssl.data.subject_split import Subject


class InsufficientNumEpochsError(Exception):
    """Used when the number of epochs is less that accepted """


class TransformationBase(abc.ABC):

    _augmentation_name: str

    # -----------------
    # Methods for saving
    # -----------------
    @abc.abstractmethod
    def _apply_and_save_single_data(self, raw, subject, config, plot_data, save_data):
        """
        Method for applying transformation and saving a single epochs object

        Parameters
        ----------
        raw : mne.io.BaseRaw
        subject : elecssl.data.subject_split.Subject
        config : dict[str, typing.Any]
        plot_data : bool
        save_data : bool

        Returns
        -------
        None
        """

    def save_data(self, datasets: Tuple[DatasetInfo, ...], config: Dict[str, Any], plot_data, save_data) -> None:
        """Method for preparing and saving input data"""
        # --------------
        # Create folders
        # --------------
        if save_data:
            print("Creating folders...")
            self._create_folders(config)

        # --------------
        # Pre-process, transform, and save (or plot, or both)
        # --------------
        num_datasets = len(datasets)
        print("Processing data...")

        # Loop through all datasets
        for i, info in enumerate(datasets):
            print(f"\t({i + 1}/{num_datasets}) {type(info.dataset).__name__}")

            # Loop through all provided subjects for the dataset
            for subject in progressbar(info.subjects, prefix=f"{type(info.dataset).__name__} ", redirect_stdout=True):
                # Load the EEG
                try:
                    eeg = info.dataset.load_single_mne_object(subject_id=subject, **info.kwargs)
                except MNELoadingError:
                    continue

                # Save the data
                try:
                    self._apply_and_save_single_data(
                        eeg, subject=Subject(subject_id=subject, dataset_name=type(info.dataset).__name__),
                        config=config, plot_data=plot_data, save_data=save_data,
                    )
                except InsufficientNumEpochsError:
                    continue

    # -----------------
    # Path methods
    # -----------------
    def _get_path(self, ocular_state):
        return get_numpy_data_storage_path() / f"preprocessed_{self._augmentation_name}_{ocular_state.lower()}"

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
    def _create_folders(self, config):
        """Method for creating all expected folders"""
