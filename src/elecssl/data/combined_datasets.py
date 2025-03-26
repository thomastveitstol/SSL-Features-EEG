import dataclasses
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy
import pandas

from elecssl.data.datasets.dataset_base import EEGDatasetBase, ChannelSystem
from elecssl.data.datasets.getter import get_dataset, get_channel_system
from elecssl.data.interpolate_datasets import interpolate_datasets
from elecssl.data.subject_split import Subject, subjects_tuple_to_dict


# -----------------
# Convenient dataclasses
# -----------------
@dataclasses.dataclass(frozen=True)
class LoadDetails:
    subject_ids: Tuple[str, ...]
    time_series_start: Optional[int] = None
    num_time_steps: Optional[int] = None
    channels: Optional[Tuple[str, ...]] = None
    pre_processed_version: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class DatasetDetails:
    dataset: EEGDatasetBase
    details: LoadDetails


# -----------------
# Classes
# -----------------
class CombinedDatasets:
    """
    Class for storing multiple datasets. I find this convenient to avoid re-loading data all the time. Less memory
    efficient, but more time efficient
    """

    __slots__ = "_subject_ids", "_data", "_targets", "_datasets", "_subjects_info", "_variable_availability"

    def __init__(self, datasets_details: Tuple[DatasetDetails, ...], variables, target, interpolation_method,
                 main_channel_system, sampling_freq, required_target):
        """
        Initialise

        Parameters
        ----------
        datasets_details : tuple[DatasetDetails, ...]
        target: str, optional
            Targets to load. If None, no targets are loaded
        interpolation_method : str, optional
        main_channel_system : str, optional
            The channel system to interpolate to. If interpolation_method is None, this argument is ignored
        sampling_freq : float, optional
            Sampling frequency which is needed for interpolation. Ignored if interpolation_method is None
        required_target : str, optional
        """
        # Store subject IDs. Organised as {dataset_name: {subject_name: row-number in data matrix}}
        self._subject_ids = _extract_subject_ids(datasets_details=datasets_details)

        # --------------
        # Load data
        # --------------
        # Load and store input data
        if interpolation_method is None:
            self._data = {
                details.dataset.name: details.dataset.load_numpy_arrays(
                    subject_ids=self._subject_ids[details.dataset.name],
                    pre_processed_version=details.details.pre_processed_version,
                    time_series_start=details.details.time_series_start, num_time_steps=details.details.num_time_steps,
                    channels=details.details.channels,required_target=required_target)
                          for details in datasets_details
            }
        else:
            # Load non-interpolated data
            non_interpolated: Dict[str, Dict[str, Union[numpy.ndarray, ChannelSystem]]] = (  # type: ignore[type-arg]
                dict())
            for details in datasets_details:
                non_interpolated[details.dataset.name] = {
                    "data": details.dataset.load_numpy_arrays(
                        subject_ids=details.details.subject_ids,
                        pre_processed_version=details.details.pre_processed_version,
                        time_series_start=details.details.time_series_start,
                        num_time_steps=details.details.num_time_steps,
                        channels=details.details.channels, required_target=required_target
                    ),
                    "channel_system": details.dataset.channel_system
                }

            # Interpolate
            if main_channel_system in non_interpolated:
                target_channel_system = non_interpolated[main_channel_system]["channel_system"]
            else:
                target_channel_system = get_channel_system(main_channel_system)
            self._data = interpolate_datasets(
                datasets=non_interpolated, method=interpolation_method, sampling_freq=sampling_freq,
                main_channel_system=target_channel_system
            )

        # Load targets
        self._targets = None if target is None \
            else {details.dataset.name: details.dataset.load_targets(subject_ids=details.details.subject_ids,
                                                                     target=target)
                  for details in datasets_details}

        # Convenient for e.g. extracting channel systems
        self._datasets: Tuple[EEGDatasetBase, ...] = tuple(details.dataset for details in datasets_details)

        # --------------
        # Extract subject info to be used for e.g. correlations after pretext training
        # This is currently unused, but I won't prioritise removing it
        # --------------
        subject_info: Dict[Subject, Dict[str, Any]] = {}
        for details in datasets_details:
            dataset_subjects = details.details.subject_ids

            # Make empty subjects info. Convenient for consistency if no variables are passed
            for _subject_id in dataset_subjects:
                subject_info[Subject(subject_id=_subject_id, dataset_name=details.dataset.name)] = {}

            # Loop through the variables (if any)
            for variable in variables[details.dataset.name]:
                info_targets = details.dataset.load_targets(subject_ids=dataset_subjects, target=variable)

                for info_target, subject_id in zip(info_targets, dataset_subjects):  # type: ignore[type-arg]
                    subject = Subject(subject_id=subject_id, dataset_name=details.dataset.name)

                    # i-th loaded target corresponds to the i-th subject
                    subject_info[subject][variable] = info_target

        self._subjects_info = subject_info
        self._variable_availability = variables

    @classmethod
    def from_config(cls, config, interpolation_config, variables, target, sampling_freq, required_target,
                    all_subjects):
        """
        Method for initialising directly from a config file

        Parameters
        ----------
        config : dict[str, typing.Any]
        variables
        interpolation_config : dict[str, typing.Any] | None
        target : str, optional
        sampling_freq : float
        required_target : str, optional
        all_subjects : pandas.DataFrame
            Must have the columns 'dataset' and 'sub_id'. All subjects are expected to be loaded without problems

        Returns
        -------
        """
        # Initialise lists and dictionaries
        datasets_details: List[DatasetDetails] = []

        # Loop through all datasets and loading details to be used
        for dataset_name, dataset_kwargs in config.items():
            # Get dataset
            dataset = get_dataset(dataset_name)
            dataset_subjects = tuple(all_subjects[all_subjects["dataset"] == dataset_name]["sub_id"])

            # Construct dataset details
            load_details = LoadDetails(
                subject_ids=dataset_subjects, time_series_start=dataset_kwargs["time_series_start"],
                num_time_steps=dataset_kwargs["num_time_steps"],
                pre_processed_version=dataset_kwargs["pre_processed_version"]
            )
            datasets_details.append(DatasetDetails(dataset=dataset, details=load_details))

        # Extract details for interpolation
        interpolation_method = None if interpolation_config is None else interpolation_config["method"]
        main_channel_system = None if interpolation_method is None else interpolation_config["main_channel_system"]

        # Load all data and return object
        return cls(datasets_details=tuple(datasets_details), target=target, interpolation_method=interpolation_method,
                   main_channel_system=main_channel_system, sampling_freq=sampling_freq,
                   required_target=required_target, variables=variables)

    # --------------
    # Methods for getting data
    # --------------
    def get_data(self, subjects: Tuple[Subject, ...]):
        """
        Method for getting data. If modifying data after calling this method, the attribute 'self._data' will not be
        changed

        (unittests in test folder)

        Parameters
        ----------
        subjects : tuple[elecssl.data.subject_split.Subject, ...]
            Subjects to extract

        Returns
        -------
        dict[str, numpy.ndarray]
        """
        # Loop through all subjects
        data: Dict[str, List[numpy.ndarray]] = dict()  # type: ignore[type-arg]
        for subject in subjects:
            dataset_name = subject.dataset_name

            # Get the data
            idx = self._subject_ids[dataset_name][subject.subject_id]
            subject_data = self._data[dataset_name][idx]

            # Add the subject data
            if dataset_name in data:
                data[dataset_name].append(subject_data)
            else:
                data[dataset_name] = [subject_data]

        # Convert to numpy arrays and return (here, we assume that the data matrices can be concatenated)
        return {dataset_name: numpy.concatenate(numpy.expand_dims(data_matrix, axis=0), axis=0)
                for dataset_name, data_matrix in data.items()}

    def get_targets(self, subjects: Tuple[Subject, ...]):
        """
        Method for getting targets. If modifying data after calling this method, the attribute 'self._data' will not be
        changed

        (unittests in test folder)

        Parameters
        ----------
        subjects : tuple[elecssl.data.data_split.Subject, ...]
            Subjects to extract

        Returns
        -------
        dict[str, numpy.ndarray]
        """
        # Input check
        if self._targets is None:
            raise ValueError("Tried to extract targets, but no targets are available")

        # Loop through all subjects
        data: Dict[str, List[numpy.ndarray]] = dict()  # type: ignore[type-arg]
        for subject in subjects:
            dataset_name = subject.dataset_name

            # Get the target
            idx = self._subject_ids[dataset_name][subject.subject_id]
            subject_data = self._targets[dataset_name][idx]  # type: ignore[index]

            # Add the subject data
            if dataset_name in data:
                data[dataset_name].append(subject_data)
            else:
                data[dataset_name] = [subject_data]

        # Convert to numpy arrays and return (here, we assume that the data matrices can be concatenated)
        return {dataset_name: numpy.concatenate(numpy.expand_dims(data_matrix, axis=0), axis=0)
                for dataset_name, data_matrix in data.items()}

    # --------------
    # Methods related to subject info (currently unused in the main HPO experiments)
    # --------------
    def get_subjects_info(self, subjects):
        """
        Method for getting the subject information for investigations

        Parameters
        ----------
        subjects : tuple[elecssl.data.subject_split.Subject, ...]

        Returns
        -------
        dict[elecssl.data.subject_split.Subject, dict[str, typing.Any]]
        """
        return {subject: self._subjects_info[subject] for subject in subjects}

    def get_expected_variables(self, subjects):
        # Get all dataset of which we will extract expected variables from
        datasets = set(subject.dataset_name for subject in subjects)

        # Get the variables as exemplified by {"age": ("SRM", "LEMON")}
        expected_variables: Dict[str, List[str]] = {}
        for dataset_name in datasets:
            for var_name in self._variable_availability[dataset_name]:
                if var_name not in expected_variables:
                    expected_variables[var_name] = []

                # Add it to the dataset name list of the variable
                expected_variables[var_name].append(dataset_name)

        # Convert from list to tuple and return
        return {var_name: tuple(dataset_list) for var_name, dataset_list in expected_variables.items()}

    # --------------
    # Convenient methods
    # --------------
    @staticmethod
    def get_subjects_dict(subjects: Tuple[Subject, ...]) -> Dict[str, Tuple[str, ...]]:
        """
        Method for getting a tuple of subjects as a dictionary

        Parameters
        ----------
        subjects : tuple[elecssl.data.subject_split.Subject, ...]
            Subjects to extract

        Returns
        -------
        dict[str, tuple[str, ...]]

        Examples
        --------
        >>> from elecssl.data.subject_split import Subject
        >>> my_drivers = (Subject(dataset_name="Merc", subject_id="LH"), Subject(dataset_name="RB", subject_id="SP"),
        ...               Subject(dataset_name="AM", subject_id="FA"), Subject(dataset_name="RB", subject_id="MV"),
        ...               Subject(dataset_name="Merc", subject_id="GR"))
        >>> CombinedDatasets.get_subjects_dict(my_drivers)
        {'Merc': ('LH', 'GR'), 'RB': ('SP', 'MV'), 'AM': ('FA',)}
        """
        subjects_dict: Dict[str, List[str]] = dict()
        for subject in subjects:
            # Add the subject data
            if subject.dataset_name in subjects_dict:
                subjects_dict[subject.dataset_name].append(subject.subject_id)
            else:
                subjects_dict[subject.dataset_name] = [subject.subject_id]
        return {dataset_name: tuple(subject_ids) for dataset_name, subject_ids in subjects_dict.items()}

    # ----------------
    # Properties
    # ----------------
    @property
    def dataset_subjects(self) -> Dict[str, Tuple[str, ...]]:
        """Get a dictionary containing the subjects available (values) in the datasets (keys)"""
        return {name: tuple(subjects.keys()) for name, subjects in self._subject_ids.items()}

    @property
    def datasets(self):
        return self._datasets

    @property
    def channel_name_to_index(self):
        """Get 'channel_name_to_index' per dataset. Note that this may be misleading if interpolation has been applied,
        as it just goes to the class implementation"""
        return {dataset.name: dataset.channel_name_to_index() for dataset in self._datasets}


# ----------------
# Functions
# ----------------
def _extract_subject_ids(datasets_details: Tuple[DatasetDetails, ...]):
    """
    Function for getting a dictionary containing information of the integer position of a subject in the data matrix

    Parameters
    ----------
    datasets_details

    Returns
    -------
    dict[str, dict[str, int]]
        Organised as {dataset_name: {subject_id: row in data matrix}}

    Examples
    --------
    >>> from elecssl.data.datasets.dortmund_vital import DortmundVital
    >>> from elecssl.data.datasets.lemon import LEMON
    >>> d1 = DatasetDetails(dataset=LEMON(), details=LoadDetails(subject_ids=("sub-1", "sub-2", "sub-3")))
    >>> d2 = DatasetDetails(dataset=DortmundVital(), details=LoadDetails(subject_ids=("sub-512", "sub-56", "sub-1")))
    >>> _extract_subject_ids(datasets_details=(d1, d2))  # doctest: +NORMALIZE_WHITESPACE
    {'LEMON': {'sub-1': 0, 'sub-2': 1, 'sub-3': 2},
     'DortmundVital': {'sub-512': 0, 'sub-56': 1, 'sub-1': 2}}
    """
    return {details.dataset.name: {sub_id: i for i, sub_id in enumerate(details.details.subject_ids)}
            for details in datasets_details}
