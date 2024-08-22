import dataclasses
import itertools
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy

from elecssl.data.datasets.dataset_base import EEGDatasetBase, ChannelSystem
from elecssl.data.datasets.getter import get_dataset
from elecssl.data.interpolate_datasets import interpolate_datasets
from elecssl.data.subject_split import Subject


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


# -----------------
# Classes
# -----------------
class CombinedDatasets:
    """
    Class for storing multiple datasets. I find this convenient to avoid re-loading data all the time. Less memory
    efficient, but more time efficient
    """

    __slots__ = "_subject_ids", "_data", "_targets", "_datasets", "_subjects_info", "_variable_availability"

    def __init__(self, datasets, variables, load_details=None, target=None, interpolation_method=None,
                 main_channel_system=None, sampling_freq=None, required_target=None):
        """
        Initialise

        todo: add tests

        Parameters
        ----------
        datasets : tuple[EEGDatasetBase, ...]
        load_details : tuple[LoadDetails, ...], optional
        target: str, optional
            Targets to load. If None, no targets are loaded
        interpolation_method : str, optional
        main_channel_system : str
            The channel system to interpolate to. If interpolation_method is None, this argument is ignored
        sampling_freq : float, optional
            Sampling frequency. Ignored if interpolation_method is None
        required_target : str, optional
        """
        # If no loading details are provided, use default
        load_details = tuple(LoadDetails(dataset.get_subject_ids()) for dataset in datasets) \
            if load_details is None else load_details

        # --------------
        # Input check
        # --------------
        if len(datasets) != len(load_details):
            raise ValueError(f"Expected number of datasets to be the same as the number of loading details, but found "
                             f"{len(datasets)} and {len(load_details)}")
        # --------------
        # Store attributes
        # --------------
        # Store subject IDs. Organised as {dataset_name: {subject_name: row-number in data matrix}}
        subject_ids: Dict[str, Dict[str, int]] = dict()
        new_details = []
        for dataset, details in zip(datasets, load_details):
            _accepted_subject_ids: List[str] = []
            _subjects = details.subject_ids
            if required_target is None:
                _targets = itertools.cycle((1,))  # This avoids not accepting a subject
            else:
                _targets = dataset.load_targets(target=required_target, subject_ids=_subjects)
            for sub_id, _target in zip(_subjects, _targets):
                if not numpy.isnan(_target):
                    _accepted_subject_ids.append(sub_id)
                else:
                    print(f"Excluded {sub_id}")
            subject_ids[dataset.name] = {sub_id: i for i, sub_id in enumerate(_accepted_subject_ids)}

            new_details.append(LoadDetails(subject_ids=tuple(_accepted_subject_ids),
                                           time_series_start=details.time_series_start,
                                           num_time_steps=details.num_time_steps,
                                           channels=details.channels,
                                           pre_processed_version=details.pre_processed_version))

        load_details = tuple(new_details)
        self._subject_ids = subject_ids

        # Load and store data  todo: can this be made faster be asyncio?
        if interpolation_method is None:
            self._data = {dataset.name: dataset.load_numpy_arrays(subject_ids=details.subject_ids,
                                                                  pre_processed_version=details.pre_processed_version,
                                                                  time_series_start=details.time_series_start,
                                                                  num_time_steps=details.num_time_steps,
                                                                  channels=details.channels,
                                                                  required_target=required_target)
                          for dataset, details in zip(datasets, load_details)}
        else:
            non_interpolated: Dict[str, Dict[str, Union[numpy.ndarray, ChannelSystem]]] = (  # type: ignore[type-arg]
                dict())

            for dataset, details in zip(datasets, load_details):
                non_interpolated[dataset.name] = {
                    "data": dataset.load_numpy_arrays(
                        subject_ids=details.subject_ids, pre_processed_version=details.pre_processed_version,
                        time_series_start=details.time_series_start, num_time_steps=details.num_time_steps,
                        channels=details.channels, required_target=required_target
                    ),
                    "channel_system": dataset.channel_system
                }

            # Interpolate
            self._data = interpolate_datasets(
                datasets=non_interpolated, method=interpolation_method, sampling_freq=sampling_freq,
                main_channel_system=non_interpolated[main_channel_system]["channel_system"]
            )

        self._targets = None if target is None \
            else {dataset.name: dataset.load_targets(subject_ids=details.subject_ids, target=target)
                  for dataset, details in zip(datasets, load_details)}

        # Convenient for e.g. extracting channel systems
        self._datasets: Tuple[EEGDatasetBase, ...] = datasets

        # --------------
        # Extract subject info to be used for e.g. correlations after pretext training
        # --------------
        subject_info: Dict[Subject, Dict[str, Any]] = {}
        for dataset, details in zip(datasets, load_details):
            dataset_subjects = details.subject_ids
            for variable in variables[dataset.name]:
                info_targets = dataset.load_targets(subject_ids=dataset_subjects, target=variable)

                for info_target, subject_id in zip(info_targets, dataset_subjects):
                    subject = Subject(subject_id=subject_id, dataset_name=dataset.name)
                    if subject_id not in subject_info:
                        subject_info[subject] = {}
                    # i-th loaded target corresponds to the i-th subject
                    subject_info[subject][variable] = info_target

        self._subjects_info = subject_info
        self._variable_availability = variables

    @classmethod
    def from_config(cls, config, interpolation_config, variables, target=None, sampling_freq=None,
                    required_target=None):
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

        Returns
        -------
        """
        # Initialise lists and dictionaries
        load_details = []
        datasets = []
        subjects = dict()
        channel_name_to_index = dict()

        # Loop through all datasets and loading details to be used
        for dataset_name, dataset_details in config.items():
            # Get dataset
            dataset = get_dataset(dataset_name)
            datasets.append(dataset)
            dataset_subjects = dataset.get_subject_ids(
                preprocessed_version=dataset_details["pre_processed_version"]
            )[:dataset_details["num_subjects"]]
            subjects[dataset_name] = dataset_subjects
            channel_name_to_index[dataset_name] = dataset.channel_name_to_index()

            # Construct loading details
            load_details.append(
                LoadDetails(subject_ids=dataset_subjects, time_series_start=dataset_details["time_series_start"],
                            num_time_steps=dataset_details["num_time_steps"],
                            pre_processed_version=dataset_details["pre_processed_version"])
            )

        # Extract details for interpolation
        interpolation_method = None if interpolation_config is None else interpolation_config["method"]
        main_channel_system = None if interpolation_method is None else interpolation_config["main_channel_system"]

        # Load all data and return object
        return cls(datasets=tuple(datasets), load_details=tuple(load_details), target=target,
                   interpolation_method=interpolation_method, main_channel_system=main_channel_system,
                   sampling_freq=sampling_freq, required_target=required_target,
                   variables=variables)

    def get_data(self, subjects):
        """
        Method for getting data

        todo: add tests

        Parameters
        ----------
        subjects : tuple[cdl_eeg.data.data_split.Subject, ...]
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

    def get_targets(self, subjects):
        """
        Method for getting targets

        todo: add tests

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
            subject_data = self._targets[dataset_name][idx]

            # Add the subject data
            if dataset_name in data:
                data[dataset_name].append(subject_data)
            else:
                data[dataset_name] = [subject_data]

        # Convert to numpy arrays and return (here, we assume that the data matrices can be concatenated)
        return {dataset_name: numpy.concatenate(numpy.expand_dims(data_matrix, axis=0), axis=0)
                for dataset_name, data_matrix in data.items()}

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

    @staticmethod
    def get_subjects_dict(subjects):
        """
        Method for the subjects as a dictionary

        Parameters
        ----------
        subjects : tuple[cdl_eeg.data.data_split.Subject, ...]
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
        return {dataset.name: dataset.channel_name_to_index() for dataset in self._datasets}

# todo: check out asyncio for loading. See mCoding at https://www.youtube.com/watch?v=ftmdDlwMwwQ and
#  https://www.youtube.com/watch?v=ueTXYhtlnjA
