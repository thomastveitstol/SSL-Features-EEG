import abc
import dataclasses
import os
import typing
import warnings
from enum import Enum
from typing import Dict, Tuple, List, Optional, Union, Any, Iterable, Set

import numpy
import pandas
import yaml  # type: ignore[import-untyped]
from matplotlib import pyplot
from mne.transforms import _cart_to_sph, _pol_to_cart
from progressbar import progressbar

from elecssl.data.paths import get_raw_data_storage_path, get_numpy_data_storage_path, get_eeg_features_storage_path
from elecssl.data.subject_split import Subject
from elecssl.models.region_based_pooling.utils import ELECTRODES_3D


# --------------------
# Convenient decorators
# --------------------
def target_method(func):
    setattr(func, "_is_target_method", True)

    def availability_decorator(availability_func):
        """Allows defining an availability method for the target method"""
        if not callable(availability_func):
            raise TypeError(f"Expected availability method to be callable, but that was not the case for "
                            f"(type={type(availability_func)}) {availability_func}")
        setattr(func, "availability_method", availability_func)
        return availability_func

    func.availability = availability_decorator  # Attach decorator
    return func


# --------------------
# Classes
# --------------------
class UnavailableOcularStateError(Exception):
    ...


class MNELoadingError(Exception):
    """This exception should be raised if there are anomalies in a dataset, which prevents loading an MNE object that
    was expected to work"""


class OcularState(Enum):
    """
    Enum class for Ocular state

    Examples
    --------
    >>> OcularState.EC
    <OcularState.EC: 'EC'>
    """

    EC = "EC"
    EO = "EO"

    def to_pascal_case(self):
        """
        Method for converting to a string Pascal Case

        Returns
        -------
        str

        Examples
        --------
        >>> OcularState.EO.to_pascal_case()
        'EyesOpen'
        >>> OcularState.EC.to_pascal_case()
        'EyesClosed'
        """
        if self == OcularState.EC:
            return "EyesClosed"
        elif self == OcularState.EO:
            return "EyesOpen"
        else:
            raise ValueError(f"Invalid OcularState {self}")


@dataclasses.dataclass(frozen=True)
class ChannelSystem:
    """Data class for channel systems"""
    name: str  # Should ideally be the same as dataset name
    channel_name_to_index: Dict[str, int]
    electrode_positions: ELECTRODES_3D
    montage_name: Optional[str] = None


class EEGDatasetBase(abc.ABC):
    """
    Base class for all datasets to be used
    """

    __slots__ = "_name"

    _ocular_states: Union[Tuple[OcularState], Tuple[OcularState, OcularState]]
    _montage_name: Optional[str] = None

    def __init__(self, name=None):
        """
        Initialisation method

        Parameters
        ----------
        name : str, optional
            Name of the EEG dataset
        """
        self._name: str = self.__class__.__name__ if name is None else name

    # ----------------
    # Loading methods
    # ----------------
    def load_single_mne_object(self, subject_id, ocular_state, derivatives=False, **kwargs):
        """
        Method for loading MNE raw object of a single subject

        Parameters
        ----------
        subject_id : str
            Subject ID
        ocular_state : OcularState
            The ocular state of which to load.
        derivatives : bool
            For datasets where an already cleaned version is available. If True, the cleaned version will be used,
            otherwise the non-cleaned data is loaded

        Returns
        -------
        mne.io.base.BaseRaw
            MNE object of the subject
        """
        # Check if the ocular state is available
        if ocular_state not in self._ocular_states:
            raise UnavailableOcularStateError(f"Tried to load data from the ocular state {ocular_state}, but only "
                                              f"{self._ocular_states} are available")

        # Load raw object
        raw = self._load_single_cleaned_mne_object(subject_id, ocular_state=ocular_state, **kwargs) if derivatives \
            else self._load_single_raw_mne_object(subject_id, ocular_state=ocular_state, **kwargs)

        # Set montage
        raw.set_montage(self.channel_system.montage_name)
        return raw

    @abc.abstractmethod
    def _load_single_raw_mne_object(self, *args, **kwargs):
        """
        Method for loading raw data

        Parameters
        ----------
        subject_id : str
            Subject ID
        Returns
        -------
        mne.io.base.BaseRaw
            MNE object of the subject
        """

    def _load_single_cleaned_mne_object(self, *args, **kwargs):
        """
        Method for loading existing pre-processed data (only relevant for some datasets)

        Parameters
        ----------
        subject_id : str
            Subject ID

        Returns
        -------
        mne.io.base.BaseRaw
            MNE object of the subject
        """
        raise NotImplementedError(f"A cleaned version is not available for this class ({self.__class__.__name__}).")

    def load_numpy_arrays(self, subject_ids, pre_processed_version, *, time_series_start, num_time_steps, channels,
                          required_target):
        """
        Method for loading numpy arrays

        Parameters
        ----------
        subject_ids : tuple[str, ...], optional
        pre_processed_version : str, optional
            The pre-processed version. That is, the numpy arrays should be stored inside
            os.path.join(self.get_numpy_arrays_path(), pre_processed_version)
        time_series_start : int, optional
        num_time_steps : int, optional
        channels: tuple[str, ...], optional
        required_target : str, optional

        Returns
        -------
        numpy.ndarray
        """
        # Maybe set defaults
        path = self.get_numpy_arrays_path(pre_processed_version)
        subject_ids = self.get_subject_ids() if subject_ids is None else subject_ids

        # If there is a required target availability, prune the subject IDs
        if required_target is not None:
            _accepted_subject_ids: List[str] = []
            _targets = self.load_targets(target=required_target, subject_ids=subject_ids)
            for sub_id, target in zip(subject_ids, _targets):  # type: ignore[arg-type]
                if not numpy.isnan(target):
                    _accepted_subject_ids.append(sub_id)
            subject_ids = tuple(_accepted_subject_ids)

        # ------------------
        # Input checks
        # ------------------
        # Check if all subjects are passed only once
        if len(set(subject_ids)) != len(subject_ids):
            _num_non_unique_subjects = len(subject_ids) - len(set(subject_ids))
            raise ValueError(f"Expected all subject IDs to be unique, but there were {_num_non_unique_subjects} "
                             f"subject IDs which were passed more than once")

        # Check if all subjects are actually available
        available_subjects = self.get_subject_ids()
        if not all(sub_id in available_subjects for sub_id in subject_ids):
            _unexpected_subjects = tuple(sub_id for sub_id in subject_ids if sub_id not in self.get_subject_ids())
            raise ValueError(f"Unexpected subject IDs for class '{type(self).__name__}' "
                             f"(N={len(_unexpected_subjects)}): {_unexpected_subjects}")

        # ------------------
        # Loop through all subjects
        # ------------------
        data = []
        for sub_id in progressbar(subject_ids, prefix="Loading data ", redirect_stdout=True):
            # Load the numpy array
            eeg_data = numpy.load(path / f"{sub_id}.npy")

            # (Maybe) crop the signal
            if time_series_start is not None:
                eeg_data = eeg_data[..., time_series_start:]
            if num_time_steps is not None:
                eeg_data = eeg_data[..., :num_time_steps]

            # (Maybe) remove unwanted signals
            if channels is not None:
                indices = channel_names_to_indices(ch_names=channels,
                                                   channel_name_to_index=self.channel_name_to_index())
                eeg_data = eeg_data[indices]

            # Add the data
            data.append(numpy.expand_dims(eeg_data, axis=0))

        # Concatenate to a single numpy ndarray
        return numpy.concatenate(data, axis=0)

    def get_subject_ids(self, preprocessed_version=None) -> Tuple[str, ...]:
        """Get the subject IDs available. If a preprocessed version is specified (not None), it only returns the IDs
        which are present in that preprocessed version"""
        # Return all available ones if preprocessed version is not specified
        if preprocessed_version is None:
            return self._get_subject_ids()

        # ----------------
        # Remove all subjects which are not contained in the specified preprocessed version
        # ----------------
        subjects = self._get_subject_ids()
        path = os.path.join(get_numpy_data_storage_path(), preprocessed_version, self.name)

        # Need to remove .npy. Can't use .removesuffix(".npy") for Python 3.8 compatibility reasons
        _available_subjects = tuple(subject_npy[:-4] for subject_npy in os.listdir(path))
        return tuple(subject for subject in subjects if subject in _available_subjects)

    def _get_subject_ids(self) -> Tuple[str, ...]:
        """Get the subject IDs available. Unless this method is overridden, it will collect the IDs from the
        participants.tsv file"""
        return tuple(pandas.read_csv(self.get_participants_tsv_path(), sep="\t",
                                     usecols=["participant_id"])["participant_id"])

    @classmethod
    def download(cls):
        """Method for downloading the dataset"""
        raise NotImplementedError

    # ----------------
    # Methods which helps to filter out subjects not reaching certain requirements
    #
    # (See 'Functions' below for 'pseudo_targets_filter_subjects')
    # ----------------
    def preprocessing_filter_subjects(self, subjects: Set[str], preprocessing_versions: Iterable[str]):
        """
        Method for including/excluding subjects which are not contained in ALL provided preprocessed versions of the
        EEG data

        Parameters
        ----------
        subjects : set[str]
        preprocessing_versions : Iterable[str]

        Returns
        -------
        tuple[set[Subject], set[Subject]]
            Included and excluded subjects
        """
        # Get subjects available in all preprocessed versions
        available_subjects_per_version: List[Set[str]] = []
        for input_version in preprocessing_versions:
            path = self.get_numpy_arrays_path(input_version)
            available_subjects_per_version.append(set(subject_npy[:-4] for subject_npy in os.listdir(path)))
        available_subjects = set.intersection(*available_subjects_per_version)

        # Get all included and excluded subjects
        included = subjects & available_subjects
        excluded = subjects - included

        # Correct the type
        included_subjects = {Subject(dataset_name=self.name, subject_id=sub_id) for sub_id in included}
        excluded_subjects = {Subject(dataset_name=self.name, subject_id=sub_id) for sub_id in excluded}
        return included_subjects, excluded_subjects

    def downstream_target_filter_subjects(self, subjects: Set[str], target: str):
        """
        Method for including/excluding subjects which has the provided downstream target

        Parameters
        ----------
        subjects : set[str]
        target : str

        Returns
        -------
        tuple[set[Subject], set[Subject]]
            Included and excluded subjects
        """
        # Get subjects availability based on the target
        available_subjects = set(self.get_downstream_target_subjects_availability(target_name=target))

        # Get all included and excluded subjects
        included = subjects & available_subjects
        excluded = subjects - included

        # Correct the type and return
        included_subjects = {Subject(dataset_name=self.name, subject_id=sub_id) for sub_id in included}
        excluded_subjects = {Subject(dataset_name=self.name, subject_id=sub_id) for sub_id in excluded}
        return included_subjects, excluded_subjects

    # ----------------
    # Target methods
    # ----------------
    def load_targets(self, target, subject_ids=None):
        """
        Method for loading targets. There are some default logarithmic transformations which are also included

        Parameters
        ----------
        target : str
        subject_ids : tuple[str, ...]

        Returns
        -------
        numpy.ndarray
        """
        subject_ids = self.get_subject_ids() if subject_ids is None else subject_ids

        # Check and get (e.g., logarithmic) transformation of the target
        transformation: typing.Callable[[numpy.ndarray], numpy.ndarray]  # type: ignore[type-arg]
        if target.startswith(("log_", "loge_", "ln_")):
            transformation = numpy.log
            target = target.split("_", 1)[1]
        elif target.startswith("log2_"):
            transformation = numpy.log2
            target = target.split("_", 1)[1]
        elif target.startswith("log10_"):
            transformation = numpy.log10
            target = target.split("_", 1)[1]
        else:
            transformation = _no_transform

        # Load the targets
        if target in self.get_available_targets(exclude_ssl=True):
            loaded_targets = getattr(self, target)(subject_ids=subject_ids)
        elif target in self._get_available_self_supervised_targets():
            loaded_targets = self._load_ssl_targets(subject_ids=subject_ids, target=target)
        else:
            raise ValueError(f"Target '{target}' was not recognised. Make sure that the method passed shares the name "
                             f"with the implemented method you want to use. The targets available for this class "
                             f"({type(self).__name__}) are: {self.get_available_targets(exclude_ssl=False)}")

        # Return with log transform (if any was specified)
        return transformation(loaded_targets)

    def _load_ssl_targets(self, subject_ids, target):
        features_path = _get_pseudo_target_csv_path(target)

        # Load csv file
        df = pandas.read_csv(features_path)

        # Filter on dataset name and subject IDs
        df = df[(df["Dataset"] == type(self).__name__) & df["Subject-ID"].isin(subject_ids)]

        # Rearrange to ensure correct ordering
        df = df.set_index("Subject-ID").reindex(subject_ids)

        # Return numpy array
        return df.iloc[:, -1].values  # Column name and target does not match, but this works for the current pipeline

    @classmethod
    def get_available_targets(cls, exclude_ssl):
        """Get all target methods available for the class. The target method must be decorated by @target_method to be
        properly registered. Furthermore, there are some default logarithmic transformations which are also included by
        adding e.g. 'log10_' to the target name, see 'load_targets' for these transformations"""
        if not isinstance(exclude_ssl, bool):
            raise TypeError(f"Expected input argument 'exclude_ssl' to be bool, but found {type(exclude_ssl)}")

        # -------------
        # Get all implemented target methods
        # -------------
        target_methods: List[str] = []
        for method in dir(cls):
            attribute = getattr(cls, method)

            # Append (as type 'str') if it is a target method
            if callable(attribute) and getattr(attribute, "_is_target_method", False):
                target_methods.append(method)

        if exclude_ssl:
            return tuple(target_methods)

        # -------------
        # Get all SSL implemented methods
        # -------------
        try:
            ssl_features = cls._get_available_self_supervised_targets()
        except FileNotFoundError:
            ssl_features = ()

        # Convert to tuple and return
        return tuple(target_methods) + ssl_features

    @classmethod
    def _get_available_self_supervised_targets(cls):
        """Get all available SSL targets. They need to exist in the correct folder"""
        # The feature names should identical to folder names inside the EEG features folder
        root_features_path = get_eeg_features_storage_path()

        # Check all features for this specific dataset
        available_features: List[str] = []
        for feature in os.listdir(root_features_path):
            # Which datasets are available for a given feature should be specified in the config file
            with open(_get_pseudo_target_config_file_path(root_features_path=root_features_path, feature=feature)) as f:
                config = yaml.safe_load(f)

            if cls.__name__ in config["Datasets"]:
                available_features.append(feature)

        # Return available SSL features as a tuple
        return tuple(available_features)

    def get_downstream_target_subjects_availability(self, target_name) -> Tuple[str, ...]:
        """Get the available subjects, given that it needs to have the specified target available"""
        # Verify that it exists
        if target_name not in self.get_available_targets(exclude_ssl=True):
            raise ValueError(f"Target {target_name!r} was not recognised. Make sure that the method passed shares the "
                             f"name with the implemented method you want to use. The targets available for this class "
                             f"({type(self).__name__}) are: {self.get_available_targets(exclude_ssl=False)}")

        # Get the target function
        target_func = getattr(self, target_name)

        # Verify that it has an availability method
        if not hasattr(target_func, "availability_method"):
            raise ValueError(f"Target method {target_name!r} does not have an availability method.")

        # Use it
        return getattr(target_func, "availability_method")(self)  # type: ignore[no-any-return]

    # ----------------
    # Properties
    # ----------------
    @property
    def name(self) -> str:
        return self._name

    @property
    def num_channels(self):
        return len(self.channel_name_to_index())

    @property
    def channel_system(self) -> ChannelSystem:
        return ChannelSystem(name=self.name, channel_name_to_index=self.channel_name_to_index(),
                             electrode_positions=self.get_electrode_positions(), montage_name=self._montage_name)

    # ----------------
    # Path methods
    # ----------------
    @classmethod
    def get_mne_path(cls):
        return get_raw_data_storage_path() / cls.__name__

    def get_numpy_arrays_path(self, pre_processed_version):
        return get_numpy_data_storage_path() / pre_processed_version / self.name

    @classmethod
    def get_participants_tsv_path(cls):
        """Get the path to the participants.tsv file"""
        return os.path.join(cls.get_mne_path(), "participants.tsv")

    # ----------------
    # Channel system  TODO: consider changing to classmethods
    # ----------------
    def get_electrode_positions(self, subject_id=None):
        """
        Method for getting the electrode positions (cartesian coordinates) of a specified subject. If this method is not
        overridden or None is passed, a template is used instead
        Parameters
        ----------
        subject_id : str, optional
            Subject ID

        Returns
        -------
        ELECTRODES_3D
            Cartesian coordinates of the channels. Keys are channel names
        """
        if subject_id is None:
            return self._get_template_electrode_positions()
        else:
            # Use subject specific coordinates. If not implemented, use template instead
            try:
                return self._get_electrode_positions(subject_id)
            except NotImplementedError:
                warnings.warn("Electrode positions are not available per subject. Trying template instead...",
                              RuntimeWarning)
                return self._get_template_electrode_positions()

    def _get_electrode_positions(self, subject_id=None):
        """
        Method for getting the electrode positions (cartesian coordinates) of a specified subject.

        Parameters
        ----------
        subject_id : str, optional
            Subject ID

        Returns
        -------
        ELECTRODES_3D
            Cartesian coordinates of the channels. Keys are channel names
        """
        raise NotImplementedError

    def _get_template_electrode_positions(self):
        """
        Method for getting the template electrode positions (cartesian coordinates)

        Returns
        -------
        ELECTRODES_3D
            Cartesian coordinates of the channels. Keys are channel names
        """
        raise NotImplementedError

    @abc.abstractmethod
    def channel_name_to_index(self):
        """
        Get the mapping from channel name to index

        Returns
        -------
        dict[str, int]
            Keys are channel name, value is the row-position in the data matrix
        """

    def plot_electrode_positions(self, subject_id=None, annotate=True, ax=None):
        """
        Method for 3D plotting the electrode positions.

        Parameters
        ----------
        subject_id : str, optional
            Subject ID
        annotate : bool
            To annotate the points with channel names (True) or not (False)
        ax: optional

        Returns
        -------
        None
        """
        # Get electrode positions
        electrode_positions = self.get_electrode_positions(subject_id=subject_id)

        # Extract coordinates
        channel_names = []
        x_vals = []
        y_vals = []
        z_vals = []
        for ch_name, (x, y, z) in electrode_positions.items():
            channel_names.append(ch_name)
            x_vals.append(x)
            y_vals.append(y)
            z_vals.append(z)

        # Maybe make new figure
        if ax is None:
            fig = pyplot.figure()
            ax = fig.add_subplot(111, projection='3d')

        # Plot
        ax.scatter(x_vals, y_vals, z_vals)

        # Annotate the channels with channel names (if desired)
        if annotate:
            for x, y, z, channel in zip(x_vals, y_vals, z_vals, channel_names):
                ax.text(x=x, y=y, z=z, s=channel)

    def plot_2d_electrode_positions(self, subject_id=None, annotate=True):
        # Get electrode positions
        electrode_positions = self.get_electrode_positions(subject_id=subject_id)

        # Apply the same steps as _auto_topomap_coordinates from MNE.transforms
        cartesian_coords = _cart_to_sph(tuple(electrode_positions.values()))  # type: ignore
        out = _pol_to_cart(cartesian_coords[:, 1:][:, ::-1])
        out *= cartesian_coords[:, [0]] / (numpy.pi / 2.)

        # Extract coordinates
        channel_names = []
        x_vals = []
        y_vals = []
        for ch_name, (x, y) in zip(electrode_positions, out):  # type: ignore
            channel_names.append(ch_name)
            x_vals.append(x)
            y_vals.append(y)

        # Plot
        pyplot.scatter(x_vals, y_vals)

        # Annotate the channels with channel names (if desired)
        if annotate:
            for x, y, channel in zip(x_vals, y_vals, channel_names):
                pyplot.text(x=x, y=y, s=channel)


# ------------------
# Functions
# ------------------
def pseudo_targets_filter_subjects(subjects: Set[Subject], pseudo_targets: Iterable[str]):
    """
    Method for including/excluding subjects which has ALL provided pseudo targets available. It assumes that the
    values are valid

    Parameters
    ----------
    subjects : set[Subject]
    pseudo_targets : Iterable[str]

    Returns
    -------
    tuple[set[Subject], set[Subject]]
        Included and excluded subjects
    """
    # Get subjects with all pseudo targets
    available_subjects_per_pseudo_target: List[Set[Subject]] = []
    for target in pseudo_targets:
        # Load the required columns (we assume that all values are ok)
        path = _get_pseudo_target_csv_path(target)
        target_df = pandas.read_csv(path, usecols=["Dataset", "Subject-ID"])
        target_df.rename(columns={"Dataset": "dataset", "Subject-ID": "sub_id"}, inplace=True)

        # Add to list
        available_subjects_per_pseudo_target.append(
            set(Subject(dataset_name=row.dataset, subject_id=row.sub_id)  # type: ignore[attr-defined]
                for row in target_df.itertuples(index=False))
        )
    available_subjects = set.intersection(*available_subjects_per_pseudo_target)

    # Get all included and excluded subjects
    included_subjects = subjects & available_subjects
    excluded_subjects = subjects - included_subjects

    return included_subjects, excluded_subjects


def _get_pseudo_target_csv_path(target):
    # The feature names should identical to folder names inside the EEG features folder
    return (get_eeg_features_storage_path() / target / target).with_suffix(".csv")


def _get_pseudo_target_config_file_path(root_features_path, feature):
    """Checks for files with names config*. If there is only one, the path is returned, otherwise and error is raised"""
    # Get file name
    config_files = tuple(file_name for file_name in os.listdir(root_features_path / feature)
                         if file_name.startswith("config"))

    # Make sure there is only one and return it
    assert len(config_files) == 1, (f"Expected only one config file in {root_features_path / feature}, but found "
                                    f"{len(config_files)}: {config_files}")
    return root_features_path / feature / config_files[0]


def _no_transform(x):
    return x


def get_channel_name_order(channel_name_to_index):
    """
    This function ensures that one obtains the correct channel name order, even if the channel_name_to_index is poorly
    sorted. Input checks are also included

    Parameters
    ----------
    channel_name_to_index : dict[str, int]

    Returns
    -------
    cdl_eeg.models.region_based_pooling.utils.CHANNELS_NAMES

    Examples
    --------
    >>> get_channel_name_order(channel_name_to_index={"C2": 2, "C0": 0, "C1": 1, "C4": 3})
    ('C0', 'C1', 'C2', 'C4')

    If there are too small or large values, a ValueError is raised

    >>> get_channel_name_order(channel_name_to_index={"C2": 2, "C0": -2, "C1": 1, "C4": 3})
    Traceback (most recent call last):
    ...
    ValueError: Expected all values to be between 0 and 3, but found (2, -2, 1, 3)
    >>> get_channel_name_order(channel_name_to_index={"C2": 2, "C0": 0, "C1": 1, "C4": 5})
    Traceback (most recent call last):
    ...
    ValueError: Expected all values to be between 0 and 3, but found (2, 0, 1, 5)

    Duplicates are not allowed

    >>> get_channel_name_order(channel_name_to_index={"C2": 1, "C0": 0, "C1": 1, "C4": 3})
    Traceback (most recent call last):
    ...
    ValueError: Expected all values to be unique, but number of channel names was 4 and number of unique values were 3
    """
    # ----------------
    # Input checks
    # ----------------
    num_channels = len(channel_name_to_index)

    # All channel names must be strings
    if not all(ch_name for ch_name in channel_name_to_index):
        raise TypeError(f"Expected all channel names to be strings, but found "
                        f"{set(ch_name for ch_name in channel_name_to_index)}")

    # All indices must be in [0, num_channels - 1]
    if not all(idx in range(num_channels) for idx in channel_name_to_index.values()):
        raise ValueError(f"Expected all values to be between 0 and {num_channels-1}, but found "
                         f"{tuple(channel_name_to_index.values())}")

    # All indices must be integers
    if not all(isinstance(idx, int) for idx in channel_name_to_index.values()):
        raise TypeError(f"Expected all values to be integers, but found "
                        f"{set(type(idx) for idx in channel_name_to_index.values())}")

    # No duplicates of indices
    if len(channel_name_to_index) != len(set(channel_name_to_index.values())):
        raise ValueError(f"Expected all values to be unique, but number of channel names was "
                         f"{len(channel_name_to_index)} and number of unique values were "
                         f"{len(set(channel_name_to_index.values()))}")

    # ----------------
    # Invert the dict
    # ----------------
    inverted_dict = {idx: ch_name for ch_name, idx in channel_name_to_index.items()}

    # Ensure correct ordering by looping through 0 to num_channels - 1
    return tuple(inverted_dict[i] for i in range(num_channels))


def channel_names_to_indices(ch_names, channel_name_to_index):
    """
    Same as channel_name_to_index, but now you can pass in a tuple of channel names

    Parameters
    ----------
    ch_names : tuple[str, ...]
        Channel names to be mapped to indices
    channel_name_to_index : dict[str, int]
        Mapping from channel name (keys) to index in data matrix (values)

    Returns
    -------
    tuple[int, ...]
        The indices of the channel names, in the same order as ch_names

    Examples
    --------
    >>> channel_names_to_indices(("A", "C", "B", "E"), channel_name_to_index={"A": 0, "B": 1, "C": 2, "D": 3, "E": 4})
    (0, 2, 1, 4)
    """
    return tuple(channel_name_to_index[channel_name] for channel_name in ch_names)


@dataclasses.dataclass
class DatasetInfo:
    dataset: EEGDatasetBase
    subjects: Tuple[str, ...]
    kwargs: Dict[str, Any]
