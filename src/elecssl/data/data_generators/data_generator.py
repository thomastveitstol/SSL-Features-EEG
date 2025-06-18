import abc
from typing import List, Dict, Any, Tuple, Optional

import numpy
import torch
from torch.utils.data import Dataset

from elecssl.data.subject_split import Subject
from elecssl.models.main_models.main_base_class import reorder_subjects


# ----------------
# Base class for all
# ----------------
class DataGenBase(Dataset):  # type: ignore[type-arg]

    def __init__(self, *, data, targets, subjects, subjects_info, expected_variables):
        super().__init__()

        self._data = data
        self._targets = targets
        self._subjects = subjects
        self._subjects_info = subjects_info
        self._expected_variables = expected_variables

    def __len__(self):
        return sum(x.shape[0] * x.shape[1] for x in self._data.values())

    # --------------
    # Methods for 'collate_fn'
    # --------------
    @classmethod
    def get_collate_fn(cls):
        return cls._collate_fn

    @staticmethod
    @abc.abstractmethod
    def _collate_fn(batch: Tuple[Any, ...]) -> Any:
        """This method will be passed to 'collate_fn' argument in DataLoader"""

    # --------------
    # Convenient methods
    # TODO: I think these are deprecated after adding collate_fn to the class
    # --------------
    def get_subject_from_idx(self, item):
        """
        Get the subject from the index. It is needed because the subject information cannot easily be returned in the
        __getitem__ method. Therefore, the index is returned instead, and the subject information can be extracted by
        passing the index to this method.

        Parameters
        ----------
        item : torch.Tensor

        Returns
        -------
        Subject
        """
        # Get the dataset name and index
        dataset_name, subject_idx, _ = _select_dataset_and_index(item=int(item), dataset_shapes=self.dataset_shapes)

        # Use correct type and return
        return Subject(subject_id=self._subjects[dataset_name][subject_idx], dataset_name=dataset_name)

    def get_subjects_from_indices(self, items):
        """
        Get the subjects from the indices returned by the __getitem__ method (and later collated).

        Parameters
        ----------
        items : torch.Tensor

        Returns
        -------
        tuple[Subject, ...]
        """
        return tuple(self.get_subject_from_idx(item=item) for item in items)

    def get_dataset_indices_from_subjects(self, subjects):
        """
        Get the dataset indices from a tuple of subjects

        Parameters
        ----------
        subjects : tuple[Subject, ...]

        Returns
        -------
        torch.Tensor
        """
        # Get the dictionary mapping from dataset name to dataset index
        dataset_mapping = self.dataset_indices

        # return indices as a torch tensor
        return torch.tensor([dataset_mapping[subject.dataset_name] for subject in subjects])

    # --------------
    # Properties
    # --------------
    @property
    def collate_fn(self):
        # Same as 'get_collate_fn'. Cannot be made to class property because it was removed in Python 3.13 :(
        return self.get_collate_fn()

    @property
    def dataset_names(self):
        """Get the dataset names included in the data. The order is as the keys of the data passed to the __init__
        method"""
        return tuple(self._data.keys())

    @property
    def dataset_shapes(self):
        return {dataset_name: x.shape for dataset_name, x in self._data.items()}

    @property
    def dataset_sizes(self):
        """Get the sizes of the datasets. The keys are the dataset names, the values are the number of subjects in the
        dataset"""
        return {dataset_name: x.shape[0] for dataset_name, x in self._data.items()}

    @property
    def dataset_indices(self):
        """Get a dictionary mapping the dataset name to the dataset index"""
        return {dataset_name: i for i, dataset_name in enumerate(self._data)}

    @property
    def subjects_info(self):
        return self._subjects_info

    @property
    def expected_variables(self):
        return self._expected_variables


# ----------------
# Data generators for RBP
# ----------------
class RBPDataGenBase(DataGenBase, abc.ABC):

    # --------------
    # Magic/dunder methods
    # --------------
    def __init__(self, data, targets, subjects, *, pre_computed=None, subjects_info, expected_variables):
        """
        Initialise

        Parameters
        ----------
        data : dict[str, numpy.ndarray]
        targets : dict[str, numpy.ndarray]
        subjects : dict[str, tuple[str, ...]]
        pre_computed : tuple[dict[str, typing.Any], ...]
        subjects_info : dict[Subject, dict[str, typing.Any]]
            To be passed into .on_epoch_end method of Histories class
        """
        # Input check
        if not all(x.ndim == 4 for x in data.values()):
            _all_sizes = set(x.ndim for x in data.values())
            raise ValueError(f"Expected all input arrays to be 4D with dimensions (subjects, EEG epochs, channels, "
                             f"time_steps), but found {_all_sizes}")

        super().__init__(data=data, targets=targets, subjects=subjects, subjects_info=subjects_info,
                         expected_variables=expected_variables)
        self._pre_computed = pre_computed


class RBPDataGenerator(RBPDataGenBase):
    """
    Pytorch dataset class for downstream training and testing of RBP models of type MainRBPModel

    (unittest in tests folder)
    """

    def __getitem__(self, item):
        # Select dataset and subject in the dataset
        dataset_name, subject_idx, epoch_idx = _select_dataset_and_index(item, dataset_shapes=self.dataset_shapes)

        # Add the data which should be used
        data = torch.tensor(self._data[dataset_name][subject_idx][epoch_idx], dtype=torch.float)
        targets = torch.unsqueeze(
            torch.tensor(self._targets[dataset_name][subject_idx], dtype=torch.float, requires_grad=False), dim=-1)
        subject = Subject(subject_id=self._subjects[dataset_name][subject_idx], dataset_name=dataset_name)

        # Fix the pre-computed features and return
        if self._pre_computed is None:
            return data, None, targets, subject

        pre_computed = []
        for pre_comp in self._pre_computed:
            if pre_comp is None:
                features = None
            else:
                features = pre_comp[dataset_name][subject_idx][epoch_idx]

            pre_computed.append(features)

        # Don't think I should convert pre_computed to tuple, as I must strip it anyway
        return data, pre_computed, targets, subject

    @staticmethod
    def _collate_fn(batch: Tuple[Any, ...]) -> Any:
        # Collate
        input_data: Dict[str, List[torch.Tensor]] = dict()
        target_data: Dict[str, List[torch.Tensor]] = dict()
        pre_computed_data: Optional[List[Optional[Dict[str, List[torch.Tensor]]]]] = None
        unordered_subjects: List[Subject] = []
        all_none = all(features is None for _, features, *_ in batch)
        for i, (tensor, features, target, subject) in enumerate(batch):
            # Fix all the 'normal' stuff
            dataset_name = subject.dataset_name
            if dataset_name not in input_data:
                input_data[dataset_name] = []
                target_data[dataset_name] = []

            input_data[dataset_name].append(tensor)
            target_data[dataset_name].append(target)
            unordered_subjects.append(subject)

            if all_none:
                continue

            # We will now loop over the pre-computed features of single subjects
            if i == 0:
                pre_computed_data = [dict() for _ in features]
            assert pre_computed_data is not None
            for j, (pre_computed, feature) in enumerate(zip(pre_computed_data, features)):
                if feature is None:
                    if i == 0:
                        pre_computed_data[j] = None
                    else:
                        assert pre_computed_data[j] is None, \
                            f"Expected 'precomputed' to be None, but received {pre_computed}"
                    continue

                assert pre_computed is not None
                if dataset_name not in pre_computed:
                    pre_computed[dataset_name] = []
                pre_computed[dataset_name].append(feature)

        # Reorder subjects and stack the tensors
        subjects = reorder_subjects(tuple(input_data), subjects=tuple(unordered_subjects))
        input_data_tensors = {name: torch.stack(tensor_list) for name, tensor_list in input_data.items()}
        target_data_tensors = {name: torch.stack(tensor_list) for name, tensor_list in target_data.items()}
        if pre_computed_data is None:
            assert all_none
            return input_data_tensors, None, target_data_tensors, subjects

        features_tensors: List[Optional[Dict[str, torch.Tensor]]] = []
        for tensors in pre_computed_data:
            if tensors is None:
                features_tensors.append(None)
            else:
                features_tensors.append({name: torch.stack(tensor_list) for name, tensor_list in tensors.items()})
        return input_data_tensors, tuple(features_tensors), target_data_tensors, subjects


class MultiTaskRBPdataGenerator(RBPDataGenBase):
    """
    Pytorch dataset class for training with the multi-task learning approach, where the first task is to predict some
    (pseudo) target the second task is to predict some downstream target from the residual of the first task. Should be
    used with RBP models of type MultiTaskRBPmodel

    (unittest in tests folder)
    """

    strip_outputs = True  # I don't think this is actually used anywhere?

    def __init__(self, *, data, downstream_targets, pretext_targets, subjects, pre_computed, subjects_info,
                 expected_variables, downstream_mask, pretext_mask):
        """
        Initialise

        Parameters
        ----------
        data : dict[str, numpy.ndarray]
        downstream_targets : dict[str, numpy.ndarray]
        pretext_targets : dict[str, numpy.ndarray]
        subjects : dict[str, tuple[str, ...]]
        pre_computed : tuple[dict[str, typing.Any], ...]
        subjects_info : dict[Subject, dict[str, typing.Any]]
            To be passed into .on_epoch_end method of Histories class
        downstream_mask : dict[str, numpy.ndarray] | None
            Boolean masks indicating which should be included in calculations of loss. If None, all data is used
        pretext_mask : dict[str, numpy.ndarray] | None
            Boolean masks indicating which should be included in calculations of loss. If None, all data is used
        """
        # Input check
        if not all(x.ndim == 4 for x in data.values()):
            _all_sizes = set(x.ndim for x in data.values())
            raise ValueError(f"Expected all input arrays to be 4D with dimensions (subjects, EEG epochs, channels, "
                             f"time_steps), but found {_all_sizes}")

        # self._targets will be the downstream targets
        super().__init__(data=data, targets=downstream_targets, subjects=subjects, pre_computed=pre_computed,
                         subjects_info=subjects_info, expected_variables=expected_variables)

        self._pretext_targets = pretext_targets

        # Fix masking if they are None
        if downstream_mask is None:
            self._downstream_mask = create_mask(
                sample_sizes={name: d.shape[0] for name, d in data.items()}, to_include=data.keys())
        else:
            self._downstream_mask = downstream_mask

        if pretext_mask is None:
            self._pretext_mask = create_mask(
                sample_sizes={name: d.shape[0] for name, d in data.items()}, to_include=data.keys())
        else:
            self._pretext_mask = pretext_mask

    def __len__(self):
        return sum(x.shape[0] * x.shape[1] for x in self._data.values())

    def __getitem__(self, item):
        # Select dataset and subject in the dataset
        dataset_name, subject_idx, epoch_idx = _select_dataset_and_index(item, dataset_shapes=self.dataset_shapes)

        # Add the data which should be used. And the masks
        data = torch.tensor(self._data[dataset_name][subject_idx][epoch_idx], dtype=torch.float)
        targets = torch.unsqueeze(
            torch.tensor(self._targets[dataset_name][subject_idx], dtype=torch.float, requires_grad=False), dim=-1)
        pretext_targets = torch.unsqueeze(
            torch.tensor(self._pretext_targets[dataset_name][subject_idx], dtype=torch.float, requires_grad=False),
            dim=-1)

        mask = torch.tensor(bool(self._downstream_mask[dataset_name][subject_idx]), dtype=torch.bool)
        pretext_mask = torch.tensor(bool(self._pretext_mask[dataset_name][subject_idx]), dtype=torch.bool)
        subject = Subject(subject_id=self._subjects[dataset_name][subject_idx], dataset_name=dataset_name)

        # Fix the pre-computed features and return
        if self._pre_computed is None:
            return data, None, (pretext_targets, pretext_mask), (targets, mask), subject

        pre_computed = []
        for pre_comp in self._pre_computed:
            if pre_comp is None:
                features = None
            else:
                features = pre_comp[dataset_name][subject_idx][epoch_idx]

            pre_computed.append(features)

        return data, pre_computed, (pretext_targets, pretext_mask), (targets, mask), subject

    @staticmethod
    def _collate_fn(batch: Tuple[Any, ...]) -> Any:
        # Collate
        input_data: Dict[str, List[torch.Tensor]] = dict()
        target_data: Dict[str, List[torch.Tensor]] = dict()
        pre_computed_data: Optional[List[Optional[Dict[str, List[torch.Tensor]]]]] = None
        pretext_target_data: Dict[str, List[torch.Tensor]] = dict()
        masks: Dict[str, List[torch.Tensor]] = dict()
        pretext_masks: Dict[str, List[torch.Tensor]] = dict()
        unordered_subjects: List[Subject] = []
        all_none = all(features is None for _, features, *_ in batch)
        for i, (input_tensor, features, (pretext_target, pretext_mask), (target, mask),
                subject) in enumerate(batch):
            # Fix all the 'normal' stuff
            dataset_name = subject.dataset_name
            if dataset_name not in input_data:
                input_data[dataset_name] = []
                target_data[dataset_name] = []
                pretext_target_data[dataset_name] = []
                masks[dataset_name] = []
                pretext_masks[dataset_name] = []

            input_data[dataset_name].append(input_tensor)
            target_data[dataset_name].append(target)
            pretext_target_data[dataset_name].append(pretext_target)
            masks[dataset_name].append(mask)
            pretext_masks[dataset_name].append(pretext_mask)
            unordered_subjects.append(subject)

            if all_none:
                continue

            # We will now loop over the pre-computed features of single subjects
            if i == 0:
                pre_computed_data = [dict() for _ in features]
            assert pre_computed_data is not None
            for j, (pre_computed, feature) in enumerate(zip(pre_computed_data, features)):
                if feature is None:
                    if i == 0:
                        pre_computed_data[j] = None
                    else:
                        assert pre_computed_data[j] is None, \
                            f"Expected 'precomputed' to be None, but received {pre_computed}"
                    continue

                assert pre_computed is not None
                if dataset_name not in pre_computed:
                    pre_computed[dataset_name] = []
                pre_computed[dataset_name].append(feature)

        # Reorder subjects and stack the tensors
        subjects = reorder_subjects(tuple(input_data), subjects=tuple(unordered_subjects))
        input_data_tensors = {name: torch.stack(tensor_list) for name, tensor_list in input_data.items()}
        target_data_tensors = {name: torch.stack(tensor_list) for name, tensor_list in target_data.items()}
        pretext_target_data_tensors = {name: torch.stack(tensor_list)
                                       for name, tensor_list in pretext_target_data.items()}
        masks_tensors = {name: torch.stack(tensor_list) for name, tensor_list in masks.items()}
        pretext_masks_tensors = {name: torch.stack(tensor_list) for name, tensor_list in pretext_masks.items()}

        if pre_computed_data is None:
            assert all_none
            return (input_data_tensors, None, (pretext_target_data_tensors, pretext_masks_tensors),
                    (target_data_tensors, masks_tensors), subjects)
        features_tensors: List[Optional[Dict[str, torch.Tensor]]] = []
        for tensors in pre_computed_data:
            if tensors is None:
                features_tensors.append(None)
            else:
                features_tensors.append({name: torch.stack(tensor_list) for name, tensor_list in tensors.items()})
        return (input_data_tensors, tuple(features_tensors), (pretext_target_data_tensors, pretext_masks_tensors),
                (target_data_tensors, masks_tensors), subjects)

    # --------------
    # Properties
    # --------------
    @property
    def downstream_dataset_size(self):
        """Sample sizes for downstream training. Inferred from the downstream mask. Datasets with N=0 will be removed
        from the dictionary"""
        sizes: Dict[str, int] = {}
        for dataset_name, mask in self._downstream_mask.items():
            sample_size = mask.sum()
            if sample_size > 0:
                sizes[dataset_name] = sample_size
        return sizes

    @property
    def pretext_dataset_size(self):
        """Sample sizes for pretext training. Inferred from the pretext mask. Datasets with N=0 will be removed
        from the dictionary"""
        sizes: Dict[str, int] = {}
        for dataset_name, mask in self._pretext_mask.items():
            sample_size = mask.sum()
            if sample_size > 0:
                sizes[dataset_name] = sample_size
        return sizes


# ----------------
# Data generators for interpolation
# ----------------
class InterpolationDataGenBase(DataGenBase, abc.ABC):

    # --------------
    # Magic/dunder methods
    # --------------
    def __init__(self, data, targets, subjects, *, subjects_info, expected_variables):
        """
        Initialise

        Parameters
        ----------
        data : dict[str, numpy.ndarray]
            The data should be interpolated prior to passing them to this method
        targets : dict[str, numpy.ndarray]
        subjects : dict[str, tuple[str, ...]]
        subjects_info : dict[Subject, dict[str, typing.Any]]
        """
        # Input check (data should already be interpolated). Thus checking spatial dimension consistency
        if not len(set(eeg_data.shape[1] for eeg_data in data.values())) == 1:
            _all_shapes = {dataset: eeg_data.shape for dataset, eeg_data in data.items()}
            raise ValueError(f"Expected spatial dimension consistency of all EEG data passed, as the data should "
                             f"already be interpolated. Instead, the following shapes were found {_all_shapes}")

        super().__init__(data=data, targets=targets, subjects=subjects, subjects_info=subjects_info,
                         expected_variables=expected_variables)


class InterpolationDataGenerator(InterpolationDataGenBase):
    """
    Pytorch dataset for downstream training of models which require interpolation for spatial dimension consistency

    (additional tests in tests folder)

    Examples
    --------
    >>> import numpy
    >>> my_data = {"d1": numpy.random.rand(3, 7, 300), "d2": numpy.random.rand(4, 7, 300),
    ...            "d3": numpy.random.rand(1, 7, 300)}
    >>> my_targets = {"d1": numpy.random.rand(3), "d2": numpy.random.rand(4), "d3": numpy.random.rand(1)}
    >>> my_subjects = {"d1": (Subject("P1", "d1"), Subject("P2", "d1"), Subject("P3", "d1")),
    ...                "d2": (Subject("P1", "d2"), Subject("P2", "d2"), Subject("P3", "d2"), Subject("P4", "d2")),
    ...                "d3": (Subject("P1", "d2"),)}
    >>> my_subjects_info = {subject_: {} for subjects_ in my_subjects.values()  # type: ignore[attr-defined]
    ...                     for subject_ in subjects_}  # type: ignore[attr-defined]
    >>> _ = InterpolationDataGenerator(my_data, my_targets, my_subjects, subjects_info=my_subjects_info,
    ...                                expected_variables={})

    A ValueError is raised if spatial dimension is inconsistent

    >>> my_data["d2"] = numpy.random.rand(4, 77, 300)
    >>> InterpolationDataGenerator(my_data, my_targets, my_subjects,subjects_info=my_subjects_info,
    ...                            expected_variables={})  # doctest: +NORMALIZE_WHITESPACE
    Traceback (most recent call last):
    ...
    ValueError: Expected spatial dimension consistency of all EEG data passed, as the data should already be
    interpolated. Instead, the following shapes were found {'d1': (3, 7, 300), 'd2': (4, 77, 300), 'd3': (1, 7, 300)}
    """

    def __getitem__(self, item):
        # Select dataset and subject in the dataset
        dataset_name, subject_idx, epoch_idx = _select_dataset_and_index(item, dataset_shapes=self.dataset_shapes)

        # Add the data which should be used
        data = torch.tensor(self._data[dataset_name][subject_idx][epoch_idx], dtype=torch.float)
        targets = torch.unsqueeze(
            torch.tensor(self._targets[dataset_name][subject_idx], dtype=torch.float, requires_grad=False),
            dim=-1)
        subject = Subject(subject_id=self._subjects[dataset_name][subject_idx], dataset_name=dataset_name)

        # Return input data, targets, and the Subject
        return data, targets, subject

    @staticmethod
    def _collate_fn(batch: Tuple[Any, ...]) -> Any:
        # Collate
        input_data: Dict[str, List[torch.Tensor]] = dict()
        target_data: Dict[str, List[torch.Tensor]] = dict()
        unordered_subjects: List[Subject] = []
        for tensor, target, subject in batch:
            dataset_name = subject.dataset_name

            if dataset_name not in input_data:
                input_data[dataset_name] = []
                target_data[dataset_name] = []

            input_data[dataset_name].append(tensor)
            target_data[dataset_name].append(target)
            unordered_subjects.append(subject)

        # Reorder subjects and stack the tensors
        subjects = reorder_subjects(tuple(input_data), subjects=tuple(unordered_subjects))
        input_data_tensors = {name: torch.stack(tensor_list) for name, tensor_list in input_data.items()}
        target_data_tensors = {name: torch.stack(tensor_list) for name, tensor_list in target_data.items()}
        return input_data_tensors, target_data_tensors, subjects


class MultiTaskInterpolationDataGenerator(InterpolationDataGenBase):
    """
    Data generator for multi-task learning when the EEG datasets have the same number of channels (when using
    interpolation to handle varied electrode configurations)

    (unittests in tests folder)
    """

    def __init__(self, *, data, downstream_targets, subjects, subjects_info, expected_variables, downstream_mask,
                 pretext_mask, pretext_targets):
        # self._targets will be the downstream targets
        super().__init__(data=data, targets=downstream_targets, subjects=subjects, subjects_info=subjects_info,
                         expected_variables=expected_variables)

        # -------------
        # Set MTL-specific attributes
        # -------------
        self._pretext_targets = pretext_targets

        # Fix masking if they are None
        if downstream_mask is None:
            self._downstream_mask = create_mask(
                sample_sizes={name: d.shape[0] for name, d in data.items()}, to_include=data.keys())
        else:
            self._downstream_mask = downstream_mask

        if pretext_mask is None:
            self._pretext_mask = create_mask(
                sample_sizes={name: d.shape[0] for name, d in data.items()}, to_include=data.keys())
        else:
            self._pretext_mask = pretext_mask

    def __getitem__(self, item):
        # Select dataset and subject in the dataset
        dataset_name, subject_idx, epoch_idx = _select_dataset_and_index(item, dataset_shapes=self.dataset_shapes)

        # Add the data which should be used
        data = torch.tensor(self._data[dataset_name][subject_idx][epoch_idx], dtype=torch.float)
        targets = torch.unsqueeze(
            torch.tensor(self._targets[dataset_name][subject_idx], dtype=torch.float, requires_grad=False),
            dim=-1)
        pretext_targets = torch.unsqueeze(
            torch.tensor(self._pretext_targets[dataset_name][subject_idx], dtype=torch.float, requires_grad=False),
            dim=-1)
        mask = torch.tensor(bool(self._downstream_mask[dataset_name][subject_idx]), dtype=torch.bool)
        pretext_mask = torch.tensor(bool(self._pretext_mask[dataset_name][subject_idx]), dtype=torch.bool)
        subject = Subject(subject_id=self._subjects[dataset_name][subject_idx], dataset_name=dataset_name)

        # Return
        return data, (pretext_targets, pretext_mask), (targets, mask), subject

    @staticmethod
    def _collate_fn(batch: Tuple[Any, ...]) -> Any:
        # Collate
        input_data: Dict[str, List[torch.Tensor]] = dict()
        target_data: Dict[str, List[torch.Tensor]] = dict()
        pretext_target_data: Dict[str, List[torch.Tensor]] = dict()
        masks: Dict[str, List[torch.Tensor]] = dict()
        pretext_masks: Dict[str, List[torch.Tensor]] = dict()
        unordered_subjects: List[Subject] = []
        for input_tensor, (pretext_target, pretext_mask), (target, mask), subject in batch:
            dataset_name = subject.dataset_name

            if dataset_name not in input_data:
                input_data[dataset_name] = []
                target_data[dataset_name] = []
                pretext_target_data[dataset_name] = []
                masks[dataset_name] = []
                pretext_masks[dataset_name] = []

            input_data[dataset_name].append(input_tensor)
            target_data[dataset_name].append(target)
            pretext_target_data[dataset_name].append(pretext_target)
            masks[dataset_name].append(mask)
            pretext_masks[dataset_name].append(pretext_mask)
            unordered_subjects.append(subject)

        # Reorder subjects and stack the tensors
        subjects = reorder_subjects(tuple(input_data), subjects=tuple(unordered_subjects))
        input_data_tensors = {name: torch.stack(tensor_list) for name, tensor_list in input_data.items()}
        target_data_tensors = {name: torch.stack(tensor_list) for name, tensor_list in target_data.items()}
        pretext_target_data_tensors = {name: torch.stack(tensor_list)
                                       for name, tensor_list in pretext_target_data.items()}
        masks_tensors = {name: torch.stack(tensor_list) for name, tensor_list in masks.items()}
        pretext_masks_tensors = {name: torch.stack(tensor_list) for name, tensor_list in pretext_masks.items()}
        return (input_data_tensors, (pretext_target_data_tensors, pretext_masks_tensors),
                (target_data_tensors, masks_tensors), subjects)

    # --------------
    # Properties
    # --------------
    @property
    def downstream_dataset_size(self):
        """Sample sizes for downstream training. Inferred from the downstream mask. Datasets with N=0 will be removed
        from the dictionary"""
        sizes: Dict[str, int] = {}
        for dataset_name, mask in self._downstream_mask.items():
            sample_size = mask.sum()
            if sample_size > 0:
                sizes[dataset_name] = sample_size
        return sizes

    @property
    def pretext_dataset_size(self):
        """Sample sizes for pretext training. Inferred from the pretext mask. Datasets with N=0 will be removed
        from the dictionary"""
        sizes: Dict[str, int] = {}
        for dataset_name, mask in self._pretext_mask.items():
            sample_size = mask.sum()
            if sample_size > 0:
                sizes[dataset_name] = sample_size
        return sizes


# ----------------
# Functions
# ----------------
def _select_dataset_and_index(item, dataset_shapes):
    """
    Function for selecting dataset. Only works for positive integer items

    Parameters
    ----------
    item : int
    dataset_shapes : dict[str, tuple[int, ...]]

    Returns
    -------
    tuple[str, int]

    Examples
    --------
    >>> my_shapes = {"a": (3, 3, 19, 3000), "b": (4, 4, 19, 3000), "c": (6, 1, 19, 3000), "d": (7, 2, 19, 3000)}
    >>> def _round(x): return x[0], int(x[1]), int(x[2])
    >>> _round(_select_dataset_and_index(item=15, dataset_shapes=my_shapes))
    ('b', 1, 2)
    >>> _round(_select_dataset_and_index(item=27, dataset_shapes=my_shapes))
    ('c', 2, 0)
    >>> _round(_select_dataset_and_index(item=36, dataset_shapes=my_shapes))
    ('d', 2, 1)
    >>> _round(_select_dataset_and_index(item=44, dataset_shapes=my_shapes))
    ('d', 6, 1)
    >>> _select_dataset_and_index(item=45, dataset_shapes=my_shapes)
    Traceback (most recent call last):
    ...
    IndexError: Index 45 exceeds the total size of the combined dataset 45
    >>> _select_dataset_and_index(item=-1, dataset_shapes=my_shapes)
    Traceback (most recent call last):
    ...
    ValueError: Expected item to be a positive integer, but found -1 (type=<class 'int'>)
    """
    # Input check
    if not isinstance(item, int) or item < 0:
        raise ValueError(f"Expected item to be a positive integer, but found {item} (type={type(item)})")

    # Find the dataset name and position
    accumulated_sizes = 0
    for name, shape in dataset_shapes.items():
        num_subjects, num_eeg_epochs, *_ = shape
        size = num_subjects * num_eeg_epochs
        accumulated_sizes += size
        if item < accumulated_sizes:
            # Now, the current dataset is the correct one. Need to extract the correct subject and EEG epoch indices
            idx = item - (accumulated_sizes - size)

            subject_idx, eeg_epoch_idx = numpy.divmod(idx, num_eeg_epochs)
            return name, subject_idx, eeg_epoch_idx

    # This should not happen...
    raise IndexError(f"Index {item} exceeds the total size of the combined dataset {accumulated_sizes}")


def create_mask(*, sample_sizes, to_include):
    """
    Function for creating boolean masks based on a dict of sample sizes and which keys to include

    Parameters
    ----------
    sample_sizes : dict[str, int]
    to_include

    Returns
    -------
    dict[str, numpy.ndarray]

    Examples
    --------
    >>> my_sizes = {"Ferrari": 7, "Mercedes": 9, "Red Bull": 2}
    >>> create_mask(sample_sizes=my_sizes, to_include=("Red Bull", "Ferrari"))  # doctest: +NORMALIZE_WHITESPACE
    {'Ferrari': array([ True,  True,  True,  True,  True,  True,  True]),
     'Mercedes': array([False, False, False, False, False, False, False, False, False]),
     'Red Bull': array([ True,  True])}
    """
    return {name: numpy.array([(name in to_include)] * size) for name, size in sample_sizes.items()}


def create_subjects_mask(subjects_dict, to_include):
    """
    Function for creating boolean masks based on a dict of subjects for evaluation, and a set of subjects to include

    Parameters
    ----------
    subjects_dict : dict[str, typing.Sequence[Subject]] | dict[str, typing.Sequence[str]]
    to_include : typing.Sequence[Subject] | set[Subject]

    Returns
    -------
    dict[str, numpy.ndarray]

    Examples
    --------
    >>> my_subjects = {"D1": ("S1", "S2", "S3", "S4", "S5"), "D2": ("S1", "S2", "S3"), "D3": ("S1", "S2")}
    >>> my_to_include = {Subject(dataset_name="D1", subject_id="S3"), Subject(dataset_name="D1", subject_id="S1"),
    ...                  Subject(dataset_name="D2", subject_id="S2"), Subject(dataset_name="D1", subject_id="S5")}
    >>> create_subjects_mask(subjects_dict=my_subjects, to_include=my_to_include)  # doctest: +NORMALIZE_WHITESPACE
    {'D1': array([ True, False,  True, False,  True]),
     'D2': array([False,  True, False]),
     'D3': array([False, False])}
    """
    mask: Dict[str, List[bool]] = dict()
    for dataset_name, subjects in subjects_dict.items():
        mask[dataset_name] = []
        for subject in subjects:
            if isinstance(subject, str):
                subject = Subject(dataset_name=dataset_name, subject_id=subject)

            mask[dataset_name].append(subject in to_include)

    return {name: numpy.array(bool_arr) for name, bool_arr in mask.items()}
