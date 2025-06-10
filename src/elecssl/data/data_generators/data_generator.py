from typing import List, Dict

import numpy
import torch
from torch.utils.data import Dataset

from elecssl.data.subject_split import Subject


# ----------------
# Data generators for RBP
# ----------------
class RBPDataGenBase(Dataset):  # type: ignore[type-arg]

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

        super().__init__()

        self._data = data
        self._targets = targets
        self._subjects = subjects
        self._pre_computed = pre_computed
        self._subjects_info = subjects_info
        self._expected_variables = expected_variables

    def __len__(self):
        return sum(x.shape[0] * x.shape[1] for x in self._data.values())

    # --------------
    # Convenient methods
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


class RBPDataGenerator(RBPDataGenBase):
    """
    Pytorch dataset class for downstream training and testing of RBP models of type MainRBPModel

    (unittest in tests folder)
    """

    # Remember to remove the yielded -1 tensors!
    strip_outputs = True  # I don't think this is actually used anywhere?

    def __getitem__(self, item):
        # Varying keys in the returned dictionary is not possible with the DataLoader of PyTorch. This solution to the
        # problem is to simply return a tensor of -1s for the datasets not used
        data = {dataset_name: torch.ones(size=x.shape[2:]) * (-1) for dataset_name, x in self._data.items()}
        targets = {dataset_name: torch.unsqueeze(torch.ones(size=y.shape[1:]) * (-1), dim=-1)
                   for dataset_name, y in self._targets.items()}

        # Select dataset and subject in the dataset
        dataset_name, subject_idx, epoch_idx = _select_dataset_and_index(item, dataset_shapes=self.dataset_shapes)

        # Add the data which should be used
        data[dataset_name] = torch.tensor(self._data[dataset_name][subject_idx][epoch_idx], dtype=torch.float)
        targets[dataset_name] = torch.unsqueeze(torch.tensor(self._targets[dataset_name][subject_idx],
                                                             dtype=torch.float, requires_grad=False),
                                                dim=-1)

        # Return
        if self._pre_computed is None:
            return data, torch.tensor(float("nan")), targets, item
        else:
            # assert False, {type(pre_comp) for pre_comp in self._pre_computed}
            pre_computed = []
            for pre_comp in self._pre_computed:
                if pre_comp is None:
                    my_dict = {data_name: torch.tensor(float("nan")) for data_name in self.dataset_names}
                else:
                    my_dict = {data_name: torch.ones(tensor.size()[2:]) * (-1)
                               for data_name, tensor in pre_comp.items()}
                    my_dict[dataset_name] = pre_comp[dataset_name][subject_idx][epoch_idx]

                pre_computed.append(my_dict)

            # Don't think I should convert pre_computed to tuple, as I must strip it anyway
            return data, pre_computed, targets, item


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
        # Varying keys in the returned dictionary is not possible with the DataLoader of PyTorch. This solution to the
        # problem is to simply return a tensor of -1s for the datasets not used
        data = {dataset_name: torch.ones(size=x.shape[2:]) * (-1) for dataset_name, x in self._data.items()}
        targets = {dataset_name: torch.unsqueeze(torch.ones(size=y.shape[1:]) * (-1), dim=-1)
                   for dataset_name, y in self._targets.items()}
        pretext_targets = {dataset_name: torch.unsqueeze(torch.ones(size=y.shape[1:]) * (-1), dim=-1)
                           for dataset_name, y in self._pretext_targets.items()}

        mask = {dataset_name: torch.ones(size=y.shape[1:], dtype=torch.int) * (-1)
                for dataset_name, y in self._downstream_mask.items()}
        pretext_mask = {dataset_name: torch.ones(size=y.shape[1:], dtype=torch.int) * (-1)
                        for dataset_name, y in self._pretext_mask.items()}

        # Select dataset and subject in the dataset
        dataset_name, subject_idx, epoch_idx = _select_dataset_and_index(item, dataset_shapes=self.dataset_shapes)

        # Add the data which should be used. And the masks
        data[dataset_name] = torch.tensor(self._data[dataset_name][subject_idx][epoch_idx], dtype=torch.float)
        targets[dataset_name] = torch.unsqueeze(torch.tensor(self._targets[dataset_name][subject_idx],
                                                             dtype=torch.float, requires_grad=False),
                                                dim=-1)
        pretext_targets[dataset_name] = torch.unsqueeze(torch.tensor(self._pretext_targets[dataset_name][subject_idx],
                                                                     dtype=torch.float, requires_grad=False),
                                                        dim=-1)

        mask[dataset_name] = torch.tensor(self._downstream_mask[dataset_name][subject_idx], dtype=torch.bool)
        pretext_mask[dataset_name] = torch.tensor(self._pretext_mask[dataset_name][subject_idx], dtype=torch.bool)

        # Return
        if self._pre_computed is None:
            return data, torch.tensor(float("nan")), (pretext_targets, pretext_mask), (targets, mask), item
        else:
            # assert False, {type(pre_comp) for pre_comp in self._pre_computed}
            pre_computed = []
            for pre_comp in self._pre_computed:
                if pre_comp is None:
                    my_dict = {data_name: torch.tensor(float("nan")) for data_name in self.dataset_names}
                else:
                    my_dict = {data_name: torch.ones(tensor.size()[2:]) * (-1)
                               for data_name, tensor in pre_comp.items()}
                    my_dict[dataset_name] = pre_comp[dataset_name][subject_idx][epoch_idx]

                pre_computed.append(my_dict)

            # Don't think I should convert pre_computed to tuple, as I must strip it anyway
            return data, pre_computed, (pretext_targets, pretext_mask), (targets, mask), item

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
class InterpolationDataGenBase(Dataset):  # type: ignore[type-arg]

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

        super().__init__()

        self._data = data
        self._targets = targets
        self._subjects = subjects
        self._subjects_info = subjects_info
        self._expected_variables = expected_variables

    def __len__(self):
        return sum(x.shape[0] * x.shape[1] for x in self._data.values())

    # --------------
    # Convenient methods
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
        dataset_name, subject_idx, epoch_idx = _select_dataset_and_index(item=int(item),
                                                                         dataset_shapes=self.dataset_shapes)

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


class InterpolationDataGenerator(InterpolationDataGenBase):  # type: ignore[type-arg]
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

    A ValueError is raise if spatial dimension is inconsistent

    >>> my_data["d2"] = numpy.random.rand(4, 77, 300)
    >>> InterpolationDataGenerator(my_data, my_targets, my_subjects,subjects_info=my_subjects_info,
    ...                            expected_variables={})  # doctest: +NORMALIZE_WHITESPACE
    Traceback (most recent call last):
    ...
    ValueError: Expected spatial dimension consistency of all EEG data passed, as the data should already be
    interpolated. Instead, the following shapes were found {'d1': (3, 7, 300), 'd2': (4, 77, 300), 'd3': (1, 7, 300)}
    """

    def __getitem__(self, item):
        # Varying keys in the returned dictionary is not possible with the DataLoader of PyTorch. This solution to the
        # problem is to simply return a tensor of -1s for the datasets not used
        data = {dataset_name: torch.ones(size=x.shape[2:]) * (-1) for dataset_name, x in self._data.items()}
        targets = {dataset_name: torch.unsqueeze(torch.ones(size=y.shape[1:]) * (-1), dim=-1)
                   for dataset_name, y in self._targets.items()}

        # Select dataset and subject in the dataset
        dataset_name, subject_idx, epoch_idx = _select_dataset_and_index(item, dataset_shapes=self.dataset_shapes)

        # Add the data which should be used
        data[dataset_name] = torch.tensor(self._data[dataset_name][subject_idx][epoch_idx], dtype=torch.float)
        targets[dataset_name] = torch.unsqueeze(torch.tensor(self._targets[dataset_name][subject_idx],
                                                             dtype=torch.float, requires_grad=False),
                                                dim=-1)

        # Return input data, targets, and the item (will be converted to Subject later)
        return data, targets, item


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
        # Varying keys in the returned dictionary is not possible with the DataLoader of PyTorch. This solution to the
        # problem is to simply return a tensor of -1s for the datasets not used
        data = {dataset_name: torch.ones(size=x.shape[2:]) * (-1) for dataset_name, x in self._data.items()}
        targets = {dataset_name: torch.unsqueeze(torch.ones(size=y.shape[1:]) * (-1), dim=-1)
                   for dataset_name, y in self._targets.items()}
        pretext_targets = {dataset_name: torch.unsqueeze(torch.ones(size=y.shape[1:]) * (-1), dim=-1)
                           for dataset_name, y in self._pretext_targets.items()}
        mask = {dataset_name: torch.ones(size=y.shape[1:], dtype=torch.int) * (-1)
                for dataset_name, y in self._downstream_mask.items()}
        pretext_mask = {dataset_name: torch.ones(size=y.shape[1:], dtype=torch.int) * (-1)
                        for dataset_name, y in self._pretext_mask.items()}

        # Select dataset and subject in the dataset
        dataset_name, subject_idx, epoch_idx = _select_dataset_and_index(item, dataset_shapes=self.dataset_shapes)

        # Add the data which should be used
        data[dataset_name] = torch.tensor(self._data[dataset_name][subject_idx][epoch_idx], dtype=torch.float)
        targets[dataset_name] = torch.unsqueeze(torch.tensor(self._targets[dataset_name][subject_idx],
                                                             dtype=torch.float, requires_grad=False),
                                                dim=-1)
        pretext_targets[dataset_name] = torch.unsqueeze(torch.tensor(self._pretext_targets[dataset_name][subject_idx],
                                                                     dtype=torch.float, requires_grad=False),
                                                        dim=-1)
        mask[dataset_name] = torch.tensor(self._downstream_mask[dataset_name][subject_idx], dtype=torch.bool)
        pretext_mask[dataset_name] = torch.tensor(self._pretext_mask[dataset_name][subject_idx], dtype=torch.bool)

        # Return
        return data, (pretext_targets, pretext_mask), (targets, mask), item

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


def strip_tensors(tensors, fill_val=-1):
    """
    Function which may be used to remove unused tensors

    This function changes the input in-place (meaning it is not actually important to use the return)

    Pro-tip: use this before sending the data to a GPU

    Parameters
    ----------
    tensors : dict[str, torch.Tensor]
    fill_val : int
        The value which was used to indicate that a tensor should not be there

    Returns
    -------
    dict[str, torch.Tensor]

    Examples
    --------
    >>> my_fill = -1
    >>> tensor_a = torch.rand(size=(1, 5, 30))
    >>> tensor_b = torch.rand(size=(1, 3, 30))
    >>> tensor_c = torch.rand(size=(1, 6, 30))
    >>> my_tensors = {"a": torch.cat((tensor_a, torch.ones(size=(1, 5, 30)) * my_fill,
    ...                               torch.ones(size=(1, 5, 30)) * my_fill), dim=0),
    ...               "b": torch.cat((torch.ones(size=(1, 3, 30)) * my_fill, torch.ones(size=(1, 3, 30)) * my_fill,
    ...                               tensor_b), dim=0),
    ...               "c": torch.cat((torch.ones(size=(1, 6, 30)) * my_fill, tensor_c,
    ...                               torch.ones(size=(1, 6, 30)) * my_fill), dim=0),
    ...               "d": torch.cat((torch.ones(size=(1, 11, 30)) * my_fill, torch.ones(size=(1, 11, 30)) * my_fill,
    ...                               torch.ones(size=(1, 11, 30)) * my_fill), dim=0)}
    >>> my_stripped_tensors = strip_tensors(my_tensors)
    >>> tuple(my_stripped_tensors.keys()), tuple(my_stripped_tensors.keys())  # Left out dataset 'd'
    (('a', 'b', 'c'), ('a', 'b', 'c'))

    The operations were also made in-place (the input dict is changed as well)

    >>> all(torch.equal(new_tensor, old_tensor) for new_tensor, old_tensor  # type: ignore[attr-defined]
    ...     in zip(my_stripped_tensors.values(), my_tensors.values()))
    True

    If all tensors only contain nan values, None is returned

    >>> strip_tensors({"d1": torch.tensor([float("nan"), float("nan"), float("nan"), float("nan")]),
    ...                "d2": torch.tensor([float("nan"), float("nan")])}) is None
    True
    """
    # -------------
    # Maybe ignore all tensors
    # -------------
    is_nan: List[bool] = []
    for tensor in tensors.values():
        # assert False, torch.isnan(tensor)
        if torch.all(torch.isnan(tensor)):
            is_nan.append(True)
        elif torch.any(torch.isnan(tensor)):
            _dict = {name: tensor for name, tensor in tensors.items()}
            raise ValueError(f"Expected all  or none of the values in the tensor to be 'nan', but found both. {_dict}")
        else:
            is_nan.append(False)

    if all(is_nan):
        return None
    elif any(is_nan):
        ValueError("Expected all all or none of the tensor to be 'nan', but found both")

    # -------------
    # Remove ghost tensors
    # -------------
    # Loop through all datasets. Changing values while iterating is ok, inserting/deleting is not. Thanks, James
    # 'mCoding' Murphy (Sec. 13): https://www.youtube.com/watch?v=E8NijUYfyus
    to_delete = set()
    for dataset_name, x in tensors.items():
        # Get the indices of which indices to keep
        ghost_tensor = torch.ones(size=x.size()[1:]) * fill_val
        kept_indices = [i for i, tensor in enumerate(x) if not torch.equal(tensor, ghost_tensor)]  # type: ignore

        # If no data is supposed to be used in the batch, the dataset should be deleted. Otherwise, keep only the real
        # ones
        if not kept_indices:
            to_delete.add(dataset_name)
        else:
            tensors[dataset_name] = x[kept_indices]

    # Delete
    for dataset_name in to_delete:
        del tensors[dataset_name]

    # Return the dictionary of tensors (although the operations also happen in-place)
    return tensors


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
