import abc
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn

from elecssl.data.subject_split import Subject
from elecssl.models.metrics import Histories


class MainModuleBase(nn.Module, abc.ABC):
    """
    Base class for all models which will used. For example, it combines spatial method (interpolation/RBP), domain
    discriminator training, and the DL architecture itself.
    """
    # ------------
    # Abstract methods
    # ------------
    @abc.abstractmethod
    def train_model(self, *args, **kwargs) -> Dict[str, Histories]:
        """Method for training the model"""

    @abc.abstractmethod
    def test_model(self, *args, **kwargs) -> Histories:
        """Method for testing the model"""

    @abc.abstractmethod
    def save_metadata(self, *, name, path):
        """Method for saving metadata, such as the frequencies in GREEN architecture"""

    # ------------
    # Model saving and loading
    # ------------
    @classmethod
    def load_model(cls, name: str, path: Path) -> "MainModuleBase":
        model = torch.load((path / name).with_suffix(".pt"))
        if not isinstance(model, cls):
            raise ModuleLoadError(f"Expected the loaded module to be from the same class as attempted loaded from "
                                  f"({cls.__name__}), but got {type(model)}")
        return model

    def save_model(self, name: str, path: Path):
        # todo: Sub-optimal to use this saving
        torch.save(self, (path / name).with_suffix(".pt"))


# ----------
# Errors
# ----------
def reorder_subjects(order, subjects):
    """
    Function for re-ordering subjects such that they align with how the input and target tensors are concatenated

    Parameters
    ----------
    order : tuple[str, ...]
        Ordering of the datasets
    subjects : tuple[Subject, ...]
        Subjects to re-order

    Returns
    -------
    tuple[Subject, ...]

    Examples
    --------
    >>> my_subjects = (Subject("P3", "D2"), Subject("P1", "D2"), Subject("P1", "D1"), Subject("P4", "D1"),
    ...                Subject("P2", "D2"))
    >>> reorder_subjects(order=("D1", "D2"), subjects=my_subjects)  # doctest: +NORMALIZE_WHITESPACE
    (Subject(subject_id='P1', dataset_name='D1'),
     Subject(subject_id='P4', dataset_name='D1'),
     Subject(subject_id='P3', dataset_name='D2'),
     Subject(subject_id='P1', dataset_name='D2'),
     Subject(subject_id='P2', dataset_name='D2'))
    """
    subjects_dict: Dict[str, List[Subject]] = {dataset_name: [] for dataset_name in order}
    for subject in subjects:
        subjects_dict[subject.dataset_name].append(subject)

    # Convert to list
    corrected_subjects = []
    for subject_list in subjects_dict.values():
        corrected_subjects.extend(subject_list)

    # return as a tuple
    return tuple(corrected_subjects)


# ----------
# Errors
# ----------
class ModuleLoadError(Exception):
    ...
