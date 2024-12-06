from typing import Type, Tuple

from elecssl.data.datasets.ai_mind import AIMind
from elecssl.data.datasets.dataset_base import EEGDatasetBase
from elecssl.data.datasets.dortmund_vital import DortmundVital
from elecssl.data.datasets.lemon import LEMON
from elecssl.data.datasets.miltiadous import Miltiadous
from elecssl.data.datasets.srm import SRM
from elecssl.data.datasets.td_brain import TDBRAIN
from elecssl.data.datasets.wang import Wang


def get_dataset(dataset_name, **kwargs):
    """
    Function for getting the specified dataset

    Parameters
    ----------
    dataset_name : str
        Dataset name
    kwargs
        Key word arguments

    Returns
    -------
    EEGDatasetBase
    """
    # All available datasets must be included here
    available_datasets: Tuple[Type[EEGDatasetBase], ...] = (SRM, Miltiadous, Wang, LEMON, TDBRAIN, AIMind,
                                                            DortmundVital)

    # Loop through and select the correct one
    for dataset in available_datasets:
        if dataset_name in (dataset.__name__, dataset().name):
            return dataset(**kwargs)

    # If no match, an error is raised
    raise ValueError(f"The dataset '{dataset_name}' was not recognised. Please select among the following: "
                     f"{tuple(dataset.__name__ for dataset in available_datasets)}")


def get_channel_system(dataset_name, **kwargs):
    """
    Function for getting the specified channel system

    Parameters
    ----------
    dataset_name : str
        Dataset name
    kwargs
        Keyword arguments

    Returns
    -------
    elecssl.data.datasets.dataset_base.ChannelSystem
    """
    return get_dataset(dataset_name, **kwargs).channel_system
