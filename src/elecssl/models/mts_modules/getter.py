"""
Contains only a function for returning a specified MTS module
"""
from typing import Type, Tuple

from elecssl.models.mts_modules.braindecode_models import Deep4NetMTS, ShallowFBCSPNetMTS, TCNMTS
from elecssl.models.mts_modules.green_model import GreenModel
from elecssl.models.mts_modules.inception_network import InceptionNetwork
from elecssl.models.mts_modules.mts_module_base import MTSModuleBase


def get_available_architectures() -> Tuple[Type[MTSModuleBase], ...]:
    """All MTS modules must be available from here"""
    return InceptionNetwork, ShallowFBCSPNetMTS, Deep4NetMTS, GreenModel, TCNMTS


def get_mts_module(mts_module_name, **kwargs):
    """
    Function for getting a specified MTS module

    Parameters
    ----------
    mts_module_name : str
        MTS Module
    kwargs
        Key word arguments, which depends on the selected MTS module

    Returns
    -------
    cdl_eeg.models.mts_modules.mts_module_base.MTSModuleBase

    Examples
    --------
    >>> _ = get_mts_module("InceptionNetwork", in_channels=5, num_classes=3)
    >>> get_mts_module("NotAnMTSModule")  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: The MTS module 'NotAnMTSModule' was not recognised. Please select among the following:
    ('InceptionNetwork',...)
    """
    # Get all available MTS modules
    available_mts_modules = get_available_architectures()

    # Loop through and select the correct one
    for mts_module in available_mts_modules:
        if mts_module_name == mts_module.__name__:
            return mts_module(**kwargs)

    # If no match, an error is raised
    raise ValueError(f"The MTS module '{mts_module_name}' was not recognised. Please select among the following: "
                     f"{tuple(mts_module.__name__ for mts_module in available_mts_modules)}")


def get_mts_module_type(mts_module_name):
    """Function for getting a specified MTS module class"""
    # Get all available MTS modules
    available_mts_modules = get_available_architectures()

    # Loop through and select the correct one
    for mts_module in available_mts_modules:
        if mts_module_name == mts_module.__name__:
            return mts_module

    # If no match, an error is raised
    raise ValueError(f"The MTS module '{mts_module_name}' was not recognised. Please select among the following: "
                     f"{tuple(mts_module.__name__ for mts_module in available_mts_modules)}")
