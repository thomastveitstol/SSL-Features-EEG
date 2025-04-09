from typing import Type, Tuple, Dict

import mne

from elecssl.data.datasets.ai_mind import AIMind
from elecssl.data.datasets.dataset_base import EEGDatasetBase, ChannelSystem
from elecssl.data.datasets.dortmund_vital import DortmundVital
from elecssl.data.datasets.lemon import LEMON
from elecssl.data.datasets.miltiadous import Miltiadous
from elecssl.data.datasets.srm import SRM
from elecssl.data.datasets.td_brain import TDBRAIN
from elecssl.data.datasets.wang import Wang


# ---------------
# Make some basic channel systems
# ---------------
def _make_ch_name_to_idx(*, channel_names):
    return {name: i for i, name in enumerate(channel_names)}


def _extract_positions(*, montage, channel_names):
    channel_positions: Dict[str, Tuple[float, float, float]] = montage.get_positions()["ch_pos"]
    return {ch_name: channel_positions[ch_name] for ch_name in channel_names}



def _get_basic_channel_systems():
    """
    Get some pre-defined basic channel systems

    Returns
    -------
    dict[str, ChannelSystem]

    Examples
    --------
    >>> my_ch_systems = _get_basic_channel_systems()
    >>> tuple(my_ch_systems.keys())
    ('Standard19', 'Standard32', 'Standard64', 'Standard126')
    >>> len(my_ch_systems["Standard19"].channel_name_to_index), len(my_ch_systems["Standard19"].electrode_positions)
    (19, 19)
    >>> len(my_ch_systems["Standard32"].channel_name_to_index), len(my_ch_systems["Standard32"].electrode_positions)
    (32, 32)
    >>> len(my_ch_systems["Standard64"].channel_name_to_index), len(my_ch_systems["Standard64"].electrode_positions)
    (64, 64)
    >>> len(my_ch_systems["Standard126"].channel_name_to_index), len(my_ch_systems["Standard126"].electrode_positions)
    (126, 126)
    """
    # Get info from MNE
    montage_name_1020 = "standard_1020"
    standard_montage_1020 = mne.channels.make_standard_montage(montage_name_1020)

    montage_name_1005 = "standard_1005"
    standard_montage_1005 = mne.channels.make_standard_montage(montage_name_1005)

    # Make all basic channel systems
    names_19 = ("Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6", "Fz",
                 "Cz", "Pz")
    names_32 = ("Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T7", "T8", "P7", "P8", "Fz",
                "Cz", "Pz", "AF3", "AF4", "FC1", "FC2", "CP1", "CP2", "PO3", "PO4", "FC5", "FC6", "CP5", "CP6", "Oz")
    names_64 = ('Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8',
                'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2',
                'PO10', 'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FT9', 'FT7', 'FC3', 'FC4', 'FT8', 'FT10',
                'C5', 'C1', 'C2', 'C6', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'P5', 'P1', 'P2', 'P6', 'PO7', 'PO3', 'POz',
                'PO4', 'PO8')  # From dortmund vital
    names_126 = ("Fp1", "Fpz", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "M1", "T7", "C3", "Cz",
                 "C4", "T8", "M2", "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8", "POz", "O1", "O2", "AF7",
                 "AF3", "AF4", "AF8", "F5", "F1", "F2", "F6", "FC3", "FCz", "FC4", "C5", "C1", "C2", "C6", "CP3", "CP4",
                 "P5", "P1", "P2", "P6", "F9", "PO3", "PO4", "F10", "FT7", "FT8", "TP7", "TP8","PO7", "PO8", "FT9",
                 "FT10", "TPP9h", "TPP10h", "PO9", "PO10", "P9", "P10", "AFF1", "AFz", "AFF2", "FFC5h", "FFC3h",
                 "FFC4h", "FFC6h", "FCC5h", "FCC3h", "FCC4h", "FCC6h", "CCP5h", "CCP3h", "CCP4h","CCP6h", "CPP5h",
                 "CPP3h", "CPP4h", "CPP6h", "PPO1", "PPO2", "I1", "Iz", "I2", "AFp3h", "AFp4h","AFF5h", "AFF6h",
                 "FFT7h", "FFC1h", "FFC2h", "FFT8h", "FTT9h", "FTT7h", "FCC1h", "FCC2h", "FTT8h", "FTT10h", "TTP7h",
                 "CCP1h", "CCP2h", "TTP8h", "TPP7h", "CPP1h", "CPP2h", "TPP8h", "PPO9h","PPO5h", "PPO6h", "PPO10h",
                 "POO9h", "POO3h", "POO4h", "POO10h", "OI1h", "OI2h")  # From AI-Mind

    channel_systems = (
        ChannelSystem(name="Standard19", channel_name_to_index=_make_ch_name_to_idx(channel_names=names_19),
                      electrode_positions=_extract_positions(montage=standard_montage_1020, channel_names=names_19),
                      montage_name=montage_name_1020),
        ChannelSystem(name="Standard32", channel_name_to_index=_make_ch_name_to_idx(channel_names=names_32),
                      electrode_positions=_extract_positions(montage=standard_montage_1020, channel_names=names_32),
                      montage_name=montage_name_1020),
        ChannelSystem(name="Standard64", channel_name_to_index=_make_ch_name_to_idx(channel_names=names_64),
                      electrode_positions=_extract_positions(montage=standard_montage_1020, channel_names=names_64),
                      montage_name=montage_name_1020),
        ChannelSystem(name="Standard126", channel_name_to_index=_make_ch_name_to_idx(channel_names=names_126),
                      electrode_positions=_extract_positions(montage=standard_montage_1005, channel_names=names_126),
                      montage_name=montage_name_1005)
    )
    return {ch_system.name: ch_system for ch_system in channel_systems}


# ---------------
# Getter functions
# ---------------
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
    basic_channel_systems = _get_basic_channel_systems()
    if dataset_name in basic_channel_systems:
        return basic_channel_systems[dataset_name]

    return get_dataset(dataset_name, **kwargs).channel_system
