"""
Functions for getting the path to different files related to the data and results
"""
import json
import os
from pathlib import Path


def get_results_dir():
    """
    Get the path to where the results of the experiments are to be stored

    Returns
    -------
    Path
    """
    return Path(os.path.join(os.path.dirname(__file__), "results"))


def get_raw_data_storage_path():
    """
    Get the path to where the newly downloaded data is (supposed to be) stored. It should be specified in the
    'config_paths.json' file

    Returns
    -------
    Path
        The path to where the data is stored, or to be stored (e.g. in the scripts for downloading)
    """
    # Load the config file for paths
    config_path = os.path.join(os.path.dirname(__file__), "config_paths.json")
    with open(config_path) as f:
        config = json.load(f)

    return Path(config["MNEPath"])


def get_numpy_data_storage_path():
    """
    Get the path to where the downloaded data is (supposed to be) stored as numpy arrays.

    Note that this will only work for me (Thomas). I could not store the datasets inside this Python-project, as they
    require too much memory
    Returns
    -------
    Path
        The path to where the data is stored as numpy arrays, or to be stored (e.g. in the scripts for saving as numpy
        arrays)
    """
    # Load the config file for paths
    config_path = os.path.join(os.path.dirname(__file__), "config_paths.json")
    with open(config_path) as f:
        config = json.load(f)

    return Path(config["PreprocessedDataPath"])


def get_eeg_features_storage_path():
    """
    Get the path to where the computed features supposed to be stored.

    Returns
    -------
    Path
    """
    # Load the config file for paths
    config_path = os.path.join(os.path.dirname(__file__), "config_paths.json")
    with open(config_path) as f:
        config = json.load(f)

    return Path(config["FeaturesDataPath"])


def get_td_brain_raw_data_storage_path():
    """
    Get the path to the TDBrain folder. I am doing this to avoid duplicating datasets, and rather just use the path it
    was stored in the Cross dataset learning project.

    Returns
    -------
    Path
    """
    # Load the config file for paths
    config_path = os.path.join(os.path.dirname(__file__), "config_paths.json")
    with open(config_path) as f:
        config = json.load(f)

    return Path(config["MNEPathTDBrain"])


def get_ai_mind_path():
    """
    Gives the path to where the AI-Mind data is stored

    Returns
    -------
    Path
    """
    # Load the config file for paths
    config_path = os.path.join(os.path.dirname(__file__), "config_paths.json")
    with open(config_path) as f:
        config = json.load(f)

    return Path(config["AIMind"])
