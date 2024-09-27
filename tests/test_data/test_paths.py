import os

from elecssl.data.paths import (get_results_dir, get_raw_data_storage_path, get_numpy_data_storage_path,
                                get_eeg_features_storage_path)


def test_get_results_dir():
    """Test if results path exists"""
    assert os.path.isdir(get_results_dir())


def test_get_raw_data_storage_path():
    """Test if path where raw data is expected to be exists"""
    assert os.path.isdir(get_raw_data_storage_path())


def test_get_numpy_data_storage_path():
    """Test if path where numpy arrays (input data) are expected to be exists"""
    assert os.path.isdir(get_numpy_data_storage_path())


def test_get_eeg_features_storage_path():
    """Test if path where features for SSL are expected to be exists"""
    assert os.path.isdir(get_eeg_features_storage_path())
