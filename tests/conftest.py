import mne
import numpy
import pytest
import torch

from elecssl.data.datasets.dataset_base import EEGDatasetBase, target_method
from elecssl.data.datasets.dortmund_vital import DortmundVital
from elecssl.data.datasets.lemon import LEMON
from elecssl.data.datasets.wang import Wang


@pytest.fixture
def dummy_num_classes():
    return 3


@pytest.fixture
def dummy_data():
    # Some configurations
    batch_size, num_channels, num_time_steps = 10, 19, 2000

    # Make a random tensor
    return torch.rand(size=(batch_size, num_channels, num_time_steps))


@pytest.fixture
def dummy_data_2():
    # Some configurations
    batch_size, num_channels, num_time_steps = 6, 25, 2000

    # Make a random tensor
    return torch.rand(size=(batch_size, num_channels, num_time_steps))


@pytest.fixture
def dummy_data_dicts_with_ch_systems():
    """Data from multiple datasets. Includes dummy data as a dict, and channel systems as a dict"""
    channel_systems = dict()
    data = dict()
    num_time_steps = 45 * 3 * 5
    for dataset, dummy_size in zip((LEMON(), Wang(), DortmundVital()), (6, 10, 1)):
        channel_systems[dataset.name] = dataset.channel_system
        data[dataset.name] = torch.rand(size=(dummy_size, dataset.num_channels, num_time_steps))
    return data, channel_systems


@pytest.fixture
def dummy_eeg_dataset(dummy_data):
    _, dummy_num_channels, dummy_num_time_steps = dummy_data.shape

    class DummyDataset(EEGDatasetBase):
        _num_channels = dummy_num_channels
        _num_time_steps = dummy_num_time_steps
        _channel_names = ("Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T3", "T4", "T5",
                          "T6", "Fz", "Cz", "Pz")
        _montage_name = "standard_1020"

        # -------------
        # Overriding abstract methods which are not required for these tests
        # -------------
        def _load_single_raw_mne_object(self, *args, **kwargs):
            raise NotImplementedError

        # -------------
        # Overriding methods to make this class suited for testing
        # -------------
        @classmethod
        def _get_template_electrode_positions(cls):
            # Following the standard 10-20 system according to the original article
            montage = mne.channels.make_standard_montage(cls._montage_name)
            channel_positions = montage.get_positions()["ch_pos"]

            # Return dict with channel positions, keeping only the ones in the data
            return {ch_name: tuple(pos) for ch_name, pos in channel_positions.items() if ch_name in cls._channel_names}

        def channel_name_to_index(self):
            return {ch_name: i for i, ch_name in enumerate(self._channel_names)}

        def load_numpy_arrays(self, subject_ids=None, pre_processed_version=None, *, time_series_start=None,
                              num_time_steps=None, channels=None, required_target=None):
            return numpy.random.normal(loc=0, scale=1.,
                                       size=(len(subject_ids), self._num_channels, self._num_time_steps))

        @target_method
        def age(self, subject_ids):
            return numpy.random.randint(18, 90, size=(len(subject_ids),))

        @target_method
        def sex(self, subject_ids):
            return numpy.random.randint(0, 2, size=(len(subject_ids),))  # 0s and 1s

    return DummyDataset()


@pytest.fixture
def dummy_eeg_dataset_2(dummy_data_2):
    _, dummy_num_channels, dummy_num_time_steps = dummy_data_2.shape

    class DummyDataset2(EEGDatasetBase):
        _num_channels = dummy_num_channels
        _num_time_steps = dummy_num_time_steps
        _channel_names = ("Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "T7", "C3", "Cz",
                          "C4", "T8", "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8", "POz", "O1", "O2",
                          "FT9", "FT10", "TP9", "TP10")
        _montage_name = "standard_1020"

        # -------------
        # Overriding abstract methods which are not required for these tests
        # -------------
        def _load_single_raw_mne_object(self, *args, **kwargs):
            raise NotImplementedError

        # -------------
        # Overriding methods to make this class suited for testing
        # -------------
        @classmethod
        def _get_template_electrode_positions(cls):
            # Following the standard 10-20 system according to the original article
            montage = mne.channels.make_standard_montage(cls._montage_name)
            channel_positions = montage.get_positions()["ch_pos"]

            # Return dict with channel positions, keeping only the ones in the data
            return {ch_name: tuple(pos) for ch_name, pos in channel_positions.items() if ch_name in cls._channel_names}

        def channel_name_to_index(self):
            return {ch_name: i for i, ch_name in enumerate(self._channel_names)}

        def load_numpy_arrays(self, subject_ids=None, pre_processed_version=None, *, time_series_start=None,
                              num_time_steps=None, channels=None, required_target=None):
            return numpy.random.normal(loc=0, scale=1.,
                                       size=(len(subject_ids), self._num_channels, self._num_time_steps))

        @target_method
        def age(self, subject_ids):
            return numpy.random.randint(50, 70, size=(len(subject_ids),))

        @target_method
        def sex(self, subject_ids):
            return numpy.random.randint(0, 2, size=(len(subject_ids),))  # 0s and 1s

    return DummyDataset2()
