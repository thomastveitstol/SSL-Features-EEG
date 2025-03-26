import pytest
import torch

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
def dummy_data_dicts_with_ch_systems():
    """Data from multiple datasets. Includes dummy data as a dict, and channel systems as a dict"""
    channel_systems = dict()
    data = dict()
    num_time_steps = 45 * 3 * 5
    for dataset, dummy_size in zip((LEMON(), Wang(), DortmundVital()), (6, 10, 1)):
        channel_systems[dataset.name] = dataset.channel_system
        data[dataset.name] = torch.rand(size=(dummy_size, dataset.num_channels, num_time_steps))
    return data, channel_systems
