import pytest
import torch

from elecssl.models.mts_modules.braindecode_models import TCNMTS, Deep4NetMTS, ShallowFBCSPNetMTS
from elecssl.models.mts_modules.green_model import GreenModel
from elecssl.models.mts_modules.inception_network import InceptionNetwork


@pytest.fixture
def dummy_num_classes():
    return 3


@pytest.fixture
def dummy_data():
    # Some configurations
    batch_size, num_channels, num_time_steps = 10, 19, 2000

    # Make a random tensor
    return torch.rand(size=(batch_size, num_channels, num_time_steps))


# ------------
# DL architectures
# ------------
@pytest.fixture
def inception_network(dummy_data, dummy_num_classes):
    return InceptionNetwork(in_channels=dummy_data.size()[1], num_classes=dummy_num_classes, cnn_units=31, depth=12)


@pytest.fixture
def green(dummy_data, dummy_num_classes):
    return GreenModel(in_channels=dummy_data.size()[1], num_classes=dummy_num_classes, sampling_freq=128,
                      hidden_dim=(123, 11, 67), n_freqs = 30,  kernel_width_s=4, dropout = 0.435,
                      pool_layer="real_covariance", bi_out=39, orth_weights=False)


@pytest.fixture
def tcn(dummy_data, dummy_num_classes):
    return TCNMTS(in_channels=dummy_data.size()[1], num_classes=dummy_num_classes, n_blocks=3, kernel_size=10,
                  n_filters=16, drop_prob=0.42)


@pytest.fixture
def deep_net(dummy_data, dummy_num_classes):
    return Deep4NetMTS(in_channels=dummy_data.size()[1], num_classes=dummy_num_classes,
                       num_time_steps=dummy_data.size()[2])


@pytest.fixture
def shallow_net(dummy_data, dummy_num_classes):
    return ShallowFBCSPNetMTS(in_channels=dummy_data.size()[1], num_classes=dummy_num_classes,
                              num_time_steps=dummy_data.size()[2])


@pytest.fixture
def models(inception_network, green, tcn, deep_net, shallow_net):
    return inception_network, green, tcn, deep_net, shallow_net
