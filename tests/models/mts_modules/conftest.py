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
# Many different versions of green
@pytest.fixture
def green_models(dummy_data, dummy_num_classes):
    num_channels = dummy_data.shape[1]

    # Create models with different single pooling layer
    m1 = GreenModel(in_channels=num_channels, num_classes=dummy_num_classes, sampling_freq=128, shrinkage_init=-4.,
                    hidden_dim=(123, 11, 67), n_freqs=30, kernel_width_s=4, dropout=0.435, pool_layer="real_covariance",
                    bi_out_perc=None)
    m2 = GreenModel(in_channels=num_channels, num_classes=dummy_num_classes, sampling_freq=128, shrinkage_init=None,
                    hidden_dim=(123, 11, 67), n_freqs=30, kernel_width_s=4, dropout=0.435, pool_layer="pw_plv",
                    bi_out_perc=None, pool_layer_kwargs={"reg": 1e-5})
    m3 = GreenModel(in_channels=num_channels, num_classes=dummy_num_classes, sampling_freq=128, shrinkage_init=None,
                    hidden_dim=(123, 11, 67), n_freqs=30, kernel_width_s=4, dropout=0.435, pool_layer="cross_pw_plv",
                    bi_out_perc=0.8,  pool_layer_kwargs={"reg": 1e-5})
    m4 = GreenModel(in_channels=num_channels, num_classes=dummy_num_classes, sampling_freq=128, shrinkage_init=-2.,
                    hidden_dim=(123, 11, 67), n_freqs=30, kernel_width_s=4, dropout=0.435,
                    pool_layer="cross_covariance", bi_out_perc=0.5)

    # Can only combine cross-frequency with cross-frequency, and vice versa
    m5 = GreenModel(in_channels=num_channels, num_classes=dummy_num_classes, sampling_freq=128, shrinkage_init=None,
                    hidden_dim=(123, 11, 67), n_freqs=30, kernel_width_s=4, dropout=0.435,
                    pool_layer="combined_pooling", bi_out_perc=0.5,
                    pool_layer_kwargs=(("pw_plv", {"reg": 3e-6}), ("real_covariance", {})))
    m6 = GreenModel(in_channels=num_channels, num_classes=dummy_num_classes, sampling_freq=128, shrinkage_init=None,
                    hidden_dim=(123, 11, 67), n_freqs=30, kernel_width_s=4, dropout=0.435,
                    pool_layer="combined_pooling", bi_out_perc=0.5,
                    pool_layer_kwargs=(("cross_pw_plv", {"reg": 1e-5}), ("cross_covariance", {})))

    return m1, m2, m3, m4, m5, m6


# The other ones
@pytest.fixture
def inception_network(dummy_data, dummy_num_classes):
    return InceptionNetwork(in_channels=dummy_data.size()[1], num_classes=dummy_num_classes, cnn_units=31, depth=12)


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
def models(inception_network, green_models, tcn, deep_net, shallow_net):
    return inception_network, *green_models, tcn, deep_net, shallow_net
