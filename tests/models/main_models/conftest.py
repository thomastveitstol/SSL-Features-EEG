import random

import optuna
import pytest
import torch

from elecssl.data.subject_split import Subject
# noinspection PyProtectedMember
from elecssl.models.hp_suggesting import _suggest_rbp, suggest_dl_architecture
from elecssl.models.main_models.main_fixed_channels_model import MainFixedChannelsModel
from elecssl.models.main_models.main_rbp_model import MainRBPModel


@pytest.fixture
def input_data():
    return {"DummyDataset": torch.rand(size=(10, 19, 200)),
            "DummyDataset2": torch.rand(size=(6, 32, 200))}


@pytest.fixture
def interpolated_input_data():
    return {"DummyDataset": torch.rand(size=(10, 32, 200)),
            "DummyDataset2": torch.rand(size=(6, 32, 200))}


@pytest.fixture
def target_data(input_data):
    return {dataset_name: torch.rand(inputs.size()[0]) for dataset_name, inputs in input_data.items()}


@pytest.fixture
def subjects(input_data):
    return {dataset_name: tuple(Subject(dataset_name=dataset_name, subject_id=f"sub-{i}")
                                for i in range(inputs.size()[0]))
            for dataset_name, inputs in input_data.items()}


@pytest.fixture
def dl_hpds(input_data):
    num_time_steps = input_data["DummyDataset"].shape[-1]
    config = {
        "InceptionNetwork": {
            "num_classes": 1,
            "cnn_units": {"low": 2, "high": 6, "log": True},
            "num_res_blocks": {"low": 1, "high": 2, "log": True},
        },
        "ShallowFBCSPNetMTS": {
            "num_classes": 1,
            "num_time_steps": num_time_steps,
            "num_filters": {"low": 20, "high": 60, "step": 1, "log": False},
            "filter_time_length": {"low": 5, "high": 45, "step": 1, "log": False},
            "pool_time_stride": {"low": 5, "high": 25, "step": 1, "log": False},
            "drop_prob": {"low": 0.0, "high": 0.5, "step": None, "log": False},
        },
        "Deep4NetMTS": {
            "num_classes": 1,
            "num_time_steps": num_time_steps,
            "num_first_filters": {"low": 10, "high": 15, "step": 1, "log": False},
            "filter_length": {"low": 5, "high": 15, "step": 1, "log": False},
            "drop_prob": {"low": 0.0, "high": 0.5, "log": False},
        },
        "TCNMTS": {
            "num_classes": 1,
            "n_blocks": {"low": 1, "high": 5, "step": 1, "log": False},
            "n_filters": {"low": 4, "high": 128, "step": 1, "log": True},
            "kernel_size": {"low": 3, "high": 8, "step": 1, "log": False},
            "drop_prob": {"low": 0.0, "high": 0.5, "log": False},
        },
        "GreenModel": {
            "num_classes": 1,
            "sampling_freq": 45,
            "n_freqs": {"low": 1, "high": 5, "step": 1, "log": True},
            "kernel_width_s": {"low": 0.5, "high": 2, "step": None, "log": False},
            "conv_stride": {"low": 1, "high": 5, "step": 1, "log": False},
            "oct_min": 0.0,
            "oct_max_addition": 5.5,
            "random_f_init": {"choices": [False, True]},
            "pool_layer": {
                "choices": [
                    "RealCovariance",
                    "CrossCovariance",
                    "PW_PLV",
                    "CrossPW_PLV",
                    "CombinedPoolingNoCross",
                    "CombinedPoolingCross",
                ]
            },
            "pool_layer_kwargs": {
                "RealCovariance": {},
                "CrossCovariance": {},
                "PW_PLV": {"reg": {"low": 0.0000001, "high": 0.00001, "log": True}},
                "CrossPW_PLV": {"reg": {"low": 0.0000001, "high": 0.00001, "log": True}},
                "CombinedPoolingNoCross": {
                    "RealCovariance": {},
                    "PW_PLV": {"reg": {"low": 0.0000001, "high": 0.00001, "log": True}},
                },
                "CombinedPoolingCross": {
                    "CrossCovariance": {},
                    "CrossPW_PLV": {"reg": {"low": 0.0000001, "high": 0.00001, "log": True}},
                },
            },
            "shrinkage_init": {"low": -5.0, "high": -1.0, "step": None, "log": False},
            "bi_out_perc": {"low": 0.5, "high": 1, "step": None, "log": False},
            "orth_weights": {"choices": [False]},
            "logref": {"choices": ["identity", "logeuclid"]},
            "reeig_reg": {"low": 0.00001, "high": 0.001, "step": None, "log": True},
            "momentum": {"low": 0.8, "high": 1.0, "step": None, "log": False},
            "num_fc_layers": {"low": 1, "high": 2, "step": 1, "log": False},
            "num_first_fc_filters": {"low": 8, "high": 16, "step": 1, "log": True},
            "drop_prob": {"low": 0.0, "high": 0.5, "log": False},
        },
    }
    return config


# --------------
# Main models
# --------------
@pytest.fixture
def rbp_main_models(dl_hpds, dummy_eeg_dataset, dummy_eeg_dataset_2):
    num_models = 25

    # Fix HP distributions
    num_kernels = ("int", {"low": 10, "high": 20, "log": True})
    max_receptive_field = ("int", {"low": 15, "high": 30, "log": True})
    rbp_hpds = {
        "num_montage_splits": {"low": 1, "high": 4, "log": False},
        "share_all_pooling_modules": {"choices": [True, False]},
        "num_pooling_modules_percentage": {"low": 0.0, "high": 1.0, "step": None, "log": False},
        "num_designs": 1,
        "pooling_type": "multi_cs",
        "PoolingMethods": {
            "MultiMSMean": {},
            "MultiMSSharedRocket": {
                "num_kernels": num_kernels,
                "max_receptive_field": max_receptive_field
            },
            "MultiMSSharedRocketHeadRegion": {
                "num_kernels": num_kernels,
                "max_receptive_field": max_receptive_field,
                "latent_search_features": ("int", {"low": 2, "high": 16, "log": True}),
                "share_search_receiver_modules": ("categorical", {"choices": [True, False]}),
                "bias": ("categorical", {"choices": [False]})
            },
        },
        "MontageSplits": {
            "CentroidPolygons": {
                "k": ("categorical_dict",
                      {"k_1": [2, 2, 2, 2, 2, 2, 2],
                       "k_2": [3, 3, 3, 3, 3, 3, 3, 3],
                       "k_3": [2, 3, 2, 3, 2, 3, 2, 3, 2],
                       "k_4": [4, 3, 2, 3, 4, 3, 2, 3, 4]}),
                "min_nodes": ("int", {"low": 1, "high": 6, "step": 1, "log": False}),
                "channel_positions": ("not_a_hyperparameter", ["Miltiadous", "LEMON"])
            }
        }
    }

    study = optuna.create_study()
    trial = study.ask()

    # Make many models
    models = []
    for _ in range(num_models):
        # Make up some HPCs
        rbp_config = _suggest_rbp(name="unimportant", config=rbp_hpds, normalisation=random.choice((True, False)),
                                  cmmn={"kwargs": {}, "use_cmmn_layer": False}, trial=trial)["kwargs"]
        dl_config = suggest_dl_architecture(name="unimportant", trial=trial, config=dl_hpds,
                                            preprocessed_config_path=None, suggested_preprocessing_steps=None,
                                            freq_band=None)

        # Create and prepare model
        model = MainRBPModel.from_config(rbp_config=rbp_config, discriminator_config=None, mts_config=dl_config)
        model.fit_channel_systems((dummy_eeg_dataset.channel_system, dummy_eeg_dataset_2.channel_system))

        models.append(model)

    return tuple(models)


@pytest.fixture
def interpolation_main_models(dl_hpds, interpolated_input_data):
    num_models = 25

    # Create a fake trial
    study = optuna.create_study()
    trial = study.ask()

    # Get number of channels
    _num_channels = set(arr.shape[1] for arr in interpolated_input_data.values())
    assert len(_num_channels) == 1
    num_channels = next(iter(_num_channels))

    # Make many models
    models = []
    for _ in range(num_models):
        # Make up some HPCs
        dl_config = suggest_dl_architecture(
            name="unimportant", trial=trial, config=dl_hpds, preprocessed_config_path=None,
            suggested_preprocessing_steps=None, freq_band=None)
        dl_config["normalise"] = random.choice((True, False))
        dl_config["kwargs"]["in_channels"] = num_channels

        # Create and prepare model
        model = MainFixedChannelsModel.from_config(
            mts_config=dl_config, discriminator_config=None, cmmn_config={"kwargs": {}, "use_cmmn_layer": False})

        models.append(model)

    return tuple(models)
