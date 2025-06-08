import copy
import enum
import os
import traceback
from typing import Any, Dict, Optional, Type

import optuna
import torch
import yaml  # type: ignore[import-untyped]
from torch import optim
from torch.utils.data import DataLoader

from elecssl.data.combined_datasets import CombinedDatasets
from elecssl.data.data_generators.data_generator import InterpolationDataGenerator, RBPDataGenerator, create_mask, \
    MultiTaskRBPdataGenerator, RBPDataGenBase, InterpolationDataGenBase, MultiTaskInterpolationDataGenerator
from elecssl.data.datasets.getter import get_channel_system
from elecssl.data.scalers.target_scalers import get_target_scaler
from elecssl.data.subject_split import get_data_split, DataSplitBase
from elecssl.models.losses import CustomWeightedLoss, get_activation_function
from elecssl.models.main_models.main_base_class import MainModuleBase
from elecssl.models.main_models.main_fixed_channels_model import MultiTaskFixedChannelsModel, \
    DownstreamFixedChannelsModel, MainFixedChannelsModelBase
from elecssl.models.main_models.main_rbp_model import DownstreamRBPModel, MainRBPModelBase, MultiTaskRBPModel
from elecssl.models.metrics import Histories, NaNValueError
from elecssl.models.mtl_strategies.multi_task_strategies import get_mtl_strategy
from elecssl.models.utils import tensor_dict_to_device


# --------------
# Small convenient classes
# --------------
class SpatialMethod(enum.Enum):
    """Enum class for indicating the method for handling varied electrode configurations"""
    RBP = "RegionBasedPooling"
    INTERPOLATION = "Interpolation"


class TrainMethod(enum.Enum):
    """Enum class for indicating the training method"""
    DOWNSTREAM = "downstream_training"
    DD = "discriminator_training"  # This is no longer maintained
    MTL = "multi_task"


# --------------
# Main class
# --------------
class SingleExperiment:
    """
    Class for running a single experiment. Note that this is a context manager, so to actually run an experiment, use a
    'with' statement and call .run_experiment()
    """

    __slots__ = ("_hp_config", "_experiments_config", "_pre_processing_config", "_results_path", "_device",
                 "_fine_tuning", "_experiment_name")

    def __init__(self, hp_config, experiments_config, pre_processing_config, results_path, fine_tuning,
                 experiment_name):
        """
        Initialise

        Parameters
        ----------
        hp_config : dict[str, Any]
        experiments_config : dict[str, Any]
        pre_processing_config : dict[str, Any]
        results_path : pathlib.Path
        fine_tuning : str | None
            This is the name of the model to fine-tune. If it is a string, then a pytorch model with this name is
            expected to exist in the 'Fold_{i}' folder for every fold that is made. If it is None, it indicates that
            this experiment is not a fine-tuning experiment
        experiment_name: str | None
            This name will be used for saving e.g. models, performance scores and predictions. Its main purpose was to
            distinguish pretraining from downstream training, and its default to None is recommended otherwise
        """
        # Create path
        if fine_tuning is None:
            os.mkdir(results_path)

        # Save config files
        prefix_name = "" if experiment_name is None else f"{experiment_name}_"
        hpc_config_path = results_path / f"{prefix_name}hpc_config.yml"
        experiments_config_path = results_path / f"{prefix_name}experiments_config.yml"
        with open(hpc_config_path, "w") as file:
            yaml.safe_dump(hp_config, file, sort_keys=False)
        with open(experiments_config_path, "w") as file:
            yaml.safe_dump(experiments_config, file, sort_keys=False)

        # Make them read-only
        os.chmod(hpc_config_path, 0o444)
        os.chmod(experiments_config_path, 0o444)

        # Store attributes
        self._hp_config = hp_config
        self._experiments_config = experiments_config
        self._pre_processing_config = pre_processing_config
        self._results_path = results_path
        self._device = torch.device(experiments_config["Device"])
        self._fine_tuning = fine_tuning
        self._experiment_name = experiment_name

    # -------------
    # Dunder methods for context manager (using the 'with' statement). See this video from mCoding for more information
    # on context managers https://www.youtube.com/watch?v=LBJlGwJ899Y&t=640s
    # -------------
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """This will execute when exiting the with statement. It will NOT execute if the run was killed by the operating
        system, which can happen if too much data is loaded into memory"""
        # If everything was as it should, just exit
        if exc_val is None:
            return None

        # Otherwise, document the error received in a text file
        prefix_name = "" if self._experiment_name is None else f"{self._experiment_name}_"
        with open((self._results_path / f"{prefix_name}{exc_type.__name__}").with_suffix(".txt"), "w") as file:
            file.write("Traceback (most recent call last):\n")
            traceback.print_tb(exc_tb, file=file)  # type: ignore
            file.write(f"{exc_type.__name__}: {exc_val}")

    # -------------
    # Methods for making and preparing model
    # -------------
    def _make_interpolation_model(self):
        """Method for defining a model which expects the same number of channels"""
        # Maybe add number of time steps
        mts_config = copy.deepcopy(self.dl_architecture_config)
        if "num_time_steps" in mts_config["kwargs"] and mts_config["kwargs"]["num_time_steps"] is None:
            mts_config["kwargs"]["num_time_steps"] = self.shared_pre_processing_config["num_time_steps"]

        # Add number of input channels
        mts_config["kwargs"]["in_channels"] = get_channel_system(
            dataset_name=self.interpolation_config["main_channel_system"]  # type: ignore[index]
        ).num_channels

        # Define model
        if self.train_method == TrainMethod.DOWNSTREAM:
            model = DownstreamFixedChannelsModel.from_config(
                mts_config=mts_config, cmmn_config=self.cmmn_config).to(self._device)
        elif self.train_method == TrainMethod.MTL:
            model = MultiTaskFixedChannelsModel.from_config(
                mts_config=mts_config, cmmn_config=self.cmmn_config).to(self._device)
        else:
            raise ValueError(f"The training method {self.train_method} is not supported")

        return model

    def _make_rbp_model(self):
        """Method for defining a model with RBP as the first layer"""
        # Maybe add number of time steps
        mts_config = copy.deepcopy(self.dl_architecture_config)
        if "num_time_steps" in mts_config["kwargs"] and mts_config["kwargs"]["num_time_steps"] is None:
            mts_config["kwargs"]["num_time_steps"] = self.shared_pre_processing_config["num_time_steps"]

        # Define model
        if self.train_method == TrainMethod.DOWNSTREAM:
            model = DownstreamRBPModel.from_config(
                rbp_config=self.rbp_config,
                mts_config=mts_config,
            ).to(self._device)
        elif self.train_method == TrainMethod.MTL:
            model = MultiTaskRBPModel.from_config(
                rbp_config=self.rbp_config,
                mts_config=mts_config,
            ).to(self._device)
        else:
            raise ValueError(f"The training method {self.train_method} is not supported")

        return model

    def _make_model(self):
        if self.spatial_method == SpatialMethod.RBP:
            return self._make_rbp_model()
        elif self.spatial_method == SpatialMethod.INTERPOLATION:
            return self._make_interpolation_model()
        raise ValueError(f"Unexpected method for handling a varied number of channels: {self.spatial_method}")

    def _load_pretrained_model(self, path):
        # Pretraining is only supported for downstream training models frameworks (ironically enough)
        if self.spatial_method == SpatialMethod.RBP:
            return DownstreamRBPModel.load_model(name=f"{self._fine_tuning}_model", path=path).to(self._device)
        elif self.spatial_method == SpatialMethod.INTERPOLATION:
            return DownstreamFixedChannelsModel.load_model(name=f"{self._fine_tuning}_model",
                                                           path=path).to(self._device)
        raise ValueError(f"Unexpected method for handling a varied number of channels: {self.spatial_method}")

    def _prepare_rbp_model(self, *, model, channel_systems, combined_dataset, train_subjects, test_subjects):
        # Fit channel systems
        self._fit_channel_systems(model=model, channel_systems=channel_systems)

        # (Maybe) fit the CMMN layers of RBP
        if model.any_rbp_cmmn_layers:
            self._fit_cmmn_layers(model=model, train_data=combined_dataset.get_data(train_subjects),
                                  channel_systems=channel_systems)

            # Fit the test data as well
            self._fit_cmmn_layers_test_data(model=model, test_data=combined_dataset.get_data(test_subjects),
                                            channel_systems=channel_systems)

        return model

    def _prepare_interpolation_model(self, *, model, combined_dataset, train_subjects, test_subjects):
        # Maybe fit CMMN layers
        if model.has_cmmn_layer:
            self._fit_cmmn_layers(model=model, train_data=combined_dataset.get_data(train_subjects))

            # Fit the test data as well (just the monge filters)
            self._fit_cmmn_layers_test_data(model=model, test_data=combined_dataset.get_data(test_subjects))

        return model

    def _prepare_model(self, *, model, channel_systems, combined_dataset, train_subjects, test_subjects):
        if isinstance(model, MainRBPModelBase):
            return self._prepare_rbp_model(model=model, channel_systems=channel_systems,
                                           combined_dataset=combined_dataset, train_subjects=train_subjects,
                                           test_subjects=test_subjects)
        elif isinstance(model, MainFixedChannelsModelBase):
            return self._prepare_interpolation_model(model=model, combined_dataset=combined_dataset,
                                                     train_subjects=train_subjects, test_subjects=test_subjects)
        raise TypeError(f"Did not recognise the model type: {type(model)}")

    @staticmethod
    def _fit_channel_systems(model, channel_systems):
        model.fit_channel_systems(tuple(channel_systems.values()))

    def _fit_cmmn_layers(self, *, model, train_data, channel_systems=None):
        if self.spatial_method == SpatialMethod.RBP:
            model.fit_psd_barycenters(data=train_data, channel_systems=channel_systems,
                                      sampling_freq=self.shared_pre_processing_config["resample"])
            model.fit_monge_filters(data=train_data, channel_systems=channel_systems)
        elif self.spatial_method == SpatialMethod.INTERPOLATION:
            model.fit_psd_barycenters(data=train_data, sampling_freq=self.shared_pre_processing_config["resample"])
            model.fit_monge_filters(data=train_data)
        else:
            raise ValueError(f"Unexpected method for handling a varied number of channels: {self.spatial_method}")

    def _fit_cmmn_layers_test_data(self, *, model, test_data, channel_systems=None):
        for name, eeg_data in test_data.items():
            if name not in model.cmmn_fitted_channel_systems:
                # As long as the channel systems for the test data are present in 'channel_systems', this works
                # fine. Redundant channel systems is not a problem
                if self.spatial_method == SpatialMethod.RBP:
                    model.fit_monge_filters(data={name: eeg_data}, channel_systems=channel_systems)
                elif self.spatial_method == SpatialMethod.INTERPOLATION:
                    model.fit_monge_filters(data={name: eeg_data})
                else:
                    raise ValueError(f"Unexpected method for handling a varied number of channels: "
                                     f"{self.spatial_method}")

    # -------------
    # Methods for creating Pytorch data loaders
    # -------------
    def _load_train_val_data_loaders(self, *, model=None, train_subjects, val_subjects, combined_dataset):
        if self.spatial_method == SpatialMethod.RBP:
            return self._load_rbp_train_val_data_loaders(
                model=model, train_subjects=train_subjects, val_subjects=val_subjects,
                combined_dataset=combined_dataset
            )
        elif self.spatial_method == SpatialMethod.INTERPOLATION:
            return self._load_interpolation_train_val_data_loaders(
                train_subjects=train_subjects, val_subjects=val_subjects, combined_dataset=combined_dataset
            )
        else:
            raise ValueError

    def _load_test_data_loader(self, *, model=None, test_subjects, combined_dataset, target_scaler,
                               pretext_target_scaler):
        if self.spatial_method == SpatialMethod.RBP:
            return self._load_rbp_test_data_loader(
                model=model, test_subjects=test_subjects, combined_dataset=combined_dataset,
                target_scaler=target_scaler, pretext_target_scaler=pretext_target_scaler
            )
        elif self.spatial_method == SpatialMethod.INTERPOLATION:
            return self._load_interpolation_test_data_loader(
                test_subjects=test_subjects, combined_dataset=combined_dataset, target_scaler=target_scaler,
                pretext_target_scaler=pretext_target_scaler
            )
        else:
            raise ValueError

    def _load_interpolation_train_val_data_loaders(self, *, train_subjects, val_subjects, combined_dataset):
        # Extract input data
        train_data = combined_dataset.get_data(subjects=train_subjects)
        val_data = combined_dataset.get_data(subjects=val_subjects)

        # Extract scaled target data and the scaler itself
        (train_targets, val_targets), target_scaler = self._get_targets_and_scalers(
            train_subjects=train_subjects, val_subjects=val_subjects, combined_dataset=combined_dataset,
            is_mtl_pretext=False
        )
        train_gen: InterpolationDataGenBase
        val_gen: InterpolationDataGenBase
        if self.train_method == TrainMethod.DOWNSTREAM:
            # Create data generators
            train_gen = InterpolationDataGenerator(
                data=train_data, targets=train_targets, subjects=combined_dataset.get_subjects_dict(train_subjects),
                subjects_info=combined_dataset.get_subjects_info(train_subjects),
                expected_variables=combined_dataset.get_expected_variables(train_subjects)
            )
            val_gen = InterpolationDataGenerator(
                data=val_data, targets=val_targets, subjects=combined_dataset.get_subjects_dict(val_subjects),
                subjects_info=combined_dataset.get_subjects_info(val_subjects),
                expected_variables=combined_dataset.get_expected_variables(val_subjects)
            )
            scalers = {"target_scaler": target_scaler}
        elif self.train_method == TrainMethod.MTL:
            (pretext_train_targets, pretext_val_targets), pretext_target_scaler = self._get_targets_and_scalers(
                train_subjects=train_subjects, val_subjects=val_subjects, combined_dataset=combined_dataset,
                is_mtl_pretext=True
            )

            # Create masks
            train_pretext_mask = create_mask(
                sample_sizes={name: data.shape[0] for name, data in train_data.items()},
                to_include=self.mtl_config["pretext_datasets"])
            val_pretext_mask = create_mask(
                sample_sizes={name: data.shape[0] for name, data in val_data.items()},
                to_include=self.mtl_config["pretext_datasets"])

            train_downstream_mask = create_mask(
                sample_sizes={name: data.shape[0] for name, data in train_data.items()},
                to_include=self.mtl_config["downstream_datasets"])
            val_downstream_mask = create_mask(
                sample_sizes={name: data.shape[0] for name, data in val_data.items()},
                to_include=self.mtl_config["downstream_datasets"])

            # Create data generators
            train_gen = MultiTaskInterpolationDataGenerator(
                data=train_data, downstream_targets=train_targets, pretext_targets=pretext_train_targets,
                pretext_mask=train_pretext_mask, downstream_mask=train_downstream_mask,
                subjects=combined_dataset.get_subjects_dict(train_subjects),
                subjects_info=combined_dataset.get_subjects_info(train_subjects),
                expected_variables=combined_dataset.get_expected_variables(train_subjects)
            )
            val_gen = MultiTaskInterpolationDataGenerator(
                data=val_data, downstream_targets=val_targets, pretext_targets=pretext_val_targets,
                pretext_mask=val_pretext_mask, downstream_mask=val_downstream_mask,
                subjects=combined_dataset.get_subjects_dict(val_subjects),
                subjects_info=combined_dataset.get_subjects_info(val_subjects),
                expected_variables=combined_dataset.get_expected_variables(val_subjects)
            )

            scalers = {"target_scaler": target_scaler, "pretext_target_scaler": pretext_target_scaler}
        else:
            raise RuntimeError(f"Train method {self.train_method} is not supported")

        # Create data loaders
        train_loader = DataLoader(dataset=train_gen, batch_size=self.train_config["batch_size"], shuffle=True)
        val_loader = DataLoader(dataset=val_gen, batch_size=self.train_config["batch_size"], shuffle=True)

        return train_loader, val_loader, scalers

    def _load_interpolation_test_data_loader(self, *, test_subjects, combined_dataset, target_scaler,
                                             pretext_target_scaler):
        # Extract input data
        test_data = combined_dataset.get_data(subjects=tuple(set(test_subjects)))
        # todo: Maybe the subjects are duplicated if they are test subjects in both splits?

        # Extract scaled targets
        test_targets = combined_dataset.get_targets(subjects=test_subjects)
        test_targets = target_scaler.transform(test_targets)
        test_gen: InterpolationDataGenBase
        if self.train_method == TrainMethod.DOWNSTREAM:
            # Create data generators
            test_gen = InterpolationDataGenerator(
                data=test_data, targets=test_targets, subjects=combined_dataset.get_subjects_dict(test_subjects),
                subjects_info=combined_dataset.get_subjects_info(test_subjects),
                expected_variables=combined_dataset.get_expected_variables(test_subjects)
            )
        elif self.train_method == TrainMethod.MTL:
            pretext_test_targets = combined_dataset.get_targets(
                subjects=test_subjects, target=self.mtl_config["pretext_target"])
            pretext_test_targets = pretext_target_scaler.transform(pretext_test_targets)

            # Create masks  # todo: should this be necessary?
            pretext_mask = create_mask(
                sample_sizes={name: data.shape[0] for name, data in test_data.items()},
                to_include=self.mtl_config["pretext_datasets"])
            downstream_mask = create_mask(
                sample_sizes={name: data.shape[0] for name, data in test_data.items()},
                to_include=self.mtl_config["downstream_datasets"])

            # Create data generator
            test_gen = MultiTaskInterpolationDataGenerator(
                data=test_data, downstream_targets=test_targets, pretext_targets=pretext_test_targets,
                pretext_mask=pretext_mask, downstream_mask=downstream_mask,
                subjects=combined_dataset.get_subjects_dict(test_subjects),
                subjects_info=combined_dataset.get_subjects_info(test_subjects),
                expected_variables=combined_dataset.get_expected_variables(test_subjects)
            )
        else:
            raise RuntimeError(f"Train method {self.train_method} is not supported")

        # Create data loader
        test_loader = DataLoader(dataset=test_gen, batch_size=self.train_config["batch_size"], shuffle=True)

        return test_loader

    def _load_rbp_train_val_data_loaders(self, *, model, train_subjects, val_subjects, combined_dataset):
        # Extract input data
        train_data = combined_dataset.get_data(subjects=train_subjects)
        val_data = combined_dataset.get_data(subjects=val_subjects)

        # Compute the pre-computed features
        if model.supports_precomputing:
            train_pre_computed, val_pre_computed = self._get_pre_computed_features(model=model,
                                                                                   train_data=train_data,
                                                                                   val_data=val_data)
        else:
            train_pre_computed, val_pre_computed = None, None

        # Extract scaled target data and the scaler itself
        (train_targets, val_targets), target_scaler = self._get_targets_and_scalers(
            train_subjects=train_subjects, val_subjects=val_subjects, combined_dataset=combined_dataset,
            is_mtl_pretext=False
        )
        train_gen: RBPDataGenBase
        val_gen: RBPDataGenBase
        if self.train_method == TrainMethod.DOWNSTREAM:
            # Create data generators
            train_gen = RBPDataGenerator(
                data=train_data, targets=train_targets, pre_computed=train_pre_computed,
                subjects=combined_dataset.get_subjects_dict(train_subjects),
                subjects_info=combined_dataset.get_subjects_info(train_subjects),
                expected_variables=combined_dataset.get_expected_variables(train_subjects)
            )
            val_gen = RBPDataGenerator(
                data=val_data, targets=val_targets, pre_computed=val_pre_computed,
                subjects=combined_dataset.get_subjects_dict(val_subjects),
                subjects_info=combined_dataset.get_subjects_info(val_subjects),
                expected_variables=combined_dataset.get_expected_variables(val_subjects)
            )
            scalers = {"target_scaler": target_scaler}
        elif self.train_method == TrainMethod.MTL:
            (pretext_train_targets, pretext_val_targets), pretext_target_scaler = self._get_targets_and_scalers(
                train_subjects=train_subjects, val_subjects=val_subjects, combined_dataset=combined_dataset,
                is_mtl_pretext=True
            )

            # Create masks
            train_pretext_mask = create_mask(
                sample_sizes={name: data.shape[0] for name, data in train_data.items()},
                to_include=self.mtl_config["pretext_datasets"])
            val_pretext_mask = create_mask(
                sample_sizes={name: data.shape[0] for name, data in val_data.items()},
                to_include=self.mtl_config["pretext_datasets"])

            train_downstream_mask = create_mask(
                sample_sizes={name: data.shape[0] for name, data in train_data.items()},
                to_include=self.mtl_config["downstream_datasets"])
            val_downstream_mask = create_mask(
                sample_sizes={name: data.shape[0] for name, data in val_data.items()},
                to_include=self.mtl_config["downstream_datasets"])

            # Create data generators
            train_gen = MultiTaskRBPdataGenerator(
                data=train_data, downstream_targets=train_targets, pretext_targets=pretext_train_targets,
                pretext_mask=train_pretext_mask, downstream_mask=train_downstream_mask, pre_computed=train_pre_computed,
                subjects=combined_dataset.get_subjects_dict(train_subjects),
                subjects_info=combined_dataset.get_subjects_info(train_subjects),
                expected_variables=combined_dataset.get_expected_variables(train_subjects)
            )
            val_gen = MultiTaskRBPdataGenerator(
                data=val_data, downstream_targets=val_targets, pretext_targets=pretext_val_targets,
                pretext_mask=val_pretext_mask, downstream_mask=val_downstream_mask, pre_computed=val_pre_computed,
                subjects=combined_dataset.get_subjects_dict(val_subjects),
                subjects_info=combined_dataset.get_subjects_info(val_subjects),
                expected_variables=combined_dataset.get_expected_variables(val_subjects)
            )

            scalers = {"target_scaler": target_scaler, "pretext_target_scaler": pretext_target_scaler}
        else:
            raise RuntimeError(f"Train method {self.train_method} is not supported")

        # Create data loaders
        train_loader = DataLoader(dataset=train_gen, batch_size=self.train_config["batch_size"], shuffle=True)
        val_loader = DataLoader(dataset=val_gen, batch_size=self.train_config["batch_size"], shuffle=True)

        return train_loader, val_loader, scalers

    def _load_rbp_test_data_loader(self, *, model, test_subjects, combined_dataset, target_scaler,
                                   pretext_target_scaler):
        # Extract input data
        test_data = combined_dataset.get_data(subjects=tuple(set(test_subjects)))
        # todo: Maybe the subjects are duplicated if they are test subjects in both splits?

        # Compute the pre-computed features
        if model.supports_precomputing:
            test_pre_computed = model.pre_compute(
                input_tensors={dataset_name: torch.tensor(data, dtype=torch.float).to(self._device)
                               for dataset_name, data in test_data.items()})
            test_pre_computed = tuple(tensor_dict_to_device(pre_comp, device=torch.device("cpu"))
                                      for pre_comp in test_pre_computed)
        else:
            test_pre_computed = None

        # Extract scaled targets
        test_targets = combined_dataset.get_targets(subjects=test_subjects)
        test_targets = target_scaler.transform(test_targets)
        test_gen: RBPDataGenBase
        if self.train_method == TrainMethod.DOWNSTREAM:
            # Create data generator
            test_gen = RBPDataGenerator(
                data=test_data, targets=test_targets, pre_computed=test_pre_computed,
                subjects=combined_dataset.get_subjects_dict(test_subjects),
                subjects_info=combined_dataset.get_subjects_info(test_subjects),
                expected_variables=combined_dataset.get_expected_variables(test_subjects)
            )
        elif self.train_method == TrainMethod.MTL:
            pretext_test_targets = combined_dataset.get_targets(
                subjects=test_subjects, target=self.mtl_config["pretext_target"])
            pretext_test_targets = pretext_target_scaler.transform(pretext_test_targets)

            # Create masks  # todo: should this be necessary?
            pretext_mask = create_mask(
                sample_sizes={name: data.shape[0] for name, data in test_data.items()},
                to_include=self.mtl_config["pretext_datasets"])
            downstream_mask = create_mask(
                sample_sizes={name: data.shape[0] for name, data in test_data.items()},
                to_include=self.mtl_config["downstream_datasets"])

            # Create data generator
            test_gen = MultiTaskRBPdataGenerator(
                data=test_data, downstream_targets=test_targets, pretext_targets=pretext_test_targets,
                pretext_mask=pretext_mask, downstream_mask=downstream_mask, pre_computed=test_pre_computed,
                subjects=combined_dataset.get_subjects_dict(test_subjects),
                subjects_info=combined_dataset.get_subjects_info(test_subjects),
                expected_variables=combined_dataset.get_expected_variables(test_subjects)
            )
        else:
            raise RuntimeError(f"Train method {self.train_method} is not supported")

        # Create data loader
        test_loader = DataLoader(dataset=test_gen, batch_size=self.train_config["batch_size"], shuffle=True)

        return test_loader

    def _create_loaders(self, *, model, combined_dataset, train_subjects, val_subjects, test_subjects):
        train_loader, val_loader, scalers = self._load_train_val_data_loaders(
            model=model, train_subjects=train_subjects, val_subjects=val_subjects, combined_dataset=combined_dataset
        )

        # Maybe create loaders for test data
        test_loader: Optional[DataLoader[Any]]
        if self.train_config["continuous_testing"]:
            test_loader = self._load_test_data_loader(
                model=model, test_subjects=test_subjects, combined_dataset=combined_dataset,
                target_scaler=scalers["target_scaler"], pretext_target_scaler=scalers.get("pretext_target_scaler")
            )
        else:
            test_loader = None

        # Some type checks
        _allowed_dataset_types = (RBPDataGenBase, InterpolationDataGenBase)
        if not isinstance(train_loader.dataset, _allowed_dataset_types):
            raise TypeError(f"Expected training Pytorch datasets to inherit from "
                            f"{tuple(data_gen.__name__ for data_gen in _allowed_dataset_types)}, but found "
                            f"{type(train_loader.dataset)}")
        if not isinstance(val_loader.dataset, _allowed_dataset_types):
            raise TypeError(f"Expected validation Pytorch datasets to inherit from "
                            f"{tuple(data_gen.__name__ for data_gen in _allowed_dataset_types)}, but found "
                            f"{type(val_loader.dataset)}")

        return (train_loader, val_loader, test_loader), scalers

    def _get_pre_computed_features(self, *, model, train_data, val_data):
        # Perform pre-computing of features
        train_pre_computed = model.pre_compute(
            input_tensors={dataset_name: torch.tensor(data, dtype=torch.float).to(self._device)
                           for dataset_name, data in train_data.items()})
        val_pre_computed = model.pre_compute(
            input_tensors={dataset_name: torch.tensor(data, dtype=torch.float).to(self._device)
                           for dataset_name, data in val_data.items()})

        # Send pre-computed features to cpu
        train_pre_computed = tuple(tensor_dict_to_device(pre_comp, device=torch.device("cpu"))
                                   for pre_comp in train_pre_computed)
        val_pre_computed = tuple(tensor_dict_to_device(pre_comp, device=torch.device("cpu"))
                                 for pre_comp in val_pre_computed)

        return train_pre_computed, val_pre_computed

    def _get_targets_and_scalers(self, *, train_subjects, val_subjects, combined_dataset, is_mtl_pretext: bool):
        target = self.mtl_config["pretext_target"] if is_mtl_pretext else None

        # Extract target data
        train_targets = combined_dataset.get_targets(subjects=train_subjects, target=target)
        val_targets = combined_dataset.get_targets(subjects=val_subjects, target=target)

        scaler_name = self.mtl_config["Scaler"]["target"]["name"] \
            if is_mtl_pretext else self.scaler_config["target"]["name"]
        scaler_kwargs = self.mtl_config["Scaler"]["target"]["kwargs"] if is_mtl_pretext \
            else self.scaler_config["target"]["kwargs"]

        # Get, fit, and scale
        target_scaler = get_target_scaler(scaler_name, **scaler_kwargs)
        target_scaler.fit(train_targets)

        train_targets = target_scaler.transform(train_targets)
        val_targets = target_scaler.transform(val_targets)

        return (train_targets, val_targets), target_scaler

    # -------------
    # Methods for creating optimisers and loss
    # -------------
    def _get_domain_discriminator_details(self, dataset_sizes):
        # Initialise domain discriminator kwargs
        dd_kwargs = copy.deepcopy(self.domain_discriminator_config)

        # Maybe add sample weighting
        if dd_kwargs["training"]["Loss"]["weighter"] is not None:
            dd_kwargs["training"]["Loss"]["weighter_kwargs"]["dataset_sizes"] = dataset_sizes

        # Set criterion, weight of domain discriminator loss, and the metrics to be used
        discriminator_criterion = CustomWeightedLoss(**dd_kwargs["training"]["Loss"])
        discriminator_weight = dd_kwargs["training"]["lambda"]
        discriminator_metrics = dd_kwargs["training"]["metrics"]

        return discriminator_criterion, discriminator_weight, discriminator_metrics

    # --------------
    # Loss, optimisers, and MTL strategies
    # --------------
    def _create_criteria(self, train_loader):
        # Create loss
        if self.train_method == TrainMethod.DOWNSTREAM:
            if self.loss_config["weighter"] is not None:
                self.loss_config["weighter_kwargs"]["dataset_sizes"] = train_loader.dataset.dataset_sizes
            criterion = CustomWeightedLoss(**self.loss_config)
            criteria = {"criterion": criterion}

        elif self.train_method == TrainMethod.MTL:
            # For MTL, I need two criteria
            assert isinstance(train_loader.dataset, (MultiTaskRBPdataGenerator, MultiTaskInterpolationDataGenerator)), \
                f"Must use and MTL compatible data generator, but received {type(train_loader.dataset)}"

            _pretext_loss_config = copy.deepcopy(self.loss_config["pretext"])
            if _pretext_loss_config["weighter"] is not None:
                _pretext_loss_config["weighter_kwargs"]["dataset_sizes"] = train_loader.dataset.pretext_dataset_size
            pretext_criterion = CustomWeightedLoss(**_pretext_loss_config)

            _downstream_loss_config = copy.deepcopy(self.loss_config["downstream"])
            if _downstream_loss_config["weighter"] is not None:
                _downstream_loss_config["weighter_kwargs"]["dataset_sizes"] = (
                    train_loader.dataset.downstream_dataset_size)
            downstream_criterion = CustomWeightedLoss(**_downstream_loss_config)

            criteria = {"pretext_criterion": pretext_criterion, "downstream_criterion": downstream_criterion}
        else:
            raise ValueError(f"Train method {self.train_method} not supported")

        return criteria

    def _create_optimiser(self, model):
        """Create optimiser / learning strategy"""
        optimiser = optim.Adam(model.parameters(), lr=self.train_config["learning_rate"],
                               betas=(self.train_config["beta_1"], self.train_config["beta_2"]),
                               eps=self.train_config["eps"])

        if self.train_method != TrainMethod.MTL:
            return {"optimiser": optimiser}

        # Create the multi-task learning strategy
        strategy = get_mtl_strategy(name=self.mtl_config["Strategy"]["name"], optimiser=optimiser, model=model,
                                    **self.mtl_config["Strategy"]["kwargs"])

        return {"mtl_strategy": strategy}

    # -------------
    # Methods for saving results
    # -------------
    def _save_results(self, *, histories: Dict[str, Histories], results_path):
        decimals = self.saving_config["performance_score_decimals"]

        prefix_name = "" if self._experiment_name is None else f"{self._experiment_name}_"

        # Save prediction histories
        for name, history in histories.items():
            history.save_main_history(history_name=f"{prefix_name}{name}_history", path=results_path, decimals=decimals)

        # Save subgroup plots
        sub_group_path = os.path.join(results_path, f"{prefix_name}sub_groups_plots")
        os.mkdir(sub_group_path)

        for name, history in histories.items():
            history.save_subgroup_metrics(history_name=name, path=sub_group_path, decimals=decimals,
                                          save_plots=self.saving_config["save_subgroups_plots"])

    # -------------
    # Methods for getting input arguments suited for the specific train_method
    # -------------
    def _get_metrics_train_method_kwargs(self):
        if self.train_method == TrainMethod.DOWNSTREAM:
            metric_kwargs = {"metrics": self.train_config["metrics"], "main_metric": self.train_config["main_metric"]}
        elif self.train_method == TrainMethod.MTL:
            metric_kwargs = {
                "downstream_metrics": self.mtl_config["Metrics"]["downstream_metrics"],
                "pretext_metrics": self.mtl_config["Metrics"]["pretext_metrics"],
                "downstream_selection_metric": self.mtl_config["Metrics"]["downstream_selection_metric"],
                "pretext_selection_metric": self.mtl_config["Metrics"]["pretext_selection_metric"]
            }
        else:
            raise ValueError(f"Train method {self.train_method} not supported")
        return metric_kwargs

    def _get_activation_functions_train_method_kwargs(self):
        if self.train_method == TrainMethod.DOWNSTREAM:
            _activation_func = get_activation_function(self.train_config["prediction_activation_function"])
            activation_fun_kwargs = {"prediction_activation_function": _activation_func}
        elif self.train_method == TrainMethod.MTL:
            _act_func_config = self.mtl_config["ActivationFunctions"]
            _downstream_act_func = get_activation_function(_act_func_config["downstream"])
            _pretext_act_func = get_activation_function(_act_func_config["pretext"])
            activation_fun_kwargs = {"downstream_prediction_activation_function": _downstream_act_func,
                                     "pretext_prediction_activation_function": _pretext_act_func}
        else:
            raise ValueError(f"Train method {self.train_method} not supported")
        return activation_fun_kwargs

    # -------------
    # Method for running experiments
    # -------------
    def _run_single_fold(self, *, train_subjects, val_subjects, test_subjects, channel_systems, channel_name_to_index,
                         combined_dataset: CombinedDatasets, results_path):
        # -----------------
        # Define or load model
        # -----------------
        model: MainModuleBase
        if self._fine_tuning is not None:
            model = self._load_pretrained_model(results_path)
        else:
            model = self._make_model()
        model = self._prepare_model(model=model, channel_systems=channel_systems, combined_dataset=combined_dataset,
                                    train_subjects=train_subjects, test_subjects=test_subjects)

        # Save some metadata
        prefix_name = "" if self._experiment_name is None else f"{self._experiment_name}_"
        model.save_metadata(name=f"{prefix_name}metadata_before", path=results_path)

        # -----------------
        # Create data loaders (and target scaler)
        # -----------------
        print("Creating data loaders...")
        (train_loader, val_loader, test_loader), target_scalers = self._create_loaders(
            model=model, combined_dataset=combined_dataset, train_subjects=train_subjects, val_subjects=val_subjects,
            test_subjects=test_subjects
        )

        # -----------------
        # Create loss and optimiser
        # -----------------
        criteria = self._create_criteria(train_loader=train_loader)
        optimiser = self._create_optimiser(model=model)  # This is a dict which is either optimiser or strategy

        # -----------------
        # Prepare input arguments
        # -----------------
        metric_kwargs = self._get_metrics_train_method_kwargs()
        activation_fun_kwargs = self._get_activation_functions_train_method_kwargs()
        channel_name_to_index_kwarg = {"channel_name_to_index": channel_name_to_index} \
            if self.spatial_method == SpatialMethod.RBP else dict()

        # -----------------
        # Train model
        # -----------------
        print(f"{' Training ':-^20}")
        try:
            # method=self.train_config["method"]
            histories, model_states, best_epochs = model.train_model(
                train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, **metric_kwargs, **criteria,
                **optimiser,  num_epochs=self.train_config["num_epochs"], verbose=self.train_config["verbose"],
                device=self._device, **target_scalers, **channel_name_to_index_kwarg, **activation_fun_kwargs,
                sub_group_splits=self.sub_groups_config["sub_groups"],
                sub_groups_verbose=self.sub_groups_config["verbose"],
                verbose_variables=self.train_config["verbose_variables"], variable_metrics=self.variables_metrics,
                patience=self._experiments_config["EarlyStopping"]["patience"]
            )
        except NaNValueError as e:
            if self._experiments_config.get("raise_upon_nan_predictions") is None:
                raise e
            else:
                selected_error = _get_error(self._experiments_config["raise_upon_nan_predictions"])
                raise selected_error("Error raised due to NaN values, most likely in the predictions")

        # Save some metadata
        for best_model_state, epoch in zip(model_states, best_epochs):
            model.load_state_dict({k: v.to(self._device)
                                   for k, v in best_model_state.items()})  # type: ignore[arg-type]
            model.save_metadata(name=f"{prefix_name}metadata_epoch_{epoch}", path=results_path)

        # -----------------
        # Test model (but only if continuous testing was not used)
        # -----------------
        if not self.train_config["continuous_testing"]:
            print(f"\n{' Testing ':-^20}")
            if "test" in histories:
                raise RuntimeError("Expected 'test' history not to be present with continuous test set to 'False', "
                                   "but that was not the case")

            for best_model_state, epoch in zip(model_states, best_epochs):
                model.load_state_dict({k: v.to(self._device) for k, v in best_model_state.items()})

                # Get test loader
                test_loader = self._load_test_data_loader(
                    model=model, test_subjects=test_subjects, combined_dataset=combined_dataset,
                    target_scaler=target_scalers["target_scaler"],
                    pretext_target_scaler=target_scalers.get("pretext_target_scaler"))

                # Test model on test data
                metric_kwargs.pop("downstream_selection_metric", None)
                metric_kwargs.pop("pretext_selection_metric", None)
                test_histories = model.test_model(
                    data_loader=test_loader, **metric_kwargs, verbose=self.train_config["verbose"],
                    **channel_name_to_index_kwarg, device=self._device, **target_scalers,
                    sub_group_splits=self.sub_groups_config["sub_groups"], **activation_fun_kwargs,
                    sub_groups_verbose=self.sub_groups_config["verbose"],
                    verbose_variables=self.train_config["verbose_variables"], variable_metrics=self.variables_metrics
                )
                if isinstance(test_histories, tuple):  # TODO: fix
                    histories[f"test_epoch_{epoch}_downstream"] = test_histories[1]
                    histories[f"test_epoch_{epoch}_pretext"] = test_histories[0]
                else:
                    histories[f"test_epoch_{epoch}"] = test_histories

        # -----------------
        # Save results
        # -----------------
        # Performance scores
        self._save_results(histories=histories, results_path=results_path)

        # (Maybe) the models itself
        if self.saving_config["save_model"]:
            for best_model_state, epoch in zip(model_states, best_epochs):
                model.load_state_dict({k: v.to(self._device) for k, v in best_model_state.items()})
                model = model.to(device=torch.device("cpu"))
                prefix_name = "" if self._experiment_name is None else f"{self._experiment_name}_"
                model.save_model(name=f"{prefix_name}model_epoch_{epoch}", path=results_path)

    def _run(self, *, splits, channel_systems, channel_name_to_index, combined_dataset):
        # Loop through all splits (e.g, folds if k-fold cross validation)
        for i, (train_subjects, val_subjects, test_subjects) in enumerate(splits):
            print(f"\nSplit {i + 1}/{len(splits)}")

            # -----------------
            # Make folder for the current fold
            # -----------------
            fold_path = self._results_path / f"split_{i}"
            if self._fine_tuning is None:
                os.mkdir(fold_path)

            # -----------------
            # Run the current split/fold
            # -----------------
            self._run_single_fold(
                train_subjects=train_subjects, val_subjects=val_subjects, test_subjects=test_subjects,
                results_path=fold_path, channel_systems=channel_systems, channel_name_to_index=channel_name_to_index,
                combined_dataset=combined_dataset
            )

    # -------------
    # Main method for running the cross validation experiment
    # -------------
    def run_experiment(self, subject_split, combined_datasets):
        """
        Method for running a single experiment. This method does not use HPO, but can be used as part of HPO by serving
        as a single trial

        Parameters
        ----------
        combined_datasets : CombinedDatasets | None
            The combined datasets to use. It can be convenient when a model has been pre-trained, to avoid loading the
            same data twice
        subject_split : elecssl.data.subject_split.DataSplitBase

        Returns
        -------
        CombinedDatasets
        """
        print(f"Running on device: {self._device}")

        # -----------------
        # Load data and extract some details
        # -----------------
        if combined_datasets is None:
            combined_datasets = self._load_data(subject_split)

        # Get some dataset details
        dataset_details = self._extract_dataset_details(combined_datasets)

        channel_systems = dataset_details["channel_systems"]
        channel_name_to_index = dataset_details["channel_name_to_index"]

        # -----------------
        # Make subject split
        # -----------------
        splits = subject_split.splits

        # -----------------
        # Run the experiment
        # -----------------
        self._run(
            splits=splits, channel_systems=channel_systems, channel_name_to_index=channel_name_to_index,
            combined_dataset=combined_datasets
        )

        # -----------------
        # Create a file indicating that everything ran as expected
        # -----------------
        prefix_name = "" if self._experiment_name is None else f"{self._experiment_name}_"
        with open(self._results_path / f"{prefix_name}finished_successfully.txt", "w"):
            pass

        return combined_datasets

    # -------------
    # Methods for preparing for cross validation
    # -------------
    def _load_data(self, subject_split: DataSplitBase):
        """Method for loading data"""
        if self.train_method == TrainMethod.DOWNSTREAM:
            return CombinedDatasets.from_config(
                config=self.datasets_config, targets=self.train_config["target"], required_target=None,
                variables=self.variables, all_subjects=subject_split.all_subjects,
                default_target=self.train_config["target"])
        elif self.train_method == TrainMethod.MTL:
            return CombinedDatasets.from_config(
                config=self.datasets_config, targets=(self.mtl_config["pretext_target"], self.train_config["target"]),
                required_target=None, variables=self.variables, all_subjects=subject_split.all_subjects,
                default_target=self.train_config["target"])
        raise ValueError(f"Train method {self.train_method} not supported")

    @staticmethod
    def _extract_dataset_details(combined_dataset: CombinedDatasets):
        datasets = combined_dataset.datasets
        channel_systems = {dataset.name: dataset.channel_system for dataset in datasets}
        channel_name_to_index = {dataset.name: dataset.channel_name_to_index() for dataset in datasets}
        return {"channel_systems": channel_systems, "channel_name_to_index": channel_name_to_index}

    def _make_subject_split(self, subjects):
        """Method for splitting subjects into multiple folds"""
        data_split = get_data_split(split=self.subject_split_config["name"], dataset_subjects=subjects,
                                    **self.subject_split_config["kwargs"])
        return data_split.splits

    # -------------
    # Properties (shortcuts to sub-part of the config file)
    # -------------
    @property
    def datasets_config(self):
        return self._experiments_config["Datasets"]

    @property
    def train_config(self):
        return {**self._hp_config["Training"], **self._experiments_config["Training"]}

    @property
    def loss_config(self):
        return self._hp_config["Loss"]

    @property
    def interpolation_config(self) -> Optional[Dict[str, Any]]:
        if self._hp_config["SpatialDimensionMismatch"]["name"] != "Interpolation":
            return None
        else:
            return self._hp_config["SpatialDimensionMismatch"]["kwargs"]  # type: ignore[no-any-return]

    @property
    def shared_pre_processing_config(self):
        """Get the dict of the pre-processing config file which contains all shared pre-processing configurations"""
        return self._pre_processing_config

    @property
    def subject_split_config(self):
        return self._experiments_config["SubjectSplit"]

    @property
    def spatial_dimension_handling_config(self):
        return self._hp_config["SpatialDimensionMismatch"]

    @property
    def dl_architecture_config(self):
        return self._hp_config["DLArchitecture"]

    @property
    def domain_discriminator_config(self):
        return self._hp_config["DomainDiscriminator"]

    @property
    def cmmn_config(self):
        """Config file for CMMN. Only relevant for non-RBP models, as CMMN configurations are part of the RBP config for
        RBP models"""
        return self.dl_architecture_config["CMMN"]

    @property
    def rbp_config(self):
        if self._hp_config["SpatialDimensionMismatch"]["name"] != "RegionBasedPooling":
            return None
        else:
            return self._hp_config["SpatialDimensionMismatch"]["kwargs"]

    @property
    def scaler_config(self):
        return self._experiments_config["Scalers"]

    @property
    def sub_groups_config(self):
        return self._experiments_config["SubGroups"]

    @property
    def variables(self):
        return self._experiments_config["PredictionErrorAssociations"]

    @property
    def variables_metrics(self):
        return self._experiments_config["VariablesMetrics"]

    @property
    def saving_config(self):
        return self._experiments_config["Saving"]

    @property
    def spatial_method(self):
        return SpatialMethod(self.spatial_dimension_handling_config["name"])

    @property
    def train_method(self):
        return TrainMethod(self.train_config["method"])

    @property
    def mtl_config(self):
        """Config file for multi-task learning"""
        return self._experiments_config["MultiTaskLearning"]


# -------------
# Functions
# -------------
def _get_error(err) -> Type[Exception]:
    """
    Function for getting an error/exception by string

    Parameters
    ----------
    err : str

    Examples
    --------
    >>> _get_error("TrialPruned")
    <class 'optuna.exceptions.TrialPruned'>
    >>> raise _get_error("TrialPruned")("This is a message")
    Traceback (most recent call last):
    ...
    optuna.exceptions.TrialPruned: This is a message
    >>> _get_error("ThisIsNotAnError")
    Traceback (most recent call last):
    ...
    ValueError: Could not recognise the error 'ThisIsNotAnError'
    """
    if err.lower() in ("trialpruned", "optuna.trialpruned", "trialpruned()", "optuna.trialpruned()"):
        # I have no idea why mypy is complaining here...
        return optuna.TrialPruned  # type: ignore[no-any-return]

    raise ValueError(f"Could not recognise the error '{err}'")
