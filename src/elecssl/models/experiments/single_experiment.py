import copy
import os
import traceback
from typing import Any, Dict, Optional, Type

import optuna
import torch
import yaml  # type: ignore[import-untyped]
from torch import optim
from torch.utils.data import DataLoader

from elecssl.data.combined_datasets import CombinedDatasets
from elecssl.data.data_generators.data_generator import InterpolationDataGenerator, RBPDataGenerator
from elecssl.data.datasets.getter import get_dataset
from elecssl.data.scalers.target_scalers import get_target_scaler
from elecssl.data.subject_split import get_data_split, DataSplitBase
from elecssl.models.losses import CustomWeightedLoss, get_activation_function
from elecssl.models.main_models.main_base_class import MainModuleBase
from elecssl.models.main_models.main_fixed_channels_model import MainFixedChannelsModel
from elecssl.models.main_models.main_rbp_model import MainRBPModel
from elecssl.models.metrics import Histories, save_discriminator_histories_plots, NaNValueError
from elecssl.models.utils import tensor_dict_to_device


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
        mts_config["kwargs"]["in_channels"] = get_dataset(
            dataset_name=self.interpolation_config["main_channel_system"]  # type: ignore[index]
        ).num_channels

        # Define model
        model = MainFixedChannelsModel.from_config(
            mts_config=mts_config,
            discriminator_config=None if self.domain_discriminator_config is None
            else self.domain_discriminator_config["discriminator"],
            cmmn_config=self.cmmn_config
        ).to(self._device)

        return model

    def _make_rbp_model(self):
        """Method for defining a model with RBP as the first layer"""
        # Maybe add number of time steps
        mts_config = copy.deepcopy(self.dl_architecture_config)
        if "num_time_steps" in mts_config["kwargs"] and mts_config["kwargs"]["num_time_steps"] is None:
            mts_config["kwargs"]["num_time_steps"] = self.shared_pre_processing_config["num_time_steps"]

        # Define model
        model = MainRBPModel.from_config(
            rbp_config=self.rbp_config,
            mts_config=mts_config,
            discriminator_config=None if self.domain_discriminator_config is None
            else self.domain_discriminator_config["discriminator"]
        ).to(self._device)

        return model

    def _make_model(self):
        if self.spatial_dimension_handling_config["name"] == "RegionBasedPooling":
            return self._make_rbp_model()
        elif self.spatial_dimension_handling_config["name"] == "Interpolation":
            return self._make_interpolation_model()
        raise ValueError(f"Unexpected method for handling a varied number of channels: "
                         f"{self.spatial_dimension_handling_config['name']}")

    def _load_pretrained_model(self, path):
        if self.spatial_dimension_handling_config["name"] == "RegionBasedPooling":
            return MainRBPModel.load_model(name=f"{self._fine_tuning}_model", path=path).to(self._device)
        elif self.spatial_dimension_handling_config["name"] == "Interpolation":
            return MainFixedChannelsModel.load_model(name=f"{self._fine_tuning}_model", path=path).to(self._device)
        raise ValueError(f"Unexpected method for handling a varied number of channels: "
                         f"{self.spatial_dimension_handling_config['name']}")

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
        if isinstance(model, MainRBPModel):
            return self._prepare_rbp_model(model=model, channel_systems=channel_systems,
                                           combined_dataset=combined_dataset, train_subjects=train_subjects,
                                           test_subjects=test_subjects)
        elif isinstance(model, MainFixedChannelsModel):
            return self._prepare_interpolation_model(model=model, combined_dataset=combined_dataset,
                                                     train_subjects=train_subjects, test_subjects=test_subjects)
        raise TypeError(f"Did not recognise the model type: {type(model)}")

    @staticmethod
    def _fit_channel_systems(model, channel_systems):
        model.fit_channel_systems(tuple(channel_systems.values()))

    def _fit_cmmn_layers(self, *, model, train_data, channel_systems=None):
        if self.spatial_dimension_handling_config["name"] == "RegionBasedPooling":
            model.fit_psd_barycenters(data=train_data, channel_systems=channel_systems,
                                      sampling_freq=self.shared_pre_processing_config["resample"])
            model.fit_monge_filters(data=train_data, channel_systems=channel_systems)
        elif self.spatial_dimension_handling_config["name"] == "Interpolation":
            model.fit_psd_barycenters(data=train_data, sampling_freq=self.shared_pre_processing_config["resample"])
            model.fit_monge_filters(data=train_data)
        else:
            raise ValueError(f"Unexpected method for handling a varied number of channels: "
                             f"{self.spatial_dimension_handling_config['name']}")

    def _fit_cmmn_layers_test_data(self, *, model, test_data, channel_systems=None):
        for name, eeg_data in test_data.items():
            if name not in model.cmmn_fitted_channel_systems:
                # As long as the channel systems for the test data are present in 'channel_systems', this works
                # fine. Redundant channel systems is not a problem
                if self.spatial_dimension_handling_config["name"] == "RegionBasedPooling":
                    model.fit_monge_filters(data={name: eeg_data}, channel_systems=channel_systems)
                elif self.spatial_dimension_handling_config["name"] == "Interpolation":
                    model.fit_monge_filters(data={name: eeg_data})
                else:
                    raise ValueError(f"Unexpected method for handling a varied number of channels: "
                                     f"{self.spatial_dimension_handling_config['name']}")

    # -------------
    # Methods for creating Pytorch data loaders
    # -------------
    def _load_train_val_data_loaders(self, *, model=None, train_subjects, val_subjects, combined_dataset):
        if self.spatial_dimension_handling_config["name"] == "RegionBasedPooling":
            return self._load_rbp_train_val_data_loaders(
                model=model, train_subjects=train_subjects, val_subjects=val_subjects,
                combined_dataset=combined_dataset
            )
        elif self.spatial_dimension_handling_config["name"] == "Interpolation":
            return self._load_interpolation_train_val_data_loaders(
                train_subjects=train_subjects, val_subjects=val_subjects, combined_dataset=combined_dataset
            )
        else:
            raise ValueError

    def _load_test_data_loader(self, *, model=None, test_subjects, combined_dataset, target_scaler):
        if self.spatial_dimension_handling_config["name"] == "RegionBasedPooling":
            return self._load_rbp_test_data_loader(
                model=model, test_subjects=test_subjects, combined_dataset=combined_dataset,
                target_scaler=target_scaler
            )
        elif self.spatial_dimension_handling_config["name"] == "Interpolation":
            return self._load_interpolation_test_data_loader(
                test_subjects=test_subjects, combined_dataset=combined_dataset, target_scaler=target_scaler
            )
        else:
            raise ValueError

    def _load_interpolation_train_val_data_loaders(self, *, train_subjects, val_subjects, combined_dataset):
        # Extract input data
        train_data = combined_dataset.get_data(subjects=train_subjects)
        val_data = combined_dataset.get_data(subjects=val_subjects)

        # Extract scaled target data and the scaler itself
        train_targets, val_targets, target_scaler = self._get_targets_and_scaler(
            train_subjects=train_subjects, val_subjects=val_subjects, combined_dataset=combined_dataset
        )

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

        # Create data loaders
        train_loader = DataLoader(dataset=train_gen, batch_size=self.train_config["batch_size"], shuffle=True)
        val_loader = DataLoader(dataset=val_gen, batch_size=self.train_config["batch_size"], shuffle=True)

        return train_loader, val_loader, target_scaler

    def _load_interpolation_test_data_loader(self, *, test_subjects, combined_dataset, target_scaler):
        # Extract input data
        test_data = combined_dataset.get_data(subjects=test_subjects)

        # Extract scaled targets
        test_targets = combined_dataset.get_targets(subjects=test_subjects)
        test_targets = target_scaler.transform(test_targets)

        # Create data generators
        test_gen = InterpolationDataGenerator(
            data=test_data, targets=test_targets, subjects=combined_dataset.get_subjects_dict(test_subjects),
            subjects_info=combined_dataset.get_subjects_info(test_subjects),
            expected_variables=combined_dataset.get_expected_variables(test_subjects)
        )

        # Create data loader
        test_loader = DataLoader(dataset=test_gen, batch_size=self.train_config["batch_size"], shuffle=True)

        return test_loader

    def _load_rbp_train_val_data_loaders(self, *, model, train_subjects, val_subjects, combined_dataset):
        # Extract input data
        train_data = combined_dataset.get_data(subjects=train_subjects)
        val_data = combined_dataset.get_data(subjects=val_subjects)

        # Extract scaled target data and the scaler itself
        train_targets, val_targets, target_scaler = self._get_targets_and_scaler(
            train_subjects=train_subjects, val_subjects=val_subjects, combined_dataset=combined_dataset
        )

        # Compute the pre-computed features
        if model.supports_precomputing:
            train_pre_computed, val_pre_computed = self._get_pre_computed_features(model=model,
                                                                                   train_data=train_data,
                                                                                   val_data=val_data)
        else:
            train_pre_computed, val_pre_computed = None, None

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

        # Create data loaders
        train_loader = DataLoader(dataset=train_gen, batch_size=self.train_config["batch_size"], shuffle=True)
        val_loader = DataLoader(dataset=val_gen, batch_size=self.train_config["batch_size"], shuffle=True)

        return train_loader, val_loader, target_scaler

    def _load_rbp_test_data_loader(self, *, model, test_subjects, combined_dataset, target_scaler):
        # Extract input data
        test_data = combined_dataset.get_data(subjects=test_subjects)

        # Extract scaled targets
        test_targets = combined_dataset.get_targets(subjects=test_subjects)
        test_targets = target_scaler.transform(test_targets)

        # Compute the pre-computed features
        if model.supports_precomputing:
            test_pre_computed = model.pre_compute(
                input_tensors={dataset_name: torch.tensor(data, dtype=torch.float).to(self._device)
                               for dataset_name, data in test_data.items()})
            test_pre_computed = tuple(tensor_dict_to_device(pre_comp, device=torch.device("cpu"))
                                      for pre_comp in test_pre_computed)
        else:
            test_pre_computed = None

        # Create data generators
        test_gen = RBPDataGenerator(
            data=test_data, targets=test_targets, pre_computed=test_pre_computed,
            subjects=combined_dataset.get_subjects_dict(test_subjects),
            subjects_info=combined_dataset.get_subjects_info(test_subjects),
            expected_variables=combined_dataset.get_expected_variables(test_subjects)
        )

        # Create data loader
        test_loader = DataLoader(dataset=test_gen, batch_size=self.train_config["batch_size"], shuffle=True)

        return test_loader

    def _create_loaders(self, *, model, combined_dataset, train_subjects, val_subjects, test_subjects):
        train_loader, val_loader, target_scaler = self._load_train_val_data_loaders(
            model=model, train_subjects=train_subjects, val_subjects=val_subjects, combined_dataset=combined_dataset
        )

        # Maybe create loaders for test data
        test_loader: Optional[DataLoader[Any]]
        if self.train_config["continuous_testing"]:
            test_loader = self._load_test_data_loader(
                model=model, test_subjects=test_subjects, combined_dataset=combined_dataset,
                target_scaler=target_scaler
            )
        else:
            test_loader = None

        # Some type checks
        _allowed_dataset_types = (RBPDataGenerator, InterpolationDataGenerator)
        if not isinstance(train_loader.dataset, _allowed_dataset_types):
            raise TypeError(f"Expected training Pytorch datasets to inherit from "
                            f"{tuple(data_gen.__name__ for data_gen in _allowed_dataset_types)}, but found "
                            f"{type(train_loader.dataset)}")
        if not isinstance(val_loader.dataset, _allowed_dataset_types):
            raise TypeError(f"Expected validation Pytorch datasets to inherit from "
                            f"{tuple(data_gen.__name__ for data_gen in _allowed_dataset_types)}, but found "
                            f"{type(val_loader.dataset)}")

        return train_loader, val_loader, test_loader, target_scaler

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

    def _get_targets_and_scaler(self, *, train_subjects, val_subjects, combined_dataset):
        # Extract target data
        train_targets = combined_dataset.get_targets(subjects=train_subjects)
        val_targets = combined_dataset.get_targets(subjects=val_subjects)

        # Fit scaler and scale
        target_scaler = get_target_scaler(self.scaler_config["target"]["name"],
                                          **self.scaler_config["target"]["kwargs"])
        target_scaler.fit(train_targets)

        train_targets = target_scaler.transform(train_targets)
        val_targets = target_scaler.transform(val_targets)

        return train_targets, val_targets, target_scaler

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

    def _create_loss_and_optimiser(self, model, dataset_sizes):
        # Create optimiser
        optimiser = optim.Adam(model.parameters(), lr=self.train_config["learning_rate"],
                               betas=(self.train_config["beta_1"], self.train_config["beta_2"]),
                               eps=self.train_config["eps"])

        # Create loss
        if self.loss_config["weighter"] is not None:
            self.loss_config["weighter_kwargs"]["dataset_sizes"] = dataset_sizes
        criterion = CustomWeightedLoss(**self.loss_config)

        return optimiser, criterion

    # -------------
    # Methods for saving results
    # -------------
    def _save_results(self, *, histories: Dict[str, Histories], results_path):
        decimals = self.saving_config["performance_score_decimals"]

        prefix_name = "" if self._experiment_name is None else f"{self._experiment_name}_"

        # Save prediction histories
        train_history = histories["train"]
        val_history = histories["val"]
        test_history = histories["test"] if "test" in histories else None

        train_history.save_main_history(history_name=f"{prefix_name}train_history", path=results_path,
                                        decimals=decimals)
        val_history.save_main_history(history_name=f"{prefix_name}val_history", path=results_path,
                                      decimals=decimals)
        if test_history is not None:
            test_history.save_main_history(history_name=f"{prefix_name}test_history", path=results_path,
                                           decimals=decimals)

        if self.domain_discriminator_config is not None:
            domain_discriminator_path = os.path.join(results_path, f"{prefix_name}domain_discriminator")
            os.mkdir(domain_discriminator_path)

            dd_train_history = histories["dd_train"]
            dd_val_history = histories["dd_val"]

            dd_train_history.save_main_history(history_name=f"{prefix_name}dd_train_history",
                                               path=domain_discriminator_path, decimals=decimals)
            dd_val_history.save_main_history(history_name=f"{prefix_name}dd_val_history",
                                             path=domain_discriminator_path, decimals=decimals)

            # Save domain discriminator metrics plots
            if self.saving_config["save_discriminator_plots"]:
                save_discriminator_histories_plots(path=domain_discriminator_path,
                                                   histories=(dd_train_history, dd_val_history))

        # Save subgroup plots
        sub_group_path = os.path.join(results_path, f"{prefix_name}sub_groups_plots")
        os.mkdir(sub_group_path)

        train_history.save_subgroup_metrics(history_name="train", path=sub_group_path, decimals=decimals,
                                            save_plots=self.saving_config["save_subgroups_plots"])
        val_history.save_subgroup_metrics(history_name="val", path=sub_group_path, decimals=decimals,
                                          save_plots=self.saving_config["save_subgroups_plots"])
        if test_history is not None:
            test_history.save_subgroup_metrics(history_name="test", path=sub_group_path, decimals=decimals,
                                               save_plots=self.saving_config["save_subgroups_plots"])

        # Save variable associations with prediction error
        _histories = (train_history, val_history) if test_history is None else (train_history, val_history,
                                                                                test_history)
        if any(history.has_variables_history for history in _histories):
            variables_history_path = results_path / f"{prefix_name}error_associations"
            os.mkdir(variables_history_path)
            train_history.save_variables_histories(history_name="train", path=variables_history_path,
                                                   decimals=decimals,
                                                   save_plots=self.saving_config["save_error_association_plots"])
            val_history.save_variables_histories(history_name="val", path=variables_history_path, decimals=decimals,
                                                 save_plots=self.saving_config["save_error_association_plots"])
            if test_history is not None:
                test_history.save_variables_histories(history_name="test", path=variables_history_path,
                                                      decimals=decimals,
                                                      save_plots=self.saving_config["save_error_association_plots"])

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

        # -----------------
        # Create data loaders (and target scaler)
        # -----------------
        print("Creating data loaders...")
        train_loader, val_loader, test_loader, target_scaler = self._create_loaders(
            model=model, combined_dataset=combined_dataset, train_subjects=train_subjects, val_subjects=val_subjects,
            test_subjects=test_subjects
        )

        # -----------------
        # Create loss and optimiser
        # -----------------
        dataset_sizes = train_loader.dataset.dataset_sizes  # type: ignore[attr-defined]

        # For the downstream model
        optimiser, criterion = self._create_loss_and_optimiser(model=model, dataset_sizes=dataset_sizes)

        # Maybe for a domain discriminator
        discriminator_criterion: Optional[CustomWeightedLoss]
        if self.domain_discriminator_config is None:
            discriminator_kwargs = dict()
        else:
            (discriminator_criterion, discriminator_weight,
             discriminator_metrics) = self._get_domain_discriminator_details(dataset_sizes=dataset_sizes)
            discriminator_kwargs = {"discriminator_criterion": discriminator_criterion,
                                    "discriminator_weight": discriminator_weight,
                                    "discriminator_metrics": discriminator_metrics}

        # -----------------
        # Train model
        # -----------------
        print(f"{' Training ':-^20}")

        channel_name_to_index_kwarg = {"channel_name_to_index": channel_name_to_index} \
            if self.spatial_dimension_handling_config["name"] == "RegionBasedPooling" else dict()

        try:
            histories = model.train_model(
                method=self.train_config["method"], train_loader=train_loader, val_loader=val_loader,
                test_loader=test_loader, metrics=self.train_config["metrics"],
                main_metric=self.train_config["main_metric"], classifier_criterion=criterion, optimiser=optimiser,
                **discriminator_kwargs, num_epochs=self.train_config["num_epochs"],
                verbose=self.train_config["verbose"],
                device=self._device, target_scaler=target_scaler, **channel_name_to_index_kwarg,
                prediction_activation_function=get_activation_function(self.train_config["prediction_activation_"
                                                                                         "function"]),
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

        # -----------------
        # Test model (but only if continuous testing was not used)
        # -----------------
        if not self.train_config["continuous_testing"]:
            print(f"\n{' Testing ':-^20}")
            if "test" in histories:
                raise RuntimeError("Expected 'test' history not to be present with continuous test set to 'False', "
                                   "but that was not the case")

            # Get test loader
            test_loader = self._load_test_data_loader(model=model, test_subjects=test_subjects,
                                                      combined_dataset=combined_dataset, target_scaler=target_scaler)

            # Test model on test data
            histories["test"] = model.test_model(
                data_loader=test_loader, metrics=self.train_config["metrics"], verbose=self.train_config["verbose"],
                **channel_name_to_index_kwarg, device=self._device, target_scaler=target_scaler,
                sub_group_splits=self.sub_groups_config["sub_groups"],
                prediction_activation_function=get_activation_function(self.train_config["prediction_activation_"
                                                                                         "function"]),
                sub_groups_verbose=self.sub_groups_config["verbose"],
                verbose_variables=self.train_config["verbose_variables"], variable_metrics=self.variables_metrics
            )

        # -----------------
        # Save results
        # -----------------
        # Performance scores
        self._save_results(histories=histories, results_path=results_path)

        # (Maybe) the model itself
        if self.saving_config["save_model"]:
            model = model.to(device=torch.device("cpu"))
            prefix_name = "" if self._experiment_name is None else f"{self._experiment_name}_"
            model.save_model(name=f"{prefix_name}model", path=results_path)

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
            This dataframe requires two columns, 'dataset' and 'sub_id'. This contains all data which is supposed to be
            loaded. Does not need to be specified if 'combined_datasets' is not None

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
        return CombinedDatasets.from_config(config=self.datasets_config, target=self.train_config["target"],
                                            required_target=None,  # Not necessary nor wanted to specify
                                            variables=self.variables, all_subjects=subject_split.all_subjects)

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
