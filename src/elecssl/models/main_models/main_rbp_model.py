import abc
import copy
from typing import Dict, Optional, Iterator, List, Tuple

import torch
from progressbar import progressbar
from torch import optim, nn
from torch.nn import Parameter

from elecssl.data.data_generators.data_generator import strip_tensors
from elecssl.data.datasets.dataset_base import ChannelSystem
from elecssl.models.domain_adaptation.domain_discriminators.getter import get_domain_discriminator
from elecssl.models.losses import CustomWeightedLoss
from elecssl.models.main_models.main_base_class import MainModuleBase, reorder_subjects
from elecssl.models.metrics import Histories, is_improved_model, PScore, is_pareto_optimal
from elecssl.models.mtl_strategies.multi_task_strategies import MultiTaskStrategy
from elecssl.models.mts_modules.getter import get_mts_module
from elecssl.models.region_based_pooling.region_based_pooling import RegionBasedPooling, RBPDesign, RBPPoolType
from elecssl.models.utils import tensor_dict_to_device, flatten_targets, ReverseLayerF, verify_type, \
    tensor_dict_to_boolean, maybe_no_grad


class MainRBPModelBase(MainModuleBase, abc.ABC):
    """
    Main model supporting use of RBP. That is, this class uses RBP as a first layer, followed by an MTS module

    PS: Merges montage splits by concatenation
    """

    def __init__(self, *, mts_module, mts_module_kwargs, rbp_designs, normalise_region_representations):
        """
        Initialise

        Parameters
        ----------
        mts_module : str
        mts_module_kwargs : dict[str, typing.Any]
        rbp_designs : tuple[elecssl.models.region_based_pooling.region_based_pooling.RBPDesign, ...]
        normalise_region_representations : bool
        """
        super().__init__()

        # -----------------
        # Create RBP layer
        # -----------------
        self._region_based_pooling = RegionBasedPooling(rbp_designs)
        self._normalise_region_representations = normalise_region_representations

        # ----------------
        # Create MTS module
        # ----------------
        self._mts_module = get_mts_module(
            mts_module_name=mts_module, **{"in_channels": self._region_based_pooling.num_regions, **mts_module_kwargs}
        )

    @staticmethod
    def _create_rbp_designs_from_config(rbp_config):
        designs_config = copy.deepcopy(rbp_config["RBPDesigns"])
        rbp_designs = []
        for name, design in designs_config.items():
            rbp_designs.append(
                RBPDesign(pooling_type=RBPPoolType(design["pooling_type"]),
                          pooling_methods=design["pooling_methods"],
                          pooling_methods_kwargs=design["pooling_methods_kwargs"],
                          split_methods=design["split_methods"],
                          split_methods_kwargs=design["split_methods_kwargs"],
                          use_cmmn_layer=design["use_cmmn_layer"],
                          cmmn_kwargs=design["cmmn_kwargs"],
                          num_designs=design["num_designs"])
            )
        return tuple(rbp_designs)

    # ----------------
    # Methods for forward pass and related
    # ----------------
    def _forward_rbp(self, input_tensors, *, channel_name_to_index, pre_computed=None):
        # ------------
        # Input check
        # ------------
        if any(tensor_.isnan().any() for tensor_ in input_tensors.values()):
            datasets_with_nans = []
            for name, tensor_ in input_tensors.items():
                if tensor_.isnan().any():
                    datasets_with_nans.append(name)
            raise ValueError(f"Input tensors from {datasets_with_nans} contain NaN values")

        # ------------
        # Forward
        # ------------
        # Pass through RBP layer
        x = self._region_based_pooling(input_tensors, channel_name_to_index=channel_name_to_index,
                                       pre_computed=pre_computed)
        # Merge by concatenation
        x = torch.cat(x, dim=1)

        # Maybe normalise region representations
        if self._normalise_region_representations:
            x = (x - torch.mean(x, dim=-1, keepdim=True)) / (torch.std(x, dim=-1, keepdim=True) + 1e-8)

        return x

    def pre_compute(self, input_tensors):
        """
        Pre-compute

        Parameters
        ----------
        input_tensors : dict[str, torch.Tensor]

        Returns
        -------
        tuple[dict[str, torch.Tensor], ...]
        """
        return self._region_based_pooling.pre_compute(input_tensors)

    # ----------------
    # Methods for fitting channel systems
    # ----------------
    def fit_channel_system(self, channel_system):
        self._region_based_pooling.fit_channel_system(channel_system)

    def fit_channel_systems(self, channel_systems):
        self._region_based_pooling.fit_channel_systems(channel_systems)

    # ----------------
    # Methods for fitting CMMN layer
    # ----------------
    def fit_psd_barycenters(self, data, *, channel_systems: Dict[str, ChannelSystem], sampling_freq=None):
        self._region_based_pooling.fit_psd_barycenters(data, channel_systems=channel_systems,
                                                       sampling_freq=sampling_freq)

    def fit_monge_filters(self, data, *, channel_systems: Dict[str, ChannelSystem]):
        self._region_based_pooling.fit_monge_filters(data, channel_systems=channel_systems)

    # ----------------
    # Some additional abstract methods
    # ----------------
    def save_metadata(self, *, name, path):
        self._mts_module.save_metadata(name=name, path=path)

    # ----------------
    # Required methods for multi-task learning and
    # multi-objective optimisation
    # ----------------
    def gradnorm_parameters(self) -> Iterator[Parameter]:
        raise NotImplementedError

    def shared_parameters(self) -> Iterator[Parameter]:
        raise NotImplementedError

    # ----------------
    # Properties
    # ----------------
    @property
    def supports_precomputing(self):
        return self._region_based_pooling.supports_precomputing

    @property
    def any_rbp_cmmn_layers(self) -> bool:
        return self._region_based_pooling.any_cmmn_layers

    @property
    def cmmn_fitted_channel_systems(self):
        return self._region_based_pooling.cmmn_fitted_channel_systems


class DownstreamRBPModel(MainRBPModelBase):
    """
    Normal single-task downstream training only
    """

    @classmethod
    def from_config(cls, *, rbp_config, mts_config):
        # -----------------
        # Read RBP designs
        # -----------------
        rbp_designs = cls._create_rbp_designs_from_config(rbp_config)

        # -----------------
        # Read MTS design
        # -----------------
        # Read configuration file
        mts_design = copy.deepcopy(mts_config)

        # -----------------
        # Make model
        # -----------------
        return cls(mts_module=mts_design["model"], mts_module_kwargs=mts_design["kwargs"], rbp_designs=rbp_designs,
                   normalise_region_representations=rbp_config["normalise_region_representations"])

    # --------------
    # Forward pass
    # --------------
    def forward(self, input_tensors, *, channel_name_to_index, pre_computed=None):
        # Pass through RBP (and maybe normalisation)
        x = self._forward_rbp(input_tensors, channel_name_to_index=channel_name_to_index, pre_computed=pre_computed)

        # Pass through MTS module
        return self._mts_module(x)

    # --------------
    # Methods for training and testing
    # --------------
    def train_model(self, *, train_loader, val_loader, test_loader, metrics, main_metric, num_epochs,
                    criterion, optimiser, device, channel_name_to_index, prediction_activation_function=None,
                    verbose=True, target_scaler, sub_group_splits, sub_groups_verbose, verbose_variables,
                    variable_metrics, patience: Optional[int]):
        # Defining histories objects
        train_history = Histories(metrics=metrics, splits=sub_group_splits, variable_metrics=variable_metrics,
                                  expected_variables=train_loader.dataset.expected_variables)
        val_history = Histories(metrics=metrics, name="val", splits=sub_group_splits, variable_metrics=variable_metrics,
                                expected_variables=val_loader.dataset.expected_variables)
        test_history = None if test_loader is None else Histories(
            metrics=metrics, name="test", splits=sub_group_splits, variable_metrics=variable_metrics,
            expected_variables=test_loader.dataset.expected_variables
        )

        # ------------------------
        # Fit model
        # ------------------------
        remaining_patience = patience
        best_metrics = None
        best_model_state = {k: v.cpu() for k, v in self.state_dict().items()}
        best_epoch = 0
        for epoch in range(num_epochs):
            # ----------------
            # Training
            # ----------------
            pbar_prefix = f"Train epoch {epoch + 1}/{num_epochs} "
            self._full_epoch_pass(
                loader=train_loader, compute_loss=True, channel_name_to_index=channel_name_to_index,
                pbar_prefix=pbar_prefix, device=device, optimiser=optimiser, criterion=criterion,
                prediction_activation_function=prediction_activation_function, target_scaler=target_scaler,
                history=train_history, verbose=verbose, verbose_variables=verbose_variables,
                sub_groups_verbose=sub_groups_verbose
            )

            # ----------------
            # Validation
            # ----------------
            pbar_prefix = f"Val epoch {epoch + 1}/{num_epochs} "
            with torch.no_grad():
                self._full_epoch_pass(
                    loader=val_loader, compute_loss=False, optimiser=None, criterion=None, history=val_history,
                    channel_name_to_index=channel_name_to_index, pbar_prefix=pbar_prefix, device=device,
                    target_scaler=target_scaler, prediction_activation_function=prediction_activation_function,
                    verbose=verbose, verbose_variables=verbose_variables, sub_groups_verbose=sub_groups_verbose
                )

            # ----------------
            # (Maybe) testing
            # ----------------
            if test_loader is not None:
                pbar_prefix = f"Test epoch {epoch + 1}/{num_epochs} "
                with torch.no_grad():
                    self._full_epoch_pass(
                        loader=test_loader, compute_loss=False, optimiser=None, criterion=None, history=test_history,
                        channel_name_to_index=channel_name_to_index, pbar_prefix=pbar_prefix, device=device,
                        target_scaler=target_scaler, prediction_activation_function=prediction_activation_function,
                        verbose=verbose, verbose_variables=verbose_variables, sub_groups_verbose=sub_groups_verbose
                    )

            # ----------------
            # If this is the highest performing model, as evaluated on the validation set, store it
            # ----------------
            if is_improved_model(old_metrics=best_metrics, new_metrics=val_history.newest_metrics,
                                 main_metric=main_metric):
                # Store the model on the cpu
                best_model_state = copy.deepcopy({k: v.cpu() for k, v in self.state_dict().items()})

                # Update the best metrics and epoch
                best_metrics = val_history.newest_metrics
                best_epoch = epoch

                # Patience revived
                if patience is not None:
                    remaining_patience = patience
            else:
                if patience is not None:
                    assert remaining_patience is not None  # really just for mypy
                    remaining_patience -= 1

            # ----------------
            # Break if we are out of patience
            # ----------------
            if patience is not None and remaining_patience == 0:
                break

        # Set the parameters back to those of the best model
        self.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})  # type: ignore[arg-type]

        # Return the histories
        histories = {"train": train_history, "val": val_history}
        if test_history is not None:
            histories["test"] = test_history
        return histories, (best_model_state,), (best_epoch,)

    def test_model(self, *, data_loader, metrics, device, channel_name_to_index, prediction_activation_function,
                   verbose=True, target_scaler, sub_group_splits, sub_groups_verbose, verbose_variables,
                   variable_metrics):
        # Defining histories objects
        history = Histories(metrics=metrics, name="test", splits=sub_group_splits, variable_metrics=variable_metrics,
                            expected_variables=data_loader.dataset.expected_variables)

        # No gradients needed
        pbar_prefix = "Testing "
        with torch.no_grad():
            self._full_epoch_pass(
                loader=data_loader, compute_loss=False, optimiser=None, criterion=None, history=history, device=device,
                channel_name_to_index=channel_name_to_index, pbar_prefix=pbar_prefix, target_scaler=target_scaler,
                prediction_activation_function=prediction_activation_function, verbose=verbose,
                sub_groups_verbose=sub_groups_verbose, verbose_variables=verbose_variables
            )

        return history

    def _full_epoch_pass(self, *, loader, compute_loss, channel_name_to_index, device, pbar_prefix, history,
                         optimiser: Optional[optim.Optimizer], criterion: Optional[CustomWeightedLoss],
                         prediction_activation_function, target_scaler, verbose, sub_groups_verbose, verbose_variables):
        # Set training/evaluation mode
        if verify_type(compute_loss, bool):
            self.train()
        else:
            self.eval()

        # -------------
        # Run for a full epoch
        # -------------
        for x, pre_computed, y, subject_indices in progressbar(loader, redirect_stdout=True, prefix=pbar_prefix):
            # Strip the dictionaries for 'ghost tensors'
            x = strip_tensors(x)
            y = strip_tensors(y)

            if isinstance(pre_computed, torch.Tensor) and torch.all(torch.isnan(pre_computed)):
                pre_computed = None
            else:
                pre_computed = [strip_tensors(pre_comp) for pre_comp in pre_computed]

            # Extract subjects and correct the ordering
            subjects = reorder_subjects(
                order=tuple(x.keys()), subjects=loader.dataset.get_subjects_from_indices(subject_indices))

            # Send data to the correct device
            x = tensor_dict_to_device(x, device=device)
            y = flatten_targets(y).to(device)
            if pre_computed is not None:
                pre_computed = tuple(tensor_dict_to_device(pre_comp, device=device) for pre_comp in pre_computed)

            # Forward pass
            output = self(x, pre_computed=pre_computed, channel_name_to_index=channel_name_to_index)

            # Maybe compute loss and apply optimiser
            if compute_loss:
                assert optimiser is not None, "When computing loss, an optimiser must be passed, but received None"
                assert criterion is not None, "When computing loss, a criterion must be passed, but received None"

                optimiser.zero_grad()
                loss = criterion(output, y, subjects=subjects)
                loss.backward()
                optimiser.step()

            # Update history object
            self._updated_history_object(output=output, y=y, target_scaler=target_scaler, subjects=subjects,
                                         history=history, prediction_activation_function=prediction_activation_function)

        # Finalise epoch for history object
        history.on_epoch_end(
            verbose=verbose, verbose_sub_groups=sub_groups_verbose, verbose_variables=verbose_variables,
            subjects_info=loader.dataset.subjects_info)


class DomainDiscriminatorRBPModel(MainRBPModelBase):
    """
    RBP with domain discriminator training
    """

    def __init__(self, *, mts_module, mts_module_kwargs, rbp_designs, normalise_region_representations,
                 domain_discriminator, domain_discriminator_kwargs):
        super().__init__(mts_module=mts_module, mts_module_kwargs=mts_module_kwargs, rbp_designs=rbp_designs,
                         normalise_region_representations=normalise_region_representations)

        # ----------------
        # Create domain discriminator
        # ----------------
        # Set kwargs to empty dict if none are passed
        domain_discriminator_kwargs = dict() if domain_discriminator_kwargs is None else domain_discriminator_kwargs

        # Need to get input features from MTS module
        domain_discriminator_kwargs["in_features"] = self._mts_module.latent_features_dim

        self._domain_discriminator = get_domain_discriminator(
            name=domain_discriminator, **domain_discriminator_kwargs
        )

    @classmethod
    def from_config(cls, *, rbp_config, mts_config, discriminator_config):
        # -----------------
        # Read RBP designs
        # -----------------
        rbp_designs = cls._create_rbp_designs_from_config(rbp_config)

        # -----------------
        # Read MTS design
        # -----------------
        # Read configuration file
        mts_design = copy.deepcopy(mts_config)

        # -----------------
        # Make model
        # -----------------
        return cls(mts_module=mts_design["model"], mts_module_kwargs=mts_design["kwargs"], rbp_designs=rbp_designs,
                   normalise_region_representations=rbp_config["normalise_region_representations"],
                   domain_discriminator=None if discriminator_config is None else discriminator_config["name"],
                   domain_discriminator_kwargs=None if discriminator_config is None else discriminator_config["kwargs"])

    # --------------
    # Forward pass
    # --------------
    def forward(self, input_tensors, *, channel_name_to_index, pre_computed=None):
        # Pass through RBP (and maybe normalisation)
        x = self._forward_rbp(input_tensors, channel_name_to_index=channel_name_to_index, pre_computed=pre_computed)

        # Pass through MTS module to extract latent features
        x = self._mts_module.extract_latent_features(x)

        # ----------------
        # Pass through both the classifier and domain discriminator
        # ----------------
        # Adding a gradient reversal layer to the features passed to domain discriminator
        gradient_reversed_x = ReverseLayerF.apply(x, 1.)

        return (self._mts_module.classify_latent_features(x),
                self._domain_discriminator(gradient_reversed_x))  # type: ignore[misc]

    # --------------
    # Methods for training and testing
    # --------------
    def train_model(self, *, train_loader, val_loader, test_loader=None, metrics, main_metric, num_epochs,
                    classifier_criterion, optimiser, discriminator_criterion, discriminator_weight,
                    discriminator_metrics, device, channel_name_to_index, prediction_activation_function=None,
                    verbose=True, target_scaler=None, sub_group_splits, sub_groups_verbose, verbose_variables,
                    variable_metrics, patience: Optional[int]):
        # Defining histories objects
        train_history = Histories(metrics=metrics, splits=sub_group_splits, variable_metrics=variable_metrics,
                                  expected_variables=train_loader.dataset.expected_variables)
        val_history = Histories(metrics=metrics, name="val", splits=sub_group_splits, variable_metrics=variable_metrics,
                                expected_variables=val_loader.dataset.expected_variables)
        test_history = None if test_loader is None else Histories(
            metrics=metrics, name="test", splits=sub_group_splits, variable_metrics=variable_metrics,
            expected_variables=test_loader.dataset.expected_variables
        )

        dd_train_history = Histories(metrics=discriminator_metrics, name="dd", splits=None)
        dd_val_history = Histories(metrics=discriminator_metrics, name="val_dd", splits=None)

        # ------------------------
        # Fit model
        # ------------------------
        remaining_patience = patience
        best_metrics = None
        best_model_state = {k: v.cpu() for k, v in self.state_dict().items()}
        best_epoch = 0
        for epoch in range(num_epochs):
            # ----------------
            # Training
            # ----------------
            pbar_prefix = f"Train epoch {epoch + 1}/{num_epochs} "
            self._full_epoch_pass(
                loader=train_loader, compute_loss=True, downstream_history=train_history,
                discriminator_history=dd_train_history, channel_name_to_index=channel_name_to_index,
                pbar_prefix=pbar_prefix, device=device, optimiser=optimiser, classifier_criterion=classifier_criterion,
                discriminator_criterion=discriminator_criterion, discriminator_weight=discriminator_weight,
                prediction_activation_function=prediction_activation_function, target_scaler=target_scaler,
                verbose=verbose, verbose_variables=verbose_variables, sub_groups_verbose=sub_groups_verbose
            )

            # ----------------
            # Validation
            # ----------------
            pbar_prefix = f"Val epoch {epoch + 1}/{num_epochs} "
            with torch.no_grad():
                self._full_epoch_pass(
                    loader=val_loader, compute_loss=False, optimiser=None, classifier_criterion=None,
                    discriminator_criterion=None, discriminator_weight=None, downstream_history=val_history,
                    discriminator_history=dd_val_history, channel_name_to_index=channel_name_to_index,
                    pbar_prefix=pbar_prefix, device=device, target_scaler=target_scaler,
                    prediction_activation_function=prediction_activation_function,  verbose=verbose,
                    verbose_variables=verbose_variables, sub_groups_verbose=sub_groups_verbose
                )

            # ----------------
            # (Maybe) testing
            # ----------------
            if test_loader is not None:
                pbar_prefix = f"Test epoch {epoch + 1}/{num_epochs} "
                with torch.no_grad():
                    # Not using discriminator for the test set. Really just because it was not used before refactoring,
                    # and training with DD is not actually maintained atm...
                    self._full_epoch_pass(
                        loader=test_loader, downstream_history=test_history, discriminator_history=None,
                        compute_loss=False, optimiser=None, classifier_criterion=None, discriminator_criterion=None,
                        discriminator_weight=None, channel_name_to_index=channel_name_to_index, pbar_prefix=pbar_prefix,
                        device=device, target_scaler=target_scaler,
                        prediction_activation_function=prediction_activation_function,  verbose=verbose,
                        verbose_variables=verbose_variables, sub_groups_verbose=sub_groups_verbose
                    )

            # ----------------
            # If this is the highest performing model, store it
            # ----------------
            if is_improved_model(old_metrics=best_metrics, new_metrics=val_history.newest_metrics,
                                 main_metric=main_metric):
                # Store the model on the cpu
                best_model_state = copy.deepcopy({k: v.cpu() for k, v in self.state_dict().items()})

                # Update the best metrics and epoch
                best_metrics = val_history.newest_metrics
                best_epoch = epoch

                # Patience revived
                if patience is not None:
                    remaining_patience = patience
            else:
                if remaining_patience is not None:
                    remaining_patience -= 1

            # ----------------
            # Break if we are out of patience
            # ----------------
            if patience is not None and remaining_patience == 0:
                break

        # Set the parameters back to those of the best model
        self.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})  # type: ignore[arg-type]

        # Return the histories
        histories = {"train": train_history, "val": val_history, "train_dd": dd_train_history, "val_dd": dd_val_history}
        if test_history is not None:
            histories["test"] = test_history
        return histories, (best_model_state,), (best_epoch,)

    def test_model(self, *, data_loader, metrics, device, channel_name_to_index, prediction_activation_function,
                   verbose=True, target_scaler, sub_group_splits, sub_groups_verbose, verbose_variables,
                   variable_metrics) -> Histories:
        # Defining histories objects
        history = Histories(metrics=metrics, name="test", splits=sub_group_splits, variable_metrics=variable_metrics,
                            expected_variables=data_loader.dataset.expected_variables)

        # No gradients needed
        pbar_prefix = "Testing "
        with torch.no_grad():
            # Not using discriminator for the test set. Really just because it was not used before refactoring,
            # and training with DD is not actually maintained atm...
            self._full_epoch_pass(
                loader=data_loader, downstream_history=history, discriminator_history=None,
                compute_loss=False, optimiser=None, classifier_criterion=None, discriminator_criterion=None,
                discriminator_weight=None, channel_name_to_index=channel_name_to_index, pbar_prefix=pbar_prefix,
                device=device, target_scaler=target_scaler,
                prediction_activation_function=prediction_activation_function, verbose=verbose,
                verbose_variables=verbose_variables, sub_groups_verbose=sub_groups_verbose
            )

        return history

    def _full_epoch_pass(self, *, loader, compute_loss, channel_name_to_index, device, pbar_prefix, downstream_history,
                         optimiser: Optional[optim.Optimizer], classifier_criterion: Optional[CustomWeightedLoss],
                         discriminator_criterion: Optional[CustomWeightedLoss], discriminator_weight: Optional[float],
                         discriminator_history, prediction_activation_function, target_scaler, verbose,
                         sub_groups_verbose, verbose_variables):
        # Set training/evaluation mode
        if verify_type(compute_loss, bool):
            self.train()
        else:
            self.eval()

        # -------------
        # Run for a full epoch
        # -------------
        for x, pre_computed, y, subject_indices in progressbar(loader, redirect_stdout=True, prefix=pbar_prefix):
            # Strip the dictionaries for 'ghost tensors'
            x = strip_tensors(x)
            y = strip_tensors(y)

            if isinstance(pre_computed, torch.Tensor) and torch.all(torch.isnan(pre_computed)):
                pre_computed = None
            else:
                pre_computed = [strip_tensors(pre_comp) for pre_comp in pre_computed]

            # Extract subjects and correct the ordering
            subjects = reorder_subjects(
                order=tuple(x.keys()), subjects=loader.dataset.get_subjects_from_indices(subject_indices))

            # Send data to the correct device
            x = tensor_dict_to_device(x, device=device)
            y = flatten_targets(y).to(device)
            if pre_computed is not None:
                pre_computed = tuple(tensor_dict_to_device(pre_comp, device=device) for pre_comp in pre_computed)

            # Forward pass
            classifier_output, discriminator_output = self(x, pre_computed=pre_computed,
                                                           channel_name_to_index=channel_name_to_index)

            # Get dataset belonging (targets for discriminator)
            discriminator_targets = loader.dataset.get_dataset_indices_from_subjects(
                subjects=subjects).to(device)

            # Maybe compute loss and apply optimiser
            if compute_loss:
                assert optimiser is not None, "When computing loss, an optimiser must be passed, but received None"
                assert classifier_criterion is not None, \
                    "When computing loss, a criterion for the classifier must be passed, but received None"
                assert discriminator_criterion is not None, \
                    "When computing loss, a criterion for the discriminator must be passed, but received None"

                loss = (classifier_criterion(classifier_output, y, subjects=subjects)
                        + discriminator_weight * discriminator_criterion(discriminator_output, discriminator_targets,
                                                                         subjects=subjects))
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

            # Update history objects
            self._updated_history_object(
                history=downstream_history, output=classifier_output, y=y,  target_scaler=target_scaler,
                subjects=subjects, prediction_activation_function=prediction_activation_function)
            if discriminator_history is not None:
                self._updated_history_object(
                    history=discriminator_history, output=discriminator_output, y=discriminator_targets,
                    target_scaler=None, subjects=subjects, prediction_activation_function=None)

        # Finalise epoch for history objects
        downstream_history.on_epoch_end(
            verbose=verbose, verbose_sub_groups=sub_groups_verbose, verbose_variables=verbose_variables,
            subjects_info=loader.dataset.subjects_info)
        if discriminator_history is not None:
            discriminator_history.on_epoch_end(verbose=verbose)


class MultiTaskRBPModel(MainRBPModelBase):
    """
    RBP with multi-task learning, where we want to (1) predict some variable, and (2) use the residual of the first loss
    to make predictions on some other task.

    For example, to (1) predict age, and (2) predict some pathological state from the brain age residual. Or as in this
    work, predict a feature from eyes open, and use the residual to predict something related to cognition
    """

    def __init__(self, *, mts_module, mts_module_kwargs, rbp_designs, normalise_region_representations):
        super().__init__(mts_module=mts_module, mts_module_kwargs=mts_module_kwargs, rbp_designs=rbp_designs,
                         normalise_region_representations=normalise_region_representations)

        # Module for making predictions from the residual
        self._residual_model = nn.Linear(1, 1, bias=True)

    @classmethod
    def from_config(cls, *, rbp_config, mts_config):
        # -----------------
        # Read RBP designs
        # -----------------
        rbp_designs = cls._create_rbp_designs_from_config(rbp_config)

        # -----------------
        # Read MTS design
        # -----------------
        # Read configuration file
        mts_design = copy.deepcopy(mts_config)

        # -----------------
        # Make model
        # -----------------
        return cls(mts_module=mts_design["model"], mts_module_kwargs=mts_design["kwargs"], rbp_designs=rbp_designs,
                   normalise_region_representations=rbp_config["normalise_region_representations"])

    # --------------
    # Required methods for multi-task learning
    # --------------
    def shared_parameters(self) -> Iterator[Parameter]:
        """This is required for MGDA"""
        for param in self._region_based_pooling.parameters():
            yield param

        for param in self._mts_module.parameters():
            yield param

    def gradnorm_parameters(self) -> Iterator[Parameter]:
        return self._mts_module.gradnorm_parameters()

    # --------------
    # Forward pass
    # --------------
    def forward(self, input_tensors, *, channel_name_to_index, pre_computed=None, pretext_y, downstream_mask):
        """
        Forward pass

        Parameters
        ----------
        input_tensors : dict[str, torch.Tensor]
        channel_name_to_index
        pre_computed
        pretext_y : dict[str, torch.Tensor]
        downstream_mask : dict[str, torch.Tensor]
            This should be a dict of boolean tensors indicating which subjects will be used for the predictive modelling
            of the residual (True), and which should not be (False)
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The predictions on the first objective, and the predictions for the second objective (using the residual of
            task 1 as input). Note that although a masking was applied to the second task, no masking is applied to the
            first task
        """
        # ----------------
        # Input checks
        # ----------------
        if not (tuple(input_tensors) == tuple(pretext_y) == tuple(downstream_mask)):
            raise RuntimeError(f"The input tensors, the targets for the pretext, and the mask, must all have the same "
                               f"dataset ordering, but received {tuple(input_tensors)}, {tuple(pretext_y)}, and "
                               f"{tuple(downstream_mask)}")

        # ----------------
        # Forward passes
        # ----------------
        # Pass through RBP (and maybe normalisation)
        x = self._forward_rbp(input_tensors, channel_name_to_index=channel_name_to_index, pre_computed=pre_computed)

        # Pass through MTS module which should solve the first objective
        pretext_predictions = self._mts_module(x)

        # Make predictions from residuals (only the subset of subjects which we want to include, as indicated by mask)
        flattened_mask = torch.cat(list(downstream_mask.values()))
        flattened_pretext_y = torch.cat(list(pretext_y.values()), dim=0)
        residuals = pretext_predictions[flattened_mask] - flattened_pretext_y[flattened_mask]

        downstream_prediction = self._residual_model(residuals)

        return pretext_predictions, downstream_prediction

    # --------------
    # Methods for training and testing
    # --------------
    def train_model(self, *, train_loader, val_loader, test_loader, downstream_metrics, pretext_metrics,
                    variable_metrics, sub_group_splits, patience, num_epochs, device, channel_name_to_index,
                    mtl_strategy, downstream_criterion, pretext_criterion, pretext_prediction_activation_function,
                    pretext_target_scaler, target_scaler, downstream_prediction_activation_function, verbose,
                    verbose_variables, sub_groups_verbose, pretext_selection_metric, downstream_selection_metric):
        # --------------
        # History objects
        # --------------
        # Defining histories objects for downstream task (second objective)
        train_history = Histories(
            metrics=downstream_metrics, splits=sub_group_splits, variable_metrics=variable_metrics,
            expected_variables=train_loader.dataset.expected_variables)
        val_history = Histories(
            metrics=downstream_metrics, name="val", splits=sub_group_splits, variable_metrics=variable_metrics,
            expected_variables=val_loader.dataset.expected_variables)
        test_history = None if test_loader is None else Histories(
            metrics=downstream_metrics, name="test", splits=sub_group_splits, variable_metrics=variable_metrics,
            expected_variables=test_loader.dataset.expected_variables
        )

        # Defining histories objects for pretext task (first objective)
        pretext_train_history = Histories(
            metrics=pretext_metrics, name="pretext", splits=sub_group_splits,  variable_metrics=variable_metrics,
            expected_variables=train_loader.dataset.expected_variables)
        pretext_val_history = Histories(
            metrics=pretext_metrics, name="pretext_val", splits=sub_group_splits, variable_metrics=variable_metrics,
            expected_variables=val_loader.dataset.expected_variables)
        pretext_test_history = None if test_loader is None else Histories(
            metrics=pretext_metrics, name="pretext_test", splits=sub_group_splits, variable_metrics=variable_metrics,
            expected_variables=test_loader.dataset.expected_variables
        )

        # --------------
        # Fit model
        # --------------
        remaining_patience = patience
        pareto_frontier: List[Tuple[PScore, ...]] = []  # Tuples should have the num_elements = num_tasks
        pareto_model_states: List[Dict[str, torch.Tensor]] = []
        best_epochs: List[int] = []
        for epoch in range(num_epochs):
            # ----------------
            # Training
            # ----------------
            pbar_prefix = f"Train epoch {epoch + 1}/{num_epochs} "
            self._full_epoch_pass(
                loader=train_loader, apply_optimiser=True, channel_name_to_index=channel_name_to_index,
                pbar_prefix=pbar_prefix, device=device, mtl_strategy=mtl_strategy,
                downstream_criterion=downstream_criterion, pretext_criterion=pretext_criterion,
                pretext_prediction_activation_function=pretext_prediction_activation_function,
                downstream_prediction_activation_function=downstream_prediction_activation_function,
                pretext_target_scaler=pretext_target_scaler, downstream_target_scaler=target_scaler,
                pretext_history=pretext_train_history, downstream_history=train_history, verbose=verbose,
                verbose_variables=verbose_variables, sub_groups_verbose=sub_groups_verbose
            )

            # ----------------
            # Validation
            # ----------------
            pbar_prefix = f"Val epoch {epoch + 1}/{num_epochs} "
            with torch.no_grad():
                self._full_epoch_pass(
                    loader=val_loader, apply_optimiser=False, mtl_strategy=None, device=device,
                    pretext_history=pretext_val_history, downstream_history=val_history, downstream_criterion=None,
                    pretext_criterion=None, channel_name_to_index=channel_name_to_index, pbar_prefix=pbar_prefix,
                    pretext_prediction_activation_function=pretext_prediction_activation_function,
                    downstream_prediction_activation_function=downstream_prediction_activation_function,
                    pretext_target_scaler=pretext_target_scaler, downstream_target_scaler=target_scaler,
                    verbose=verbose, verbose_variables=verbose_variables, sub_groups_verbose=sub_groups_verbose
                )

            # ----------------
            # (Maybe) testing
            # ----------------
            if test_loader is not None:
                pbar_prefix = f"Test epoch {epoch + 1}/{num_epochs} "
                with torch.no_grad():
                    self._full_epoch_pass(
                        loader=test_loader, apply_optimiser=False, pretext_history=pretext_test_history,
                        downstream_history=test_history, mtl_strategy=mtl_strategy, device=device,
                        downstream_criterion=None, pretext_criterion=None, channel_name_to_index=channel_name_to_index,
                        pbar_prefix=pbar_prefix,
                        pretext_prediction_activation_function=pretext_prediction_activation_function,
                        downstream_prediction_activation_function=downstream_prediction_activation_function,
                        pretext_target_scaler=pretext_target_scaler, downstream_target_scaler=target_scaler,
                        verbose=verbose, verbose_variables=verbose_variables, sub_groups_verbose=sub_groups_verbose
                    )

            # ----------------
            # If this is a pareto-optimal solution, store it
            # ----------------
            curr_scores = (
                PScore(score=pretext_val_history.newest_metrics[pretext_selection_metric],
                       metric=pretext_selection_metric),
                PScore(score=val_history.newest_metrics[downstream_selection_metric],
                       metric=downstream_selection_metric),
            )
            _pareto_optimal, to_remove = is_pareto_optimal(new_scores=curr_scores, pareto_frontier=pareto_frontier)
            if _pareto_optimal:
                # Remove non-pareto optimal solutions
                for idx in reversed(to_remove):
                    del pareto_model_states[idx]
                    del pareto_frontier[idx]
                    del best_epochs[idx]

                # Store the model on the cpu. A memory optimistic solution...
                pareto_model_states.append(copy.deepcopy({k: v.cpu() for k, v in self.state_dict().items()}))

                # Store its performance scores and epoch
                pareto_frontier.append(curr_scores)
                best_epochs.append(epoch)

                # Patience revived
                if patience is not None:
                    remaining_patience = patience

            else:
                if patience is not None:
                    remaining_patience -= 1

            # ----------------
            # Break if we are out of patience
            # ----------------
            if patience is not None and remaining_patience == 0:
                break

        # Return the histories and best models
        histories = {"train": train_history, "val": val_history, "train_pretext": pretext_train_history,
                     "val_pretext": pretext_val_history}
        if test_history is not None:
            histories["test"] = test_history
            assert pretext_test_history is not None  # mypy complained
            histories["test_pretext"] = pretext_test_history
        return histories, tuple(pareto_model_states), tuple(best_epochs)

    def test_model(self, *, data_loader, downstream_metrics, pretext_metrics, sub_group_splits, variable_metrics,
                   device, channel_name_to_index, pretext_prediction_activation_function,
                   downstream_prediction_activation_function, pretext_target_scaler, target_scaler, verbose,
                   verbose_variables, sub_groups_verbose) -> Tuple[Histories, Histories]:
        """Remember to set model state dict"""
        # Defining histories objects
        downstream_history = Histories(
            metrics=downstream_metrics, name="test", splits=sub_group_splits, variable_metrics=variable_metrics,
            expected_variables=data_loader.dataset.expected_variables)
        pretext_test_history = Histories(
            metrics=pretext_metrics, name="pretext_test", splits=sub_group_splits, variable_metrics=variable_metrics,
            expected_variables=data_loader.dataset.expected_variables)

        # No gradients needed
        pbar_prefix = "Testing "
        with torch.no_grad():
            self._full_epoch_pass(
                loader=data_loader, apply_optimiser=False, pretext_history=pretext_test_history,
                downstream_history=downstream_history, mtl_strategy=None, device=device,
                downstream_criterion=None, pretext_criterion=None, channel_name_to_index=channel_name_to_index,
                pbar_prefix=pbar_prefix,
                pretext_prediction_activation_function=pretext_prediction_activation_function,
                downstream_prediction_activation_function=downstream_prediction_activation_function,
                pretext_target_scaler=pretext_target_scaler, downstream_target_scaler=target_scaler,
                verbose=verbose, verbose_variables=verbose_variables, sub_groups_verbose=sub_groups_verbose
            )

        return pretext_test_history, downstream_history

    def _full_epoch_pass(self, *, loader, apply_optimiser, pretext_criterion, downstream_criterion, pretext_history,
                         downstream_history, mtl_strategy: Optional[MultiTaskStrategy], device, channel_name_to_index,
                         pretext_target_scaler, pbar_prefix, pretext_prediction_activation_function,
                         downstream_target_scaler, downstream_prediction_activation_function, verbose,
                         verbose_variables, sub_groups_verbose):
        # Set training/evaluation mode
        if verify_type(apply_optimiser, bool):
            self.train()
        else:
            self.eval()

        # -------------
        # Run for a full epoch
        # -------------
        for (x, pre_computed, (pretext_y, pretext_mask), (downstream_y, downstream_mask),
             subject_indices) in progressbar(loader, redirect_stdout=True, prefix=pbar_prefix):
            # TODO: Should masks be required? If validation or test set?

            # Strip the dictionaries for 'ghost tensors'
            x = strip_tensors(x)
            pretext_y = strip_tensors(pretext_y)
            downstream_y = strip_tensors(downstream_y)  # todo: must skip input check for nans
            pretext_mask = tensor_dict_to_boolean(strip_tensors(pretext_mask))
            downstream_mask = tensor_dict_to_boolean(strip_tensors(downstream_mask))

            if isinstance(pre_computed, torch.Tensor) and torch.all(torch.isnan(pre_computed)):
                pre_computed = None
            else:
                pre_computed = [strip_tensors(pre_comp) for pre_comp in pre_computed]

            # Extract subjects and correct the ordering
            subjects = reorder_subjects(
                order=tuple(x.keys()), subjects=loader.dataset.get_subjects_from_indices(subject_indices))

            # Send data to the correct device
            x = tensor_dict_to_device(x, device=device)
            pretext_y = tensor_dict_to_device(pretext_y, device=device)
            downstream_y = flatten_targets(downstream_y).to(device)
            if pre_computed is not None:
                pre_computed = tuple(tensor_dict_to_device(pre_comp, device=device) for pre_comp in pre_computed)

            # Forward, loss, and maybe apply optimiser
            with maybe_no_grad(apply_optimiser):
                # Forward pass
                pretext_yhat, downstream_yhat = self(
                    x, pre_computed=pre_computed, pretext_y=pretext_y, channel_name_to_index=channel_name_to_index,
                    downstream_mask=downstream_mask)

                # -------------
                # Compute losses
                # -------------
                # Flatten out mask (if some subjects should not count on the pretext loss)
                flattened_pretext_mask = torch.cat(list(pretext_mask.values()), dim=0)
                flattened_pretext_y = torch.cat(list(pretext_y.values()), dim=0)
                flattened_downstream_mask = torch.cat(list(downstream_mask.values()))

                # Maybe compute gradients and apply a step
                if apply_optimiser:
                    assert pretext_criterion is not None
                    assert downstream_criterion is not None
                    assert mtl_strategy is not None

                    loss_1 = pretext_criterion(pretext_yhat[flattened_pretext_mask],
                                               flattened_pretext_y[flattened_pretext_mask])
                    loss_2 = downstream_criterion(downstream_yhat, downstream_y[flattened_downstream_mask])

                    mtl_strategy.zero_grad()
                    mtl_strategy.backward(losses=(loss_1, loss_2))
                    mtl_strategy.step()

            # Update history objects. Not masking pretext because it could be interesting to see for the downstream
            # subjects too. Have to mask on the downstream because targets are not expected to be available for the
            # masked ones
            self._updated_history_object(
                history=pretext_history, subjects=subjects, output=pretext_yhat, y=flattened_pretext_y,
                target_scaler=pretext_target_scaler,
                prediction_activation_function=pretext_prediction_activation_function)
            _downstream_subjects = tuple(subject for subject, mask in zip(subjects, flattened_downstream_mask) if mask)
            self._updated_history_object(
                history=downstream_history, subjects=_downstream_subjects, output=downstream_yhat,
                y=downstream_y[flattened_downstream_mask], target_scaler=downstream_target_scaler,
                prediction_activation_function=downstream_prediction_activation_function)

        # Finalise epoch for history objects. 'subjects_info' is no longer maintained
        pretext_history.on_epoch_end(verbose=verbose, verbose_sub_groups=sub_groups_verbose,
                                     verbose_variables=verbose_variables)
        downstream_history.on_epoch_end(verbose=verbose)
