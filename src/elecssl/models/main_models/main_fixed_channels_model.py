import abc
import copy
from typing import Optional, Iterator, Dict, Tuple, Union, List

import torch
from progressbar import progressbar
from torch import optim
from torch.nn import Parameter

from elecssl.data.data_generators.data_generator import strip_tensors
from elecssl.models.domain_adaptation.cmmn import ConvMMN
from elecssl.models.domain_adaptation.domain_discriminators.getter import get_domain_discriminator
from elecssl.models.losses import CustomWeightedLoss
from elecssl.models.main_models.main_base_class import MainModuleBase, reorder_subjects
from elecssl.models.metrics import Histories, is_improved_model, PScore, is_pareto_optimal
from elecssl.models.mtl_strategies.multi_task_strategies import MultiTaskStrategy
from elecssl.models.mtl_strategies.residual_modules import ResidualHead
from elecssl.models.mts_modules.getter import get_mts_module
from elecssl.models.utils import ReverseLayerF, tensor_dict_to_device, flatten_targets, verify_type, \
    tensor_dict_to_boolean, maybe_no_grad


# ----------------
# Convenient decorators
# ----------------
def train_method(func):
    setattr(func, "_is_train_method", True)
    return func


# ----------------
# Classes
# ----------------
class MainFixedChannelsModelBase(MainModuleBase, abc.ABC):
    """
    Main model when the number of input channels is fixed

    Examples
    --------
    >>> my_mts_kwargs = {"in_channels": 19, "num_classes": 7, "num_res_blocks": 1, "cnn_units": 32}
    >>> my_dd_kwargs = {"hidden_units": (8, 4), "num_classes": 3, "activation_function": "relu"}
    >>> my_cmmn_kwargs = {"kernel_size": 128}
    >>> MainFixedChannelsModel("InceptionNetwork", mts_module_kwargs=my_mts_kwargs, domain_discriminator="FCModule",
    ...                        domain_discriminator_kwargs=my_dd_kwargs, use_cmmn_layer=True,
    ...                        cmmn_kwargs=my_cmmn_kwargs, normalise=True)
    MainFixedChannelsModel(
      (_mts_module): InceptionNetwork(
        (_inception_modules): ModuleList(
          (0): _InceptionModule(
            (_input_conv): Conv1d(19, 32, kernel_size=(1,), stride=(1,), padding=same, bias=False)
            (_conv_list): ModuleList(
              (0): Conv1d(32, 32, kernel_size=(40,), stride=(1,), padding=same, bias=False)
              (1): Conv1d(32, 32, kernel_size=(20,), stride=(1,), padding=same, bias=False)
              (2): Conv1d(32, 32, kernel_size=(10,), stride=(1,), padding=same, bias=False)
            )
            (_max_pool): MaxPool1d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
            (_conv_after_max_pool): Conv1d(19, 32, kernel_size=(1,), stride=(1,), padding=same, bias=False)
            (_batch_norm): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1-2): 2 x _InceptionModule(
            (_input_conv): Conv1d(128, 32, kernel_size=(1,), stride=(1,), padding=same, bias=False)
            (_conv_list): ModuleList(
              (0): Conv1d(32, 32, kernel_size=(40,), stride=(1,), padding=same, bias=False)
              (1): Conv1d(32, 32, kernel_size=(20,), stride=(1,), padding=same, bias=False)
              (2): Conv1d(32, 32, kernel_size=(10,), stride=(1,), padding=same, bias=False)
            )
            (_max_pool): MaxPool1d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
            (_conv_after_max_pool): Conv1d(128, 32, kernel_size=(1,), stride=(1,), padding=same, bias=False)
            (_batch_norm): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (_shortcut_layers): ModuleList(
          (0): _ShortcutLayer(
            (_conv): Conv1d(19, 128, kernel_size=(1,), stride=(1,), padding=same)
            (_batch_norm): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (_fc_layer): Linear(in_features=128, out_features=7, bias=True)
      )
      (_domain_discriminator): FCModule(
        (_model): ModuleList(
          (0): Linear(in_features=128, out_features=8, bias=True)
          (1): ReLU()
          (2): Linear(in_features=8, out_features=4, bias=True)
          (3): ReLU()
          (4): Linear(in_features=4, out_features=3, bias=True)
        )
      )
    )
    """

    def __init__(self, *, mts_module, mts_module_kwargs, use_cmmn_layer, cmmn_kwargs, normalise):
        """
        Initialise

        Parameters
        ----------
        mts_module : str
        mts_module_kwargs : dict[str, typing.Any]
        use_cmmn_layer : bool
        cmmn_kwargs : dict[str, typing.Any] | None
        normalise : bool
        """
        super().__init__()

        # ----------------
        # (Maybe) create CMMN layer
        # ----------------
        self._cmmn_layer = None if not use_cmmn_layer else ConvMMN(**cmmn_kwargs)

        # ----------------
        # Create MTS module
        # ----------------
        self._normalise = normalise
        self._mts_module = get_mts_module(mts_module_name=mts_module, **mts_module_kwargs)

    # ---------------
    # Methods for forward propagation
    # ---------------
    def _first_forward(self, x):
        # Maybe run CMMN
        if self._cmmn_layer is not None:
            x = self._cmmn_layer(x)

        # Concatenate all tensors. This should be possible, as this class should ony be used with a fixed number
        # of input channels
        if isinstance(x, dict):
            x = torch.cat(list(x.values()), dim=0)

        # Normalise
        assert isinstance(x, torch.Tensor)
        if self._normalise:
            x = (x - torch.mean(x, dim=-1, keepdim=True)) / (torch.std(x, dim=-1, keepdim=True) + 1e-8)

        return x

    # ----------------
    # Methods for fitting CMMN layer
    # ----------------
    def fit_psd_barycenters(self, data, sampling_freq):
        if self._cmmn_layer is None:
            raise RuntimeError("Cannot fit PSD barycenters of the CMMN layers, when none is used")

        self._cmmn_layer.fit_psd_barycenters(data=data, sampling_freq=sampling_freq)

    def fit_monge_filters(self, data):
        if self._cmmn_layer is None:
            raise RuntimeError("Cannot fit monge filters of the CMMN layers, when none is used")

        self._cmmn_layer.fit_monge_filters(data=data, is_psds=False)

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

    # ---------------
    # Properties
    # ---------------
    @property
    def has_cmmn_layer(self) -> bool:  # inconsistent property name with respect to the RBP version
        """Boolean indicating if the model uses a CMMN layer (True) or not (False)"""
        return self._cmmn_layer is not None

    @property
    def cmmn_fitted_channel_systems(self):
        if not self.has_cmmn_layer:
            raise RuntimeError(f"{type(self).__name__} has no property 'cmmn_fitted_channel_systems' when no CMMN "
                               f"layer is used")
        return self._cmmn_layer.fitted_monge_filters  # type: ignore[union-attr]


class DownstreamFixedChannelsModel(MainFixedChannelsModelBase):

    @classmethod
    def from_config(cls, mts_config, cmmn_config):
        """
        Initialise from config file

        Parameters
        ----------
        mts_config : dict[str, typing.Any]
        cmmn_config : dict[str, typing.Any]
        """
        use_cmmn_layer = cmmn_config["use_cmmn_layer"]
        return cls(mts_module=mts_config["model"], mts_module_kwargs=mts_config["kwargs"],
                   use_cmmn_layer=cmmn_config["use_cmmn_layer"], normalise=mts_config["normalise"],
                   cmmn_kwargs=None if not use_cmmn_layer else cmmn_config["kwargs"])

    # --------------
    # Forward pass
    # --------------
    def forward(self, x: Dict[str, torch.Tensor]):
        x = self._first_forward(x)
        return self._mts_module(x)

    # --------------
    # Methods for training and testing
    # --------------
    def train_model(self, *, train_loader, val_loader, test_loader, metrics, main_metric, num_epochs,
                    classifier_criterion, optimiser, device, prediction_activation_function=None,
                    verbose=True, target_scaler, sub_group_splits, sub_groups_verbose, verbose_variables,
                    variable_metrics, patience):
        # Defining histories objects
        train_history = Histories(metrics=metrics, splits=sub_group_splits, variable_metrics=variable_metrics,
                                  expected_variables=train_loader.dataset.expected_variables)
        val_history = Histories(metrics=metrics, name="val", splits=sub_group_splits, variable_metrics=variable_metrics,
                                expected_variables=val_loader.dataset.expected_variables)
        test_history = None if test_loader is None else Histories(
            metrics=metrics, name="test", splits=sub_group_splits, variable_metrics=variable_metrics,
            expected_variables=test_loader.dataset.expected_variables
        )

        # ---------------
        # Fit model
        # ---------------
        remaining_patience = patience
        best_metrics = None
        best_model_state = {k: v.cpu() for k, v in self.state_dict().items()}
        best_epoch = 0
        for epoch in range(num_epochs):
            # ---------------
            # Training
            # ---------------
            pbar_prefix = f"Train epoch {epoch + 1}/{num_epochs} "
            self._full_epoch_pass(
                loader=train_loader, compute_loss=True, history=train_history, criterion=classifier_criterion,
                optimiser=optimiser, target_scaler=target_scaler, pbar_prefix=pbar_prefix, device=device,
                prediction_activation_function=prediction_activation_function, verbose=verbose,
                verbose_variables=verbose_variables, sub_groups_verbose=sub_groups_verbose

            )

            # ---------------
            # Validation
            # ---------------
            pbar_prefix = f"Val epoch {epoch + 1}/{num_epochs} "
            with torch.no_grad():
                self._full_epoch_pass(
                    loader=val_loader, compute_loss=False, optimiser=None, criterion=None, history=val_history,
                    target_scaler=target_scaler, pbar_prefix=pbar_prefix, device=device,
                    prediction_activation_function=prediction_activation_function, verbose=verbose,
                    verbose_variables=verbose_variables, sub_groups_verbose=sub_groups_verbose
                )

            # ----------------
            # (Maybe) testing
            # ----------------
            if test_loader is not None:
                pbar_prefix = f"Test epoch {epoch + 1}/{num_epochs} "
                with torch.no_grad():
                    self._full_epoch_pass(
                        loader=test_loader, compute_loss=False, optimiser=None, criterion=None, history=test_history,
                        target_scaler=target_scaler, pbar_prefix=pbar_prefix, device=device,
                        prediction_activation_function=prediction_activation_function, verbose=verbose,
                        verbose_variables=verbose_variables, sub_groups_verbose=sub_groups_verbose
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

    def test_model(self, *, data_loader, metrics, device, prediction_activation_function, verbose=True,
                   target_scaler, sub_group_splits, sub_groups_verbose, verbose_variables, variable_metrics):
        # Defining histories objects
        history = Histories(metrics=metrics, name="test", splits=sub_group_splits, variable_metrics=variable_metrics,
                            expected_variables=data_loader.dataset.expected_variables)

        # No gradients needed
        pbar_prefix = "Testing "
        with torch.no_grad():
            self._full_epoch_pass(
                loader=data_loader, compute_loss=False, optimiser=None, criterion=None, history=history,
                target_scaler=target_scaler, pbar_prefix=pbar_prefix, device=device,
                prediction_activation_function=prediction_activation_function, verbose=verbose,
                verbose_variables=verbose_variables, sub_groups_verbose=sub_groups_verbose
            )

        return history

    def _full_epoch_pass(self, *, loader, compute_loss, device, prediction_activation_function, history, target_scaler,
                         optimiser: Optional[optim.Optimizer], criterion: Optional[CustomWeightedLoss], pbar_prefix,
                         verbose, sub_groups_verbose, verbose_variables):
        # Set training/evaluation mode
        if verify_type(compute_loss, bool):
            self.train()
        else:
            self.eval()

        # -------------
        # Run for a full epoch
        # -------------
        for x, y, subject_indices in progressbar(loader, redirect_stdout=True, prefix=pbar_prefix):
            # Strip the dictionaries for 'ghost tensors'
            x = strip_tensors(x)
            y = strip_tensors(y)

            # Extract subjects and correct the ordering
            subjects = reorder_subjects(
                order=tuple(x.keys()), subjects=loader.dataset.get_subjects_from_indices(subject_indices))

            # Send data to the correct device
            x = tensor_dict_to_device(x, device=device)
            y = flatten_targets(y).to(device)

            # Forward pass
            output = self(x, use_domain_discriminator=False)

            # Maybe compute loss and apply optimiser
            if compute_loss:
                assert optimiser is not None, "When computing loss, an optimiser must be passed, but received None"
                assert criterion is not None, "When computing loss, a criterion must be passed, but received None"

                optimiser.zero_grad()
                loss = criterion(output, y, subjects=subjects)
                loss.backward()
                optimiser.step()

            # Update history object
            with torch.no_grad():
                y_pred = torch.clone(output)
                if prediction_activation_function is not None:
                    y_pred = prediction_activation_function(y_pred)

                # (Maybe) re-scale targets and predictions before computing metrics
                if target_scaler is not None:
                    y_pred = target_scaler.inv_transform(scaled_data=y_pred)
                    y = target_scaler.inv_transform(scaled_data=y)
                history.store_batch_evaluation(y_pred=y_pred, y_true=y, subjects=subjects)

        # Finalise epoch for the history object
        history.on_epoch_end(
            verbose=verbose, verbose_sub_groups=sub_groups_verbose, verbose_variables=verbose_variables,
            subjects_info=loader.dataset.subjects_info)


MainFixedChannelsModel = DownstreamFixedChannelsModel


class DomainDiscriminatorFixedChannelsModel(MainFixedChannelsModelBase):
    """
    Training with domain discriminator is no longer maintained

    The interested researcher can check out a former implementation at
    https://github.com/thomastveitstol/CrossDatasetLearningEEG/blob/master/src/cdl_eeg/models/main_models/
        main_fixed_channels_model.py
    """

    def __init__(self, *, mts_module, mts_module_kwargs, use_cmmn_layer, cmmn_kwargs, normalise, domain_discriminator,
                 domain_discriminator_kwargs):
        super().__init__(mts_module=mts_module, mts_module_kwargs=mts_module_kwargs, use_cmmn_layer=use_cmmn_layer,
                         cmmn_kwargs=cmmn_kwargs, normalise=normalise)

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
    def from_config(cls, mts_config, discriminator_config, cmmn_config):
        """
        Initialise from config file

        Parameters
        ----------
        mts_config : dict[str, typing.Any]
        discriminator_config : dict[str, typing.Any] | None
        cmmn_config : dict[str, typing.Any]
        """
        use_cmmn_layer = cmmn_config["use_cmmn_layer"]
        return cls(mts_module=mts_config["model"], mts_module_kwargs=mts_config["kwargs"],
                   domain_discriminator=None if discriminator_config is None else discriminator_config["name"],
                   domain_discriminator_kwargs=None if discriminator_config is None else discriminator_config["kwargs"],
                   use_cmmn_layer=cmmn_config["use_cmmn_layer"], normalise=mts_config["normalise"],
                   cmmn_kwargs=None if not use_cmmn_layer else cmmn_config["kwargs"])

    # --------------
    # Forward pass
    # --------------
    def forward(self, x: Dict[str, torch.Tensor]):
        # Pass through MTS module to extract latent features
        x = self._first_forward(x)
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
    def train_model(
            self, *args, **kwargs) -> Tuple[Dict[str, Histories], Tuple[Dict[str, torch.Tensor], ...], Tuple[int, ...]]:
        raise NotImplementedError("Training with domain discriminator is no longer maintained")

    def test_model(self, *args, **kwargs) -> Union[Histories, Tuple[Histories, ...]]:
        raise NotImplementedError("Training with domain discriminator is no longer maintained")


class MultiTaskFixedChannelsModel(MainFixedChannelsModelBase):
    """
    Multi-task learning model which assumes fixed number of channels. For multi-task learning, we want to (1) predict
    some variable, and (2) use the residual of the first loss to make predictions on some other task.

    For example, to (1) predict age, and (2) predict some pathological state from the brain age residual. Or as in this
    work, predict a feature from eyes open, and use the residual to predict something related to cognition
    """

    def __init__(self, *, mts_module, mts_module_kwargs, use_cmmn_layer, cmmn_kwargs, normalise):
        super().__init__(mts_module=mts_module, mts_module_kwargs=mts_module_kwargs, use_cmmn_layer=use_cmmn_layer,
                         cmmn_kwargs=cmmn_kwargs, normalise=normalise)

        # Module for making predictions from the residual
        self._residual_model = ResidualHead()

    @classmethod
    def from_config(cls, mts_config, cmmn_config):
        """
        Initialise from config file

        Parameters
        ----------
        mts_config : dict[str, typing.Any]
        cmmn_config : dict[str, typing.Any]
        """
        use_cmmn_layer = cmmn_config["use_cmmn_layer"]
        return cls(mts_module=mts_config["model"], mts_module_kwargs=mts_config["kwargs"],
                   use_cmmn_layer=cmmn_config["use_cmmn_layer"], normalise=mts_config["normalise"],
                   cmmn_kwargs=None if not use_cmmn_layer else cmmn_config["kwargs"])

    def shared_parameters(self) -> Iterator[Parameter]:
        for module in self.modules():
            if hasattr(module, "shared") and not module.shared:
                continue
            for params in module.parameters():
                yield params

    # --------------
    # Forward pass
    # --------------
    def forward(self, input_tensors, *, pretext_y, downstream_mask):
        """
        Forward pass

        Parameters
        ----------
        input_tensors : dict[str, torch.Tensor]
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
        x = self._first_forward(input_tensors)

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
    def train_model(self, *, train_loader, val_loader, test_loader, mtl_strategy, downstream_criterion,
                    pretext_criterion, device, target_scaler, pretext_target_scaler,
                    pretext_prediction_activation_function, downstream_prediction_activation_function,
                    downstream_metrics, pretext_metrics, pretext_selection_metric, downstream_selection_metric,
                    patience, num_epochs, variable_metrics, sub_group_splits, verbose, verbose_variables,
                    sub_groups_verbose):
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
            metrics=pretext_metrics, name="pretext", splits=sub_group_splits, variable_metrics=variable_metrics,
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
            # ---------------
            # Training
            # ---------------
            pbar_prefix = f"Train epoch {epoch + 1}/{num_epochs} "
            self._full_epoch_pass(
                loader=train_loader, apply_optimiser=True, device=device, mtl_strategy=mtl_strategy,
                downstream_criterion=downstream_criterion, pretext_criterion=pretext_criterion,
                pretext_prediction_activation_function=pretext_prediction_activation_function,
                downstream_prediction_activation_function=downstream_prediction_activation_function,
                pretext_target_scaler=pretext_target_scaler, downstream_target_scaler=target_scaler,
                pretext_history=pretext_train_history, downstream_history=train_history, verbose=verbose,
                verbose_variables=verbose_variables, sub_groups_verbose=sub_groups_verbose, pbar_prefix=pbar_prefix
            )

            # ----------------
            # Validation
            # ----------------
            pbar_prefix = f"Val epoch {epoch + 1}/{num_epochs} "
            with torch.no_grad():
                self._full_epoch_pass(
                    loader=val_loader, apply_optimiser=False, downstream_criterion=None, pretext_criterion=None,
                    mtl_strategy=None, pretext_history=pretext_val_history, downstream_history=val_history,
                    pretext_prediction_activation_function=pretext_prediction_activation_function,
                    downstream_prediction_activation_function=downstream_prediction_activation_function,
                    pretext_target_scaler=pretext_target_scaler, downstream_target_scaler=target_scaler, device=device,
                    verbose=verbose, verbose_variables=verbose_variables, sub_groups_verbose=sub_groups_verbose,
                    pbar_prefix=pbar_prefix
                )

            # ----------------
            # (Maybe) testing
            # ----------------
            if test_loader is not None:
                pbar_prefix = f"Test epoch {epoch + 1}/{num_epochs} "
                with torch.no_grad():
                    self._full_epoch_pass(
                        loader=test_loader, apply_optimiser=False, downstream_criterion=None, pretext_criterion=None,
                        mtl_strategy=None, pretext_history=pretext_test_history, downstream_history=test_history,
                        pretext_prediction_activation_function=pretext_prediction_activation_function,
                        downstream_prediction_activation_function=downstream_prediction_activation_function,
                        pretext_target_scaler=pretext_target_scaler, downstream_target_scaler=target_scaler,
                        device=device, verbose=verbose, verbose_variables=verbose_variables,
                        sub_groups_verbose=sub_groups_verbose, pbar_prefix=pbar_prefix
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
                   device, pretext_prediction_activation_function, downstream_prediction_activation_function,
                   pretext_target_scaler, target_scaler, verbose, verbose_variables,
                   sub_groups_verbose) -> Tuple[Histories, Histories]:
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
                loader=data_loader, apply_optimiser=False, downstream_criterion=None, pretext_criterion=None,
                mtl_strategy=None, pretext_history=pretext_test_history, downstream_history=downstream_history,
                pretext_prediction_activation_function=pretext_prediction_activation_function,
                downstream_prediction_activation_function=downstream_prediction_activation_function,
                pretext_target_scaler=pretext_target_scaler, downstream_target_scaler=target_scaler,
                device=device, verbose=verbose, verbose_variables=verbose_variables,
                sub_groups_verbose=sub_groups_verbose, pbar_prefix=pbar_prefix
            )

        return pretext_test_history, downstream_history

    def _full_epoch_pass(self, *, loader, apply_optimiser, pretext_criterion, downstream_criterion, pretext_history,
                         downstream_history, mtl_strategy: Optional[MultiTaskStrategy], device, pretext_target_scaler,
                         pbar_prefix, pretext_prediction_activation_function, downstream_target_scaler,
                         downstream_prediction_activation_function, verbose, verbose_variables, sub_groups_verbose):
        # Set training/evaluation mode
        if verify_type(apply_optimiser, bool):
            self.train()
        else:
            self.eval()

            # -------------
            # Run for a full epoch
            # -------------
            for (x, (pretext_y, pretext_mask), (downstream_y, downstream_mask),
                 subject_indices) in progressbar(loader, redirect_stdout=True, prefix=pbar_prefix):
                # TODO: Should masks be required? If validation or test set?

                # Strip the dictionaries for 'ghost tensors'
                x = strip_tensors(x)
                pretext_y = strip_tensors(pretext_y)
                downstream_y = strip_tensors(downstream_y)  # todo: must skip input check for nans
                pretext_mask = tensor_dict_to_boolean(strip_tensors(pretext_mask))
                downstream_mask = tensor_dict_to_boolean(strip_tensors(downstream_mask))

                # Extract subjects and correct the ordering
                subjects = reorder_subjects(
                    order=tuple(x.keys()), subjects=loader.dataset.get_subjects_from_indices(subject_indices))

                # Send data to the correct device
                x = tensor_dict_to_device(x, device=device)
                pretext_y = tensor_dict_to_device(pretext_y, device=device)
                downstream_y = flatten_targets(downstream_y).to(device)

                # Forward, loss, and maybe apply optimiser
                with maybe_no_grad(apply_optimiser):
                    # Forward pass
                    pretext_yhat, downstream_yhat = self(x, pretext_y=pretext_y, downstream_mask=downstream_mask)

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
                _downstream_subjects = tuple(
                    subject for subject, mask in zip(subjects, flattened_downstream_mask) if mask)
                self._updated_history_object(
                    history=downstream_history, subjects=_downstream_subjects, output=downstream_yhat,
                    y=downstream_y[flattened_downstream_mask], target_scaler=downstream_target_scaler,
                    prediction_activation_function=downstream_prediction_activation_function)

            # Finalise epoch for history objects. 'subjects_info' is no longer maintained
            pretext_history.on_epoch_end(verbose=verbose, verbose_sub_groups=sub_groups_verbose,
                                         verbose_variables=verbose_variables)
            downstream_history.on_epoch_end(verbose=verbose)
