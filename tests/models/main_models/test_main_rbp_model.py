from pathlib import Path
from tempfile import TemporaryDirectory

import numpy
import optuna
import pytest
import torch
from torch import optim
from torch.utils.data import DataLoader

from elecssl.data.data_generators.data_generator import RBPDataGenerator, MultiTaskRBPdataGenerator, create_mask
from elecssl.models.losses import CustomWeightedLoss
from elecssl.models.main_models.main_rbp_model import DownstreamRBPModel, MultiTaskRBPModel
from elecssl.models.metrics import NaNValueError
from elecssl.models.mtl_strategies.multi_task_strategies import get_mtl_strategy_type


# --------------
# Tests for downstream model
# --------------
def test_save_load_model_reproducibility(input_data, target_data, subjects, rbp_downstream_models, dummy_eeg_dataset,
                                         dummy_eeg_dataset_2):
    """Test if the model produces the same output after saving and loading, as before the model was saved"""
    channel_name_to_index = {dataset.name: dataset.channel_name_to_index()
                             for dataset in (dummy_eeg_dataset, dummy_eeg_dataset_2)}

    for i, model in enumerate(rbp_downstream_models):
        if i > 10:
            break

        assert isinstance(model, DownstreamRBPModel)
        if model.supports_precomputing:
            pre_computed = model.pre_compute(input_data)
            data_gen_pre_computed = model.pre_compute({name: torch.unsqueeze(inputs, dim=1)
                                                       for name, inputs in input_data.items()})
        else:
            pre_computed = None
            data_gen_pre_computed = None

        # -------------
        # Training preparations
        # -------------
        # Create dummy dataloaders. The dataloaders have en EEG epochs dimension too
        train_gen = RBPDataGenerator(
            data={name: numpy.expand_dims(inputs.numpy(), axis=1) for name, inputs in input_data.items()},
            targets=target_data, subjects=subjects, pre_computed=data_gen_pre_computed, subjects_info=dict(),
            expected_variables=None
        )
        train_loader = DataLoader(train_gen, batch_size=4, shuffle=True, collate_fn=train_gen.collate_fn)

        val_gen = RBPDataGenerator(
            data={name: numpy.expand_dims(inputs.numpy(), axis=1) for name, inputs in input_data.items()},
            targets=target_data, subjects=subjects, pre_computed=data_gen_pre_computed, subjects_info=dict(),
            expected_variables=None
        )
        val_loader = DataLoader(val_gen, batch_size=4, shuffle=True, collate_fn=val_gen.collate_fn)

        # Optimiser and criterion
        optimiser = optim.Adam(model.parameters(), lr=0.001, betas=(0.8, 0.9), eps=1e-8)
        criterion = CustomWeightedLoss("MSELoss", weighter=None, weighter_kwargs={}, loss_kwargs={})

        # Test if forward method is reproducible
        model.eval()
        try:
            # Do some training of the model and get the outputs
            model.train_model(
                train_loader=train_loader, val_loader=val_loader, test_loader=None,
                metrics="regression", main_metric="r2_score", num_epochs=2, criterion=criterion,
                optimiser=optimiser, device=torch.device("cpu"), channel_name_to_index=channel_name_to_index,
                prediction_activation_function=None, verbose=False, target_scaler=None, sub_group_splits=None,
                sub_groups_verbose=False, verbose_variables=False, variable_metrics=None, patience=3,
                use_progressbar=False
            )
            outputs_1 = model(input_data, pre_computed=pre_computed, channel_name_to_index=channel_name_to_index)

            # Save model and then load it
            with TemporaryDirectory() as d:
                model.save_model("my_test_model", path=Path(d))
                loaded_model = DownstreamRBPModel.load_model("my_test_model", path=Path(d))

            outputs_2 = loaded_model(input_data, pre_computed=pre_computed, channel_name_to_index=channel_name_to_index)
            # Test
            assert torch.equal(outputs_1, outputs_2), \
                (f"Model prediction were not the same before and after saving model ({i}) {model}\n"
                 f"{outputs_1 - outputs_2}")

        except (optuna.TrialPruned, NaNValueError, RuntimeError):
            continue


def test_forward_reproducibility(input_data, rbp_downstream_models, dummy_eeg_dataset, dummy_eeg_dataset_2):
    """Test if the model predictions are reproducible when model is set to evaluate mode"""
    channel_name_to_index = {dataset.name: dataset.channel_name_to_index()
                             for dataset in (dummy_eeg_dataset, dummy_eeg_dataset_2)}

    for model in rbp_downstream_models:
        assert isinstance(model, DownstreamRBPModel)
        if model.supports_precomputing:
            pre_computed = model.pre_compute(input_data)
        else:
            pre_computed = None

        # Test if forward method is reproducible
        model.eval()
        try:
            outputs_1 = model(input_data, pre_computed=pre_computed, channel_name_to_index=channel_name_to_index)
            outputs_2 = model(input_data, pre_computed=pre_computed, channel_name_to_index=channel_name_to_index)
        except (optuna.TrialPruned, RuntimeError):
            continue

        assert torch.equal(outputs_1, outputs_2), f"Model predictions were not reproducible for model {model}"


def test_forward_manipulation(input_data, rbp_downstream_models, dummy_eeg_dataset, dummy_eeg_dataset_2):
    """Test if manipulating the input of an EEG changes the predictions made on that and only that EEG"""
    channel_name_to_index = {dataset.name: dataset.channel_name_to_index()
                             for dataset in (dummy_eeg_dataset, dummy_eeg_dataset_2)}

    for i, model in enumerate(rbp_downstream_models):
        assert isinstance(model, DownstreamRBPModel)
        if model.supports_precomputing:
            pre_computed = model.pre_compute(input_data)
        else:
            pre_computed = None

        # --------------
        # Test
        # --------------
        model.eval()
        try:
            outputs_1 = model(input_data, pre_computed=pre_computed, channel_name_to_index=channel_name_to_index)

            # Make a change to the input data. The keys should ("DummyDataset", "DummyDataset2"), in that order
            new_input_data = {"DummyDataset": input_data["DummyDataset"].clone(),
                              "DummyDataset2": input_data["DummyDataset2"].clone()}
            new_input_data["DummyDataset2"][-3] = 10 * torch.rand(size=(new_input_data["DummyDataset2"][-3].size()))

            outputs_2 = model(new_input_data, pre_computed=pre_computed, channel_name_to_index=channel_name_to_index)
        except (optuna.TrialPruned, RuntimeError):
            continue

        assert not torch.equal(outputs_1[-3], outputs_2[-3]), \
            f"Model prediction was the same after changing the input in model ({i}) {model}\n{outputs_1-outputs_2}"
        assert torch.equal(outputs_1[:-3], outputs_2[:-3]), \
            (f"Changing the input of a subject lead to changes for other subjects for model ({i}) {model}\n"
             f"{outputs_1-outputs_2}")
        assert torch.equal(outputs_1[-2:], outputs_2[-2:]), \
            (f"Changing the input of a subject lead to changes for other subjects for model ({i}) {model}\n"
             f"{outputs_1-outputs_2}")


# --------------
# Tests for multi-task model
# --------------
@pytest.mark.parametrize("strategy_name,strategy_kwargs", (
        ("EqualWeighting", dict()),
        ("PCGrad", dict()),
        ("GradNorm", {"alpha": 1.5, "learning_rate": 0.01}),
        ("UncertaintyWeighting", dict()),
        ("MGDA", dict()))
)
def test_mtl_save_load_model_reproducibility(input_data, target_data, subjects, rbp_multi_task_models,
                                             dummy_eeg_dataset, dummy_eeg_dataset_2, pretext_target_data, strategy_name,
                                             strategy_kwargs):
    """Test if the model produces the same output after saving and loading, as before the model was saved"""
    channel_name_to_index = {dataset.name: dataset.channel_name_to_index()
                             for dataset in (dummy_eeg_dataset, dummy_eeg_dataset_2)}

    for i, model in enumerate(rbp_multi_task_models):
        assert isinstance(model, MultiTaskRBPModel)
        if model.supports_precomputing:
            pre_computed = model.pre_compute(input_data)
            data_gen_pre_computed = model.pre_compute({name: torch.unsqueeze(inputs, dim=1)
                                                       for name, inputs in input_data.items()})
        else:
            pre_computed = None
            data_gen_pre_computed = None

        # -------------
        # Training preparations
        # -------------
        # Create dummy dataloaders. The dataloaders have en EEG epochs dimension too
        data = {name: numpy.expand_dims(inputs.numpy(), axis=1) for name, inputs in input_data.items()}
        train_gen = MultiTaskRBPdataGenerator(
            data=data, downstream_targets=target_data, subjects=subjects, pre_computed=data_gen_pre_computed,
            subjects_info=dict(), expected_variables=None, pretext_targets=pretext_target_data, downstream_mask=None,
            pretext_mask=None
        )
        train_loader = DataLoader(train_gen, batch_size=4, shuffle=True, collate_fn=train_gen.collate_fn)

        data = {name: numpy.expand_dims(inputs.numpy(), axis=1) for name, inputs in input_data.items()}
        val_gen = MultiTaskRBPdataGenerator(
            data=data, downstream_targets=target_data, subjects=subjects, pre_computed=data_gen_pre_computed,
            subjects_info=dict(), expected_variables=None, pretext_targets=pretext_target_data, downstream_mask=None,
            pretext_mask=None
        )
        val_loader = DataLoader(val_gen, batch_size=4, shuffle=True, collate_fn=val_gen.collate_fn)

        # Optimiser and criterion
        optimiser = optim.Adam(model.parameters(), lr=0.001, betas=(0.8, 0.9), eps=1e-8)
        strategy = get_mtl_strategy_type(strategy_name)(optimiser=optimiser, model=model, **strategy_kwargs)
        pretext_criterion = CustomWeightedLoss("MSELoss", weighter=None, weighter_kwargs={}, loss_kwargs={})
        downstream_criterion = CustomWeightedLoss("MSELoss", weighter=None, weighter_kwargs={}, loss_kwargs={})

        # Test if forward method is reproducible
        downstream_mask = create_mask(sample_sizes={n: d.shape[0] for n, d in input_data.items()},
                                      to_include=input_data.keys())
        downstream_mask_tensor = {n: torch.tensor(d, dtype=torch.bool) for n, d in downstream_mask.items()}
        model.eval()
        try:
            # Do some training of the model and get the outputs
            model.train_model(
                train_loader=train_loader, val_loader=val_loader, test_loader=None,
                downstream_metrics="regression", downstream_selection_metric="r2_score", num_epochs=2,
                downstream_criterion=downstream_criterion, mtl_strategy=strategy, device=torch.device("cpu"),
                channel_name_to_index=channel_name_to_index, downstream_prediction_activation_function=None,
                verbose=False, target_scaler=None, sub_group_splits=None, pretext_target_scaler=None,
                pretext_prediction_activation_function=None, pretext_metrics="regression",
                pretext_criterion=pretext_criterion, pretext_selection_metric="r2_score",
                sub_groups_verbose=False, verbose_variables=False, variable_metrics=None, patience=3,
                use_progressbar=False
            )
            outputs_1 = model(input_data, pre_computed=pre_computed, channel_name_to_index=channel_name_to_index,
                              pretext_y=pretext_target_data, downstream_mask=downstream_mask_tensor)
            pretext_predictions_1, downstream_prediction_1 = outputs_1

            # Save model and then load it
            with TemporaryDirectory() as d:
                model.save_model("my_test_model", path=Path(d))
                loaded_model = DownstreamRBPModel.load_model("my_test_model", path=Path(d))

            outputs_2 = loaded_model(input_data, pre_computed=pre_computed, channel_name_to_index=channel_name_to_index,
                                     pretext_y=pretext_target_data, downstream_mask=downstream_mask_tensor)
            pretext_predictions_2, downstream_prediction_2 = outputs_2

            # Test
            assert torch.equal(pretext_predictions_1, pretext_predictions_2), \
                (f"Model pretext prediction were not the same before and after saving model ({i}) {model}\n"
                 f"{pretext_predictions_1 - pretext_predictions_2}")
            assert torch.equal(downstream_prediction_1, downstream_prediction_2), \
                (f"Model downstream prediction were not the same before and after saving model ({i}) {model}\n"
                 f"{downstream_prediction_1 - downstream_prediction_2}")

        except (optuna.TrialPruned, NaNValueError, RuntimeError):
            continue


def test_mtl_forward_reproducibility(input_data, rbp_multi_task_models, dummy_eeg_dataset, dummy_eeg_dataset_2,
                                     pretext_target_data):
    """Test if the model predictions are reproducible when model is set to evaluate mode"""
    channel_name_to_index = {dataset.name: dataset.channel_name_to_index()
                             for dataset in (dummy_eeg_dataset, dummy_eeg_dataset_2)}

    for i, model in enumerate(rbp_multi_task_models):
        assert isinstance(model, MultiTaskRBPModel)
        if model.supports_precomputing:
            pre_computed = model.pre_compute(input_data)
        else:
            pre_computed = None

        # Test if forward method is reproducible
        downstream_mask = create_mask(sample_sizes={n: d.shape[0] for n, d in input_data.items()},
                                      to_include=input_data.keys())
        downstream_mask_tensor = {n: torch.tensor(d, dtype=torch.bool) for n, d in downstream_mask.items()}
        model.eval()
        try:
            pretext_predictions_1, downstream_prediction_1 = model(
                input_data, pre_computed=pre_computed, channel_name_to_index=channel_name_to_index,
                pretext_y=pretext_target_data, downstream_mask=downstream_mask_tensor)
            pretext_predictions_2, downstream_prediction_2 = model(
                input_data, pre_computed=pre_computed, channel_name_to_index=channel_name_to_index,
                pretext_y=pretext_target_data, downstream_mask=downstream_mask_tensor)
        except (optuna.TrialPruned, RuntimeError):
            continue

        assert torch.equal(pretext_predictions_1, pretext_predictions_2), \
            (f"Model pretext predictions were not reproducible for model ({i}) {model}\n"
             f"{pretext_predictions_1 - pretext_predictions_2}")
        assert torch.equal(downstream_prediction_1, downstream_prediction_2), \
            (f"Model downstream predictions were not reproducible for model ({i}) {model}\n"
             f"{downstream_prediction_1 - downstream_prediction_2}")


def test_mtl_forward_manipulation(input_data, rbp_multi_task_models, dummy_eeg_dataset, dummy_eeg_dataset_2,
                                  pretext_target_data):
    """Test if manipulating the input of an EEG changes the predictions made on that and only that EEG"""
    channel_name_to_index = {dataset.name: dataset.channel_name_to_index()
                             for dataset in (dummy_eeg_dataset, dummy_eeg_dataset_2)}

    for i, model in enumerate(rbp_multi_task_models):
        assert isinstance(model, MultiTaskRBPModel)
        if model.supports_precomputing:
            pre_computed = model.pre_compute(input_data)
        else:
            pre_computed = None

        # --------------
        # Test
        # --------------
        downstream_mask = create_mask(sample_sizes={n: d.shape[0] for n, d in input_data.items()},
                                      to_include=input_data.keys())
        downstream_mask_tensor = {n: torch.tensor(d, dtype=torch.bool) for n, d in downstream_mask.items()}
        model.eval()
        try:
            pretext_predictions_1, downstream_prediction_1 = model(
                input_data, pre_computed=pre_computed, channel_name_to_index=channel_name_to_index,
                pretext_y=pretext_target_data, downstream_mask=downstream_mask_tensor)

            # Make a change to the input data. The keys should ("DummyDataset", "DummyDataset2"), in that order
            new_input_data = {"DummyDataset": input_data["DummyDataset"].clone(),
                              "DummyDataset2": input_data["DummyDataset2"].clone()}
            new_input_data["DummyDataset2"][-3] = 10 * torch.rand(size=(new_input_data["DummyDataset2"][-3].size()))

            pretext_predictions_2, downstream_prediction_2 = model(
                new_input_data, pre_computed=pre_computed, channel_name_to_index=channel_name_to_index,
                pretext_y=pretext_target_data, downstream_mask=downstream_mask_tensor)
        except (optuna.TrialPruned, RuntimeError):
            continue

        # Pretext predictions
        assert not torch.equal(pretext_predictions_1[-3], pretext_predictions_2[-3]), \
            (f"Model pretext prediction was the same after changing the input in model ({i}) {model}\n"
             f"{pretext_predictions_1-pretext_predictions_2}")
        assert torch.equal(pretext_predictions_1[:-3], pretext_predictions_2[:-3]), \
            (f"Changing the input of a subject lead to changes for other subjects for model ({i}) {model}\n"
             f"{pretext_predictions_1-pretext_predictions_2}")
        assert torch.equal(pretext_predictions_1[-2:], pretext_predictions_2[-2:]), \
            (f"Changing the input of a subject lead to changes for other subjects for model ({i}) {model}\n"
             f"{pretext_predictions_1-pretext_predictions_2}")

        # Downstream predictions
        assert not torch.equal(downstream_prediction_1[-3], downstream_prediction_2[-3]), \
            (f"Model downstream prediction was the same after changing the input in model ({i}) {model}\n"
             f"{downstream_prediction_1 - downstream_prediction_2}")
        assert torch.equal(downstream_prediction_1[:-3], downstream_prediction_2[:-3]), \
            (f"Changing the input of a subject lead to changes for other subjects for model ({i}) {model}\n"
             f"{downstream_prediction_1 - downstream_prediction_2}")
        assert torch.equal(downstream_prediction_1[-2:], downstream_prediction_2[-2:]), \
            (f"Changing the input of a subject lead to changes for other subjects for model ({i}) {model}\n"
             f"{downstream_prediction_1 - downstream_prediction_2}")
