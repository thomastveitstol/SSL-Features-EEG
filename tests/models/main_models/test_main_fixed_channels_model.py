from pathlib import Path
from tempfile import TemporaryDirectory

import numpy
import optuna
import torch
from torch import optim
from torch.utils.data import DataLoader

from elecssl.data.data_generators.data_generator import InterpolationDataGenerator
from elecssl.models.losses import CustomWeightedLoss
from elecssl.models.main_models.main_fixed_channels_model import DownstreamFixedChannelsModel
from elecssl.models.metrics import NaNValueError


# --------------
# Tests for downstream model
# --------------
def test_save_load_model_reproducibility(interpolation_downstream_models, interpolated_input_data, subjects,
                                         target_data):
    """Test if the model produces the same output after saving and loading, as before the model was saved"""
    for i, model in enumerate(interpolation_downstream_models):
        assert isinstance(model, DownstreamFixedChannelsModel)

        # -------------
        # Training preparations
        # -------------
        # Create dummy dataloaders. The dataloaders have en EEG epochs dimension too
        train_loader = DataLoader(
            InterpolationDataGenerator(data={name: numpy.expand_dims(inputs.numpy(), axis=1)
                                             for name, inputs in interpolated_input_data.items()},
                                       targets=target_data, subjects=subjects, subjects_info=dict(),
                                       expected_variables=None),
            batch_size=4, shuffle=True
        )
        val_loader = DataLoader(
            InterpolationDataGenerator(data={name: numpy.expand_dims(inputs.numpy(), axis=1)
                                             for name, inputs in interpolated_input_data.items()},
                                       targets=target_data, subjects=subjects, subjects_info=dict(),
                                       expected_variables=None),
            batch_size=4, shuffle=True
        )

        # Optimiser and criterion
        optimiser = optim.Adam(model.parameters(), lr=0.001, betas=(0.8, 0.9), eps=1e-8)
        criterion = CustomWeightedLoss("MSELoss", weighter=None, weighter_kwargs={}, loss_kwargs={})

        # Test if forward method is reproducible
        model.eval()
        try:
            # Do some training of the model and get the outputs
            model.train_model(
                train_loader=train_loader, val_loader=val_loader, test_loader=None,
                metrics="regression", main_metric="r2_score", num_epochs=2, classifier_criterion=criterion,
                optimiser=optimiser, device=torch.device("cpu"), patience=3,
                prediction_activation_function=None, verbose=False, target_scaler=None, sub_group_splits=None,
                sub_groups_verbose=False, verbose_variables=False, variable_metrics=None
            )
            outputs_1 = model(interpolated_input_data)

            # Save model and then load it
            with TemporaryDirectory() as d:
                model.save_model("my_test_model", path=Path(d))
                loaded_model = DownstreamFixedChannelsModel.load_model("my_test_model", path=Path(d))

            outputs_2 = loaded_model(interpolated_input_data)

            # Test
            assert torch.equal(outputs_1, outputs_2), \
                (f"Model prediction were not the same before and after saving model ({i}) {model}\n"
                 f"{outputs_1 - outputs_2}")

        except (optuna.TrialPruned, NaNValueError, RuntimeError):
            continue


def test_forward_reproducibility(interpolated_input_data, interpolation_downstream_models):
    """Test if the model predictions are reproducible when model is set to evaluate mode"""
    for i, model in enumerate(interpolation_downstream_models):
        assert isinstance(model, DownstreamFixedChannelsModel)

        # Test if forward method is reproducible (in eval mode. In train mode it is not expected)
        model.eval()
        try:
            outputs_1 = model(interpolated_input_data)
            outputs_2 = model(interpolated_input_data)
        except (optuna.TrialPruned, RuntimeError):
            continue

        assert torch.equal(outputs_1, outputs_2), \
            f"Model predictions were not reproducible for model ({i}) {model}\n{outputs_1 - outputs_2}"


def test_forward_manipulation(interpolated_input_data, interpolation_downstream_models):
    """Test if manipulating the input of an EEG changes the predictions made on that and only that EEG"""
    for i, model in enumerate(interpolation_downstream_models):
        assert isinstance(model, DownstreamFixedChannelsModel)

        # --------------
        # Test
        # --------------
        model.eval()
        try:
            outputs_1 = model(interpolated_input_data)

            # Make a change to the input data. The keys should ("DummyDataset", "DummyDataset2"), in that order
            new_input_data = {"DummyDataset": interpolated_input_data["DummyDataset"].clone(),
                              "DummyDataset2": interpolated_input_data["DummyDataset2"].clone()}
            new_input_data["DummyDataset2"][-3] = torch.rand(size=(new_input_data["DummyDataset2"][-3].size()))

            outputs_2 = model(new_input_data)
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
