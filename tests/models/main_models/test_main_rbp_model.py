from pathlib import Path
from tempfile import TemporaryDirectory

import numpy
import optuna
import torch
from torch import optim
from torch.utils.data import DataLoader

from elecssl.data.data_generators.data_generator import RBPDataGenerator
from elecssl.models.losses import CustomWeightedLoss
from elecssl.models.main_models.main_rbp_model import MainRBPModel


def test_save_load_model_reproducibility(input_data, target_data, subjects, rbp_main_models, dummy_eeg_dataset,
                                         dummy_eeg_dataset_2):
    """Test if the model produces the same output after saving and loading, as before the model was saved"""
    channel_name_to_index = {dataset.name: dataset.channel_name_to_index()
                             for dataset in (dummy_eeg_dataset, dummy_eeg_dataset_2)}

    for i, model in enumerate(rbp_main_models):
        if i > 10:
            break

        assert isinstance(model, MainRBPModel)
        if model.supports_precomputing:
            pre_computed = model.pre_compute({name: torch.unsqueeze(inputs, dim=1)
                                              for name, inputs in input_data.items()})
        else:
            pre_computed = None

        # -------------
        # Training preparations
        # -------------
        # Create dummy dataloaders. The dataloaders have en EEG epochs dimension too
        train_loader = DataLoader(
            RBPDataGenerator(data={name: numpy.expand_dims(inputs.numpy(), axis=1) for name, inputs in input_data.items()},
                             targets=target_data, subjects=subjects, pre_computed=pre_computed, subjects_info=dict(),
                             expected_variables=None),
            batch_size=4, shuffle=True
        )
        val_loader = DataLoader(
            RBPDataGenerator(data={name: numpy.expand_dims(inputs.numpy(), axis=1) for name, inputs in input_data.items()},
                             targets=target_data, subjects=subjects, pre_computed=pre_computed, subjects_info=dict(),
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
                method="downstream_training", train_loader=train_loader, val_loader=val_loader, test_loader=None,
                metrics="regression", main_metric="r2_score", num_epochs=2, classifier_criterion=criterion,
                optimiser=optimiser, device=torch.device("cpu"), channel_name_to_index=channel_name_to_index,
                prediction_activation_function=None, verbose=False, target_scaler=None, sub_group_splits=None,
                sub_groups_verbose=False, verbose_variables=False, variable_metrics=None
            )
            outputs_1 = model(input_data, pre_computed=pre_computed, channel_name_to_index=channel_name_to_index,
                              use_domain_discriminator=False)

            # Save model and then load it
            with TemporaryDirectory() as d:
                model.save_model("my_test_model", path=Path(d))
                loaded_model = MainRBPModel.load_model("my_test_model", path=Path(d))

            outputs_2 = loaded_model(input_data, pre_computed=pre_computed, channel_name_to_index=channel_name_to_index,
                                     use_domain_discriminator=False)

            # Test
            assert torch.equal(outputs_1, outputs_2), \
                (f"Model prediction were not the same before and after saving model ({i}) {model}\n"
                 f"{outputs_1 - outputs_2}")

        except optuna.TrialPruned:
            continue


def test_forward_reproducibility(input_data, rbp_main_models, dummy_eeg_dataset, dummy_eeg_dataset_2):
    """Test if the model predictions are reproducible when model is set to evaluate mode"""
    channel_name_to_index = {dataset.name: dataset.channel_name_to_index()
                             for dataset in (dummy_eeg_dataset, dummy_eeg_dataset_2)}

    for model in rbp_main_models:
        assert isinstance(model, MainRBPModel)
        if model.supports_precomputing:
            pre_computed = model.pre_compute(input_data)
        else:
            pre_computed = None

        # Test if forward method is reproducible
        model.eval()
        try:
            outputs_1 = model(input_data, pre_computed=pre_computed, channel_name_to_index=channel_name_to_index,
                              use_domain_discriminator=False)
            outputs_2 = model(input_data, pre_computed=pre_computed, channel_name_to_index=channel_name_to_index,
                              use_domain_discriminator=False)
        except optuna.TrialPruned:
            continue

        assert torch.equal(outputs_1, outputs_2), f"Model predictions were not reproducible for model {model}"


def test_forward_manipulation(input_data, rbp_main_models, dummy_eeg_dataset, dummy_eeg_dataset_2):
    """Test if manipulating the input of an EEG changes the predictions made on that and only that EEG"""
    channel_name_to_index = {dataset.name: dataset.channel_name_to_index()
                             for dataset in (dummy_eeg_dataset, dummy_eeg_dataset_2)}

    for i, model in enumerate(rbp_main_models):
        assert isinstance(model, MainRBPModel)
        if model.supports_precomputing:
            pre_computed = model.pre_compute(input_data)
        else:
            pre_computed = None

        # --------------
        # Test
        # --------------
        model.eval()
        try:
            outputs_1 = model(input_data, pre_computed=pre_computed, channel_name_to_index=channel_name_to_index,
                              use_domain_discriminator=False)

            # Make a change to the input data. The keys should ("DummyDataset", "DummyDataset2"), in that order
            new_input_data = {"DummyDataset": input_data["DummyDataset"].clone(),
                              "DummyDataset2": input_data["DummyDataset2"].clone()}
            new_input_data["DummyDataset2"][-3] = torch.rand(size=(new_input_data["DummyDataset2"][-3].size()))

            outputs_2 = model(new_input_data, pre_computed=pre_computed, channel_name_to_index=channel_name_to_index,
                              use_domain_discriminator=False)
        except optuna.TrialPruned:  # Usually raised instead of _LinAlgError for GREEN
            continue

        assert not torch.equal(outputs_1[-3], outputs_2[-3]), \
            f"Model prediction was the same after changing the input in model ({i}) {model}\n{outputs_1-outputs_2}"
        assert torch.equal(outputs_1[:-3], outputs_2[:-3]), \
            (f"Changing the input of a subject lead to changes for other subjects for model ({i}) {model}\n"
             f"{outputs_1-outputs_2}")
        assert torch.equal(outputs_1[-2:], outputs_2[-2:]), \
            (f"Changing the input of a subject lead to changes for other subjects for model ({i}) {model}\n"
             f"{outputs_1-outputs_2}")
