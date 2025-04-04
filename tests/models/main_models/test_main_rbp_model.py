import optuna
import pytest
import torch

from elecssl.models.main_models.main_rbp_model import MainRBPModel


@pytest.mark.skip(reason="The test has not been implemented yet...")
def test_save_load_model_reproducibility():
    """Test if the model produces the same output after saving and loading, as before the model was saved"""
    assert False


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
