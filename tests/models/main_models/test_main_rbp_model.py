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
        outputs_1 = model(input_data, pre_computed=pre_computed, channel_name_to_index=channel_name_to_index,
                          use_domain_discriminator=False)
        outputs_2 = model(input_data, pre_computed=pre_computed, channel_name_to_index=channel_name_to_index,
                          use_domain_discriminator=False)
        assert torch.equal(outputs_1, outputs_2), f"Model predictions were not reproducible for model {model}"


@pytest.mark.skip(reason="The test has not been implemented yet...")
def test_forward_manipulation():
    """Test if manipulating the input of an EEG changes the predictions made on that and only that EEG"""
    assert False
