import torch


def test_forward(green_models, dummy_data, dummy_num_classes):
    """Test that the forward method runs and that the output shape is as expected"""
    batch_size = dummy_data.size()[0]
    expected_shape = torch.Size([batch_size, dummy_num_classes])
    for model in green_models:
        # Verify that it runs
        output_shape = model(dummy_data).size()

        # Verify output shape
        assert output_shape == expected_shape, f"Output shape ({output_shape}) was not as expected ({expected_shape})"
