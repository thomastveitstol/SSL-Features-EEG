import torch


def test_forward_output_shape(dummy_data, dummy_num_classes, models):
    """Test if the forward pass produce correct output shapes"""
    batch_size = dummy_data.size()[0]
    for model in models:
        output_shape = model(dummy_data).size()
        expected_shape = torch.Size([batch_size, dummy_num_classes])
        assert output_shape == expected_shape, (f"Model {model.__class__.__name__} output shape {output_shape} doesn't "
                                                f"match expected {expected_shape}")


def test_extract_latent_features(dummy_data, models):
    """Test if the extract_latent_features method works as expected. It should output a tensor and have match expected
    dimensionality"""
    batch_size = dummy_data.size()[0]
    for model in models:
        latent_features = model.extract_latent_features(dummy_data)

        # Check if the latent features are a tensor
        assert isinstance(latent_features, torch.Tensor), (f"Model {model.__class__.__name__} extract_latent_features "
                                                           f"did not return a tensor, but {type(latent_features)}.")

        # Check if the shape is correct
        latent_shape = latent_features.size()
        expected_shape = torch.Size([batch_size, model.latent_features_dim])
        assert latent_shape == expected_shape,(f"Model {model.__class__.__name__} latent features shape mismatch. "
                                               f"Expected {expected_shape}, but received {latent_shape}")


def test_classify_latent_features(dummy_data, models):
    """Test if the extract_latent_features followed by classify_latent_features is the same as just running forward
    method as usual"""
    for model in models:
        model.eval()
        expected_output = model(dummy_data)

        latent_features = model.extract_latent_features(dummy_data)
        actual_output = model.classify_latent_features(latent_features)

        # Check if the two methods are the same
        assert torch.equal(expected_output, actual_output), (f"Expected output was not the same as actual output for "
                                                             f"the model {type(model)}. Expected, actual: "
                                                             f"{expected_output}, {actual_output}")
