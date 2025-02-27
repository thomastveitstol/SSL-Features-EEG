def test_forward(rbp_modules, dummy_data_dicts_with_ch_systems):
    """Test that the forward method of RBP runs and that the output does not contain nan values. This was originally
    used as an attempt to find bugs, as I've received error saying that the tensor is NaN. However, after failing to
    reproduce this error with HP configurations that crashed (and scanning the web), it seems like it is a numerical
    problem with gradient blow up or bad learning rate policy and params
    (see https://stackoverflow.com/questions/33962226/common-causes-of-nans-during-training-of-neural-networks)"""
    data, channel_systems = dummy_data_dicts_with_ch_systems

    # Loop through all the RBP modules
    for model in rbp_modules:
        # Preparing before testing forward
        model.fit_channel_systems(tuple(channel_systems.values()))

        # (Maybe) precompute the features
        precomputed = model.pre_compute(data) if model.supports_precomputing else None

        # Run forward method
        channel_name_to_index = {name: ch_system.channel_name_to_index
                                 for name, ch_system in channel_systems.items()}
        outputs = model(data, channel_name_to_index=channel_name_to_index, pre_computed=precomputed)

        # Test that the output is valid
        for i, pooling_module_output in enumerate(outputs):
            assert not pooling_module_output.isnan().any(), f"NaN values were found in pooling module number {i}"
