import torch
from torch.utils.data import DataLoader

from elecssl.data.data_generators.data_generator import MultiTaskRBPdataGenerator, strip_tensors, RBPDataGenerator, \
    InterpolationDataGenerator, MultiTaskInterpolationDataGenerator


# ---------------
# Tests for RBP-based data generators
# ---------------
def test_normal_rbp_data_gen(dummy_input_data, dummy_targets, dummy_pseudo_targets, dummy_expected_variables,
                             dummy_dataset_subjects, dummy_pretext_mask, dummy_downstream_mask, dummy_subjects_info,
                             dummy_pretext_datasets, dummy_downstream_datasets):
    """Test if RBPDataGenerator can be used with DataLoader, and yield expected outcomes"""
    all_datasets = set(dummy_dataset_subjects)

    # Create data loader
    loader = DataLoader(
        RBPDataGenerator(
            data=dummy_input_data, targets=dummy_targets, expected_variables=dummy_expected_variables,
            subjects=dummy_dataset_subjects, pre_computed=None,subjects_info=dummy_subjects_info
        )
    )

    # Loop over it
    for x, pre_computed, y, subject_indices in loader:
        # Check that the dataset order is consistent
        assert tuple(x) == tuple(y), f"Inconsistent dataset ordering: {tuple(x)} and {tuple(y)}"

        # Check that all datasets are present
        assert set(x) == all_datasets, f"Expected and actual datasets were difference: {set(x)}, {all_datasets}"

        # ------------
        # Tests before stripping the dictionaries for 'ghost tensors'
        # ------------
        # Sample size test
        _x_samples_sizes = {name: tensor.size()[0] for name, tensor in x.items()}
        _y_samples_sizes = {name: tensor.size()[0] for name, tensor in y.items()}
        assert _x_samples_sizes == _y_samples_sizes, \
            (f"Input data and target data does not have consistent sample sizes: {_x_samples_sizes} and "
             f"{_y_samples_sizes}")

        # EEG data
        for dataset_name, x_tensor in x.items():
            assert isinstance(x_tensor, torch.Tensor)
            assert x_tensor.size()[-2:] == torch.Size(dummy_input_data[dataset_name].shape[-2:])
            assert (((x_tensor >= 100) & (x_tensor <= 150)) |((x_tensor > -1.1) & (x_tensor < -0.9))).all()

        # Targets
        for dataset_name, targets in y.items():
            assert isinstance(targets, torch.Tensor)
            assert targets.ndim == 2
            assert targets.size()[-1] == 1

            assert (((targets >= 0) & (targets <= 1)) | ((targets > -1.1) & (targets < -0.9))).all()

        # ------------
        # Tests after stripping the dictionaries for 'ghost tensors'
        # ------------
        x = strip_tensors(x)
        y = strip_tensors(y)

        # Sample size test
        _x_samples_sizes = {name: tensor.size()[0] for name, tensor in x.items()}
        _y_samples_sizes = {name: tensor.size()[0] for name, tensor in y.items()}
        assert _x_samples_sizes == _y_samples_sizes, \
            (f"Input data and target data does not have consistent sample sizes after stripping tensors: "
             f"{_x_samples_sizes} and {_y_samples_sizes}")

        # Check that the dataset order is consistent
        assert tuple(x) == tuple(y), f"Inconsistent dataset ordering after stripping tensors: {tuple(x)} and {tuple(y)}"

        # EEG data
        for dataset_name, x_tensor in x.items():
            assert isinstance(x_tensor, torch.Tensor)
            assert x_tensor.size()[-2:] == torch.Size(dummy_input_data[dataset_name].shape[-2:])
            assert ((x_tensor >= 100) & (x_tensor <= 150)).all()

        # Downstream targets
        for dataset_name, targets in y.items():
            assert isinstance(targets, torch.Tensor)
            assert targets.ndim == 2
            assert targets.size()[-1] == 1

            assert ((targets >= 0) & (targets <= 1)).all()


def test_multi_task_rbp_data_gen(dummy_input_data, dummy_targets, dummy_pseudo_targets, dummy_expected_variables,
                                 dummy_dataset_subjects, dummy_pretext_mask, dummy_downstream_mask,
                                 dummy_subjects_info, dummy_pretext_datasets, dummy_downstream_datasets):
    """Test if MultiTaskRBPdataGenerator can be used with DataLoader, and yield expected outcomes"""
    all_datasets = set(dummy_dataset_subjects)

    # Create data loader
    loader = DataLoader(
        MultiTaskRBPdataGenerator(
            data=dummy_input_data, downstream_targets=dummy_targets, pretext_targets=dummy_pseudo_targets,
            expected_variables=dummy_expected_variables, subjects=dummy_dataset_subjects, pre_computed=None,
            downstream_mask=dummy_downstream_mask, pretext_mask=dummy_pretext_mask, subjects_info=dummy_subjects_info
        )
    )

    # Loop over it
    for x, pre_computed, (pretext_y, pretext_mask), (downstream_y, downstream_mask), subject_indices in loader:
        # Check that the dataset order is consistent
        assert tuple(x) == tuple(pretext_y) == tuple(pretext_mask) == tuple(downstream_y) == tuple(downstream_mask), \
            "Inconsistent dataset ordering"

        # Check that all datasets are present
        assert set(x) == all_datasets, f"Expected and actual datasets were difference: {set(x)}, {all_datasets}"

        # ------------
        # Tests before stripping the dictionaries for 'ghost tensors'
        # ------------
        # Sample size test
        _x_samples_sizes = {name: tensor.size()[0] for name, tensor in x.items()}
        _pretext_y_samples_sizes = {name: tensor.size()[0] for name, tensor in pretext_y.items()}
        _downstream_y_samples_sizes = {name: tensor.size()[0] for name, tensor in downstream_y.items()}
        _pretext_mask_samples_sizes = {name: tensor.size()[0] for name, tensor in pretext_mask.items()}
        _downstream_mask_samples_sizes = {name: tensor.size()[0] for name, tensor in downstream_mask.items()}
        assert (_x_samples_sizes == _pretext_y_samples_sizes == _downstream_y_samples_sizes
                == _pretext_mask_samples_sizes == _downstream_mask_samples_sizes)

        # EEG data
        for dataset_name, x_tensor in x.items():
            assert isinstance(x_tensor, torch.Tensor)
            assert x_tensor.size()[-2:] == torch.Size(dummy_input_data[dataset_name].shape[-2:])
            assert (((x_tensor >= 100) & (x_tensor <= 150)) |((x_tensor > -1.1) & (x_tensor < -0.9))).all()

        # Pretext targets
        for dataset_name, pseudo_targets in pretext_y.items():
            assert isinstance(pseudo_targets, torch.Tensor)
            assert pseudo_targets.ndim == 2
            assert pseudo_targets.size()[-1] == 1

            assert (((pseudo_targets >= 10) & (pseudo_targets <= 20)) |
                    ((pseudo_targets > -1.1) & (pseudo_targets < -0.9))).all()  # Before stripping, many values are -1

        # Pretext mask
        for dataset_name, p_mask in pretext_mask.items():
            assert isinstance(p_mask, torch.Tensor)
            assert p_mask.ndim == 1

        # Downstream targets
        for dataset_name, targets in downstream_y.items():
            assert isinstance(targets, torch.Tensor)
            assert targets.ndim == 2
            assert targets.size()[-1] == 1

            assert (((targets >= 0) & (targets <= 1)) | ((targets > -1.1) & (targets < -0.9))).all()

        # Downstream mask
        for dataset_name, d_mask in downstream_mask.items():
            assert isinstance(d_mask, torch.Tensor)
            assert d_mask.ndim == 1

        # ------------
        # Tests after stripping the dictionaries for 'ghost tensors'
        # ------------
        x = strip_tensors(x)
        pretext_y = strip_tensors(pretext_y)
        downstream_y = strip_tensors(downstream_y)
        pretext_mask = strip_tensors(pretext_mask)
        downstream_mask = strip_tensors(downstream_mask)

        # Sample size test
        _x_samples_sizes = {name: tensor.size()[0] for name, tensor in x.items()}
        _pretext_y_samples_sizes = {name: tensor.size()[0] for name, tensor in pretext_y.items()}
        _downstream_y_samples_sizes = {name: tensor.size()[0] for name, tensor in downstream_y.items()}
        _pretext_mask_samples_sizes = {name: tensor.size()[0] for name, tensor in pretext_mask.items()}
        _downstream_mask_samples_sizes = {name: tensor.size()[0] for name, tensor in downstream_mask.items()}
        assert (_x_samples_sizes == _pretext_y_samples_sizes == _downstream_y_samples_sizes
                == _pretext_mask_samples_sizes == _downstream_mask_samples_sizes)

        # Check that the dataset order is consistent
        assert tuple(x) == tuple(pretext_y) == tuple(pretext_mask) == tuple(downstream_y) == tuple(downstream_mask), \
            "Inconsistent dataset ordering after stripping torch tensors"

        # EEG data
        for dataset_name, x_tensor in x.items():
            assert isinstance(x_tensor, torch.Tensor)
            assert x_tensor.size()[-2:] == torch.Size(dummy_input_data[dataset_name].shape[-2:])
            assert ((x_tensor >= 100) & (x_tensor <= 150)).all()

        # Pretext targets
        for dataset_name, pseudo_targets in pretext_y.items():
            assert isinstance(pseudo_targets, torch.Tensor)
            assert pseudo_targets.ndim == 2
            assert pseudo_targets.size()[-1] == 1

            assert ((pseudo_targets >= 10) & (pseudo_targets <= 20)).all()

        # Pretext mask
        for dataset_name, p_mask in pretext_mask.items():
            assert isinstance(p_mask, torch.Tensor)
            assert p_mask.ndim == 1
            assert p_mask.dtype == torch.bool

            if dataset_name in dummy_pretext_datasets:
                assert p_mask.all()
            else:
                assert not p_mask.any()

        # Downstream targets
        for dataset_name, targets in downstream_y.items():
            assert isinstance(targets, torch.Tensor)
            assert targets.ndim == 2
            assert targets.size()[-1] == 1

            assert ((targets >= 0) & (targets <= 1)).all()

        # Downstream mask
        for dataset_name, d_mask in downstream_mask.items():
            assert isinstance(d_mask, torch.Tensor)
            assert d_mask.ndim == 1

            if dataset_name in dummy_downstream_datasets:
                assert d_mask.all()
            else:
                assert not d_mask.any()


# ---------------
# Tests for interpolation-based data generators
# ---------------
def test_normal_interpolation_data_gen(dummy_input_data, dummy_targets, dummy_pseudo_targets, dummy_expected_variables,
                                       dummy_dataset_subjects, dummy_pretext_mask, dummy_downstream_mask,
                                       dummy_subjects_info, dummy_pretext_datasets, dummy_downstream_datasets):
    """Test if InterpolationDataGenerator can be used with DataLoader, and yield expected outcomes"""
    all_datasets = set(dummy_dataset_subjects)

    # Create data loader
    loader = DataLoader(
        InterpolationDataGenerator(
            data=dummy_input_data, targets=dummy_targets, expected_variables=dummy_expected_variables,
            subjects=dummy_dataset_subjects, subjects_info=dummy_subjects_info
        )
    )

    # Loop over it
    for x, y, subject_indices in loader:
        # Check that the dataset order is consistent
        assert tuple(x) == tuple(y), f"Inconsistent dataset ordering: {tuple(x)} and {tuple(y)}"

        # Check that all datasets are present
        assert set(x) == all_datasets, f"Expected and actual datasets were difference: {set(x)}, {all_datasets}"

        # ------------
        # Tests before stripping the dictionaries for 'ghost tensors'
        # ------------
        # Sample size test
        _x_samples_sizes = {name: tensor.size()[0] for name, tensor in x.items()}
        _y_samples_sizes = {name: tensor.size()[0] for name, tensor in y.items()}
        assert _x_samples_sizes == _y_samples_sizes, \
            (f"Input data and target data does not have consistent sample sizes: {_x_samples_sizes} and "
             f"{_y_samples_sizes}")

        # EEG data
        for dataset_name, x_tensor in x.items():
            assert isinstance(x_tensor, torch.Tensor)
            assert x_tensor.size()[-2:] == torch.Size(dummy_input_data[dataset_name].shape[-2:])
            assert (((x_tensor >= 100) & (x_tensor <= 150)) | ((x_tensor > -1.1) & (x_tensor < -0.9))).all()

        # Targets
        for dataset_name, targets in y.items():
            assert isinstance(targets, torch.Tensor)
            assert targets.ndim == 2
            assert targets.size()[-1] == 1

            assert (((targets >= 0) & (targets <= 1)) | ((targets > -1.1) & (targets < -0.9))).all()


        # ------------
        # Tests after stripping the dictionaries for 'ghost tensors'
        # ------------
        x = strip_tensors(x)
        y = strip_tensors(y)

        # Sample size test
        _x_samples_sizes = {name: tensor.size()[0] for name, tensor in x.items()}
        _y_samples_sizes = {name: tensor.size()[0] for name, tensor in y.items()}
        assert _x_samples_sizes == _y_samples_sizes, \
            (f"Input data and target data does not have consistent sample sizes after stripping tensors: "
             f"{_x_samples_sizes} and {_y_samples_sizes}")

        # Check that the dataset order is consistent
        assert tuple(x) == tuple(y), f"Inconsistent dataset ordering after stripping tensors: {tuple(x)} and {tuple(y)}"

        # EEG data
        for dataset_name, x_tensor in x.items():
            assert isinstance(x_tensor, torch.Tensor)
            assert x_tensor.size()[-2:] == torch.Size(dummy_input_data[dataset_name].shape[-2:])
            assert ((x_tensor >= 100) & (x_tensor <= 150)).all()

        # Downstream targets
        for dataset_name, targets in y.items():
            assert isinstance(targets, torch.Tensor)
            assert targets.ndim == 2
            assert targets.size()[-1] == 1

            assert ((targets >= 0) & (targets <= 1)).all()


def test_multi_task_interpolation_data_gen(dummy_input_data, dummy_targets, dummy_pseudo_targets,
                                           dummy_expected_variables, dummy_dataset_subjects, dummy_pretext_mask,
                                           dummy_downstream_mask, dummy_subjects_info, dummy_pretext_datasets,
                                           dummy_downstream_datasets):
    """Test if MultiTaskInterpolationDataGenerator can be used with DataLoader, and yield expected outcomes"""
    all_datasets = set(dummy_dataset_subjects)

    # Create data loader
    loader = DataLoader(
        MultiTaskInterpolationDataGenerator(
            data=dummy_input_data, downstream_targets=dummy_targets, pretext_targets=dummy_pseudo_targets,
            expected_variables=dummy_expected_variables, subjects=dummy_dataset_subjects,
            downstream_mask=dummy_downstream_mask, pretext_mask=dummy_pretext_mask, subjects_info=dummy_subjects_info
        )
    )

    # Loop over it
    for x, (pretext_y, pretext_mask), (downstream_y, downstream_mask), subject_indices in loader:
        # Check that the dataset order is consistent
        assert tuple(x) == tuple(pretext_y) == tuple(pretext_mask) == tuple(downstream_y) == tuple(downstream_mask), \
            "Inconsistent dataset ordering"

        # Check that all datasets are present
        assert set(x) == all_datasets, f"Expected and actual datasets were difference: {set(x)}, {all_datasets}"

        # ------------
        # Tests before stripping the dictionaries for 'ghost tensors'
        # ------------
        # Sample size test
        _x_samples_sizes = {name: tensor.size()[0] for name, tensor in x.items()}
        _pretext_y_samples_sizes = {name: tensor.size()[0] for name, tensor in pretext_y.items()}
        _downstream_y_samples_sizes = {name: tensor.size()[0] for name, tensor in downstream_y.items()}
        _pretext_mask_samples_sizes = {name: tensor.size()[0] for name, tensor in pretext_mask.items()}
        _downstream_mask_samples_sizes = {name: tensor.size()[0] for name, tensor in downstream_mask.items()}
        assert (_x_samples_sizes == _pretext_y_samples_sizes == _downstream_y_samples_sizes
                == _pretext_mask_samples_sizes == _downstream_mask_samples_sizes)

        # EEG data
        for dataset_name, x_tensor in x.items():
            assert isinstance(x_tensor, torch.Tensor)
            assert x_tensor.size()[-2:] == torch.Size(dummy_input_data[dataset_name].shape[-2:])
            assert (((x_tensor >= 100) & (x_tensor <= 150)) |((x_tensor > -1.1) & (x_tensor < -0.9))).all()

        # Pretext targets
        for dataset_name, pseudo_targets in pretext_y.items():
            assert isinstance(pseudo_targets, torch.Tensor)
            assert pseudo_targets.ndim == 2
            assert pseudo_targets.size()[-1] == 1

            assert (((pseudo_targets >= 10) & (pseudo_targets <= 20)) |
                    ((pseudo_targets > -1.1) & (pseudo_targets < -0.9))).all()  # Before stripping, many values are -1

        # Pretext mask
        for dataset_name, p_mask in pretext_mask.items():
            assert isinstance(p_mask, torch.Tensor)
            assert p_mask.ndim == 1

        # Downstream targets
        for dataset_name, targets in downstream_y.items():
            assert isinstance(targets, torch.Tensor)
            assert targets.ndim == 2
            assert targets.size()[-1] == 1

            assert (((targets >= 0) & (targets <= 1)) | ((targets > -1.1) & (targets < -0.9))).all()

        # Downstream mask
        for dataset_name, d_mask in downstream_mask.items():
            assert isinstance(d_mask, torch.Tensor)
            assert d_mask.ndim == 1

        # ------------
        # Tests after stripping the dictionaries for 'ghost tensors'
        # ------------
        x = strip_tensors(x)
        pretext_y = strip_tensors(pretext_y)
        downstream_y = strip_tensors(downstream_y)
        pretext_mask = strip_tensors(pretext_mask)
        downstream_mask = strip_tensors(downstream_mask)

        # Sample size test
        _x_samples_sizes = {name: tensor.size()[0] for name, tensor in x.items()}
        _pretext_y_samples_sizes = {name: tensor.size()[0] for name, tensor in pretext_y.items()}
        _downstream_y_samples_sizes = {name: tensor.size()[0] for name, tensor in downstream_y.items()}
        _pretext_mask_samples_sizes = {name: tensor.size()[0] for name, tensor in pretext_mask.items()}
        _downstream_mask_samples_sizes = {name: tensor.size()[0] for name, tensor in downstream_mask.items()}
        assert (_x_samples_sizes == _pretext_y_samples_sizes == _downstream_y_samples_sizes
                == _pretext_mask_samples_sizes == _downstream_mask_samples_sizes)

        # Check that the dataset order is consistent
        assert tuple(x) == tuple(pretext_y) == tuple(pretext_mask) == tuple(downstream_y) == tuple(downstream_mask), \
            "Inconsistent dataset ordering after stripping torch tensors"

        # EEG data
        for dataset_name, x_tensor in x.items():
            assert isinstance(x_tensor, torch.Tensor)
            assert x_tensor.size()[-2:] == torch.Size(dummy_input_data[dataset_name].shape[-2:])
            assert ((x_tensor >= 100) & (x_tensor <= 150)).all()

        # Pretext targets
        for dataset_name, pseudo_targets in pretext_y.items():
            assert isinstance(pseudo_targets, torch.Tensor)
            assert pseudo_targets.ndim == 2
            assert pseudo_targets.size()[-1] == 1

            assert ((pseudo_targets >= 10) & (pseudo_targets <= 20)).all()

        # Pretext mask
        for dataset_name, p_mask in pretext_mask.items():
            assert isinstance(p_mask, torch.Tensor)
            assert p_mask.ndim == 1
            assert p_mask.dtype == torch.bool

            if dataset_name in dummy_pretext_datasets:
                assert p_mask.all()
            else:
                assert not p_mask.any()

        # Downstream targets
        for dataset_name, targets in downstream_y.items():
            assert isinstance(targets, torch.Tensor)
            assert targets.ndim == 2
            assert targets.size()[-1] == 1

            assert ((targets >= 0) & (targets <= 1)).all()

        # Downstream mask
        for dataset_name, d_mask in downstream_mask.items():
            assert isinstance(d_mask, torch.Tensor)
            assert d_mask.ndim == 1

            if dataset_name in dummy_downstream_datasets:
                assert d_mask.all()
            else:
                assert not d_mask.any()
