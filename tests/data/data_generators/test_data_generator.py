import random
from typing import Dict, Sequence

import pytest
import torch
from torch.utils.data import DataLoader

from elecssl.data.data_generators.data_generator import MultiTaskRBPdataGenerator, RBPDataGenerator, \
    InterpolationDataGenerator, MultiTaskInterpolationDataGenerator
from elecssl.data.subject_split import Subject


def _count_subjects_per_dataset(subjects: Sequence[Subject]):
    counts: Dict[str, int] = dict()
    for subject in subjects:
        dataset_name = subject.dataset_name
        if dataset_name not in counts:
            counts[dataset_name] = 1
        else:
            counts[dataset_name] += 1
    return counts


def _create_dummy_precomputed_features(input_data, feature_dims):
    features = []
    for feature_dim in feature_dims:
        if feature_dim is None:
            new_tensors = None
        else:
            new_tensors = {name: 200 + 25 * torch.rand(size=(*data.shape[:-1], feature_dim))
                           for name, data in input_data.items()}
        features.append(new_tensors)
    return features


# ---------------
# Tests for RBP-based data generators
# ---------------
@pytest.mark.parametrize("batch_size", (1, 2, 3, 4, 5, 6, 7, 8))
def test_normal_rbp_data_gen_without_precomputed(
        dummy_input_data, dummy_targets, dummy_pseudo_targets, dummy_expected_variables, dummy_dataset_subjects,
        dummy_pretext_mask, dummy_downstream_mask, dummy_subjects_info, dummy_pretext_datasets,
        dummy_downstream_datasets, batch_size):
    """Test if RBPDataGenerator can be used with DataLoader, and yield expected outcomes. This tests without
    pre-computing"""
    # Create data loader
    data_gen = RBPDataGenerator(
        data=dummy_input_data, targets=dummy_targets, expected_variables=dummy_expected_variables,
        subjects=dummy_dataset_subjects, pre_computed=None, subjects_info=dummy_subjects_info
    )
    loader = DataLoader(data_gen, shuffle=True, batch_size=batch_size, collate_fn=data_gen.collate_fn)

    # Loop over it
    for x, pre_computed, y, subjects in loader:
        subject_sample_sizes = _count_subjects_per_dataset(subjects=subjects)

        # Check that the dataset order is consistent
        assert tuple(x) == tuple(y) == tuple(subject_sample_sizes), \
            f"Inconsistent dataset ordering: {tuple(x)} and {tuple(y)} and {tuple(subject_sample_sizes)}"

        # Sample size test
        _x_samples_sizes = {name: tensor.size()[0] for name, tensor in x.items()}
        _y_samples_sizes = {name: tensor.size()[0] for name, tensor in y.items()}
        assert _x_samples_sizes == _y_samples_sizes == subject_sample_sizes, \
            (f"Input data and target data does not have consistent sample sizes after stripping tensors: "
             f"{_x_samples_sizes} and {_y_samples_sizes}")

        # EEG data
        for dataset_name, x_tensor in x.items():
            assert isinstance(x_tensor, torch.Tensor)
            assert x_tensor.size()[-2:] == torch.Size(dummy_input_data[dataset_name].shape[-2:])
            assert ((x_tensor >= 100) & (x_tensor <= 150)).all()

        # Precomputed features
        assert pre_computed is None

        # Downstream targets
        for dataset_name, targets in y.items():
            assert isinstance(targets, torch.Tensor)
            assert targets.ndim == 2
            assert targets.size()[-1] == 1

            assert ((targets >= 0) & (targets <= 1)).all()

        # Subjects
        idx = 0
        for expected_dataset, expected_count in subject_sample_sizes.items():
            for i in range(expected_count):
                assert subjects[idx].dataset_name == expected_dataset
                assert idx < len(subjects)
                idx += 1


@pytest.mark.parametrize("batch_size,num_rbp_modules,none_dim", (
        (1, 2, None), (2, 5, 2), (7, 1, None), (2, 2, 0), (9, 9, 4), (4, 3, 2), (3, 2, 0), (6, 8, 5)
))
def test_normal_rbp_data_gen_with_precomputed(
        dummy_input_data, dummy_targets, dummy_pseudo_targets, dummy_expected_variables, dummy_dataset_subjects,
        dummy_pretext_mask, dummy_downstream_mask, dummy_subjects_info, dummy_pretext_datasets,
        dummy_downstream_datasets, batch_size, num_rbp_modules, none_dim):
    """Test if RBPDataGenerator can be used with DataLoader, and yield expected outcomes. This tests with
    pre-computing"""
    # Create some fake pre-computed features
    feature_dims = [random.randint(0, 30) for _ in range(num_rbp_modules)]
    if none_dim is not None:
        feature_dims[none_dim] = None  # type: ignore
    pre_computed = _create_dummy_precomputed_features(dummy_input_data, feature_dims)

    # Create data loader
    data_gen = RBPDataGenerator(
        data=dummy_input_data, targets=dummy_targets, expected_variables=dummy_expected_variables,
        subjects=dummy_dataset_subjects, pre_computed=pre_computed, subjects_info=dummy_subjects_info
    )
    loader = DataLoader(data_gen, shuffle=True, batch_size=batch_size, collate_fn=data_gen.collate_fn)

    # Loop over it
    for x, pre_computed, y, subjects in loader:
        subject_sample_sizes = _count_subjects_per_dataset(subjects=subjects)

        # Check that the dataset order is consistent
        assert tuple(x) == tuple(y) == tuple(subject_sample_sizes), \
            f"Inconsistent dataset ordering: {tuple(x)} and {tuple(y)} and {tuple(subject_sample_sizes)}"
        assert (tuple(features) == tuple(subject_sample_sizes) for features in pre_computed), \
            "Not all precomputed features had correct dictionary ordering"

        # Sample size test
        _x_samples_sizes = {name: tensor.size()[0] for name, tensor in x.items()}
        _y_samples_sizes = {name: tensor.size()[0] for name, tensor in y.items()}
        assert _x_samples_sizes == _y_samples_sizes == subject_sample_sizes, \
            (f"Input data and target data does not have consistent sample sizes: "
             f"{_x_samples_sizes} and {_y_samples_sizes}")
        for tensors in pre_computed:
            if tensors is None:
                continue
            assert {name: tensor.size()[0] for name, tensor in tensors.items()}, \
                "Not all non-None precomputed features had correct sample size"

        # EEG data
        for dataset_name, x_tensor in x.items():
            assert isinstance(x_tensor, torch.Tensor)
            assert x_tensor.size()[-2:] == torch.Size(dummy_input_data[dataset_name].shape[-2:])
            assert ((x_tensor >= 100) & (x_tensor <= 150)).all()

        # Precomputed features
        assert pre_computed is None or isinstance(pre_computed, tuple)
        for feature_dim, features in zip(feature_dims, pre_computed):
            if features is None:
                assert feature_dim is None
                continue

            assert isinstance(features, dict)
            for dataset_name, features_tensor in features.items():
                assert isinstance(features_tensor, torch.Tensor)
                expected_shape = (x[dataset_name].shape[0], x[dataset_name].shape[1], feature_dim)
                assert features_tensor.size() == torch.Size(expected_shape)
                assert ((features_tensor >= 200) & (features_tensor <= 225)).all()

        # Downstream targets
        for dataset_name, targets in y.items():
            assert isinstance(targets, torch.Tensor)
            assert targets.ndim == 2
            assert targets.size()[-1] == 1

            assert ((targets >= 0) & (targets <= 1)).all()

        # Subjects
        idx = 0
        for expected_dataset, expected_count in subject_sample_sizes.items():
            for i in range(expected_count):
                assert subjects[idx].dataset_name == expected_dataset
                assert idx < len(subjects)
                idx += 1


@pytest.mark.parametrize("batch_size", (1, 2, 3, 4, 5, 6, 7, 8))
def test_multi_task_rbp_data_gen_without_precomputed(
        dummy_input_data, dummy_targets, dummy_pseudo_targets, dummy_expected_variables, dummy_dataset_subjects,
        dummy_pretext_mask, dummy_downstream_mask, dummy_subjects_info, dummy_pretext_datasets,
        dummy_downstream_datasets, batch_size):
    """Test if MultiTaskRBPdataGenerator can be used with DataLoader, and yield expected outcomes"""
    # Create data loader
    data_gen = MultiTaskRBPdataGenerator(
        data=dummy_input_data, downstream_targets=dummy_targets, pretext_targets=dummy_pseudo_targets,
        expected_variables=dummy_expected_variables, subjects=dummy_dataset_subjects, pre_computed=None,
        downstream_mask=dummy_downstream_mask, pretext_mask=dummy_pretext_mask, subjects_info=dummy_subjects_info
    )
    loader = DataLoader(data_gen, shuffle=True, batch_size=batch_size, collate_fn=data_gen.collate_fn)

    # Loop over it
    for x, pre_computed, (pretext_y, pretext_mask), (downstream_y, downstream_mask), subjects in loader:
        subject_sample_sizes = _count_subjects_per_dataset(subjects=subjects)

        # Check that the dataset order is consistent
        assert (tuple(x) == tuple(pretext_y) == tuple(pretext_mask) == tuple(downstream_y) == tuple(downstream_mask)
                == tuple(subject_sample_sizes)), \
            "Inconsistent dataset ordering"

        # Sample size test
        _x_samples_sizes = {name: tensor.size()[0] for name, tensor in x.items()}
        _pretext_y_samples_sizes = {name: tensor.size()[0] for name, tensor in pretext_y.items()}
        _downstream_y_samples_sizes = {name: tensor.size()[0] for name, tensor in downstream_y.items()}
        _pretext_mask_samples_sizes = {name: tensor.size()[0] for name, tensor in pretext_mask.items()}
        _downstream_mask_samples_sizes = {name: tensor.size()[0] for name, tensor in downstream_mask.items()}
        assert (_x_samples_sizes == _pretext_y_samples_sizes == _downstream_y_samples_sizes
                == _pretext_mask_samples_sizes == _downstream_mask_samples_sizes == subject_sample_sizes)

        # EEG data
        for dataset_name, x_tensor in x.items():
            assert isinstance(x_tensor, torch.Tensor)
            assert x_tensor.size()[-2:] == torch.Size(dummy_input_data[dataset_name].shape[-2:])
            assert ((x_tensor >= 100) & (x_tensor <= 150)).all()

        # Precomputed features
        assert pre_computed is None

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

        # Subjects
        idx = 0
        for expected_dataset, expected_count in subject_sample_sizes.items():
            for i in range(expected_count):
                assert subjects[idx].dataset_name == expected_dataset
                assert idx < len(subjects)
                idx += 1


@pytest.mark.parametrize("batch_size,num_rbp_modules,none_dim", (
        (1, 2, None), (2, 5, 2), (7, 1, None), (2, 2, 0), (9, 9, 4), (4, 3, 2), (3, 2, 0), (6, 8, 5)
))
def test_multi_task_rbp_data_gen_with_precomputed(
        dummy_input_data, dummy_targets, dummy_pseudo_targets, dummy_expected_variables, dummy_dataset_subjects,
        dummy_pretext_mask, dummy_downstream_mask, dummy_subjects_info, dummy_pretext_datasets,
        dummy_downstream_datasets, batch_size, num_rbp_modules, none_dim):
    """Test if MultiTaskRBPdataGenerator can be used with DataLoader, and yield expected outcomes"""
    # Create some fake pre-computed features
    feature_dims = [random.randint(0, 30) for _ in range(num_rbp_modules)]
    if none_dim is not None:
        feature_dims[none_dim] = None  # type: ignore
    pre_computed = _create_dummy_precomputed_features(dummy_input_data, feature_dims)

    # Create data loader
    data_gen = MultiTaskRBPdataGenerator(
        data=dummy_input_data, downstream_targets=dummy_targets, pretext_targets=dummy_pseudo_targets,
        expected_variables=dummy_expected_variables, subjects=dummy_dataset_subjects, pre_computed=pre_computed,
        downstream_mask=dummy_downstream_mask, pretext_mask=dummy_pretext_mask, subjects_info=dummy_subjects_info
    )
    loader = DataLoader(data_gen, shuffle=True, batch_size=batch_size, collate_fn=data_gen.collate_fn)

    # Loop over it
    for x, pre_computed, (pretext_y, pretext_mask), (downstream_y, downstream_mask), subjects in loader:
        subject_sample_sizes = _count_subjects_per_dataset(subjects=subjects)

        # Check that the dataset order is consistent
        assert (tuple(x) == tuple(pretext_y) == tuple(pretext_mask) == tuple(downstream_y) == tuple(downstream_mask)
                == tuple(subject_sample_sizes)), \
            "Inconsistent dataset ordering"
        assert (tuple(features) == tuple(subject_sample_sizes) for features in pre_computed), \
            "Not all precomputed features had correct dictionary ordering"

        # Sample size test
        _x_samples_sizes = {name: tensor.size()[0] for name, tensor in x.items()}
        _pretext_y_samples_sizes = {name: tensor.size()[0] for name, tensor in pretext_y.items()}
        _downstream_y_samples_sizes = {name: tensor.size()[0] for name, tensor in downstream_y.items()}
        _pretext_mask_samples_sizes = {name: tensor.size()[0] for name, tensor in pretext_mask.items()}
        _downstream_mask_samples_sizes = {name: tensor.size()[0] for name, tensor in downstream_mask.items()}
        assert (_x_samples_sizes == _pretext_y_samples_sizes == _downstream_y_samples_sizes
                == _pretext_mask_samples_sizes == _downstream_mask_samples_sizes == subject_sample_sizes)
        for tensors in pre_computed:
            if tensors is None:
                continue
            assert {name: tensor.size()[0] for name, tensor in tensors.items()}, \
                "Not all non-None precomputed features had correct sample size"

        # EEG data
        for dataset_name, x_tensor in x.items():
            assert isinstance(x_tensor, torch.Tensor)
            assert x_tensor.size()[-2:] == torch.Size(dummy_input_data[dataset_name].shape[-2:])
            assert ((x_tensor >= 100) & (x_tensor <= 150)).all()

        # Precomputed features
        assert pre_computed is None or isinstance(pre_computed, tuple)
        for feature_dim, features in zip(feature_dims, pre_computed):
            if features is None:
                assert feature_dim is None
                continue

            assert isinstance(features, dict)
            for dataset_name, features_tensor in features.items():
                assert isinstance(features_tensor, torch.Tensor)
                expected_shape = (x[dataset_name].shape[0], x[dataset_name].shape[1], feature_dim)
                assert features_tensor.size() == torch.Size(expected_shape)
                assert ((features_tensor >= 200) & (features_tensor <= 225)).all()

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

            # Subjects
            idx = 0
            for expected_dataset, expected_count in subject_sample_sizes.items():
                for i in range(expected_count):
                    assert subjects[idx].dataset_name == expected_dataset
                    assert idx < len(subjects)
                    idx += 1


# ---------------
# Tests for interpolation-based data generators
# ---------------
@pytest.mark.parametrize("batch_size", (1, 2, 3, 4, 5, 6, 7, 8))
def test_normal_interpolation_data_gen(dummy_input_data, dummy_targets, dummy_pseudo_targets,
                                       dummy_expected_variables, dummy_dataset_subjects, dummy_pretext_mask,
                                       dummy_downstream_mask, dummy_subjects_info, dummy_pretext_datasets,
                                       dummy_downstream_datasets, batch_size):
    """Test if InterpolationDataGenerator can be used with DataLoader, and yield expected outcomes"""
    # Create data loader
    data_gen = InterpolationDataGenerator(
        data=dummy_input_data, targets=dummy_targets, expected_variables=dummy_expected_variables,
        subjects=dummy_dataset_subjects, subjects_info=dummy_subjects_info
    )
    loader = DataLoader(data_gen, shuffle=True, batch_size=batch_size, collate_fn=data_gen.collate_fn)

    # Loop over it
    for x, y, subjects in loader:
        subject_sample_sizes = _count_subjects_per_dataset(subjects=subjects)

        # Check that the dataset order is consistent
        assert tuple(x) == tuple(y) == tuple(subject_sample_sizes), \
            f"Inconsistent dataset ordering: {tuple(x)} and {tuple(y)} and {tuple(subject_sample_sizes)}"

        # Sample size test
        _x_samples_sizes = {name: tensor.size()[0] for name, tensor in x.items()}
        _y_samples_sizes = {name: tensor.size()[0] for name, tensor in y.items()}
        assert _x_samples_sizes == _y_samples_sizes == subject_sample_sizes, \
            (f"Input data, target data, and subjects does not have consistent sample sizes: {_x_samples_sizes} and "
             f"{_y_samples_sizes} and {subject_sample_sizes}")

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

        # Subjects
        idx = 0
        for expected_dataset, expected_count in subject_sample_sizes.items():
            for i in range(expected_count):
                assert subjects[idx].dataset_name == expected_dataset
                assert idx < len(subjects)
                idx += 1


@pytest.mark.parametrize("batch_size", (1, 2, 3, 4, 5, 6, 7, 8))
def test_multi_task_interpolation_data_gen(dummy_input_data, dummy_targets, dummy_pseudo_targets,
                                           dummy_expected_variables, dummy_dataset_subjects, dummy_pretext_mask,
                                           dummy_downstream_mask, dummy_subjects_info, dummy_pretext_datasets,
                                           dummy_downstream_datasets, batch_size):
    """Test if MultiTaskInterpolationDataGenerator can be used with DataLoader, and yield expected outcomes"""
    # Create data loader
    data_gen = MultiTaskInterpolationDataGenerator(
        data=dummy_input_data, downstream_targets=dummy_targets, pretext_targets=dummy_pseudo_targets,
        expected_variables=dummy_expected_variables, subjects=dummy_dataset_subjects,
        downstream_mask=dummy_downstream_mask, pretext_mask=dummy_pretext_mask, subjects_info=dummy_subjects_info
    )
    loader = DataLoader(data_gen, shuffle=True, batch_size=batch_size, collate_fn=data_gen.collate_fn)

    # Loop over it
    for x, (pretext_y, pretext_mask), (downstream_y, downstream_mask), subjects in loader:
        subject_sample_sizes = _count_subjects_per_dataset(subjects=subjects)

        # Check that the dataset order is consistent
        assert (tuple(x) == tuple(pretext_y) == tuple(pretext_mask) == tuple(downstream_y) == tuple(downstream_mask)
                == tuple(subject_sample_sizes)), \
            "Inconsistent dataset ordering"

        # Sample size test
        _x_samples_sizes = {name: tensor.size()[0] for name, tensor in x.items()}
        _pretext_y_samples_sizes = {name: tensor.size()[0] for name, tensor in pretext_y.items()}
        _downstream_y_samples_sizes = {name: tensor.size()[0] for name, tensor in downstream_y.items()}
        _pretext_mask_samples_sizes = {name: tensor.size()[0] for name, tensor in pretext_mask.items()}
        _downstream_mask_samples_sizes = {name: tensor.size()[0] for name, tensor in downstream_mask.items()}
        assert (_x_samples_sizes == _pretext_y_samples_sizes == _downstream_y_samples_sizes
                == _pretext_mask_samples_sizes == _downstream_mask_samples_sizes == subject_sample_sizes)

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

        # Subjects
        idx = 0
        for expected_dataset, expected_count in subject_sample_sizes.items():
            for i in range(expected_count):
                assert subjects[idx].dataset_name == expected_dataset
                assert idx < len(subjects)
                idx += 1
