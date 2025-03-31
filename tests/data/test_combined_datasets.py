import numpy
import pytest

from elecssl.data.combined_datasets import CombinedDatasets
from elecssl.data.subject_split import Subject


# -------------
# Tests for removing datasets and switching target to load
# -------------
def test_remove_data(dummy_dataset_details):
    # Initialise object
    dataset_name_1 = dummy_dataset_details.dataset.name
    combined_dataset = CombinedDatasets(datasets_details=(dummy_dataset_details,), variables={dataset_name_1: []},
                                        current_target="age", required_target=None)

    # Select a subject of subjects
    subjects_subset = tuple(Subject(dataset_name=dataset_name_1, subject_id=sub_id)
                            for sub_id in dummy_dataset_details.details.subject_ids[:20])

    # Data and targets contain dataset
    assert dataset_name_1 in combined_dataset.get_data(subjects_subset)
    assert dataset_name_1 in combined_dataset.get_targets(subjects_subset)
    assert dataset_name_1 in combined_dataset.dataset_subjects
    assert dataset_name_1 in combined_dataset.channel_name_to_index
    assert dataset_name_1 in tuple(d.name for d in combined_dataset.datasets)

    # -------------
    # Remove dataset and do tests
    # -------------
    combined_dataset.remove_datasets(to_remove=dataset_name_1)

    with pytest.raises(KeyError):
        combined_dataset.get_data(subjects_subset)
    with pytest.raises(KeyError):
        combined_dataset.get_targets(subjects_subset)
    assert dataset_name_1 not in combined_dataset.dataset_subjects
    assert dataset_name_1 not in combined_dataset.channel_name_to_index
    assert dataset_name_1 not in tuple(d.name for d in combined_dataset.datasets)


def test_remove_and_switch_targets(dummy_dataset_details):
    # Initialise object
    dataset_name = dummy_dataset_details.dataset.name
    combined_dataset = CombinedDatasets(datasets_details=(dummy_dataset_details,), variables={dataset_name: []},
                                        current_target="age", required_target=None)

    # Select a subject of subjects
    subjects_subset = tuple(Subject(dataset_name=dataset_name, subject_id=sub_id)
                            for sub_id in dummy_dataset_details.details.subject_ids[:20])

    # -------------
    # Checks before removing targets
    # -------------
    assert combined_dataset.current_target == "age"

    # Data and targets contain dataset
    assert "age" in combined_dataset.target_names
    assert "sex" in combined_dataset.target_names
    assert "age" in combined_dataset._targets
    assert "sex" in combined_dataset._targets

    # -------------
    # Remove targets and do testing
    # -------------
    combined_dataset.remove_targets(to_remove="age")

    # age is still recognised as the current target, so it needs to be switched
    assert combined_dataset.current_target == "age"
    with pytest.raises(KeyError):
        combined_dataset.get_targets(subjects_subset)  # This fails because the current target was removed

    assert combined_dataset.target_names == ("sex",)

    # -------------
    # Switch targets and do testing
    # -------------
    combined_dataset.current_target = "sex"

    # When calling .get_target(subjects), the sex of the subjects should now be loaded
    assert combined_dataset.current_target == "sex"
    targets = combined_dataset.get_targets(subjects_subset)
    assert targets[dataset_name].shape == (20,)
    assert all(target_value in (0, 1) for target_value in targets[dataset_name])


# -------------
# As the data is copied when calling .get_data() and .get_targets()
# makes manipulating the data afterwards safe
# -------------
def test_get_data_copy(dummy_dataset_details):
    """Verify that changing data after calling .get_data() does not change the data provided by .get_data() upon a new
    call"""
    # Initialise object
    dataset_name = dummy_dataset_details.dataset.name
    combined_dataset = CombinedDatasets(datasets_details=(dummy_dataset_details,), variables={dataset_name: []},
                                        current_target="age", required_target=None)

    # Select a subject of subjects
    subjects_subset = tuple(Subject(dataset_name=dataset_name, subject_id=sub_id)
                            for sub_id in dummy_dataset_details.details.subject_ids[:20])

    # -----------
    # Try changing the EEG signal values
    # -----------
    # Load and alter
    data_0 = combined_dataset.get_data(subjects_subset)
    data_original = data_0[dataset_name].copy()
    assert not numpy.array_equal(data_0[dataset_name], numpy.zeros(shape=data_0[dataset_name].shape))  # Noise

    data_0[dataset_name] *= 0
    assert numpy.array_equal(data_0[dataset_name], numpy.zeros(shape=data_0[dataset_name].shape))  # Zeroed

    # Check if this influenced combined_dataset
    data_1 = combined_dataset.get_data(subjects_subset)
    assert numpy.array_equal(data_1[dataset_name], data_original)

    # -----------
    # Try inserting and removing keys
    # -----------
    data_2 = combined_dataset.get_data(subjects_subset)
    del data_2[dataset_name]
    data_2["FakeDataset"] = numpy.array([1, 2, 3, 4, 5])

    # Nothing dramatic has happened inplace with combined_dataset object
    data_3 = combined_dataset.get_data(subjects_subset)
    assert dataset_name in data_3
    assert "FakeDataset" not in data_3


def test_get_target_copy(dummy_dataset_details):
    """Verify that changing data after calling .get_targets() does not change the data provided by .get_targets() upon
    a new call"""
    # Initialise object
    dataset_name = dummy_dataset_details.dataset.name
    combined_dataset = CombinedDatasets(datasets_details=(dummy_dataset_details,), variables={dataset_name: []},
                                        current_target="age", required_target=None)

    # Select a subject of subjects
    subjects_subset = tuple(Subject(dataset_name=dataset_name, subject_id=sub_id)
                            for sub_id in dummy_dataset_details.details.subject_ids[:20])

    # -----------
    # Try changing the target values
    # -----------
    # Load and alter
    data_0 = combined_dataset.get_targets(subjects_subset)
    data_original = data_0[dataset_name].copy()
    assert not numpy.array_equal(data_0[dataset_name], numpy.zeros(shape=data_0[dataset_name].shape))  # Random

    data_0[dataset_name] *= 0
    assert numpy.array_equal(data_0[dataset_name], numpy.zeros(shape=data_0[dataset_name].shape))  # Zeroed

    # Check if this influenced combined_dataset
    data_1 = combined_dataset.get_targets(subjects_subset)
    assert numpy.array_equal(data_1[dataset_name], data_original)

    # -----------
    # Try inserting and removing keys
    # -----------
    data_2 = combined_dataset.get_targets(subjects_subset)
    del data_2[dataset_name]
    fake_dataset_name = "FakeDataset"
    data_2[fake_dataset_name] = numpy.array([1, 2, 3, 4, 5])

    # Nothing dramatic has happened inplace with combined_dataset object
    data_3 = combined_dataset.get_targets(subjects_subset)
    assert dataset_name in data_3
    assert fake_dataset_name not in data_3
