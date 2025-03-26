import numpy

from elecssl.data.combined_datasets import CombinedDatasets
from elecssl.data.subject_split import Subject


# -------------
# Tests
# -------------
def test_get_data_immutability(dummy_dataset_details):
    """Verify that changing data after calling .get_data() does not change the data provided by .get_data() upon a new
    call"""
    # Initialise object
    dataset_name = dummy_dataset_details.dataset.name
    combined_dataset = CombinedDatasets(datasets_details=(dummy_dataset_details,), variables={dataset_name: []},
                                        target="age", required_target="age", interpolation_method=None,
                                        main_channel_system=None, sampling_freq=None)

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


def test_get_target_immutability(dummy_dataset_details):
    """Verify that changing data after calling .get_targets() does not change the data provided by .get_targets() upon
    a new call"""
    # Initialise object
    dataset_name = dummy_dataset_details.dataset.name
    combined_dataset = CombinedDatasets(datasets_details=(dummy_dataset_details,), variables={dataset_name: []},
                                        target="age", required_target="age", interpolation_method=None,
                                        main_channel_system=None, sampling_freq=None)

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
