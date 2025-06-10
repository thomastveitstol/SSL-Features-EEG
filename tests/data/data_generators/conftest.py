import numpy
import pytest

from elecssl.data.subject_split import Subject

_NUM_EPOCHS = 7
_NUM_CHANNELS = {"Mercedes": 19, "McLaren": 32, "Haas": 11, "Aston Martin": 128, "Red Bull": 64, "Ferrari": 61}
_NUM_TIME_STEPS = 200

# Mimic the main intended purpose, which is that there are 'pretext' datasets and 'downstream' datasets
_PRETEXT_DATASETS = ("Mercedes", "McLaren", "Haas", "Aston Martin")
_DOWNSTREAM_DATASETS = ("Red Bull", "Ferrari")


@pytest.fixture
def dummy_input_data(dummy_dataset_subjects):
    """All values are between 100 and 150"""
    return {dataset_name: numpy.random.uniform(
        low=100, high=150, size=(len(subjects), _NUM_EPOCHS, _NUM_CHANNELS[dataset_name], _NUM_TIME_STEPS))
        for dataset_name, subjects in dummy_dataset_subjects.items()}


@pytest.fixture
def dummy_targets(dummy_dataset_subjects):
    """All targets are between 0 and 1"""
    return {dataset_name: numpy.random.uniform(size=(len(subjects)))
            for dataset_name, subjects in dummy_dataset_subjects.items()}


@pytest.fixture
def dummy_pseudo_targets(dummy_dataset_subjects):
    """All targets are between 10 and 20"""
    return {dataset_name: numpy.random.uniform(low=10, high=20, size=(len(subjects)))
            for dataset_name, subjects in dummy_dataset_subjects.items()}


@pytest.fixture
def dummy_expected_variables(dummy_dataset_subjects):
    """'expected_variables' is not maintained, but still required"""
    return {dataset_name: [] for dataset_name in dummy_dataset_subjects}


@pytest.fixture
def dummy_subjects_info(dummy_dataset_subjects):
    """'subjects_info' is not maintained, but still required"""
    return {Subject(subject_id=sub_id, dataset_name=dataset_name): dict()
            for dataset_name, subjects in dummy_dataset_subjects.items() for sub_id in subjects}


# -------------
# Target masking for multi-task learning
# -------------
@pytest.fixture
def dummy_pretext_datasets():
    return _PRETEXT_DATASETS


@pytest.fixture
def dummy_downstream_datasets():
    return _DOWNSTREAM_DATASETS


@pytest.fixture
def dummy_downstream_mask(dummy_dataset_subjects):
    return {dataset_name: numpy.array([(dataset_name in _DOWNSTREAM_DATASETS)] * len(subjects))
            for dataset_name, subjects in dummy_dataset_subjects.items()}


@pytest.fixture
def dummy_pretext_mask(dummy_dataset_subjects):
    return {dataset_name: numpy.array([(dataset_name in _PRETEXT_DATASETS)] * len(subjects))
            for dataset_name, subjects in dummy_dataset_subjects.items()}
