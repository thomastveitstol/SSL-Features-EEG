import numpy
import pytest

from elecssl.data.combined_datasets import DatasetDetails, LoadDetails
from elecssl.data.datasets.dataset_base import EEGDatasetBase, target_method
from elecssl.data.subject_split import RandomSplitsTVTestHoldout, RandomSplitsTV, KFoldDataSplit, LODOCV, \
    KeepDatasetsOutRandomSplits


@pytest.fixture
def dummy_dataset_subjects():
    """Get a dummy dict of subjects as they should be passed to the __init__ of the subject split classes"""
    # This was copied from a doctest made when this was true ;)
    f1_drivers = {"Mercedes": ("Hamilton", "Russel", "Wolff"), "Red Bull": ("Verstappen", "Checo"),
                  "Ferrari": ("Leclerc", "Smooth Sainz"), "McLaren": ("Norris", "Piastri"),
                  "Aston Martin": ("Alonso", "Stroll"), "Haas": ("Magnussen", "Hülkenberg")}
    return f1_drivers


@pytest.fixture
def splits_and_kwargs(dummy_dataset_subjects):
    return (
        (RandomSplitsTVTestHoldout, {"dataset_subjects": dummy_dataset_subjects, "val_split": 0.2, "test_split": 0.3,
                                     "num_random_splits": 4, "seed": 42, "sort_first": True}),
        (RandomSplitsTV, {"dataset_subjects": dummy_dataset_subjects, "val_split": 0.2, "num_random_splits": 4,
                          "seed": 42, "sort_first": True}),
        (KFoldDataSplit, {"dataset_subjects": dummy_dataset_subjects, "val_split": 0.2, "num_folds": 4,
                          "seed": 42, "sort_first": True}),
        (LODOCV, {"dataset_subjects": dummy_dataset_subjects, "val_split": 0.2, "seed": 42, "sort_first": True}),
        (KeepDatasetsOutRandomSplits, {"dataset_subjects": dummy_dataset_subjects, "val_split": 0.2, "seed": 42,
                                       "num_random_splits": 7, "sort_first": True, "left_out_datasets": "Mercedes"})
    )


@pytest.fixture
def dummy_eeg_dataset(dummy_data):
    _, dummy_num_channels, dummy_num_time_steps = dummy_data.shape

    class DummyDataset(EEGDatasetBase):
        _num_channels = dummy_num_channels
        _num_time_steps = dummy_num_time_steps

        # -------------
        # Overriding abstract methods which are not required for these tests
        # -------------
        def _load_single_raw_mne_object(self, *args, **kwargs):
            raise NotImplementedError

        def channel_name_to_index(self):
            raise NotImplementedError

        # -------------
        # Overriding methods to make this class suited for testing
        # -------------
        def load_numpy_arrays(self, subject_ids=None, pre_processed_version=None, *, time_series_start=None,
                              num_time_steps=None, channels=None, required_target=None):
            return numpy.random.normal(loc=0, scale=1.,
                                       size=(len(subject_ids), self._num_channels, self._num_time_steps))

        @target_method
        def age(self, subject_ids):
            return numpy.random.randint(18, 90, size=(len(subject_ids),))

    return DummyDataset()


@pytest.fixture
def dummy_dataset_details(dummy_eeg_dataset):
    num_subjects = 34
    load_details = LoadDetails(subject_ids=tuple(f"sub-{i}" for i in range(num_subjects)), time_series_start=None,
                               num_time_steps=None, channels=None, pre_processed_version=None)
    return DatasetDetails(dummy_eeg_dataset, load_details)
