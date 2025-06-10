import pytest

from elecssl.data.combined_datasets import DatasetDetails, LoadDetails
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
def dummy_dataset_subjects_2():
    """Get a dummy dict of subjects as they should be passed to the __init__ of the subject split classes"""
    return {"D1": ("P1", "P2", "P3"), "D2": ("P1", "P2"), "D3": ("P1", "P2"), "D4": ("P1", "P2"), "D5": ("P1", "P2"),
            "D6": ("P1", "P2"), "D7": ("P1", "P2", "P3", "P4", "P5")}


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
def combined_splits(dummy_dataset_subjects, dummy_dataset_subjects_2):
    combined_splits = []

    # Two splits
    split_1 = RandomSplitsTVTestHoldout(dataset_subjects=dummy_dataset_subjects, val_split=0.2, test_split=0.3,
                                        num_random_splits=7, seed=42, sort_first=True)
    split_2 = RandomSplitsTVTestHoldout(dataset_subjects=dummy_dataset_subjects_2, val_split=0.3, test_split=0.25,
                                        num_random_splits=7, seed=42, sort_first=True)
    combined_splits.append((split_1, split_2))

    # Three splits
    split_1 = RandomSplitsTVTestHoldout(dataset_subjects=dummy_dataset_subjects, val_split=0.2, test_split=0.3,
                                        num_random_splits=5, seed=3, sort_first=True)
    split_2 = RandomSplitsTVTestHoldout(dataset_subjects=dummy_dataset_subjects_2, val_split=0.3, test_split=0.25,
                                        num_random_splits=5, seed=4, sort_first=True)
    split_3 = RandomSplitsTVTestHoldout(dataset_subjects=dummy_dataset_subjects, val_split=0.5, test_split=0.2,
                                        num_random_splits=5, seed=1, sort_first=True)

    combined_splits.append((split_1, split_2, split_3))

    return tuple(combined_splits)


@pytest.fixture
def dummy_dataset_details(dummy_eeg_dataset):
    num_subjects = 34
    load_details = LoadDetails(subject_ids=tuple(f"sub-{i}" for i in range(num_subjects)), time_series_start=None,
                               num_time_steps=None, channels=None, pre_processed_version=None, targets=("age", "sex"))
    return DatasetDetails(dummy_eeg_dataset, load_details)
