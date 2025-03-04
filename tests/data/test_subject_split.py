import random

import pytest

from tests.data.conftest import splits_and_kwargs


def test_non_bool_sort_error(splits_and_kwargs):
    """Test if passing 'sort_first' string as string correctly raises TypeError"""
    for subject_class, kwargs in splits_and_kwargs:
        # Change argument to string
        kwargs["sort_first"] = "False"
        with pytest.raises(TypeError):
            subject_class(**kwargs)


def test_seeding(splits_and_kwargs):
    """Test if seeding gives reproducible splits"""
    for subject_class, kwargs in splits_and_kwargs:
        assert kwargs["seed"] is not None

        # Make splits twice
        splits_1 = subject_class(**kwargs).splits
        splits_2 = subject_class(**kwargs).splits

        assert splits_1 == splits_2, "Seeding is not sufficient to make the splits reproducible"


def test_missing_seeding(splits_and_kwargs):
    """Test if not seeding gives non-reproducible splits"""
    random.seed(1)
    for subject_class, kwargs in splits_and_kwargs:
        # Remove seeding
        kwargs["seed"] = None

        # Make splits twice
        splits_1 = subject_class(**kwargs).splits
        splits_2 = subject_class(**kwargs).splits

        assert splits_1 != splits_2, "Not seeding still gave reproducible splits"


def test_order_invariance(splits_and_kwargs):
    """Test if setting 'sort_first' to True gives reproducible and order invariant splits"""
    random.seed(1)
    for split_class, kwargs in splits_and_kwargs:
        kwargs["sort_first"] = True

        # Try many shuffled versions of the input
        original_dataset_subjects = kwargs["dataset_subjects"].copy()
        splits = []
        for _ in range(10):
            # Shuffle
            shuffled_dataset_subjects = dict()
            shuffled_dataset_names = list(original_dataset_subjects)
            random.shuffle(shuffled_dataset_names)
            for dataset_name in shuffled_dataset_names:
                shuffled_subjects = list(original_dataset_subjects[dataset_name])
                random.shuffle(shuffled_subjects)

                shuffled_dataset_subjects[dataset_name] = tuple(shuffled_subjects)

            # Create split
            new_kwargs = kwargs.copy()
            new_kwargs["dataset_subjects"] = shuffled_dataset_subjects
            splits.append(split_class(**new_kwargs).splits)

        # All splits should still be the same
        assert all(split == splits[0] for split in splits)


def test_order_non_invariance(splits_and_kwargs):
    """Test if setting 'sort_first' to False gives non-order invariant splits"""
    random.seed(1)
    for split_class, kwargs in splits_and_kwargs:
        kwargs["sort_first"] = False

        # Try many shuffled versions of the input
        original_dataset_subjects = kwargs["dataset_subjects"].copy()
        splits = []
        for _ in range(10):
            # Shuffle
            shuffled_dataset_subjects = dict()
            shuffled_dataset_names = list(original_dataset_subjects)
            random.shuffle(shuffled_dataset_names)
            for dataset_name in shuffled_dataset_names:
                shuffled_subjects = list(original_dataset_subjects[dataset_name])
                random.shuffle(shuffled_subjects)

                shuffled_dataset_subjects[dataset_name] = tuple(shuffled_subjects)

            # Create split
            new_kwargs = kwargs.copy()
            new_kwargs["dataset_subjects"] = shuffled_dataset_subjects
            splits.append(split_class(**new_kwargs).splits)

        # All splits should not be the same
        assert not all(split == splits[0] for split in splits)
