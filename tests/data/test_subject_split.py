import random

import pytest

from elecssl.data.subject_split import simple_random_split, RandomSplitsTVTestHoldout, Subject, CombinedSplits


# --------------
# Functions
# --------------
def _get_shuffled_dataset(original_dataset_subjects):
    shuffled_dataset_subjects = dict()
    shuffled_dataset_names = list(original_dataset_subjects)
    random.shuffle(shuffled_dataset_names)
    for dataset_name in shuffled_dataset_names:
        shuffled_subjects = list(original_dataset_subjects[dataset_name])
        random.shuffle(shuffled_subjects)

        shuffled_dataset_subjects[dataset_name] = tuple(shuffled_subjects)
    return shuffled_dataset_subjects


# --------------
# Tests
# --------------
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

        assert splits_1 == splits_2, (f"Seeding is not sufficient to make the splits reproducible for class "
                                      f"{subject_class}")


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
            shuffled_dataset_subjects = _get_shuffled_dataset(original_dataset_subjects.copy())

            # Create split
            new_kwargs = kwargs.copy()
            new_kwargs["dataset_subjects"] = shuffled_dataset_subjects
            splits.append(split_class(**new_kwargs).splits)

        # All splits should still be the same
        assert all(split == splits[0] for split in splits), f"Split were not reproducible for class {split_class}"


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
            shuffled_dataset_subjects = _get_shuffled_dataset(original_dataset_subjects.copy())

            # Create split
            new_kwargs = kwargs.copy()
            new_kwargs["dataset_subjects"] = shuffled_dataset_subjects
            splits.append(split_class(**new_kwargs).splits)

        # All splits should not be the same
        assert not all(split == splits[0] for split in splits)


def test_two_combined_subject_splits(dummy_dataset_subjects, dummy_dataset_subjects_2):
    """Test properties of CombinedSplits, and that looping over combinedSplits is the same as looping over the provided
    splits"""
    # ------------
    # Create objects
    # ------------
    splits_1 = RandomSplitsTVTestHoldout(dataset_subjects=dummy_dataset_subjects, val_split=0.2, test_split=0.3,
                                        num_random_splits=7, seed=42, sort_first=True)
    splits_2 = RandomSplitsTVTestHoldout(dataset_subjects=dummy_dataset_subjects_2, val_split=0.3, test_split=0.25,
                                        num_random_splits=7, seed=42, sort_first=True)

    combined_splits = CombinedSplits(splits_1, splits_2)

    # ------------
    # Tests
    # ------------
    print((combined_splits.splits[0][0]))
    # Properties
    assert (splits_1.all_datasets.union(splits_2.all_datasets)) == combined_splits.all_datasets
    assert (splits_1.all_subjects.union(splits_2.all_subjects)) == combined_splits.all_subjects

    # Check that the looping works as expected. Order does not matter, as long as the train, val and test sets are
    # similar for all splits
    for ((train_1, val_1, test_1), (train_2, val_2, test_2),
         (train_comb, val_comb, test_comb)) in zip(splits_1.splits, splits_2.splits, combined_splits.splits):
        assert set(train_1).union(set(train_2)) == set(train_comb), "Training set was incorrect"
        assert set(val_1).union(set(val_2)) == set(val_comb), "Validation set was incorrect"
        assert set(test_1).union(set(test_2)) == set(test_comb), "Test set was incorrect"


@pytest.mark.parametrize("func_kwargs,class_kwargs", [
    ({"split_percent": 0.3, "seed": 2, "require_seeding": True, "sort_first": True},
     {"val_split": 0.5, "seed": 2, "test_split": 0.3, "num_random_splits": 4, "sort_first": True}),
    ({"split_percent": 0.37, "seed": 9, "require_seeding": True, "sort_first": True},
     {"val_split": 0.1, "seed": 9, "test_split": 0.37, "num_random_splits": 1, "sort_first": True}),
    ({"split_percent": 0.09, "seed": 7, "require_seeding": True, "sort_first": True},
     {"val_split": 0.41, "seed": 7, "test_split": 0.09, "num_random_splits": 9, "sort_first": True})
])
def test_subject_split_equality(dummy_dataset_subjects, func_kwargs, class_kwargs):
    """Test if 'simple_random_split' and 'RandomSplitsTVTestHoldout' produces the same test set when provided similar
    input arguments. Only expected to work when 'sort_first' is True and the same seed is provided"""
    for _ in range(10):
        shuffled_subjects_1 = _get_shuffled_dataset(dummy_dataset_subjects.copy())
        shuffled_subjects_2 = _get_shuffled_dataset(dummy_dataset_subjects.copy())

        # Use the 'simple_random_split' and 'RandomSplitsTVTestHoldout'
        flattened_subjects = [Subject(dataset_name=dataset_name, subject_id=sub_id)
                              for dataset_name, sub_ids in shuffled_subjects_1.items() for sub_id in sub_ids]
        random.shuffle(flattened_subjects)
        test_set_1 = simple_random_split(subjects=flattened_subjects, **func_kwargs)[-1]

        splits_2 = RandomSplitsTVTestHoldout(dataset_subjects=shuffled_subjects_2, **class_kwargs).splits
        test_sets_2 = tuple(split[-1] for split in splits_2)
        assert all(test_set == test_sets_2[0] for test_set in test_sets_2)
        test_set_2 = test_sets_2[0]

        # They should be the same
        assert test_set_1 == test_set_2, ("The set sets are not the same when using 'RandomSplitsTVTestHoldout' and "
                                          "'simple_random_split'")
