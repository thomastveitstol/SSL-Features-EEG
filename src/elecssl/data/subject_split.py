import abc
import dataclasses
import itertools
import random
from typing import List, Tuple, Sequence, Dict, Set

import numpy

from elecssl.models.utils import verify_type


# -----------------
# Convenient dataclasses
# -----------------
@dataclasses.dataclass(frozen=True)
class Subject:
    """
    Class for defining a subject. Convenient particularly when different datasets use the same subject IDs

    Examples
    --------
    >>> Subject("Person", "Dataset")
    Subject(subject_id='Person', dataset_name='Dataset')

    Can be used as keys in a dict

    >>> my_subject = {Subject("P1", "D1"): "this_is_a_value"}
    >>> my_subject[Subject("P1", "D1")]
    'this_is_a_value'

    Attributes can also be obtained as if the class was a dict

    >>> Subject("P1", "D1")["dataset_name"]
    'D1'

    The '<' follows that of strings, first checking 'dataset_name' attribute, followed by 'subject_id'

    >>> Subject("a", "dataset") < Subject("b", "dataset")
    True
    >>> Subject("a", "dataset") < Subject("a", "dataset")
    False
    >>> Subject("sub-001", "dataset") < Subject("sub-002", "dataset")
    True
    >>> Subject("sub-001", "adataset") < Subject("sub-001", "bdataset")
    True
    >>> Subject("sub-001", "bdataset") < Subject("sub-001", "adataset")
    False
    >>> sorted((Subject("b", "x"), Subject("a", "x"), Subject("a", "x"), Subject("a", "w"), Subject("a", "w"),
    ...         Subject("c", "y"), Subject("y", "a"), Subject("x", "b"), Subject("w", "a"), Subject("b", "x")))
    ... # doctest: +NORMALIZE_WHITESPACE
    [Subject(subject_id='w', dataset_name='a'), Subject(subject_id='y', dataset_name='a'),
     Subject(subject_id='x', dataset_name='b'), Subject(subject_id='a', dataset_name='w'),
     Subject(subject_id='a', dataset_name='w'), Subject(subject_id='a', dataset_name='x'),
     Subject(subject_id='a', dataset_name='x'), Subject(subject_id='b', dataset_name='x'),
     Subject(subject_id='b', dataset_name='x'), Subject(subject_id='c', dataset_name='y')]
    """
    subject_id: str
    dataset_name: str

    def __getitem__(self, item):
        return getattr(self, item)

    def __lt__(self, other):
        if not isinstance(other, Subject):
            raise TypeError(f"The operand '<' is not implemented with rhs of type {type(other)}")

        # First, it is based on the dataset, second on the subject ID
        return (self.dataset_name, self.subject_id) < (other.dataset_name, other.subject_id)


# -----------------
# Base classes
# -----------------
class DataSplitBase(abc.ABC):

    __slots__ = ()

    @property
    @abc.abstractmethod
    def splits(self) -> Sequence[Tuple[Tuple[Subject, ...], Tuple[Subject, ...], Tuple[Subject, ...]]]:
        """
        Get the splits. First element is training, second is validation, third is testing. It is convenient with
        Sequence rather than Iterable because __len__ is needed to know how many splits (e.g., k in k-fold CV) there are
        """

    @property
    def all_subjects(self) -> Set[Subject]:
        """Get all subjects that are contained in the subject split"""
        splits = self.splits
        subjects = set(itertools.chain(*splits[0]))

        # Verify that it is consistent
        for split in splits[1:]:  # todo: consider removing this, as it can be tested instead...
            if set(itertools.chain(*split)) != subjects:
                raise RuntimeError("The union of subjects were inconsistent across splits")
        return subjects

    @property
    def all_datasets(self):
        # Todo: consider just using the first split, because it should work and this three for-loops...
        datasets: Set[str] = set()
        for single_split in self.splits:  # E.g., looping through many random splits
            for subject_set in single_split:  # Loop though train, val, and test set
                for subject in subject_set:
                    datasets.add(subject.dataset_name)
        return datasets


# -----------------
# Classes
# -----------------
class RandomSplitsTVTestHoldout(DataSplitBase):
    """
    Random splits where the test set is always the same

    Examples
    --------
    >>> f1_drivers = {"Mercedes": ("Hamilton", "Russel", "Wolff"), "Red Bull": ("Verstappen", "Checo"),
    ...               "Ferrari": ("Leclerc", "Smooth Sainz"), "McLaren": ("Norris", "Piastri"),
    ...               "Aston Martin": ("Alonso", "Stroll"), "Haas": ("Magnussen", "Hülkenberg")}
    >>> my_num_splits = 4
    >>> my_splits_obj = RandomSplitsTVTestHoldout(f1_drivers, val_split=0.2, test_split=0.3, sort_first=False,
    ...                                           num_random_splits=my_num_splits, seed=42)
    >>> len(my_splits_obj.all_subjects), type(my_splits_obj.all_subjects)
    (13, <class 'set'>)
    >>> my_splits = my_splits_obj.splits
    >>> len(my_splits) == my_num_splits, type(my_splits)
    (True, <class 'tuple'>)
    >>> tuple((len(my_split), type(my_split)) for my_split in my_splits)  # type: ignore
    ((3, <class 'tuple'>), (3, <class 'tuple'>), (3, <class 'tuple'>), (3, <class 'tuple'>))
    >>> all(isinstance(my_sub, Subject) for my_split in my_splits for my_split_subset in my_split  # type: ignore
    ...     for my_sub in my_split_subset)  # type: ignore
    True

    Train, val and test never overlaps

    >>> for my_train, my_val, my_test in my_splits:
    ...     len(my_train), len(my_val), len(my_test)
    ...     assert not set(my_train) & set(my_val)
    ...     assert not set(my_train) & set(my_test)
    ...     assert not set(my_val) & set(my_test)
    (7, 2, 4)
    (7, 2, 4)
    (7, 2, 4)
    (7, 2, 4)

    Test set is always the same

    >>> my_test_subjects = tuple(my_split[-1] for my_split in my_splits)  # type: ignore
    >>> for test_subjects in my_test_subjects:
    ...     assert test_subjects == my_test_subjects[0]

    But train and val are not

    >>> my_train_subjects = tuple(my_split[0] for my_split in my_splits)  # type: ignore
    >>> for train_subjects in my_train_subjects[1:]:
    ...     assert train_subjects != my_train_subjects[0]
    >>> my_val_subjects = tuple(my_split[1] for my_split in my_splits)  # type: ignore
    >>> for val_subjects in my_val_subjects:
    ...     val_subjects
    (Subject(subject_id='Verstappen', dataset_name='Red Bull'), Subject(subject_id='Norris', dataset_name='McLaren'))
    (Subject(subject_id='Magnussen', dataset_name='Haas'), Subject(subject_id='Verstappen', dataset_name='Red Bull'))
    (Subject(subject_id='Leclerc', dataset_name='Ferrari'), Subject(subject_id='Magnussen', dataset_name='Haas'))
    (Subject(subject_id='Smooth Sainz', dataset_name='Ferrari'), Subject(subject_id='Magnussen', dataset_name='Haas'))
    """

    __slots__ = ("_splits",)

    def __init__(self, dataset_subjects, *, val_split, test_split, num_random_splits, seed, sort_first):
        """
        Initialise

        Parameters
        ----------
        dataset_subjects : dict[str, tuple[str, ...]]
        val_split : float
        test_split : float
        num_random_splits : int
        seed : int, optional
        """
        # Maybe make data split reproducible
        rng = random.Random(seed)

        # ------------
        # Generate random splits for training/validation
        # ------------
        # Get all subjects to a list
        subjects: List[Subject] = []
        for dataset_name, subject_ids in dataset_subjects.items():
            # Add all subjects from the current non-test dataset
            subjects.extend(
                [Subject(dataset_name=dataset_name, subject_id=subject_id) for subject_id in subject_ids]
            )

        # Maybe sort it first
        if verify_type(sort_first, bool):
            subjects.sort()

        # Create splits
        splits: List[Tuple[Tuple[Subject, ...], Tuple[Subject, ...], Tuple[Subject, ...]]] = []

        # Extract test set (shuffling happens in '_split_randomly')
        non_test_subjects, test_subjects = _split_randomly(subjects=subjects, split_percent=test_split, rng=rng)
        for i in range(num_random_splits):
            non_test_subjects = non_test_subjects.copy()

            # Split into training/validation and test
            train_subjects, val_subjects = _split_randomly(subjects=non_test_subjects, split_percent=val_split, rng=rng)

            # Add to splits. The test set will be empty instead of non-existent for consistency reasons
            splits.append((tuple(train_subjects), tuple(val_subjects), tuple(test_subjects)))

        # Set the attribute
        self._splits = tuple(splits)

    # ---------------
    # Properties
    # ---------------
    @property
    def splits(self):
        return self._splits

    @property
    def test_set(self):
        """Get the test set"""
        # Get the test set per split (should be the same)
        test_sets = {split[-1] for split in self.splits}

        # Make check
        if len(test_sets) != 1:
            raise RuntimeError(f"Expected the test set to be consistent across splits, but found {len(test_sets)} "
                               f"unique ones")

        return tuple(test_sets)[0]


class RandomSplitsTV(DataSplitBase):
    """
    Random splits without really having a test set

    Examples
    --------
    >>> f1_drivers = {"Mercedes": ("Hamilton", "Russel", "Wolff"), "Red Bull": ("Verstappen", "Checo"),
    ...               "Ferrari": ("Leclerc", "Smooth Sainz"), "McLaren": ("Norris", "Piastri"),
    ...               "Aston Martin": ("Alonso", "Stroll"), "Haas": ("Magnussen", "Hülkenberg")}
    >>> my_num_splits = 4
    >>> my_splits = RandomSplitsTV(f1_drivers, val_split=0.2, num_random_splits=my_num_splits, seed=42,
    ...                            sort_first=False).splits
    >>> len(my_splits) == my_num_splits, type(my_splits)
    (True, <class 'tuple'>)
    >>> tuple((len(my_split), type(my_split)) for my_split in my_splits)  # type: ignore
    ((3, <class 'tuple'>), (3, <class 'tuple'>), (3, <class 'tuple'>), (3, <class 'tuple'>))
    >>> all(isinstance(my_sub, Subject) for my_split in my_splits for my_split_subset in my_split  # type: ignore
    ...     for my_sub in my_split_subset)  # type: ignore
    True

    Train, val and test never overlaps

    >>> for my_train, my_val, my_test in my_splits:
    ...     assert not set(my_train) & set(my_val)
    ...     assert not set(my_train) & set(my_test)
    ...     assert not set(my_val) & set(my_test)

    Test set is always empty

    >>> my_test_subjects = tuple(my_split[-1] for my_split in my_splits)  # type: ignore
    >>> for test_subjects in my_test_subjects:
    ...     test_subjects
    ()
    ()
    ()
    ()
    """

    __slots__ = ("_splits",)

    def __init__(self, dataset_subjects, *, val_split, num_random_splits, seed=None, sort_first):
        """
        Initialise

        Parameters
        ----------
        dataset_subjects : dict[str, tuple[str, ...]]
        val_split : float
        num_random_splits : int
        seed : int
        sort_first : bool
        """
        # Maybe make data split reproducible
        rng = random.Random(seed)

        # ------------
        # Generate random splits for training/validation
        # ------------
        subjects = []
        for dataset_name, subject_ids in dataset_subjects.items():
            # Add all subjects from the current non-test dataset
            subjects.extend(
                [Subject(dataset_name=dataset_name, subject_id=subject_id) for subject_id in subject_ids]
            )

        # Maybe sort it
        if verify_type(sort_first, bool):
            subjects.sort()

        # Create splits
        splits: List[Tuple[Tuple[Subject, ...], Tuple[Subject, ...], Tuple[Subject, ...]]] = []
        for i in range(num_random_splits):
            # Randomly shuffle the non-test subjects
            rng.shuffle(subjects)

            # Split into training and validation
            train_subjects, val_subjects = _split_randomly(subjects=subjects, split_percent=val_split, rng=rng)

            # Add to splits. The test set will be empty instead of non-existent for consistency reasons
            splits.append((tuple(train_subjects), tuple(val_subjects), ()))

        # Set the attribute
        self._splits = tuple(splits)

    # ---------------
    # Properties
    # ---------------
    @property
    def splits(self):
        return self._splits


class KFoldDataSplit(DataSplitBase):
    """
    Class for splitting the data into k folds. The different datasets are neglected

    Examples
    --------
    >>> f1_drivers = {"Mercedes": ("Hamilton", "Russel", "Wolff"), "Red Bull": ("Verstappen", "Checo"),
    ...               "Ferrari": ("Leclerc", "Smooth Sainz"), "McLaren": ("Norris", "Piastri"),
    ...               "Aston Martin": ("Alonso", "Stroll"), "Haas": ("Magnussen", "Hülkenberg")}
    >>> meaning_of_life = 42
    >>> my_splits = KFoldDataSplit(num_folds=4, dataset_subjects=f1_drivers, val_split=0.2, seed=meaning_of_life,
    ...                            sort_first=False).splits
    >>> len(my_splits), type(my_splits)
    (4, <class 'tuple'>)
    >>> tuple((len(my_split), type(my_split)) for my_split in my_splits)  # type: ignore
    ((3, <class 'tuple'>), (3, <class 'tuple'>), (3, <class 'tuple'>), (3, <class 'tuple'>))
    >>> all(isinstance(my_sub, Subject) for my_split in my_splits for my_split_subset in my_split  # type: ignore
    ...     for my_sub in my_split_subset)  # type: ignore
    True

    Train, val and test never overlaps

    >>> for my_train, my_val, my_test in my_splits:
    ...     assert not set(my_train) & set(my_val)
    ...     assert not set(my_train) & set(my_test)
    ...     assert not set(my_val) & set(my_test)

    Test subjects are never repeated

    >>> my_test_subjects = tuple(my_split[-1] for my_split in my_splits)  # type: ignore
    >>> for i, test_subjects in enumerate(my_test_subjects):
    ...     assert not set(test_subjects) & set(_leave_1_fold_out(i, my_test_subjects))
    """

    __slots__ = "_splits",

    def __init__(self, *, num_folds, dataset_subjects, val_split, seed=None, sort_first):
        """
        Initialise

        Parameters
        ----------
        num_folds : int
            Number of folds
        dataset_subjects : dict[str, tuple[str, ...]]
            Subject IDs. The keys are dataset names, the values are the subject IDs of the corresponding dataset
        seed : int, optional
            Seed for making the data split reproducible. If None, no seed is set
        val_split : float
        """
        # Pool all subjects together
        subjects = []
        for dataset_name, subject_ids in dataset_subjects.items():
            for sub_id in subject_ids:
                subjects.append(Subject(subject_id=sub_id, dataset_name=dataset_name))

        # Maybe make data split reproducible
        rng = random.Random(seed)

        # Maybe sort it to make order it invariant
        if verify_type(sort_first, bool):
            subjects.sort()

        # Shuffle
        rng.shuffle(subjects)

        # Perform split
        split = numpy.array_split(subjects, num_folds)  # type: ignore[arg-type, var-annotated]

        # Set attribute (and some type fix, type hinting and mypy stuff)
        folds: List[Tuple[Tuple[Subject, ...], Tuple[Subject, ...], Tuple[Subject, ...]]] = []
        for i, fold in enumerate(split):
            test = tuple(fold)
            non_test_subjects = _leave_1_fold_out(i, split)
            train, val = _split_randomly(subjects=non_test_subjects, split_percent=val_split, rng=rng)
            folds.append((tuple(train), tuple(val), test))

        # Make the k folds
        self._splits = tuple(folds)

    # ---------------
    # Properties
    # ---------------
    @property
    def splits(self):
        return self._splits


class LODOCV(DataSplitBase):
    """
    Class for leave-one-dataset-out cross validation

    Examples
    --------
    >>> f1_drivers = {"Mercedes": ("Hamilton", "Russel", "Wolff"), "Red Bull": ("Verstappen", "Checo"),
    ...               "Ferrari": ("Leclerc", "Smooth Sainz"), "McLaren": ("Norris", "Piastri"),
    ...               "Aston Martin": ("Alonso", "Stroll"), "Haas": ("Magnussen", "Hülkenberg")}
    >>> meaning_of_life = 42
    >>> my_splits = LODOCV(dataset_subjects=f1_drivers, seed=meaning_of_life, val_split=0.2, sort_first=False).splits
    >>> len(my_splits), type(my_splits)
    (6, <class 'tuple'>)
    >>> tuple((len(my_split), type(my_split)) for my_split in my_splits)  # type: ignore
    ... # doctest: +NORMALIZE_WHITESPACE
    ((3, <class 'tuple'>), (3, <class 'tuple'>), (3, <class 'tuple'>), (3, <class 'tuple'>), (3, <class 'tuple'>),
     (3, <class 'tuple'>))
    >>> all(isinstance(my_sub, Subject) for my_split in my_splits for my_split_subset in my_split  # type: ignore
    ...     for my_sub in my_split_subset)  # type: ignore
    True

    Train, val and test never overlaps

    >>> for my_train, my_val, my_test in my_splits:
    ...     assert not set(my_train) & set(my_val)
    ...     assert not set(my_train) & set(my_test)
    ...     assert not set(my_val) & set(my_test)

    Test datasets are not overlapping and contains all data from the left out dataset

    >>> for *_, my_test_subjects in my_splits:  # doctest: +NORMALIZE_WHITESPACE
    ...     my_test_subjects
    (Subject(subject_id='Piastri', dataset_name='McLaren'), Subject(subject_id='Norris', dataset_name='McLaren'))
    (Subject(subject_id='Checo', dataset_name='Red Bull'), Subject(subject_id='Verstappen', dataset_name='Red Bull'))
    (Subject(subject_id='Leclerc', dataset_name='Ferrari'), Subject(subject_id='Smooth Sainz', dataset_name='Ferrari'))
    (Subject(subject_id='Stroll', dataset_name='Aston Martin'), Subject(subject_id='Alonso',
                                                                        dataset_name='Aston Martin'))
    (Subject(subject_id='Russel', dataset_name='Mercedes'), Subject(subject_id='Hamilton', dataset_name='Mercedes'),
     Subject(subject_id='Wolff', dataset_name='Mercedes'))
    (Subject(subject_id='Hülkenberg', dataset_name='Haas'), Subject(subject_id='Magnussen', dataset_name='Haas'))
    """

    __slots__ = "_splits",

    def __init__(self, dataset_subjects, *, val_split, seed=None, sort_first):
        """
        Initialise

        Parameters
        ----------
        dataset_subjects : dict[str, tuple[str, ...]]
            Subject IDs. The keys are dataset names, the values are the subject IDs of the corresponding dataset
        seed : int, optional
            Seed for making the data split reproducible. If None, no seed is set
        val_split : float
            todo: setting val_split=0.2 does not force that all datasets have 20% in validation. This should be an
             option
        sort_first : bool
        """
        # Maybe make data split reproducible
        rng = random.Random(seed)

        # Loop though the datasets
        folds = []
        if verify_type(sort_first, bool):
            dataset_subjects = dict(sorted(dataset_subjects.items()))
        for dataset_name, subject_ids in dataset_subjects.items():
            # Fix type
            sub_ids = [Subject(dataset_name=dataset_name, subject_id=subject_id) for subject_id in subject_ids]

            # Maybe sort it to make order it invariant
            if verify_type(sort_first, bool):
                sub_ids.sort()

            # Shuffle
            rng.shuffle(sub_ids)

            # Add it as a tuple to the folds
            folds.append(tuple(sub_ids))

        # Shuffle the folds
        rng.shuffle(folds)

        # Now that each element contains all subjects from a single dataset, create the folds
        splits: List[Tuple[Tuple[Subject, ...], Tuple[Subject, ...], Tuple[Subject, ...]]] = []
        for i, fold in enumerate(folds):
            test_subjects = fold
            non_test_subjects = _leave_1_fold_out(i, folds)
            train, val = _split_randomly(subjects=non_test_subjects, split_percent=val_split, rng=rng)
            splits.append((tuple(train), tuple(val), test_subjects))

        # Set attribute
        self._splits = tuple(splits)

    # ---------------
    # Properties
    # ---------------
    @property
    def splits(self):
        return self._splits


class KeepDatasetsOutRandomSplits(DataSplitBase):
    """
    Class for keeping one or more specified datasets out as the test set. The remaining ones are used for training and
    validation with as many random splits as specified

    Examples
    --------
    >>> f1_drivers = {"Mercedes": ("Hamilton", "Russel", "Wolff"), "Red Bull": ("Verstappen", "Checo"),
    ...               "Ferrari": ("Leclerc", "Smooth Sainz"), "McLaren": ("Norris", "Piastri"),
    ...               "Aston Martin": ("Alonso", "Stroll"), "Haas": ("Magnussen", "Hülkenberg")}
    >>> my_num_splits = 5
    >>> my_splits = KeepDatasetsOutRandomSplits(f1_drivers, left_out_datasets="McLaren", val_split=0.2,
    ...                                         num_random_splits=my_num_splits, seed=42, sort_first=False).splits
    >>> len(my_splits), type(my_splits)
    (5, <class 'tuple'>)
    >>> tuple((len(my_split), type(my_split)) for my_split in my_splits)  # type: ignore
    ((3, <class 'tuple'>), (3, <class 'tuple'>), (3, <class 'tuple'>), (3, <class 'tuple'>), (3, <class 'tuple'>))
    >>> all(isinstance(my_sub, Subject) for my_split in my_splits for my_split_subset in my_split  # type: ignore
    ...     for my_sub in my_split_subset)  # type: ignore
    True

    Train, val and test never overlaps

    >>> for my_train, my_val, my_test in my_splits:
    ...     assert not set(my_train) & set(my_val)
    ...     assert not set(my_train) & set(my_test)
    ...     assert not set(my_val) & set(my_test)

    The test set will always be the same, and contain the entire test dataset

    >>> my_test_subjects = tuple(my_split[-1] for my_split in my_splits)  # type: ignore
    >>> for test_subjects in my_test_subjects:
    ...     test_subjects
    (Subject(subject_id='Norris', dataset_name='McLaren'), Subject(subject_id='Piastri', dataset_name='McLaren'))
    (Subject(subject_id='Norris', dataset_name='McLaren'), Subject(subject_id='Piastri', dataset_name='McLaren'))
    (Subject(subject_id='Norris', dataset_name='McLaren'), Subject(subject_id='Piastri', dataset_name='McLaren'))
    (Subject(subject_id='Norris', dataset_name='McLaren'), Subject(subject_id='Piastri', dataset_name='McLaren'))
    (Subject(subject_id='Norris', dataset_name='McLaren'), Subject(subject_id='Piastri', dataset_name='McLaren'))
    """

    __slots__ = "_splits",

    def __init__(self, dataset_subjects, *, left_out_datasets, val_split, num_random_splits, seed, sort_first):
        """
        Initialise

        Parameters
        ----------
        dataset_subjects : dict[str, tuple[str, ...]]
        left_out_datasets : str | tuple[str, ...]
            Dataset(s) to leave out as test set
        val_split : float
        num_random_splits : int
        seed : int, optional
        sort_first : bool
        """
        # Maybe make data split reproducible
        rng = random.Random(seed)

        if isinstance(left_out_datasets, str):
            left_out_datasets = (left_out_datasets,)
        # Just fix the test set
        test_subjects = tuple(Subject(dataset_name=left_out_dataset, subject_id=subject_id)
                              for left_out_dataset in left_out_datasets
                              for subject_id in dataset_subjects[left_out_dataset])

        # Maybe sort it first
        if verify_type(sort_first, bool):
            test_subjects = tuple(sorted(test_subjects))

        # ------------
        # Generate random splits for training/validation
        # ------------
        # Collect all non-test subjects
        non_test_subjects = []
        for dataset_name, subject_ids in dataset_subjects.items():
            # Skipping the test datasets
            if dataset_name in left_out_datasets:
                continue

            # Add all subjects from the current non-test dataset
            non_test_subjects.extend(
                [Subject(dataset_name=dataset_name, subject_id=subject_id) for subject_id in subject_ids]
            )

        if verify_type(sort_first, bool):
            non_test_subjects.sort()

        # Create splits
        splits: List[Tuple[Tuple[Subject, ...], Tuple[Subject, ...], Tuple[Subject, ...]]] = []
        for i in range(num_random_splits):
            # Randomly shuffle the non-test subjects
            rng.shuffle(non_test_subjects)

            # Split into training and validation
            train_subjects, val_subjects = _split_randomly(subjects=non_test_subjects, split_percent=val_split, rng=rng)

            # Add to splits
            splits.append((tuple(train_subjects), tuple(val_subjects), test_subjects))

        # Set the attribute
        self._splits = tuple(splits)

    # ---------------
    # Properties
    # ---------------
    @property
    def splits(self):
        return self._splits


# -----------------
# Functions
# -----------------
def get_available_splits():
    """All available data splits must be included here"""
    return KFoldDataSplit, LODOCV, KeepDatasetsOutRandomSplits, RandomSplitsTV, RandomSplitsTVTestHoldout


def get_data_split(split, **kwargs):
    """
    Function for getting the specified data split

    Parameters
    ----------
    split : str
    kwargs

    Returns
    -------
    DataSplitBase
    """
    available_splits = get_available_splits()

    # Loop through and select the correct one
    for split_class in available_splits:
        if split == split_class.__name__:
            return split_class(**kwargs)

    # If no match, an error is raised
    raise ValueError(f"The data split '{split}' was not recognised. Please select among the following: "
                     f"{tuple(split_class.__name__ for split_class in available_splits)}")


def subjects_tuple_to_dict(subjects) -> Dict[str, Tuple[str, ...]]:
    """
    Function for converting from a tuple of subjects to a dictionary with dataset name as keys and tuple of subject IDs
    as values. This function is convenient for converting to the format expected in the classes inheriting from
    'DataSplitBase'

    Parameters
    ----------
    subjects : typing.Iterable[Subject]

    Returns
    -------
    dict[str, tuple[str, ...]]

    Examples
    --------
    >>> my_subjects = (Subject("S1", "D1"), Subject("S2", "D1"), Subject("S3", "D1"), Subject("S1", "D2"),
    ...     Subject("S2", "D2"), Subject("P1", "D3"), Subject("P2", "D3"), Subject("P3", "D3"), Subject("P1", "D4"))
    >>> subjects_tuple_to_dict(my_subjects)
    {'D1': ('S1', 'S2', 'S3'), 'D2': ('S1', 'S2'), 'D3': ('P1', 'P2', 'P3'), 'D4': ('P1',)}

    Works for set as well (although, the order gets arbitrary for sets)

    >>> my_subjects_set = {Subject("S1", "D1"), Subject("S2", "D1"), Subject("S3", "D1"), Subject("S1", "D2"),
    ...     Subject("S2", "D2"), Subject("P1", "D3"), Subject("P2", "D3"), Subject("P3", "D3"), Subject("P1", "D4")}
    >>> my_subjects_dict = subjects_tuple_to_dict(my_subjects_set)
    >>> all((isinstance(my_subjects_dict["D1"], tuple), (set(my_subjects_dict["D1"]) == {"S1", "S2", "S3"}),
    ...      isinstance(my_subjects_dict["D2"], tuple), (set(my_subjects_dict["D2"]) == {"S1", "S2"}),
    ...      isinstance(my_subjects_dict["D3"], tuple), (set(my_subjects_dict["D3"]) == {"P1", "P2", "P3"}),
    ...      isinstance(my_subjects_dict["D4"], tuple), (set(my_subjects_dict["D4"]) == {"P1"})))
    True

    It even works when the subjects are keys in dictionaries (the keys are looped over by default)

    >>> my_subjects_dict = {sub: "Does not matter the values" for sub in my_subjects}  # type: ignore[attr-defined]
    >>> subjects_tuple_to_dict(my_subjects_dict)
    {'D1': ('S1', 'S2', 'S3'), 'D2': ('S1', 'S2'), 'D3': ('P1', 'P2', 'P3'), 'D4': ('P1',)}
    """
    dataset_dict: Dict[str, List[str]] = {}
    for subject in subjects:
        # Do a type check
        if not isinstance(subject, Subject):
            raise TypeError(f"Expected all subjects to be of type {Subject.__name__}, but found {subject}")

        # Add to dict
        dataset_name = subject.dataset_name

        if dataset_name not in dataset_dict:
            dataset_dict[dataset_name] = [subject.subject_id]
        else:
            dataset_dict[dataset_name].append(subject.subject_id)

    # Return with tuple values
    return {name: tuple(subject_ids) for name, subject_ids in dataset_dict.items()}


def simple_random_split(subjects, split_percent, seed, require_seeding, sort_first):
    """
    Function for splitting subjects into two. Given that 'sort_first' is True, it will reproduce the same test set as
    the test set of 'RandomSplitsTVTestHoldout' (see test folder)

    Parameters
    ----------
    subjects : tuple[Subject, ...]
    split_percent : float
    seed : int, optional
        A seed which is passed to initialise the random number generator of random
    require_seeding : bool
        A boolean which indicates if seeding should be required (True) or not (False). Ignored if a seed is given.
    sort_first : bool
        A boolean which indicates if the subjects should be sorted prior to the splitting. This ensures that the
        splitting is invariant to the input order of the subjects

    Returns
    -------
    tuple[tuple[Subject, ...], tuple[Subject, ...]]

    Examples
    --------
    >>> my_subjects = (Subject("S1", "D1"), Subject("S2", "D1"), Subject("S3", "D1"), Subject("S1", "D2"),
    ...                Subject("S2", "D2"), Subject("P1", "D3"), Subject("P2", "D3"), Subject("S3", "D3"),
    ...                Subject("S4", "D3"), Subject("S1", "D4"), Subject("S2", "D4"), Subject("P1", "D5"))
    >>> simple_random_split(subjects=my_subjects, split_percent=0.2, seed=42,
    ...                     require_seeding=False, sort_first=False)  # doctest: +NORMALIZE_WHITESPACE
    ((Subject(subject_id='S3', dataset_name='D3'), Subject(subject_id='P1', dataset_name='D3'),
      Subject(subject_id='S3', dataset_name='D1'), Subject(subject_id='S4', dataset_name='D3'),
      Subject(subject_id='S1', dataset_name='D4'), Subject(subject_id='P2', dataset_name='D3'),
      Subject(subject_id='P1', dataset_name='D5'), Subject(subject_id='S1', dataset_name='D2'),
      Subject(subject_id='S2', dataset_name='D2')),
     (Subject(subject_id='S1', dataset_name='D1'), Subject(subject_id='S2', dataset_name='D1'),
      Subject(subject_id='S2', dataset_name='D4')))

    If 'sort_first' is True, then the split is invariant to input order

    >>> my_subjects_2 = list(my_subjects)
    >>> random.seed(42)
    >>> random.shuffle(my_subjects_2)
    >>> my_split_1 = simple_random_split(subjects=my_subjects, split_percent=0.2, seed=42,
    ...                                  require_seeding=False, sort_first=True)
    >>> my_split_2 = simple_random_split(subjects=tuple(my_subjects_2), split_percent=0.2, seed=42,
    ...                                  require_seeding=False, sort_first=True)
    >>> my_split_1 == my_split_2
    True

    If 'sort_first' is False, then the split is NOT invariant to input order

    >>> my_split_1 = simple_random_split(subjects=my_subjects, split_percent=0.2, seed=42,
    ...                                  require_seeding=False, sort_first=False)
    >>> my_split_2 = simple_random_split(subjects=tuple(my_subjects_2), split_percent=0.2, seed=42,
    ...                                  require_seeding=False, sort_first=False)
    >>> my_split_1 == my_split_2
    False
    """
    # Maybe sort it first
    if verify_type(sort_first, bool):
        subjects = tuple(sorted(subjects))

    # Maybe make the split reproducible
    if seed is None:
        if not isinstance(require_seeding, bool):
            raise TypeError(f"Expected 'require_seeding' argument to be boolean, but found {type(require_seeding)}")
        if require_seeding:
            raise RuntimeError("Seeding for reproducibility was required, but not given")
    rng = random.Random(seed)

    # Generate splits
    set_1, set_2 = _split_randomly(subjects=subjects, split_percent=split_percent, rng=rng)
    return tuple(set_1), tuple(set_2)


def _leave_1_fold_out(i, folds) -> Tuple[Subject, ...]:
    """
    Method for selecting all subjects except for one fold (the i-th fold)

    Parameters
    ----------
    i : int
        Fold to not include
    folds : tuple[tuple[Subject, ...], ...]

    Returns
    -------
    tuple[Subject, ...]

    Examples
    --------
    >>> my_splits = ((Subject("TW", "Merc"), Subject("MV", "RB"), Subject("LN", "McL")),
    ...              (Subject("YT", "AT"), Subject("CS", "F")), (Subject("CL", "F"), Subject("VB", "AR")),
    ...              (Subject("FA", "AM"), Subject("LS", "AM"), Subject("DH", "RB")))
    >>> _leave_1_fold_out(2, my_splits)  # doctest: +NORMALIZE_WHITESPACE
    (Subject(subject_id='TW', dataset_name='Merc'), Subject(subject_id='MV', dataset_name='RB'),
     Subject(subject_id='LN', dataset_name='McL'), Subject(subject_id='YT', dataset_name='AT'),
     Subject(subject_id='CS', dataset_name='F'), Subject(subject_id='FA', dataset_name='AM'),
     Subject(subject_id='LS', dataset_name='AM'), Subject(subject_id='DH', dataset_name='RB'))
    >>> _leave_1_fold_out(-1, my_splits)  # doctest: +NORMALIZE_WHITESPACE
    (Subject(subject_id='TW', dataset_name='Merc'), Subject(subject_id='MV', dataset_name='RB'),
     Subject(subject_id='LN', dataset_name='McL'), Subject(subject_id='YT', dataset_name='AT'),
     Subject(subject_id='CS', dataset_name='F'), Subject(subject_id='CL', dataset_name='F'),
     Subject(subject_id='VB', dataset_name='AR'))

    """
    # Handle negative index
    i = len(folds) + i if i < 0 else i

    # Return as unpacked tuple
    return tuple(itertools.chain(*tuple(fold for j, fold in enumerate(folds) if j != i)))


def _split_randomly(subjects, split_percent, rng: random.Random):
    # Input checks
    assert all(isinstance(subject, (Subject, str)) for subject in subjects)
    assert isinstance(split_percent, float)
    assert 0 < split_percent < 1

    # Make a list and a copy
    subjects = list(subjects)

    # Shuffle randomly
    rng.shuffle(subjects)

    # Split by the percentage
    num_subjects = len(subjects)
    split_idx = int(num_subjects * (1 - split_percent))

    return subjects[:split_idx], subjects[split_idx:]
