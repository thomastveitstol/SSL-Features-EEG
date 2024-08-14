import dataclasses


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
    """
    subject_id: str
    dataset_name: str

    def __getitem__(self, item):
        return getattr(self, item)

# todo: removed classes and functions for splitting
