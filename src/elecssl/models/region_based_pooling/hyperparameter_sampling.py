import random
from typing import Any, Dict


def generate_partition_sizes(*, n, k):
    """
    Function for randomly assigning cardinalities to subsets of a set of length n to partition. Any solution in positive
    integers to the of the equation x_1 + x_2 + ... + x_k = n is ok.

    Parameters
    ----------
    n : int
        Number of montage splits
    k : int
        Number of partitions

    Returns
    -------
    tuple[int, ...]

    Examples
    --------
    >>> random.seed(2)
    >>> generate_partition_sizes(n=10, k=3)
    (5, 2, 3)

    The sum will always equal n

    >>> all(sum(generate_partition_sizes(n=n_, k=k_)) == n_  # type: ignore[attr-defined]
    ...         for n_, k_ in zip((10, 20, 15, 64), (5, 10, 5, 33)))
    True
    """
    # Generate k 'cardinalities'
    cardinalities = [1 for _ in range(k)]

    # Iteratively increment the sizes
    for _ in range(n-k):
        # Increment a randomly selected cardinality
        cardinalities[random.randint(0, k-1)] += 1

    # Return as tuple
    return tuple(cardinalities)


# ------------
# yaml constructors
# ------------
def yaml_generate_partition_sizes(loader, node):
    n, k = loader.construct_sequence(node)
    return generate_partition_sizes(n=n, k=k)


def _traverse_and_call(obj):
    """Function which is used for making calls when a list/dict contains callables. This function was written by
    ChatGPT"""
    if isinstance(obj, dict):
        # Recursively traverse dictionaries
        return {key: _traverse_and_call(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        # Recursively traverse lists
        return [_traverse_and_call(item) for item in obj]
    elif callable(obj):
        # Call the function if the value is callable
        return _traverse_and_call(obj())
    else:
        # Return the value as is if it's not a dictionary, list, or callable
        return obj


def yaml_sample_rbp(loader, node):
    dict_ = loader.construct_mapping(node, deep=True)

    partitions = dict_["partitions"]
    designs = dict_["designs"]

    rbp_designs: Dict[str, Dict[str, Any]] = dict()

    for i, k in enumerate(partitions):
        rbp_name = f"RBPDesign{i}"
        rbp_designs[rbp_name] = dict()

        # Number of designs
        rbp_designs[rbp_name]["num_designs"] = designs["num_designs"]  # Should be 1

        # Pooling type
        rbp_designs[rbp_name]["pooling_type"] = designs["pooling_type"]  # Should be multi_cs

        # Pooling modules
        _pooling_module = _traverse_and_call(designs["pooling_module"])
        rbp_designs[rbp_name]["pooling_methods"] = _pooling_module["pooling_method"]
        rbp_designs[rbp_name]["pooling_methods_kwargs"] = _pooling_module["pooling_method_kwargs"]

        # Montage splits
        _montage_splits = [_traverse_and_call(designs["montage_splits"]) for _ in range(k)]
        rbp_designs[rbp_name]["split_methods"] = [split["name"] for split in _montage_splits]
        rbp_designs[rbp_name]["split_methods_kwargs"] = [split["kwargs"] for split in _montage_splits]

        # CMMN
        _cmmn = _traverse_and_call(designs["cmmn"])
        rbp_designs[rbp_name]["use_cmmn_layer"] = _cmmn["use_cmmn_layer"]
        rbp_designs[rbp_name]["cmmn_kwargs"] = _cmmn["cmmn_kwargs"]

    return rbp_designs
