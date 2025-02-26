import random


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
