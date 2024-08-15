"""
Contains only a function for returning a specified region split
"""
from elecssl.models.region_based_pooling.montage_splits.centroid_polygon import CentroidPolygons


def get_montage_split(split_method, **kwargs):
    """
    Function for getting the specified montage split.

    Parameters
    ----------
    split_method : str
        Name of the region split class name
    kwargs
        Key word arguments, which depends on the selected region split

    Returns
    -------
    elecssl.models.region_based_pooling.montage_splits.montage_split_base.MontageSplitBase
        The region splits

    Examples
    --------
    >>> _ = get_montage_split("CentroidPolygons", k=(5, 3), channel_positions=("SRM", "LEMON"),
    ...                       min_nodes=2)
    >>> get_montage_split("NotASplitMethod")  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: The split method 'NotASplitMethod' was not recognised. Please select among the following:
    ('CentroidPolygons',...)
    """
    # All available regions splits must be included here
    available_region_splits = (CentroidPolygons,)

    # Loop through and select the correct one
    for region_split in available_region_splits:
        if split_method == region_split.__name__:
            return region_split(**kwargs)

    # If no match, an error is raised
    raise ValueError(f"The split method '{split_method}' was not recognised. Please select among the following: "
                     f"{tuple(pooling_module.__name__ for pooling_module in available_region_splits)}")
