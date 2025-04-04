import abc
import copy
from collections.abc import Mapping
from functools import reduce
from typing import Any, Dict

import numpy
import torch
from torch.autograd import Function

from elecssl.models.hp_suggesting import get_optuna_sampler


# ---------------
# Classes
# ---------------
class ReverseLayerF(Function):
    """
    Gradient reversal layer. This is simply copypasted (with added '# noqa') from the implementation at
    https://github.com/wogong/pytorch-dann/blob/master/models/functions.py

    See LICENSE.txt for their original copyright notice and disclaimer
    """

    @staticmethod
    def forward(ctx, x, alpha):  # noqa
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):  # noqa
        output = grad_output.neg() * ctx.alpha
        return output, None


# ---------------
# Functions for training
# ---------------
def tensor_dict_to_device(tensors, device):
    """
    Send a dictionary containing tensors to device

    Parameters
    ----------
    tensors : dict[str, torch.Tensor] | None
    device : torch.device

    Returns
    -------
    dict[str, torch.Tensor]
    """
    # If the tensor is None, then None is returned
    if tensors is None:
        return None

    # Input check
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors.values()):
        raise TypeError(f"Expected all values in the dictionary to be torch tensors, but found "
                        f"{set(type(tensor) for tensor in tensors.values())}")

    # Send to device and return
    return {dataset_name: tensor.to(device) for dataset_name, tensor in tensors.items()}


def flatten_targets(tensors):
    """
    Flatten the targets. The sorting is determined from the ordering of the keys

    Parameters
    ----------
    tensors : dict[str, torch.Tensor | numpy.ndarray]

    Returns
    -------
    torch.Tensor

    Examples
    --------
    >>> my_targets = {"d1": torch.ones(size=(10, 1)), "d2": torch.ones(size=(11, 1)) * 2,
    ...               "d3": torch.ones(size=(5, 1)) * 3}
    >>> my_output = flatten_targets(my_targets)
    >>> my_output.size()
    torch.Size([26, 1])

    The ordering is determined from the keys

    >>> torch.equal(my_output[:10], torch.ones(size=(10, 1)))
    True
    >>> torch.equal(my_output[10:21], 2 * torch.ones(size=(11, 1)))
    True
    >>> torch.equal(my_output[21:], 3 * torch.ones(size=(5, 1)))
    True
    """
    # Maybe convert to torch tensors
    if all(isinstance(tensor, numpy.ndarray) for tensor in tensors.values()):
        tensors = {dataset_name: torch.tensor(tensor, dtype=torch.float) for dataset_name, tensor in tensors.items()}

    # Flatten
    targets = torch.cat(tuple(tensor for tensor in tensors.values()), dim=0)

    return targets


# -------------------------
# Random number classes
# -------------------------
class RandomBase(abc.ABC):

    __slots__ = ()

    @abc.abstractmethod
    def draw(self, seed=None):
        """
        Draw a sample from the distribution

        Parameters
        ----------
        seed : int
            A seed can be passed for reproducibility

        Returns
        -------
        float
            A sample from the distribution
        """

    @abc.abstractmethod
    def scale(self, draw):
        """
        Method for scaling a random  draw

        Parameters
        ----------
        draw : float

        Returns
        -------
        float
        """


class UnivariateNormal(RandomBase):
    """
    Class for drawing samples from a univariate normal distribution. To reproduce, numpy.random.seed must be called
    """

    __slots__ = "_mean", "_std"

    def __init__(self, mean=0, std=1):
        """
        Initialisation method

        Parameters
        ----------
        std : float
            Standard deviation of the normal distribution
        mean : float
            Mean of the normal distribution
        """
        # ------------------
        # Set attributes
        # ------------------
        self._mean = mean
        self._std = std

    def draw(self, seed=None):
        """
        Examples
        -------
        >>> UnivariateNormal(3, 0.5).draw(seed=1)  # doctest: +ELLIPSIS
        3.812...

        It is the same to set seed outside, as passing the seed to the method

        >>> numpy.random.seed(1)
        >>> UnivariateNormal(3, 0.5).draw()  # doctest: +ELLIPSIS
        3.812...
        """
        # Maybe set seed
        if seed is not None:
            numpy.random.seed(seed)

        # Draw a sample from the distribution and return
        return numpy.random.normal(loc=self._mean, scale=self._std)

    def scale(self, draw):
        """
        Scaling method which subtracts the mean and divide by standard deviation

        Parameters
        ----------
        draw : float

        Returns
        -------
        float

        Examples
        --------
        >>> UnivariateNormal(3, 0.5).scale(3.5)
        1.0
        >>> round(UnivariateNormal(3, 0.5).scale(2.7), 5)
        -0.6
        """
        return (draw - self._mean) / self._std


class UnivariateUniform(RandomBase):

    __slots__ = "_lower", "_upper"

    def __init__(self, lower, upper):
        """
        Initialisation

        Parameters
        ----------
        lower : float
            Lower bound for the uniform distribution
        upper : float
            Upper bound for the uniform distribution
        """
        # ------------------
        # Set attributes
        # ------------------
        self._lower = lower
        self._upper = upper

    def draw(self, seed=None):
        """
        Examples:
        >>> UnivariateUniform(-1, 1).draw(seed=1)  # doctest: +ELLIPSIS
        -0.165...

        It is the same to set seed outside, as passing the seed to the method

        >>> numpy.random.seed(1)
        >>> UnivariateUniform(-1, 1).draw()  # doctest: +ELLIPSIS
        -0.165...
        """
        # Maybe set seed
        if seed is not None:
            numpy.random.seed(seed)

        # Draw a sample from the distribution and return
        return numpy.random.uniform(low=self._lower, high=self._upper)

    def scale(self, draw):
        """
        Scales such that the distribution is U[-1, 1]

        Parameters
        ----------
        draw : float

        Returns
        -------
        float

        Examples
        --------
        >>> UnivariateUniform(9, 10).scale(9)
        -1.0
        >>> UnivariateUniform(9, 10).scale(10)
        1.0
        >>> UnivariateUniform(9, 10).scale(9.5)
        0.0
        >>> UnivariateUniform(9, 10).scale(9.75)
        0.5
        """
        # Subtract mean
        draw -= (self._upper + self._lower) / 2

        # Scale interval
        draw /= (self._upper - self._lower) / 2

        return draw

    # --------------
    # Properties
    # --------------
    @property
    def lower(self):
        return self._lower

    @property
    def upper(self):
        return self._upper


def get_random_distribution(distribution, **kwargs):
    """
    Function for getting the specified random distribution

    Parameters
    ----------
    distribution : str
    kwargs

    Returns
    -------
    RandomBase
    """
    # All available distributions must be included here
    available_distributions = (UnivariateUniform, UnivariateNormal)

    # Loop through and select the correct one
    for dist in available_distributions:
        if distribution == dist.__name__:
            return dist(**kwargs)

    # If no match, an error is raised
    raise ValueError(f"The random distribution '{distribution}' was not recognised. Please select among the following: "
                     f"{tuple(dist.__name__ for dist in available_distributions)}")


# ------------
# Convenient functions
# ------------
def remove_prefix(s, *, prefix, case_sensitive):
    """
    Remove a prefix from a string.

    This function removes the specified prefix from the beginning of the string.
    If the string does not start with the prefix, a ValueError is raised.

    Parameters
    ----------
    s : str
        The input string.
    prefix : str
        The prefix to remove.
    case_sensitive : bool
        If True (default), the prefix comparison is case-sensitive. If False, the prefix is matched case-insensitively.

    Returns
    -------
    str
        The string with the prefix removed.

    Raises
    ------
    ValueError
        If the string does not start with the given prefix.

    Examples
    --------
    >>> remove_prefix("HelloWorld", prefix="Hello", case_sensitive=True)
    'World'
    >>> remove_prefix("Python3.8", prefix="Python", case_sensitive=False)
    '3.8'
    >>> remove_prefix("HelloWorld", prefix="hello", case_sensitive=True)
    Traceback (most recent call last):
    ...
    ValueError: String does not start with prefix: 'hello'
    >>> remove_prefix("HelloWorld", prefix="hello", case_sensitive=False)
    'World'
    >>> remove_prefix("HelloWorld", prefix="Bye", case_sensitive=False)
    Traceback (most recent call last):
    ...
    ValueError: String does not start with prefix: 'Bye'
    """
    if case_sensitive:
        if not s.startswith(prefix):
            raise ValueError(f"String does not start with prefix: {prefix!r}")
    else:
        if not s.lower().startswith(prefix.lower()):
            raise ValueError(f"String does not start with prefix: {prefix!r}")
    return s[len(prefix):]


def remove_prefix_from_keys(d, prefix):
    """
    Remove a given prefix from all keys in the dictionary.

    If a key does not start with the prefix, it remains unchanged.
    If removing the prefix results in a duplicate key, a `ValueError` is raised.

    Parameters
    ----------
    d : dict[str, Any]
        The input dictionary with string keys.
    prefix : str
        The prefix to remove from keys.

    Returns
    -------
    dict[str, Any]
        A new dictionary with the prefix removed where applicable.

    Raises
    ------
    ValueError
        If removing the prefix results in duplicate keys.

    Examples
    --------
    >>> remove_prefix_from_keys({"pre_a": 1, "pre_b": 2, "c": 3}, "pre_")
    {'a': 1, 'b': 2, 'c': 3}

    Keys that do not start with the prefix remain unchanged

    >>> remove_prefix_from_keys({"x_a": 1, "x_b": 2, "c": 3}, "pre_")
    {'x_a': 1, 'x_b': 2, 'c': 3}

    If removing the prefix causes a duplicate key, a `ValueError` is raised

    >>> remove_prefix_from_keys({"pre_x": 1, "pre_x_y": 2, "x": 3}, "pre_")
    Traceback (most recent call last):
    ...
    ValueError: Key collision detected: 'x' already exists.
    """
    new_dict = {}
    prefix_len = len(prefix)
    d = copy.deepcopy(d)

    for key, value in d.items():
        if key.startswith(prefix):
            new_key = key[prefix_len:]
            if new_key in d:
                raise ValueError(f"Key collision detected: '{new_key}' already exists.")
            new_dict[new_key] = value
        else:
            new_dict[key] = value  # Keep the original key if no prefix

    return new_dict


def verify_type(data, types):
    """Function which checks the type of the object. Returns the object if the type is as expected, otherwise raises a
    TypeError"""
    if isinstance(data, types):
        return data
    raise TypeError(f"Failed when trying to verify type. Expected input to be of type(s) {types}, but found "
                    f"{type(data)}")


def verified_performance_score(score, metric):
    """
    Returns a verified performance score, handling NaN values appropriately.

    If the score is NaN:
    - Returns `0.0` for correlation-based metrics.
    - Raises a `ValueError` for other metrics.

    Parameters
    ----------
    score : float
        The performance score to verify.
    metric : str
        The name of the metric associated with the score.

    Returns
    -------
    float
        The original score if it is valid, or a default value if the score is NaN.

    Raises
    ------
    ValueError
        If the score is NaN and no default behavior is implemented for the given metric.

    Examples
    --------
    >>> verified_performance_score(0.85, "accuracy")
    0.85
    >>> verified_performance_score(float("nan"), "pearson_r")
    0.0
    >>> verified_performance_score(float("nan"), "unknown_metric")
    Traceback (most recent call last):
    ...
    ValueError: Received the non-numeric score 'nan' for a metric 'unknown_metric' without an implemented default
    """
    # Not a problem
    if not numpy.isnan(score):
        return score

    # We set correlations to 0
    if metric in ("pearson_r", "spearman_rho"):
        return 0.0

    raise ValueError(f"Received the non-numeric score '{score}' for a metric '{metric}' without an implemented default")


def merge_dicts(*dicts):
    """
    Recursively merges multiple nested dictionaries without modifying the inputs. Convenient for merging dictionaries
    loaded from .yaml files, where there is a hierarchical system. It was mostly written by ChatGPT

    If two dictionaries have the same key:
    - If the values are dictionaries, they are merged recursively.
    - Otherwise, the value from the last dictionary is used.

    Notes
    -----
    This function ensures that the input dictionaries remain unchanged.

    Parameters
    ----------
    *dicts : dict
        One or more dictionaries to be merged.

    Returns
    -------
    dict
        A new dictionary with merged values.

    Examples
    --------
    >>> d1 = {"a": {"x": 1, "y": 2}, "b": {"z": 3}}
    >>> d2 = {"a": {"y": 20, "z": 30}, "b": {"w": 40}}
    >>> d3 = {"c": {"m": 50}}
    >>> merge_dicts(d1, d2, d3)
    {'a': {'x': 1, 'y': 20, 'z': 30}, 'b': {'z': 3, 'w': 40}, 'c': {'m': 50}}
    >>> merge_dicts(d1)
    {'a': {'x': 1, 'y': 2}, 'b': {'z': 3}}
    >>> merge_dicts(d1, {})
    {'a': {'x': 1, 'y': 2}, 'b': {'z': 3}}

    The input dictionaries remain unchanged

    >>> d1, d2, d3  # doctest: +NORMALIZE_WHITESPACE
    ({'a': {'x': 1, 'y': 2}, 'b': {'z': 3}},
     {'a': {'y': 20, 'z': 30}, 'b': {'w': 40}},
     {'c': {'m': 50}})
    """
    def merge_two(d1, d2):
        merged = copy.deepcopy(d1)  # Ensure immutability
        for key, value in d2.items():
            if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
                merged[key] = merge_two(merged[key], value)  # Recursive merge
            else:
                merged[key] = copy.deepcopy(value)  # Copy value to avoid mutation
        return merged

    return reduce(merge_two, dicts, {})


def merge_dicts_strict(*dicts):
    """
    Merge multiple dictionaries into a new dictionary without modifying inputs. Written by ChatGPT

    Ensures that no duplicate keys exist between the dictionaries.
    Raises a KeyError if any duplicate keys are found.

    Parameters
    ----------
    *dicts : dict
        Any number of dictionaries to be merged.

    Returns
    -------
    dict
        A new dictionary containing all key-value pairs from input dictionaries.

    Raises
    ------
    KeyError
        If any duplicate keys are found across the input dictionaries.

    Examples
    --------
    >>> merge_dicts_strict({"a": 1, "b": 2}, {"c": 3, "d": 4})
    {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    >>> merge_dicts_strict({"x": 10}, {"y": 20}, {"z": 30})
    {'x': 10, 'y': 20, 'z': 30}
    >>> merge_dicts_strict({"p": 1, "q": 2}, {"q": 3, "r": 4})
    Traceback (most recent call last):
    ...
    KeyError: "Duplicate keys found: {'q'}"
    """
    merged: Dict[str, Any] = {}  # Create a new dictionary to avoid modifying inputs
    for d in dicts:
        duplicate_keys = merged.keys() & d.keys()
        if duplicate_keys:
            raise KeyError(f"Duplicate keys found: {duplicate_keys}")
        merged = {**merged, **copy.deepcopy(d)}  # Merge safely without modifying inputs
    return merged


# -------------------------
# Functions for yaml loader
# -------------------------
# For experiments config file (not related to HP sampling/suggestions)
def _yaml_get_keys(loader, node):
    """Get the keys of a dictionary"""
    return tuple(loader.construct_mapping(node).keys())


def _yaml_get_hpo_sampler(loader, node):
    """Get an optuna.sampler"""
    sampler, kwargs = loader.construct_sequence(node, deep=True)
    return get_optuna_sampler(sampler, **kwargs)


# For HPO distributions config file
def _yaml_optuna_categorical(loader, node):
    """To be used in combination with optuna suggest categorical"""
    loader.construct_mapping(node, deep=True)
    return "categorical", loader.construct_mapping(node, deep=True)


def _yaml_optuna_float(loader, node):
    """To be used in combination with optuna suggest float"""
    loader.construct_mapping(node, deep=True)
    return "float", loader.construct_mapping(node, deep=True)


def _yaml_optuna_int(loader, node):
    """To be used in combination with optuna suggest int"""
    loader.construct_mapping(node, deep=True)
    return "int", loader.construct_mapping(node, deep=True)


def _yaml_optuna_categorical_dict(loader, node):
    """This one is supposed to be used when a categorical suggestion should be made, but the type is inconvenient. For
    example, if the choices are three lists, you should give them names (keys) and sample the names instead. Because
    optuna documentation prefers 'CategoricalChoiceType', which (at the time of writing this code) includes None, bool,
    int, float, and str"""
    loader.construct_mapping(node, deep=True)
    return "categorical_dict", loader.construct_mapping(node, deep=True)


def _yaml_optuna_not_a_hyperparameter_list(loader, node):
    """This is supposed to be used if it is convenient to loop through HPs, but there are some configurations which
    should not be registered by the trial object"""
    return "not_a_hyperparameter", loader.construct_sequence(node, deep=True)


def _yaml_optuna_not_a_hyperparameter_scalar(loader, node):
    """This is supposed to be used if it is convenient to loop through HPs (or store something in the HPD config file),
    but you don't actually want it to be registered as a HP by the trial object"""
    return "not_a_hyperparameter", loader.construct_scalar(node)


def add_yaml_constructors(loader):
    """
    Function for adding varied needed formatters to yaml loader

    Parameters
    ----------
    loader

    Returns
    -------
    typing.Type[yaml.SafeLoader]
    """
    # Convenient non-NPO distributions related ones
    loader.add_constructor("!GetKeys", _yaml_get_keys)
    loader.add_constructor("!HPOSampler", _yaml_get_hpo_sampler)

    # To be used with optuna
    loader.add_constructor("!Categorical", _yaml_optuna_categorical)
    loader.add_constructor("!Float", _yaml_optuna_float)
    loader.add_constructor("!Int", _yaml_optuna_int)
    loader.add_constructor("!CategoricalDict", _yaml_optuna_categorical_dict)
    loader.add_constructor("!NotAHyperparameterList", _yaml_optuna_not_a_hyperparameter_list)
    loader.add_constructor("!NotAHyperparameterScalar", _yaml_optuna_not_a_hyperparameter_scalar)
    return loader
