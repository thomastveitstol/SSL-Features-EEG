import abc
import random

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


# -------------------------
# Functions for yaml loader
# -------------------------
def _yaml_str_format(loader, node):
    """Formatting strings"""
    string_to_format = loader.construct_mapping(node, deep=True)
    return string_to_format["string"].format(**string_to_format["kwargs"])


def _yaml_multiplier_int(loader, node):
    """Multiplies all arguments together"""
    return int(numpy.prod(loader.construct_sequence(node)))


def _yaml_select_from_dict(loader, node):
    """Select the correct element from a dictionary"""
    dict_, key_ = loader.construct_sequence(node, deep=True)
    return dict_[key_]


def _yaml_mapping_length(loader, node):
    """Return the length of a mapping. With probably too many constructors, anchors, and aliases, it was easier to send
    the aliased dict as an element to a list of length one"""
    return len(loader.construct_sequence(node)[0])


def _yaml_sum(loader, node):
    """Computes the sum of objects"""
    return sum(loader.construct_sequence(node))


def _yaml_if_none_else(loader, node):
    """Returns one of two alternatives, depending on if a condition is None or not"""
    options = loader.construct_mapping(node, deep=True)
    if options["condition"] is None:
        return options[True]
    else:
        return options[False]


def _yaml_if_zero_else(loader, node):
    """Returns one of two alternatives, depending on if a condition is zero or not"""
    options = loader.construct_mapping(node, deep=True)
    if options["condition"] == 0:
        return options[True]
    else:
        return options[False]


def _yaml_if_else(loader, node):
    """Returns one of two alternatives, depending on if a condition is True or False"""
    options = loader.construct_mapping(node, deep=True)
    if options["condition"]:
        return options[True]
    else:
        return options[False]


def _yaml_tuple(loader, node):
    """Convert sequence to tuple"""
    return tuple(loader.construct_sequence(node))


def _yaml_list_intersection(loader, node):
    """Get the intersection of two lists"""
    list_1, list_2 = loader.construct_sequence(node)
    return list(set(list_1) & set(list_2))


def _yaml_multi_select_from_dict(loader, node):
    """Selects the correct elements from a dictionary"""
    dict_, keys_ = loader.construct_sequence(node, deep=True)
    return {key_: dict_[key_] for key_ in keys_}


def _yaml_get_dict(loader, node):
    """yaml and nested structures is difficult, so sometimes using this is a nice quick fix"""
    return loader.construct_mapping(node, deep=True)


def _yaml_get_list(loader, node):
    """Quick fix for returning a list"""
    return loader.construct_sequence(node, deep=True)


def _yaml_random_choice_from_list(loader, node):
    """Convenient when you must pass the list"""
    return random.choice(loader.construct_sequence(node, deep=True)[0])

# ---------------
# For experiments config file (not related to HP sampling/suggestions)
# ---------------
def _yaml_get_keys(loader, node):
    """Get the keys of a dictionary"""
    return loader.construct_mapping(node).keys()


def _yaml_get_hpo_sampler(loader, node):
    """Get an optuna.sampler"""
    sampler, kwargs = loader.construct_sequence(node, deep=True)
    return get_optuna_sampler(sampler, **kwargs)


# ---------------
# For HPO distributions config file
# ---------------
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
    # loader.add_constructor("!StrFormat", _yaml_str_format)
    #     loader.add_constructor("!MultiplierInt", _yaml_multiplier_int)
    #     loader.add_constructor("!SelectFromDict", _yaml_select_from_dict)
    #     loader.add_constructor("!MappingLength", _yaml_mapping_length)
    #     loader.add_constructor("!Sum", _yaml_sum)
    #     loader.add_constructor("!IfIsNoneElse", _yaml_if_none_else)
    #     loader.add_constructor("!IfZeroElse", _yaml_if_zero_else)
    #     loader.add_constructor("!IfElse", _yaml_if_else)
    #     loader.add_constructor("!Tuple", _yaml_tuple)
    #     loader.add_constructor("!ListIntersection", _yaml_list_intersection)
    #     loader.add_constructor("!MultiSelectFromDict", _yaml_multi_select_from_dict)
    #     loader.add_constructor("!CreatePartitionSizes", yaml_generate_partition_sizes)
    #     loader.add_constructor("!SampleRBPDesigns", yaml_sample_rbp)
    #     loader.add_constructor("!GetDict", _yaml_get_dict)
    #     loader.add_constructor("!RandomChoiceFromList", _yaml_random_choice_from_list)

    # Convenient non-NPO distributions related ones
    loader.add_constructor("!GetKeys", _yaml_get_keys)
    loader.add_constructor("!HPOSampler", _yaml_get_hpo_sampler)

    # To be used with optuna
    loader.add_constructor("!Categorical", _yaml_optuna_categorical)
    loader.add_constructor("!Float", _yaml_optuna_float)
    loader.add_constructor("!Int", _yaml_optuna_int)
    loader.add_constructor("!CategoricalDict", _yaml_optuna_categorical_dict)
    return loader
