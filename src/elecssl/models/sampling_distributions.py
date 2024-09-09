"""
Classes and functions for sampling from different distributions
"""
import random
from typing import List

import numpy
import yaml  # type: ignore[import-untyped]


# ------------------
# Convenient decorators
# ------------------
def sampling_distribution(func):
    setattr(func, "_is_sampling_distribution", True)
    return func


# ------------------
# Classes and methods
# ------------------
class _SampleDistribution:
    """
    Convenience class for sampling from a specified distribution

    Examples
    --------
    >>> _SampleDistribution.get_available_distributions()
    ('log_uniform', 'log_uniform_int', 'n_log_uniform_int', 'normal', 'uniform', 'uniform_discrete', 'uniform_int')
    >>> numpy.random.seed(1)
    >>> round(_SampleDistribution.sample("log_uniform", base=10, a=0, b=3), 3)  # 10^x, x ~ U[a, b]
    17.826
    """

    @classmethod
    def sample(cls, distribution, *args, **kwargs):
        # Input check
        if distribution not in cls.get_available_distributions():
            raise ValueError(f"The distribution {distribution} was not recognised. The available ones are: "
                             f"{cls.get_available_distributions()}")

        # Sample from the provided distribution
        return getattr(cls, distribution)(*args, **kwargs)

    @classmethod
    def get_available_distributions(cls):
        """Get all sampling distributions available for the class. The distributions must be a method decorated by
        @sampling_distribution to be properly registered"""
        # Get all regression metrics
        distributions: List[str] = []
        for method in dir(cls):
            attribute = getattr(cls, method)

            # Append (as type 'str') if it is a sampling distribution
            if callable(attribute) and getattr(attribute, "_is_sampling_distribution", False):
                distributions.append(method)

        # Convert to tuple and return
        return tuple(distributions)

    # ------------
    # Distributions
    # ------------
    @staticmethod
    @sampling_distribution
    def log_uniform(base, a, b):
        """Samples a value x uniformly from [a, b], and outputs base^x"""
        # Input checks
        assert a < b, f"Expected the value 'a' to be greater than 'b', but found {a} and {b}"
        assert base > 0, f"Expected a positive base, but found {base}"

        return base ** numpy.random.uniform(a, b)

    @staticmethod
    @sampling_distribution
    def log_uniform_int(base, a, b):
        """Samples a value x uniformly from [a, b], and outputs round(base^x)"""
        # Input checks
        assert a < b, f"Expected the value 'a' to be greater than 'b', but found {a} and {b}"
        assert base > 0, f"Expected a positive base, but found {base}"

        return round(base ** numpy.random.uniform(a, b))

    @staticmethod
    @sampling_distribution
    def n_log_uniform_int(n, base, a, b):
        """Sample a value x uniformly from [a, b], and output n*round(base^x)"""
        # Input checks
        assert a < b, f"Expected the value 'a' to be greater than 'b', but found {a} and {b}"
        assert base > 0, f"Expected a positive base, but found {base}"

        return n * round(base ** numpy.random.uniform(a, b))

    @staticmethod
    @sampling_distribution
    def uniform(a, b):
        # Input checks
        assert a < b, f"Expected the value 'a' to be greater than 'b', but found {a} and {b}"

        return numpy.random.uniform(a, b)

    @staticmethod
    @sampling_distribution
    def uniform_int(a, b):
        """Random integer, including 'a', excluding 'b'"""
        # Input checks
        assert a < b, f"Expected the value 'a' to be greater than 'b', but found {a} and {b}"

        return numpy.random.randint(a, b)

    @staticmethod
    @sampling_distribution
    def normal(mean, std):
        return numpy.random.normal(mean, std)

    @staticmethod
    @sampling_distribution
    def random_choice(*domain):
        return random.choice(domain)


def _snake_case_to_pascal_case(string):
    """
    Convert string from snake_case to PascalCase

    Parameters
    ----------
    string : str

    Returns
    -------
    str

    Examples
    --------
    >>> _snake_case_to_pascal_case("random_choice")
    'RandomChoice'
    >>> _snake_case_to_pascal_case("n_log_uniform_int")
    'NLogUniformInt'
    """
    return string.replace("_", " ").title().replace(" ", "")


def sample_hyperparameter(distribution, *args, **kwargs):
    """
    Function for sampling hyperparameters

    Parameters
    ----------
    distribution : str
    kwargs

    Returns
    -------
    typing.Any
    """
    return _SampleDistribution.sample(distribution, *args, **kwargs)


def get_yaml_loader():
    """Method for creating a loader which can interpret all distributions. Note that this only works when arguments are
    passed as positional"""
    yaml_loader = yaml.SafeLoader

    for sampling_dist in _SampleDistribution.get_available_distributions():
        # Create the constructor
        def make_constructor(dist):
            def constructor(loader, node):
                return sample_hyperparameter(dist, *loader.construct_sequence(node, deep=True))

            return constructor

        def make_type_constructor(dist):
            def type_constructor(loader, node):
                # todo: is this smelly lambda in for-loop?
                return lambda: sample_hyperparameter(dist, *loader.construct_sequence(node, deep=True))

            return type_constructor

        # Add the constructor
        _name  = _snake_case_to_pascal_case(sampling_dist)
        yaml_loader.add_constructor(f"!{_name}", make_constructor(sampling_dist))
        yaml_loader.add_constructor(f"!Type{_name}", make_type_constructor(sampling_dist))

    return yaml_loader
