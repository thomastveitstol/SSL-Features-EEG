import abc
from typing import Tuple

import torch.nn as nn


class MTSModuleBase(nn.Module, abc.ABC):

    # The expected exceptions from incompatible arguments to __init__
    expected_init_errors: Tuple[Exception, ...] = ()

    @classmethod
    def successful_initialisation(cls, *args, **kwargs):
        """Method for checking if the initialisation of a model will fail or not. It will return False if the error from
        __init__ is expected, raise an error if it is unexpected, and True if no error is raised by __init__ """
        try:
            cls(*args, **kwargs)
        except cls.expected_init_errors:
            return False
        return True

    @staticmethod
    @abc.abstractmethod
    def suggest_hyperparameters(name, trial, config):
        """
        Method for suggesting hyperparameters from a config file containing distributions of which to sample from

        Parameters
        ----------
        name : str
            A string which will be added to all HP names
        trial : optuna.Trial
        config : dict[str, typing.Any]

        Returns
        -------
        dict[str, typing.Any]
        """

    def extract_latent_features(self, input_tensor):
        """
        Method for extracting latent features

        Parameters
        ----------
        input_tensor : torch.Tensor

        Returns
        -------
        torch.Tensor
            Latent features
        """
        raise NotImplementedError

    def classify_latent_features(self, input_tensor):
        """
        Method for classifying the latent features extracted

        Parameters
        ----------
        input_tensor : torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        raise NotImplementedError

    @classmethod
    def successful_initialisation(cls, *args, **kwargs):
        """Method which returns True if the provided hyperparameters will give a successful initialisation, False if a
        ValueError or ZeroDivisionError is raised. This was implemented as the braindecode models are not always able to
        handle the input dimensionality, and tend to raise a ValueError or ZeroDivisionError if the input time series is
        too short for the architecture to handle"""
        try:
            cls(*args, **kwargs)  # type: ignore[call-arg]
        except (ValueError, ZeroDivisionError):
            return False
        return True

    @classmethod
    def get_latent_features_dim(cls, *args, **kwargs):
        """Get the expected latent feature dimension"""
        return cls(*args, **kwargs).latent_features_dim  # type: ignore[call-arg]

    # ----------------
    # Properties
    # ----------------
    @property
    def latent_features_dim(self) -> int:
        raise NotImplementedError
