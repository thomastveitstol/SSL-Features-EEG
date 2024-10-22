import abc

import torch.nn as nn


class MTSModuleBase(nn.Module, abc.ABC):

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
    def get_latent_features_dim(cls, *args, **kwargs):
        """Get the expected latent feature dimension"""
        return cls(*args, **kwargs).latent_features_dim  # type: ignore[call-arg]

    # ----------------
    # Properties
    # ----------------
    @property
    def latent_features_dim(self) -> int:
        raise NotImplementedError
