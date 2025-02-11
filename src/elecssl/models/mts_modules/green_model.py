"""
The GREEN model implemented to suit the pipeline of this project

TODO: update LICENCE file
"""

import torch.nn as nn

from elecssl.models.mts_modules.green.pl_utils import get_green
from elecssl.models.mts_modules.mts_module_base import MTSModuleBase


class GreenModel(MTSModuleBase):
    """
    The GREEN architecture

    Original implementation:
        https://github.com/Roche/neuro-green/tree/main

    Paper:
        Paillard, J., Hipp, J. F., & Engemann, D. A. (2024). GREEN: a lightweight architecture using learnable wavelets
        and Riemannian geometry for biomarker exploration. bioRxiv. https://doi.org/10.1101/2024.05.14.594142

    Examples
    --------
    >>> GreenModel(in_channels=20, num_classes=3, sampling_freq=200)  # A default model
    GreenModel(
      (_model): Green(
        (conv_layers): Sequential(
          (0): ComplexWavelet(kernel_width_s=5, sfreq=200, n_wavelets=15, stride=5, padding=0, scaling=oct)
        )
        (pooling_layers): RealCovariance()
        (spd_layers): Sequential(
          (0): LedoitWold(n_freqs=15, init_shrinkage=-3.0, learnable=True)
        )
        (proj): LogEig(ref=logeuclid, reg=0.0001, n_freqs=15, size=20
        (head): Sequential(
          (0): BatchNorm1d(3150, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (1): Dropout(p=0.333, inplace=False)
          (2): Linear(in_features=3150, out_features=32, bias=True)
          (3): GELU(approximate='none')
          (4): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (5): Dropout(p=0.333, inplace=False)
          (6): Linear(in_features=32, out_features=3, bias=True)
        )
      )
    )

    An example with different HPCs

    >>> my_model = GreenModel(in_channels=3, num_classes=1, sampling_freq=128, hidden_dim=(123, 11, 67), n_freqs=30,
    ...                       kernel_width_s=4, dropout=0.435, pool_layer="pw_plv", bi_out=(39, 51))
    >>> my_model
    GreenModel(
      (_model): Green(
        (conv_layers): Sequential(
          (0): ComplexWavelet(kernel_width_s=4, sfreq=128, n_wavelets=30, stride=5, padding=0, scaling=oct)
        )
        (pooling_layers): PW_PLV()
        (spd_layers): Sequential(
          (0): LedoitWold(n_freqs=30, init_shrinkage=-3.0, learnable=True)
          (1): BiMap(d_in=3, d_out=39, n_freqs=30
          (2): BiMap(d_in=39, d_out=51, n_freqs=30
        )
        (proj): LogEig(ref=logeuclid, reg=0.0001, n_freqs=30, size=51
        (head): Sequential(
          (0): BatchNorm1d(39780, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (1): Dropout(p=0.435, inplace=False)
          (2): Linear(in_features=39780, out_features=123, bias=True)
          (3): GELU(approximate='none')
          (4): BatchNorm1d(123, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (5): Dropout(p=0.435, inplace=False)
          (6): Linear(in_features=123, out_features=11, bias=True)
          (7): GELU(approximate='none')
          (8): BatchNorm1d(11, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (9): Dropout(p=0.435, inplace=False)
          (10): Linear(in_features=11, out_features=67, bias=True)
          (11): GELU(approximate='none')
          (12): BatchNorm1d(67, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (13): Dropout(p=0.435, inplace=False)
          (14): Linear(in_features=67, out_features=1, bias=True)
        )
      )
    )
    >>> my_model.latent_features_dim
    67
    """

    def __init__(self, in_channels, num_classes, sampling_freq, **kwargs):
        super().__init__()

        # ----------------
        # Initialise the model
        # ----------------
        self._model = get_green(n_ch=in_channels, out_dim=num_classes, sfreq=sampling_freq, **kwargs)

    def forward(self, x, return_features=False):
        """
        Forward method

        Parameters
        ----------
        x : torch.Tensor
        return_features : bool

        Returns
        -------
        torch.Tensor

        Examples
        --------
        >>> # TODO: Doesn't work with 'pw_plv'
        >>> import torch
        >>> my_num_seconds = 10
        >>> my_sfreq = 200
        >>> my_batch, my_channels, my_time_steps = 10, 103, my_num_seconds * my_sfreq
        >>> my_model = GreenModel(in_channels=my_channels, num_classes=3, sampling_freq=my_sfreq, hidden_dim=(123, 45),
        ...                       kernel_width_s=my_num_seconds // 2)
        >>> my_model(torch.rand(size=(my_batch, my_channels, my_time_steps))).size()
        torch.Size([10, 3])
        >>> my_model(torch.rand(size=(my_batch, my_channels, my_time_steps)), return_features=True).size()
        torch.Size([10, 45])
        >>> my_model = GreenModel(in_channels=my_channels, num_classes=3, sampling_freq=my_sfreq, hidden_dim=(123, 45),
        ...                       kernel_width_s=my_num_seconds // 2)
        >>> my_model(torch.rand(size=(my_batch, my_channels, my_time_steps))).size()
        torch.Size([10, 3])
        >>> my_model(torch.rand(size=(my_batch, my_channels, my_time_steps)), return_features=True).size()
        torch.Size([10, 45])

        An example with different HPCs

        >>> my_model = GreenModel(in_channels=my_channels, num_classes=3, sampling_freq=128, hidden_dim=(123, 11, 67),
        ...                       n_freqs=30, kernel_width_s=4, dropout=0.435, pool_layer="real_covariance",
        ...                       bi_out=(39,))
        >>> my_model(torch.rand(size=(my_batch, my_channels, my_time_steps))).size()
        torch.Size([10, 3])
        """
        # If predictions are to be made, just run forward method of the roche made model
        if not return_features:
            return self._model(x)

        # Return the latent features instead
        activations_name = "latent_features"
        activation = dict()

        # noinspection PyUnusedLocal
        def hook(model, inputs):
            if len(inputs) != 1:
                raise ValueError(f"Expected only one input, but received {len(inputs)}")
            activation[activations_name] = inputs[0].detach()

        # Get the layer which receives the features we are interested in, and pre-hook it
        pre_layer = self._get_feature_extraction_layer()
        pre_layer.register_forward_pre_hook(hook)

        # Run forward method, but get the latent features
        _ = self._model(x)
        return activation[activations_name]

    # ----------------
    # Methods which are needed for training with a domain discriminator
    # ----------------
    def extract_latent_features(self, input_tensor):
        return self(input_tensor, return_features=True)

    def classify_latent_features(self, input_tensor):
        """
        Consistent with how the features were extracted, we will pass it through the dropout layer and the final
        layer

        Parameters
        ----------
        input_tensor : torch.Tensor

        Returns
        -------
        torch.Tensor

        Examples
        --------
        >>> import torch
        >>> my_batch, my_channels, my_time_steps = 10, 103, 600*3
        >>> my_model = GreenModel(in_channels=my_channels, num_classes=3, sampling_freq=128, hidden_dim=(123, 11, 67),
        ...                       n_freqs=30, kernel_width_s=4, dropout=0.435, pool_layer="real_covariance",
        ...                       bi_out=(39, 51))
        >>> my_model.classify_latent_features(torch.rand(size=(10, 67))).size()
        torch.Size([10, 3])

        Running (1) feature extraction and (2) classifying is the excact same as just running forward

        >>> my_model = GreenModel(in_channels=my_channels, num_classes=3, sampling_freq=128, hidden_dim=(123, 11, 67),
        ...                       n_freqs=30, kernel_width_s=4, dropout=0.435, pool_layer="real_covariance",
        ...                       bi_out=(39, 51))
        >>> _ = my_model.eval()
        >>> my_input = torch.rand(size=(my_batch, my_channels, my_time_steps))
        >>> my_output_1 = my_model.classify_latent_features(my_model.extract_latent_features(my_input))
        >>> my_output_2 = my_model(my_input)
        >>> torch.equal(my_output_1, my_output_2)
        True
        """
        # Sanity checks
        if not isinstance(self._model.head[-2], nn.Dropout):
            raise RuntimeError(f"Expected the second last layer to be dropout layer, but found {self._model.head[-2]}")
        if not isinstance(self._model.head[-1], nn.Linear):
            raise RuntimeError(f"Expected the second last layer to be dropout layer, but found {self._model.head[-1]}")

        # Pass through the layers and return
        return self._model.head[-1](self._model.head[-2](input_tensor))

    def _get_feature_extraction_layer(self):
        """Method for getting the layer which receives the features which can be used by a domain discriminator. We will
        use the features passed to the final layer, but before drop-out"""
        pre_layer = self._model.head[-2]
        if not isinstance(pre_layer, nn.Dropout):
            raise RuntimeError(f"Expected the second last layer to be dropout layer, but found {pre_layer}")
        return pre_layer

    # ----------------
    # Methods used for HPO
    # ----------------
    @classmethod
    def suggest_hyperparameters(cls, name, trial, config):
        raise NotImplementedError

    # ----------------
    # Properties
    # ----------------
    @property
    def latent_features_dim(self) -> int:
        # Infer it from the final layer
        return self._model.head[-1].in_features  # type: ignore[no-any-return]
