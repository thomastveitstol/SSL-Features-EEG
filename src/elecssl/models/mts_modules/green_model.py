"""
The GREEN model implemented to suit the pipeline of this project
"""
import os
from typing import Iterator

import pandas
import torch.nn as nn
from torch.nn import Parameter

from elecssl.models.mts_modules.green.pl_utils import get_green
from elecssl.models.mts_modules.green.wavelet_layers import get_pooling_layer_type
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

    >>> my_model = GreenModel(in_channels=20, num_classes=1, sampling_freq=128, hidden_dim=(123, 11, 67), n_freqs=30,
    ...                       kernel_width_s=4, dropout=0.435, pool_layer="pw_plv", bi_out_perc=0.7)
    >>> my_model
    GreenModel(
      (_model): Green(
        (conv_layers): Sequential(
          (0): ComplexWavelet(kernel_width_s=4, sfreq=128, n_wavelets=30, stride=5, padding=0, scaling=oct)
        )
        (pooling_layers): PW_PLV()
        (spd_layers): Sequential(
          (0): LedoitWold(n_freqs=30, init_shrinkage=-3.0, learnable=True)
          (1): BiMap(d_in=20, d_out=13, n_freqs=30
        )
        (proj): LogEig(ref=logeuclid, reg=0.0001, n_freqs=30, size=13
        (head): Sequential(
          (0): BatchNorm1d(2730, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (1): Dropout(p=0.435, inplace=False)
          (2): Linear(in_features=2730, out_features=123, bias=True)
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
    >>> len(my_model.foi), len(my_model.fwhm)  # 30 frequencies
    (30, 30)
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
        ...                       bi_out_perc=0.8)
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
        ...                       bi_out_perc=0.8)
        >>> my_model.classify_latent_features(torch.rand(size=(10, 67))).size()
        torch.Size([10, 3])

        Running (1) feature extraction and (2) classifying is the excact same as just running forward

        >>> my_model = GreenModel(in_channels=my_channels, num_classes=3, sampling_freq=128, hidden_dim=(123, 11, 67),
        ...                       n_freqs=30, kernel_width_s=4, dropout=0.435, pool_layer="real_covariance",
        ...                       bi_out_perc=0.8)
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
    # Multi-task learning
    # ----------------
    def gradnorm_parameters(self) -> Iterator[Parameter]:
        """
        Parameters for GradNorm

        Examples
        --------
        >>> my_model = GreenModel(in_channels=20, num_classes=3, sampling_freq=128, hidden_dim=(123, 11, 67),
        ...                       n_freqs=30, kernel_width_s=4, dropout=0.435, pool_layer="pw_plv", bi_out_perc=0.7)
        >>> for my_params in my_model.gradnorm_parameters():
        ...     type(my_params), my_params.requires_grad, my_params.data.size()
        (<class 'torch.nn.parameter.Parameter'>, True, torch.Size([3, 67]))
        (<class 'torch.nn.parameter.Parameter'>, True, torch.Size([3]))
        """
        for params in self._model.head[-1].parameters():
            yield params

    # ----------------
    # Methods used for HPO
    # ----------------
    @classmethod
    def suggest_hyperparameters(cls, name, trial, config):
        # ----------------
        # The parametrised convolutions
        # ----------------
        n_freqs = trial.suggest_int(f"{name}_n_freqs", **config["n_freqs"])
        kernel_width_s = trial.suggest_float(f"{name}_kernel_width_s", **config["kernel_width_s"])
        conv_stride = trial.suggest_int(f"{name}_conv_stride", **config["conv_stride"])
        if isinstance(config["oct_min"], float):
            oct_min = config["oct_min"]
        else:
            oct_min = trial.suggest_float(f"{name}_oct_min", **config["oct_min"])
        if isinstance(config["oct_max_addition"], float):
            oct_max = oct_min + config["oct_max_addition"]
        else:
            oct_max_addition = trial.suggest_float(f"{name}_oct_max_addition", **config["oct_max_addition"])
            oct_max = oct_min + oct_max_addition
        random_f_init = trial.suggest_categorical(f"{name}_random_f_init", **config["random_f_init"])
        sampling_freq = config["sampling_freq"]  # Just copy sampling frequency

        # ----------------
        # The pooling layer
        # ----------------
        pool_layer = trial.suggest_categorical(f"{name}_pool_layer", **config["pool_layer"])
        pool_layer_kwargs = _suggest_pooling_layer_hpcs(trial=trial, name=name, config=config["pool_layer_kwargs"],
                                                        pooling_layer=pool_layer)

        # ----------------
        # Shrinkage layer
        # ----------------
        shrinkage_init = trial.suggest_float(f"{name}_shrinkage_init", **config["shrinkage_init"])

        # ----------------
        # BiMap layer
        # ----------------
        bi_out_perc = trial.suggest_float(f"{name}_bi_out_perc", **config["bi_out_perc"])
        orth_weights = trial.suggest_categorical(f"{name}_orth_weights", **config["orth_weights"])

        # ----------------
        # Combined ReEig and LogMap layer
        # ----------------
        logref = trial.suggest_categorical(f"{name}_logref", **config["logref"])
        reeig_reg = trial.suggest_float(f"{name}_reeig_reg", **config["reeig_reg"])
        momentum = trial.suggest_float(f"{name}_momentum", **config["momentum"])

        # ----------------
        # Fully connected module
        # ----------------
        # Going for a 'decrease by factor of two'
        dropout = trial.suggest_float(f"{name}_drop_prob", **config["drop_prob"])
        num_fc_layers = trial.suggest_int(f"{name}_num_fc_layers", **config["num_fc_layers"])
        num_first_fc_filters = trial.suggest_int(f"{name}_num_first_fc_filters", **config["num_first_fc_filters"])
        hidden_dim = tuple(num_first_fc_filters // (2 ** i) for i in range(num_fc_layers))

        return {"num_classes": config["num_classes"],
                "n_freqs": n_freqs,
                "sampling_freq": sampling_freq,
                "kernel_width_s": kernel_width_s,
                "conv_stride": conv_stride,
                "oct_min": oct_min,
                "oct_max": oct_max,
                "random_f_init": random_f_init,
                "pool_layer": pool_layer,
                "pool_layer_kwargs": pool_layer_kwargs,
                "shrinkage_init": shrinkage_init,
                "bi_out_perc": bi_out_perc,
                "orth_weights": orth_weights,
                "logref": logref,
                "reeig_reg": reeig_reg,
                "momentum": momentum,
                "dropout": dropout,
                "hidden_dim": hidden_dim}

    def save_metadata(self, *, name, path):
        """Save foi and fwhm"""
        file_path = (path / f"green_{name}").with_suffix(".csv")
        df = pandas.DataFrame({"foi": self.foi, "fwhm": self.fwhm})
        df.to_csv(file_path, index=False)
        os.chmod(file_path, 0o444)

    # ----------------
    # Properties
    # ----------------
    @property
    def latent_features_dim(self) -> int:
        # Infer it from the final layer
        return self._model.head[-1].in_features  # type: ignore[no-any-return]

    @property
    def foi(self):
        return tuple(self._model.conv_layers[0].foi.tolist())

    @property
    def fwhm(self):
        return tuple(self._model.conv_layers[0].fwhm.tolist())


# ----------------
# Functions
# ----------------
def _suggest_pooling_layer_hpcs(trial, name, config, pooling_layer):
    return get_pooling_layer_type(pool_layer=pooling_layer).suggest_hyperparameters(
            name, trial, config[pooling_layer]
    )
