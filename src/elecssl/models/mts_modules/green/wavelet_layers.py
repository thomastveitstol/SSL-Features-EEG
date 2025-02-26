"""
This file has been modified from the original implementation at
https://github.com/Roche/neuro-green/blob/main/green/wavelet_layers.py
"""
from typing import Union, Dict, Any, List, Tuple

# mypy: disable-error-code="assignment,arg-type,type-arg"
import numpy as np
import optuna
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn


def _compute_gaborwavelet(
        tt: nn.Parameter,
        foi: nn.Parameter,
        fwhm: nn.Parameter,
        dtype: torch.dtype = torch.complex64,
        sfreq: int = 250,
        scaling='oct',
        min_foi_oct=-2,
        max_foi_oct=6,
        min_fwhm_oct=-6,
        max_fwhm_oct=1
):
    """
    Compute the Gabor wavelet filterbank for a given set of frequencies and
    full-width at half-maximums.

    Parameters
    ----------
    tt : torch.Tensor
        The time vector at expected sampling frequency.
    foi : torch.Tensor
        The center frequencies of the wavelets in octaves.
    fwhm : torch.Tensor
        The full-width at half-maximums of the wavelets in octaves. This
        parameter is a time domain parameter. It matches the formalism
        described in "A betterway to define and describe Morlet wavelets for
        time-frequency analysis"by Michael X Cohen. It is related to the time
        domain standard deviationby fwhm =  std / 2 * sqrt(2 * log(2))  # todo: this should actually be multiplication
    dtype : torch.dtype, optional
        The dtype of the output wavelets, by default torch.complex64
    sfreq : int, optional
        The sampling frequency of the data, by default 250
    scaling : str, optional
        The scaling of the wavelets, if 'oct' the wavelets are scaled using
        the octave scaling described in meeglette, by default 'oct'
    min_foi_oct : int, optional
        The minimum center frequency in octaves used to ensure gradient flow,
        by default -2
    max_foi_oct : int, optional
        by default 6
    min_fwhm_oct : int, optional
        by default -6
    max_fwhm_oct : int, optional
        by default 1
    """
    # Looks like we are (1) resetting values outside the desired range, and (2) going from octave to Hz and seconds
    foi_oct = 2**torch.clamp(foi, min_foi_oct, max_foi_oct)
    fwhm_oct = 2**torch.clamp(fwhm, min_fwhm_oct, max_fwhm_oct)

    # Compute un-normalised wavelet by Eq. 3 in "A better way to define and describe Morlet wavelets for
    # time-frequency analysis" by Michael X Cohen
    wavelets = torch.stack([
        torch.exp(2j * np.pi * f * tt) * torch.exp(
            -4 * np.log(2) * tt**2 / h**2
        ) for f, h in zip(foi_oct, fwhm_oct)
    ], dim=0).to(dtype)

    # Normalise the wavelet
    wav_norm = wavelets / torch.linalg.norm(wavelets, dim=-1, keepdim=True)
    if scaling == 'oct':
        wav_norm *= np.sqrt(2.0 / sfreq) * torch.sqrt(np.log(2) * foi_oct).unsqueeze(1)
    return wav_norm


class WaveletConv(nn.Module):

    def __init__(self,
                 kernel_width_s: float,
                 sfreq: float = None,
                 foi_init: np.ndarray = None,
                 fwhm_init: np.ndarray = None,
                 padding: str = 0,
                 dtype: torch.dtype = torch.complex64,
                 stride: int = 1,
                 scaling: str = 'oct'):
        """Parametrized complex wavelet convolution layer.


        Parameters
        ----------
        kernel_width_s : float
            The width of the wavelet kernel in seconds.
        sfreq : float, optional
            The sampling frequency of the data, by default None
        foi_init : np.ndarray, optional
            The initial center frequencies of the wavelets in octaves,
            by default None
        fwhm_init : np.ndarray, optional
            The initial full-width at half-maximums of the wavelets in
            octaves, by default None
        padding : str, optional
            Padding mode for the convolution, by default 0
        dtype : torch.dtype, optional
            The data type of the wavelets, by default torch.complex64
        stride : int, optional
           The stride of the convolution, by default 1
        scaling : str, optional
            The scaling of the wavelets, if 'oct' the wavelets are scaled
            using the octave scaling described in meeglette, by default 'oct'

        """

        super(WaveletConv, self).__init__()

        tmax = kernel_width_s / 2
        tmin = -tmax
        self.tt = nn.Parameter(torch.linspace(tmin, tmax, int(
            kernel_width_s * sfreq)), requires_grad=False)
        self.n_wavelets = len(foi_init)
        self.sfreq = sfreq
        self.kernel_width_s = kernel_width_s
        self.tmax = tmax
        self.tmin = tmin
        self.dtype = dtype
        self.padding = padding
        self.stride = stride
        self.scaling = scaling

        self.foi = nn.Parameter(torch.Tensor(foi_init), requires_grad=True)
        self.fwhm = nn.Parameter(torch.Tensor(fwhm_init), requires_grad=True)

    def forward(self, X: Tensor):
        """
        Forward pass of the complex wavelet module.

        Parameters:
        -----------
        X : Tensor
            Input data of shape (batch_size, epochs, in_channels, times).

        Returns:
        --------
        X_conv : Tensor
            Convolved complex output of shape
            (batch_size, n_freqs, in_channels, times)

        Notes:
        ------
        The multiple epochs are concatenated along the frequency dimension
        after the convolution.

        Examples
        --------
        >>> import numpy
        >>> my_num_freqs = 7
        >>> my_model = WaveletConv(kernel_width_s=3, scaling="oct", stride=1, sfreq=128,
        ...                        foi_init=numpy.linspace(0, 5.5, my_num_freqs),
        ...                        fwhm_init=numpy.linspace(-1, 4.5, my_num_freqs))
        >>> my_model(torch.rand(10, 19, 2000)).size()
        torch.Size([10, 7, 19, 1617])
        """
        # -------------
        # Compute the Gabor wavelets
        # -------------
        wavelets = _compute_gaborwavelet(
            tt=self.tt,
            foi=self.foi,
            fwhm=self.fwhm,
            dtype=self.dtype,
            sfreq=self.sfreq,
            scaling=self.scaling
        )
        n_freqs = wavelets.shape[0]

        # -------------
        # Apply the Gabor wavelets (perform convolution)
        # -------------
        # If single epoch
        if X.dim() == 3:
            batch_size, in_channels, times = X.shape
            X_conv = F.conv1d(
                # channels to batch element
                X.to(self.dtype).view(-1, 1, times),
                wavelets.unsqueeze(1),
                padding=self.padding,
                stride=self.stride
            )
            # restore channels dimension
            X_conv = X_conv.view(batch_size, in_channels, n_freqs, -1)
            # swap frequency and channels dimension
            X_conv = X_conv.swapaxes(1, 2)

        # If multiple epochs
        elif X.dim() == 4:
            batch_size, n_epochs, in_channels, times = X.shape
            X_conv = F.conv1d(
                # channels to batch element
                X.to(
                    self.dtype).view(
                    batch_size *
                    n_epochs *
                    in_channels,
                    1,
                    times),
                wavelets.unsqueeze(1),
                padding=self.padding,
                stride=self.stride
            )
            X_conv = X_conv.view(
                batch_size, n_epochs, in_channels, n_freqs, -1)
            X_conv = X_conv.permute(0, 3, 2, 1, 4).contiguous()
            n_batch, n_freqs, n_sensors, n_epochs, n_times = X_conv.shape
            X_conv = X_conv.view(
                n_batch,
                n_freqs,
                n_sensors,
                n_epochs *
                n_times)

        return X_conv

    def __repr__(self):
        # This is where you define the representation of your module
        return f"ComplexWavelet(kernel_width_s={self.kernel_width_s}, " \
               f"sfreq={self.sfreq}, n_wavelets={self.n_wavelets}, " \
               f"stride={self.stride}, padding={self.padding}, " \
               f"scaling={self.scaling})"


class RealCovariance(nn.Module):
    """
    Compute the real covariance matrix of the wavelet transformed eeg signals.
    Input shape: (N x F x P x T)
    Output shape: (N x F x P x P)
    """

    def __init__(self,):
        super(RealCovariance, self).__init__()

    def forward(self, X):
        """
        Forward method

        Examples
        --------
        >>> my_layer = RealCovariance()
        >>> my_layer(torch.rand(10, 7, 19, 1617)).size()
        torch.Size([10, 7, 19, 19])
        """
        # X: (N x F x P x T)
        assert X.dim() == 4
        cplx_cov = X @ torch.transpose(X, -1, -2).conj() / X.shape[-1]
        return cplx_cov.real

    @staticmethod
    def suggest_hyperparameters(name, trial: optuna.Trial, config):
        return dict()


class CrossCovariance(nn.Module):
    """
    Compute the real covariance matrix with cross-frequency interactions of
    the wavelet transformed eeg signals.
    Input shape: (N x F x P x T)
    Output shape: (N x FP x FP)
    """

    def __init__(self,):
        super(CrossCovariance, self).__init__()

    def forward(self, X):
        """
        Forward method

        Examples
        --------
        >>> my_layer = CrossCovariance()
        >>> my_layer(torch.rand(10, 7, 19, 1617)).size()
        torch.Size([10, 133, 133])
        """
        # X: (N x F x P x T)
        assert X.dim() == 4
        n_batch, n_freqs, n_sensors, n_times = X.shape
        cross_cov = X.reshape(  # TT: changed to .reshape
            n_batch, n_freqs * n_sensors, n_times
        ) @ X.reshape(  # TT: changed to .reshape
            n_batch, n_freqs * n_sensors, n_times
        ).transpose(-1, -2).conj() / n_times
        return cross_cov.real

    @staticmethod
    def suggest_hyperparameters(name, trial: optuna.Trial, config):
        return dict()


class PW_PLV(nn.Module):
    """Pairwise phase locking value.
    Compute the sensor pairwise phase locking value of the wavelet transformed
    eeg signals.
    Inspired by https://onlinelibrary.wiley.com/doi/abs/10.1002/%28SICI%291097-0193%281999%298%3A4%3C194%3A%3AAID-HBM4%3E3.0.CO%3B2-C # noqa
    PLV between two sensors i and j measures the consistency of the phase lag. It is defined as:
    PLV_{ij} = |<exp(j(\\phi_i - \\phi_j)(t)) >_t|

    Examples
    --------
    >>> PW_PLV(n_ch=5, reg=1e-3).reg_mat
    Parameter containing:
    tensor([[[[0.0010, 0.0000, 0.0000, 0.0000, 0.0000],
              [0.0000, 0.0010, 0.0000, 0.0000, 0.0000],
              [0.0000, 0.0000, 0.0010, 0.0000, 0.0000],
              [0.0000, 0.0000, 0.0000, 0.0010, 0.0000],
              [0.0000, 0.0000, 0.0000, 0.0000, 0.0010]]]])
    """

    def __init__(self, reg=None, n_ch=None) -> None:
        super(PW_PLV, self).__init__()
        if reg is not None:
            self.reg_mat = torch.nn.Parameter(
                torch.eye(n_ch).reshape(1, 1, n_ch, n_ch) * reg,
                requires_grad=False)
        else:
            # register None parameter
            self.register_parameter('reg_mat', None)

    def forward(self, X: Tensor) -> Tensor:
        """
        Forward method

        Examples
        --------
        >>> my_layer = PW_PLV(n_ch=19, reg=1e-3)
        >>> my_layer(torch.rand(10, 7, 19, 1617)).size()
        torch.Size([10, 7, 19, 19])
        """
        # X: (batch_size, num_freqs, num_channels, num_time_steps)
        assert X.dim() == 4
        plv_tensor = torch.abs(
            (X / torch.abs(X)) @ torch.transpose(
                X / torch.abs(X), -1, -2).conj()
            / X.shape[-1])
        if self.reg_mat is not None:
            plv_tensor += self.reg_mat
        return plv_tensor

    @staticmethod
    def suggest_hyperparameters(name, trial: optuna.Trial, config):
        reg = trial.suggest_float(f"{name}_reg", **config["reg"])
        return {"reg": reg}


class CrossPW_PLV(nn.Module):
    """Cross-frequency pairwise phase locking value.
    """

    def __init__(self, reg=None, n_ch=None, n_freqs=None) -> None:
        super(CrossPW_PLV, self).__init__()
        if reg is not None:
            n_compo = n_ch * n_freqs
            self.reg_mat = torch.nn.Parameter(
                torch.eye(n_compo).reshape(1, n_compo, n_compo) * reg,
                requires_grad=False)
        else:
            # register None parameter
            self.register_parameter('reg_mat', None)

    def forward(self, X: Tensor) -> Tensor:
        """
        Forward method

        Examples
        --------
        >>> my_layer = CrossPW_PLV(n_ch=19, reg=1e-3, n_freqs=7)
        >>> my_layer(torch.rand(10, 7, 19, 1617)).size()
        torch.Size([10, 133, 133])
        """
        # X: (N x F x P x T)
        assert X.dim() == 4
        n_batch, n_freqs, n_sensors, n_times = X.shape
        X = X.reshape(n_batch, n_freqs * n_sensors, n_times)  # TT: changed to .reshape because I got an error with
        # .view

        plv_tensor = torch.abs(
            (X / torch.abs(X)) @ torch.transpose(
                X / torch.abs(X), -1, -2).conj()
            / n_times)
        if self.reg_mat is not None:
            plv_tensor += self.reg_mat
        return plv_tensor

    @staticmethod
    def suggest_hyperparameters(name, trial: optuna.Trial, config):
        reg = trial.suggest_float(f"{name}_reg", **config["reg"])
        return {"reg": reg}


class CombinedPooling(nn.Module):
    """Concatenate along the first axis the features computed by multiple
    pooling layers (Covariance, PLV, etc.)
    """

    def __init__(self, pooling_layers: list) -> None:
        super(CombinedPooling, self).__init__()
        self.pooling_layers = nn.ModuleList(pooling_layers)

    def forward(self, X: Tensor) -> Tensor:
        if isinstance(self.pooling_layers[0], (RealCovariance, PW_PLV)):
            return torch.cat([pool(X) for pool in self.pooling_layers], dim=1)
        elif isinstance(self.pooling_layers[0], (CrossCovariance, CrossPW_PLV)):
            return torch.cat([pool(X).unsqueeze(1)
                             for pool in self.pooling_layers], dim=1)
        raise TypeError(f"Unexpected type of first pooling layer: {type(self.pooling_layers[0])}")

    def __getitem__(self, idx):
        return self.pooling_layers[idx]

    @staticmethod
    def suggest_hyperparameters(name, trial: optuna.Trial, config):
        hyperparameters: List[Tuple[str, Dict[str, Any]]] = []
        for pooling_layer, kwargs in config.items():
            hpcs = get_pooling_layer_type(pooling_layer).suggest_hyperparameters(
                name=f"{name}_{pooling_layer}", trial=trial, config=kwargs
            )
            hyperparameters.append((pooling_layer, hpcs))
        return tuple(hyperparameters)


# -------------
# Functions
# -------------
def get_pooling_layer_type(pool_layer):
    # Hard-coding some alternatives
    if pool_layer.lower() in ("combinedpoolingcross", "combinedpoolingnocross"):
        return CombinedPooling

    # The actual ones
    available_pool_layers = (RealCovariance, CrossCovariance, CrossPW_PLV, PW_PLV, CombinedPooling)

    # Loop through and select the correct one
    for layer in available_pool_layers:
        if pool_layer == layer.__name__:
            return layer

    # If no match, an error is raised
    raise ValueError(f"The pooling layer '{pool_layer}' was not recognised. Please select among the following: "
                     f"{tuple(layer.__name__ for layer in available_pool_layers)}")


def get_pooling_layer(pool_layer: Union[str, nn.Module], n_ch, n_freqs, pool_layer_kwargs):
    """
    Helper function to allow specifying desired pooling layer by a string. Function added by TT

    Parameters
    ----------
    pool_layer : str
        Name of the pooling layer
    n_ch : int
        number of input channels
    n_freqs : int
        Number of wavelet convolutions
    pool_layer_kwargs : typing.Iterable[tuple[str, dict[str, typing.Any]]] | dict[str, typing.Any]
        Kwargs to be passed to __init__. Only CombinedPooling supports an iterable, see Examples for usage

    Examples
    --------
    >>> get_pooling_layer("real_covariance", 11, 7, {})
    RealCovariance()
    >>> get_pooling_layer("cross_covariance", 11, 7, {})
    CrossCovariance()
    >>> get_pooling_layer("pw_plv", 11, 7, {"reg": 1e-7})
    PW_PLV()
    >>> get_pooling_layer("cross_pw_plv", 11, 7, {"reg": 1e-7})
    CrossPW_PLV()

    In combined pooling, the kwargs must be provided per module

    >>> get_pooling_layer("combined_pooling", 11, 7, (("real_covariance", {}), ("cross_pw_plv", {"reg": 1e-7})))
    CombinedPooling(
      (pooling_layers): ModuleList(
        (0): RealCovariance()
        (1): CrossPW_PLV()
      )
    )
    """
    if isinstance(pool_layer, nn.Module):
        return pool_layer

    if pool_layer.lower() in ("real_covariance", "realcovariance"):
        assert isinstance(pool_layer_kwargs, dict)
        return RealCovariance(**pool_layer_kwargs)  # type: ignore[call-arg]
    elif pool_layer.lower() in ("pw_plv", "pwplv"):
        assert isinstance(pool_layer_kwargs, dict)
        return PW_PLV(n_ch=n_ch, **pool_layer_kwargs)
    elif pool_layer.lower() in ("cross_pw_plv", "crosspwplv", "cross_pwplv", "crosspw_plv"):
        assert isinstance(pool_layer_kwargs, dict)
        return CrossPW_PLV(n_ch=n_ch, n_freqs=n_freqs, **pool_layer_kwargs)
    elif pool_layer.lower() in ("cross_covariance", "crosscovariance"):
        assert isinstance(pool_layer_kwargs, dict)
        return CrossCovariance(**pool_layer_kwargs)  # type: ignore[call-arg]
    elif pool_layer.lower() in ("combined_pooling", "combinedpooling", "combinedpoolingcross",
                                "combinedpoolingnocross"):  # todo: quite un-elegant
        pooling_modules = []
        for layer_name, kwargs in pool_layer_kwargs:
            pooling_modules.append(
                get_pooling_layer(pool_layer=layer_name, n_ch=n_ch, n_freqs=n_freqs,  pool_layer_kwargs=kwargs)
            )
        return CombinedPooling(pooling_modules)

    raise ValueError(f"Unexpected pool layer: '{pool_layer}'")
