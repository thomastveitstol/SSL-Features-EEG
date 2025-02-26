"""
This file has been modified from the original implementation at
https://github.com/Roche/neuro-green/blob/main/green/research_code/pl_utils.py
"""
# mypy: disable-error-code="assignment,unreachable,attr-defined"
from typing import Optional, Tuple, Union, Any, Dict

import geotorch
import numpy as np
import torch
from torch import Tensor
from torch import nn

from elecssl.models.mts_modules.green.spd_layers import BiMap, Shrinkage, LogMap
from elecssl.models.mts_modules.green.wavelet_layers import RealCovariance, CombinedPooling, CrossCovariance, \
    CrossPW_PLV, WaveletConv, PW_PLV, get_pooling_layer


class Green(nn.Module):
    def __init__(self,
                 conv_layers: nn.Module,
                 pooling_layers: nn.Module,
                 spd_layers: nn.Module,
                 head: nn.Module,
                 proj: nn.Module,
                 use_age: bool = False
                 ):
        """
        Neural network model that processes EEG epochs using convolutional
        layers, follwed by the computation of SPD features.

        Parameters:
        -----------
        conv_layers : nn.Module
            The convolutional layers that operate on the raw EEG signals.
        pooling_layers : nn.Module
            The pooling layers that convert the the convolved signals
            to SPD (Symmetric Positive Definite) features.
        spd_layers : nn.Module
            The SPD layers that operate on the SPD features.
        head : nn.Module
            The head layer that acts in the Euclidean space.
        proj : nn.Module
            The projection layer that projects the SPD features to the
            Euclidean
            space.
        age : bool, optional
            Whether to include age in the model, by default False
        """
        super(Green, self).__init__()
        self.conv_layers = conv_layers
        self.pooling_layers = pooling_layers
        self.spd_layers = spd_layers
        self.proj = proj
        self.head = head
        self.use_age = use_age

    def forward(self, X: Tensor, age: Tensor = None):
        """
        Parameters
        ----------
        X : Tensor
            N x P x T
        age : _type_, optional
            N, by default None
        """
        X_hat = self.conv_layers(X)
        X_hat = self.pooling_layers(X_hat)
        X_hat = self.spd_layers(X_hat)
        X_hat = self.proj(X_hat)
        if isinstance(self.pooling_layers, (RealCovariance, PW_PLV, CombinedPooling)):
            X_hat = vectorize_upper(X_hat)  # todo: I have a feeling it should be PW_PLV? So I just added it

        elif isinstance(self.pooling_layers, (CrossCovariance, CrossPW_PLV)):
            X_hat = vectorize_upper_one(X_hat)

        X_hat = torch.flatten(X_hat, start_dim=1)
        if self.use_age:
            X_hat = torch.cat([X_hat, age.unsqueeze(-1)], dim=-1)
        X_hat = self.head(X_hat)
        return X_hat


def get_green(
    n_freqs: int = 15,
    kernel_width_s: int = 5,
    conv_stride: int = 5,
    oct_min: float = 0,
    oct_max: float = 5.5,
    random_f_init: bool = False,
    shrinkage_init: float = -3.,
    logref: str = 'logeuclid',
    dropout: float = .333,
    n_ch: int = 21,
    hidden_dim: Optional[Union[int, Tuple[int, ...]]] = 32,  # Modified by TT: changed type hint from 'int' only
    sfreq: int = 125,
    dtype: torch.dtype = torch.float32,
    pool_layer: Union[str, nn.Module] = RealCovariance(),  # Modified by TT: added str as allowed type
    pool_layer_kwargs: Dict[str, Any] = None,  # Added by TT
    bi_out_perc: Optional[Union[float, Tuple[float, ...]]] = None,  # Modified by TT
    out_dim: int = 1,
    use_age: bool = False,
    orth_weights=True,
    reeig_reg: float = 1e-4,  # Added by TT
    momentum: float = 0.9  # added by TT
):
    """
    Helper function to get a Green model.

    Parameters
    ----------
    n_freqs : int, optional
        Number of main frequencies in the wavelet family, by default 15
    kernel_width_s : int, optional
        Width of the kernel in seconds for the wavelets, by default 5
    conv_stride : int, optional
        Stride of the convolution operation for the wavelets, by default 5
    oct_min : float, optional
        Minimum foi in octave, by default 0
    oct_max : float, optional
        Maximum foi in octave, by default 5.5
    random_f_init : bool, optional
        Whether to randomly initialize the foi, by default False
    shrinkage_init : float, optional
        Initial shrinkage value before applying sigmoid funcion, by default -3.
    logref : str, optional
        Reference matrix used for LogEig layer, by default 'logeuclid'
    dropout : float, optional
        Dropout rate for FC layers, by default .333
    n_ch : int, optional
        Number of channels, by default 21
    hidden_dim : int, optional
        Dimension of the hidden layer, if None no hidden layer, by default 32
    sfreq : int, optional
        Sampling frequency, by default 125
    dtype : torch.dtype, optional
        Data type of the tensors, by default torch.float32
    pool_layer : nn.Module, optional
        Pooling layer, by default RealCovariance()
    pool_layer_kwargs
        Kwargs to be passed to the poling layer, if it was specified as a string
    bi_out_perc : float, optional
        Determines the dimension of the output layer after BiMap
    out_dim : int, optional
        Dimension of the output layer, by default 1
    use_age : bool, optional
        Whether to include age in the model, by default False

    Returns
    -------
    Green
        The Green model
    """
    bi_out_perc = (bi_out_perc,) if isinstance(bi_out_perc, float) else bi_out_perc
    bi_out = None if bi_out_perc is None else tuple(round(perc * (n_ch - 1)) for perc in bi_out_perc)
    pool_layer_kwargs = dict() if pool_layer_kwargs is None else pool_layer_kwargs  # Added by TT

    # -------------
    # Convolution
    # -------------
    cplx_dtype = torch.complex128 if (
        dtype == torch.float64) else torch.complex64
    if random_f_init:
        foi_init = np.random.uniform(oct_min, oct_max, size=n_freqs)
        fwhm_init = -np.random.uniform(oct_min - 1, oct_max - 1, size=n_freqs)
    else:
        foi_init = np.linspace(oct_min, oct_max, n_freqs)
        fwhm_init = -np.linspace(oct_min - 1, oct_max - 1, n_freqs)

    conv_layers = nn.Sequential(*[
        WaveletConv(
            kernel_width_s=kernel_width_s,
            sfreq=sfreq,
            foi_init=foi_init,
            fwhm_init=fwhm_init,
            stride=conv_stride,
            dtype=cplx_dtype,
            scaling='oct'
        )])

    # -------------
    # Pooling layer
    # -------------
    pool_layer = get_pooling_layer(pool_layer, n_ch=n_ch, n_freqs=n_freqs, pool_layer_kwargs=pool_layer_kwargs)
    if isinstance(pool_layer, (RealCovariance, PW_PLV)):
        n_compo = n_ch
        feat_dim = int(n_freqs * n_compo * (n_compo + 1) / 2)

    elif isinstance(pool_layer, (CrossCovariance, CrossPW_PLV)):
        n_compo = int(n_ch * n_freqs)
        feat_dim = int(n_compo * (n_compo + 1) / 2)
        n_freqs = None

    elif isinstance(pool_layer, CombinedPooling):
        pool_layer_0 = pool_layer.pooling_layers[0]
        if isinstance(pool_layer_0, (RealCovariance, PW_PLV)):
            n_compo = n_ch
            feat_dim = int(n_freqs * n_compo * (n_compo + 1) / 2) * len(pool_layer.pooling_layers)
            n_freqs = n_freqs * len(pool_layer.pooling_layers)
        elif isinstance(pool_layer_0, (CrossCovariance, CrossPW_PLV)):
            n_compo = int(n_ch * n_freqs)
            feat_dim = int(n_compo * (n_compo + 1) / 2) * len(pool_layer.pooling_layers)
            n_freqs = len(pool_layer.pooling_layers)
    else:
        raise TypeError(f"Unexpected pooling layer: {type(pool_layer)}")
    # -------------
    # SPD layers
    # -------------
    # Shrinkage
    if shrinkage_init is None:
        spd_layers_list = [nn.Identity()]
    else:
        spd_layers_list = [Shrinkage(n_freqs=n_freqs, size=n_compo, init_shrinkage=shrinkage_init, learnable=True)]

    # BiMap
    if bi_out is not None:
        for bo in bi_out:
            bimap = BiMap(d_in=n_compo, d_out=bo, n_freqs=n_freqs)
            if orth_weights:
                geotorch.orthogonal(bimap, 'weight')
            spd_layers_list.append(bimap)

            n_compo = bo

        if n_freqs is None:
            feat_dim = int(n_compo * (n_compo + 1) / 2)
        else:
            feat_dim = int(n_freqs * n_compo * (n_compo + 1) / 2)

    # If age should be used as feature too
    if use_age:
        feat_dim += 1
    spd_layers = nn.Sequential(*spd_layers_list)

    # Projection to tangent space. Modified by TT: using 'reg' and 'momentum' input arguments now
    proj = LogMap(size=n_compo, n_freqs=n_freqs, ref=logref, momentum=momentum, reeig_reg=reeig_reg)

    # -------------
    # Head
    # -------------
    if hidden_dim is None:
        head = torch.nn.Sequential(*[
            torch.nn.BatchNorm1d(feat_dim,
                                 dtype=dtype),
            torch.nn.Dropout(
                p=dropout) if dropout is not None else nn.Identity(),
            torch.nn.Linear(feat_dim,
                            out_dim,
                            dtype=dtype),
        ])
    else:
        # add multiple FC layers
        sequential_list = []
        hidden_dim = (hidden_dim,) if isinstance(hidden_dim, int) else hidden_dim  # Modified (added) by TT
        for hd in hidden_dim:
            sequential_list.extend([
                torch.nn.BatchNorm1d(feat_dim,
                                     dtype=dtype),
                torch.nn.Dropout(
                    p=dropout) if dropout is not None else nn.Identity(),
                torch.nn.Linear(feat_dim,
                                hd,
                                dtype=dtype),
                torch.nn.GELU()
            ])
            feat_dim = hd
        sequential_list.extend([
            torch.nn.BatchNorm1d(feat_dim,
                                 dtype=dtype),
            torch.nn.Dropout(
                p=dropout) if dropout is not None else nn.Identity(),
            torch.nn.Linear(feat_dim,
                            out_dim,
                            dtype=dtype)
        ])
        head = torch.nn.Sequential(*sequential_list)

    # -------------
    # Gather everything
    # -------------
    model = Green(
        conv_layers=conv_layers,
        pooling_layers=pool_layer,
        spd_layers=spd_layers,
        head=head,
        proj=proj,
        use_age=use_age
    )
    return model


# -------------
# Functions
# -------------
def vectorize_upper_one(X: Tensor):
    """
    Upper vectorisation of a single SPD matrix with multiplication of
    off-diagonal terms by sqrt(2) to preserve norm.

    Parameters:
    -----------
    X : Tensor
        The covariance matrix of shape (N x P x P).

    Returns:
    --------
    X_vec : Tensor
        The vectorized covariance matrix of shape (N x P * (P + 1) / 2).
    """
    assert X.dim() == 3
    _, size, _ = X.shape
    triu_indices = torch.triu_indices(size, size, offset=1)
    if X.dim() == 3:  # batch of matrices
        X_vec = torch.cat([
            torch.diagonal(X, dim1=-2, dim2=-1),
            X[:, triu_indices[0], triu_indices[1]] * np.sqrt(2)
        ], dim=-1)
    elif X.dim() == 2:  # single matrix
        X_vec = torch.cat([
            torch.diagonal(X, dim1=-2, dim2=-1),
            X[triu_indices[0], triu_indices[1]] * np.sqrt(2)
        ], dim=-1)
    return X_vec


def vectorize_upper(X: Tensor) -> Tensor:
    """Upper vectorisation of F SPD matrices with multiplication of
    off-diagonal terms by sqrt(2) to preserve norm.

    Parameters
    ----------
    X : Tensor
        (N) x F x C x C

    Returns
    -------
    Tensor
        (N) x (C (C + 1) / 2)
    """
    # Upper triangular
    d = X.shape[-1]
    triu_idx = torch.triu_indices(d, d, 1)
    if X.dim() == 4:  # batch
        X_out = torch.cat([
            torch.diagonal(X, dim1=-2, dim2=-1),
            X[:, :, triu_idx[0], triu_idx[1]] * np.sqrt(2)
        ], dim=-1)
        return X_out
    elif X.dim() == 3:  # single tensor
        return torch.cat([torch.diagonal(X, dim1=-2, dim2=-1),
                         X[triu_idx[0], triu_idx[1]] * np.sqrt(2)],
                         dim=-1)
