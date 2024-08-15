"""
Implementation of Convolutional Monge Mapping Normalisation.

Original implementation:
    https://github.com/PythonOT/convolutional-monge-mapping-normalization/tree/main

Paper:
    T. Gnassounou, R. Flamary, A. Gramfort, Convolutional Monge Mapping Normalization for learning on biosignals,
    Neural Information Processing Systems (NeurIPS), 2023.
"""
import itertools
import warnings
from typing import Dict, Optional, Tuple, List

import numpy
import torch
import torch.nn as nn
from scipy import fft
from scipy import signal

from elecssl.data.datasets.dataset_base import ChannelSystem, channel_names_to_indices
from elecssl.models.region_based_pooling.utils import RegionID, CHANNELS_IN_MONTAGE_SPLIT


class ConvMMN:
    """
    Implementation of CMMN. It is important that the input tensors/arrays are consistent in which EEG channel is the
    i-th channel in the data matrix

    Examples
    --------
    >>> my_source_data = {"d1": numpy.random.normal(0, 1, size=(10, 32, 2000)),
    ...                   "d2": numpy.random.normal(0, 1, size=(8, 32, 2000)),
    ...                   "d3": numpy.random.normal(0, 1, size=(17, 32, 2000))}
    >>> my_cmmn = ConvMMN(kernel_size=128)
    >>> my_cmmn.fit_psd_barycenters(data=my_source_data, sampling_freq=100)
    >>> my_cmmn.fit_monge_filters(data=my_source_data)
    >>> my_convoluted_source_data = my_cmmn(my_source_data)
    >>> {name: data_.shape for name, data_ in my_convoluted_source_data.items()}  # type: ignore[attr-defined]
    {'d1': (10, 32, 2000), 'd2': (8, 32, 2000), 'd3': (17, 32, 2000)}

    When new datasets are introduced at test time, the monge filters must be fit

    >>> my_target_data = {"t1": numpy.random.normal(0, 1, size=(6, 32, 2000)),
    ...                   "t2": numpy.random.normal(0, 1, size=(4, 32, 2000)),
    ...                   "t3": numpy.random.normal(0, 1, size=(7, 32, 2000))}
    >>> my_cmmn(my_target_data)  # doctest: +NORMALIZE_WHITESPACE
    Traceback (most recent call last):
    ...
    RuntimeError: The monge filters must be computed prior to applying them, but this was not the case for the following
    datasets: ('t1', 't2', 't3')
    >>> my_cmmn.fit_monge_filters(data=my_target_data)
    >>> my_convoluted_target_data = my_cmmn(my_target_data)
    >>> {name: data_.shape for name, data_ in my_convoluted_target_data.items()}  # type: ignore[attr-defined]
    {'t1': (6, 32, 2000), 't2': (4, 32, 2000), 't3': (7, 32, 2000)}
    """

    __slots__ = "_sampling_freq", "_kernel_size", "_psd_barycenters", "_monge_filters"

    def __init__(self, *, kernel_size, sampling_freq=None):
        """Initialise object"""
        # Store/initialise attributes
        self._sampling_freq = sampling_freq
        self._kernel_size = kernel_size

        # Will have shape=(num_channels, frequencies)
        self._psd_barycenters: Optional[numpy.ndarray] = None  # type: ignore[type-arg]
        # Values will have shape=(num_channels, kernel_size)
        self._monge_filters: Dict[str, numpy.ndarray] = dict()  # type: ignore[type-arg]

    # ---------------
    # Methods for applying CMMN
    # ---------------
    def _apply_monge_filters_torch(self, data):
        # --------------
        # Apply filters
        # --------------
        # Loop through all datasets
        convoluted = dict()
        for dataset_name, x in data.items():
            # Extract monge filter (one per channel) and convert to torch tenors
            monge_filter = torch.tensor(self._monge_filters[dataset_name], dtype=x.dtype,
                                        requires_grad=False).to(x.device)

            # Channel dimension check
            if x.size()[1] != monge_filter.size()[0]:
                raise ValueError(f"Expected channel dimension to be the same as number of monge filters, but received "
                                 f"{x.size()[1]} and {monge_filter.size()[0]}")

            # Apply monge filter channel-wise and store it
            convoluted[dataset_name] = self._compute_single_eeg_convolution_torch(x=x, monge_filter=monge_filter)

        return convoluted

    @staticmethod
    def _compute_single_eeg_convolution_torch(*, x, monge_filter):
        """
        Method for computing convolution when using torch tensors

        Parameters
        ----------
        x : torch.Tensor
            Should have shape=(batch, channels, time_steps)
        monge_filter : torch.Tensor
            Should have shape=(channels, frequencies)

        Returns
        -------
        torch.Tensor
            Should have same shape as x

        Examples
        --------
        >>> my_x = torch.rand(size=(10, 8, 2000))
        >>> my_monge_filter = torch.rand(size=(8, 65))
        >>> my_output = ConvMMN._compute_single_eeg_convolution_torch(x=my_x, monge_filter=my_monge_filter)
        >>> my_output.size()
        torch.Size([10, 8, 2000])

        Permuting a channel does not change the output of the others

        >>> my_permuted_x = torch.clone(my_x)
        >>> my_permuted_x[:, 3] = torch.rand(size=(10, 2000))
        >>> my_permuted_output = ConvMMN._compute_single_eeg_convolution_torch(x=my_permuted_x,
        ...                                                                    monge_filter=my_monge_filter)
        >>> torch.equal(my_output[:, 3], my_permuted_output[:, 3])
        False
        >>> torch.equal(my_output[:, :3], my_permuted_output[:, :3])
        True
        >>> torch.equal(my_output[:, 4:], my_permuted_output[:, 4:])
        True

        Works with 4D as well

        >>> my_x = torch.rand(size=(10, 5, 8, 2000))
        >>> my_monge_filter = torch.rand(size=(8, 65))
        >>> my_output = ConvMMN._compute_single_eeg_convolution_torch(x=my_x, monge_filter=my_monge_filter)
        >>> my_output.size()
        torch.Size([10, 5, 8, 2000])

        It does not matter if 3D or 4D with a single EEG epoch

        >>> my_x = torch.rand(size=(10, 8, 2000))
        >>> my_monge_filter = torch.rand(size=(8, 65))
        >>> my_out1 = ConvMMN._compute_single_eeg_convolution_torch(x=my_x, monge_filter=my_monge_filter)
        >>> my_out2 = ConvMMN._compute_single_eeg_convolution_torch(x=torch.unsqueeze(my_x, dim=1),
        ...                                                         monge_filter=my_monge_filter)
        >>> torch.equal(my_out1, torch.squeeze(my_out2, dim=1))
        True
        """
        if x.dim() == 3:
            return nn.functional.conv1d(
                input=x, weight=torch.unsqueeze(monge_filter, dim=1), bias=None, stride=1,
                padding="same", dilation=1, groups=monge_filter.size()[0]
            )
        elif x.dim() == 4:
            return torch.transpose(
                nn.functional.conv2d(input=torch.transpose(x, dim0=1, dim1=2),
                                     weight=torch.unsqueeze(torch.unsqueeze(monge_filter, dim=1), dim=1),
                                     bias=None, stride=1, padding="same", dilation=1, groups=monge_filter.size()[0]),
                dim0=1, dim1=2)
        else:
            raise ValueError(f"Expected input tensor to tbe 3D or 4D, but found {x.dim()}D")

    def _apply_monge_filters_numpy(self, data):
        """
        Method for applying CMMN, after it is fit

        Parameters
        ----------
        data : dict[str, numpy.ndarray]

        Returns
        -------
        dict[str, numpy.ndarray]
            The datasets convolved with their respective monge filters
        """
        # --------------
        # Apply filters
        # --------------
        # Loop through all datasets
        convoluted = dict()
        for dataset_name, x in data.items():
            # Extract monge filter (one per channel)
            monge_filter = self._monge_filters[dataset_name]

            # Channel dimension check  todo: this must likely be more flexible in the future
            if x.shape[1] != monge_filter.shape[0]:
                raise ValueError(f"Expected the number of channels to be the same as number of monge filters, but "
                                 f"received {x.shape[1]} and {monge_filter.shape[0]}")

            # Apply monge filter channel-wise and store it
            # todo: so many for-loops, should be possible to improve this
            convoluted[dataset_name] = numpy.concatenate(
                [numpy.expand_dims(self._compute_single_eeg_convolution(mts, monge_filter), axis=0) for mts in x],
                axis=0
            )

        return convoluted

    @staticmethod
    def _compute_single_eeg_convolution(x, monge_filter):
        """
        Compute the convolution

        Examples
        --------
        >>> my_x = numpy.random.normal(0, 1, size=(32, 2000))
        >>> my_monge_filter = numpy.random.normal(0, 1, size=(32, 128))
        >>> ConvMMN._compute_single_eeg_convolution(my_x, my_monge_filter).shape
        (32, 2000)
        """
        warnings.warn(message="Please use the Pytorch implementation instead", category=DeprecationWarning)

        # Input checks
        assert x.ndim == 2, (f"Expected the single EEG to only have 2 dimensions (channel and temporal dimensions), "
                             f"but found {x.ndim}")
        assert monge_filter.ndim == 2, (f"Expected the monge filter to have 2 dimensions (channel and frequency "
                                        f"dimensions), but found {monge_filter.ndim}")

        # Perform convolution channel-wise. The original implementation uses mode='same'
        return numpy.concatenate(
            [numpy.expand_dims(numpy.convolve(signal_, filter_, mode="same"), axis=0)
             for signal_, filter_ in zip(x, monge_filter)],
            axis=0
        )

    def __call__(self, data):
        """
        Method which applies monge filters (see _apply_monge_filters)

        todo: add unittest which checks if numpy and pytorch gets the same results, at least on CPU

        Parameters
        ----------
        data : dict[str, numpy.ndarray | torch.Tensor]

        Returns
        -------
        data : dict[str, numpy.ndarray | torch.Tensor]
        """
        # --------------
        # Input checks
        # --------------
        # Check that the PSD barycenter is fit
        if self._psd_barycenters is None:
            raise RuntimeError("The barycenters must be fit before mapping applying the monge filters")

        # Check that monge filter has been fit on all provided datasets
        _unfit_datasets = tuple(dataset for dataset in data if dataset not in self._monge_filters)
        if _unfit_datasets:
            raise RuntimeError(f"The monge filters must be computed prior to applying them, but this was not the case "
                               f"for the following datasets: {_unfit_datasets}")

        # Check that the type of the values is consistent
        if all(isinstance(arr, numpy.ndarray) for arr in data.values()):
            data_type = "numpy"
        elif all(isinstance(arr, torch.Tensor) for arr in data.values()):
            data_type = "torch"
        else:
            raise TypeError(f"Expected all data values to be either all numpy arrays or all torch tensors, but found "
                            f"{set(type(arr) for arr in data.values())}")

        # --------------
        # Apply filters
        # --------------
        if data_type == "numpy":
            return self._apply_monge_filters_numpy(data=data)
        elif data_type == "torch":
            return self._apply_monge_filters_torch(data=data)
        else:
            raise ValueError("This should never happen")

    # ---------------
    # Methods for fitting CMMN
    # ---------------
    def fit_psd_barycenters(self, *, data=None, psds=None, sampling_freq=None):
        # If a sampling frequency is passed, set it (currently overriding, may want to do a similarity check and raise
        # error)
        self._sampling_freq = self._sampling_freq if sampling_freq is None else sampling_freq
        if self._sampling_freq is None:
            raise ValueError("A sampling frequency must be provided either in the __init__ or fit method, but None was "
                             "found")

        # -------------
        # Compute the PSD barycenters
        # -------------
        # Compute representative PSDs of all datasets, if not provided
        if psds is None:
            psds = _compute_representative_psds(data=data, sampling_freq=self._sampling_freq,
                                                kernel_size=self._kernel_size)

        # Compute PSD barycenters
        self._psd_barycenters = _compute_psd_barycenters(psds)

    def fit_monge_filters(self, data, is_psds=False):
        """
        Method for fitting the filter used for monge mapping (fitting h_k in the CMMN paper). With this implementation,
        a monge filter is fit per dataset. An interesting idea though could be to have different monge filter for
        different subgroups (not only fitting monge filters on 'dataset' level)

        Sampling frequency should be the same as was used during fitting, although no checks are made

        Parameters
        ----------
        data : dict[str, numpy.ndarray]
            The data for which monge filters will be fitted. Keys are dataset names, values are EEG numpy arrays with
            shape=(subjects, channels, time_steps)
        is_psds : bool
            Set this to True if the input data are PSDs

        Returns
        -------
        None
        """
        # Checks
        if self._psd_barycenters is None:
            raise RuntimeError("The PSD barycenters must be fit before fitting the monge filters")

        # Loop through all provided datasets
        for dataset_name, x in data.items():
            if is_psds:
                dataset_psd = x
            else:
                # Compute representative PSD
                dataset_psd = _compute_representative_psd(x, sampling_freq=self._sampling_freq,
                                                          kernel_size=self._kernel_size)

            # Compute the monge filter as in Eq. 5
            monge_filter = fft.irfftn(numpy.sqrt(self._psd_barycenters) / numpy.sqrt(dataset_psd), axes=(-1,))

            # The original implementation does this, therefore I am doing it as well
            monge_filter = numpy.fft.fftshift(monge_filter, axes=-1)

            # Store the monge filter
            self._monge_filters[dataset_name] = monge_filter

    # ---------------
    # Properties
    # ---------------
    @property
    def has_fit_barycenter(self) -> bool:
        """Checks if the barycenter has been fit by checking if it is not None"""
        return self._psd_barycenters is not None

    @property
    def sampling_freq(self):
        return self._sampling_freq

    @property
    def kernel_size(self):
        return self._kernel_size

    @property
    def fitted_monge_filters(self):
        """Get the dataset names which has been monge filters"""
        return tuple(self._monge_filters.keys())


class RBPConvMMN:
    """
    Implementation of CMMN operating on region representations of Multi-Montage RBP

    The PSD barycenters will have shape=(num_region_representations, frequencies)
    The monge filters will have shape=(num_region_representations, frequencies)
    """

    __slots__ = ("_cmmn_layers", "_channel_splits")

    def __init__(self, *, num_montage_splits, kernel_size, sampling_freq=None):
        """
        Initialise

        Parameters
        ----------
        num_montage_splits : int
        kernel_size: int
        sampling_freq : float
        """
        # ---------------
        # Set attributes
        # ---------------
        # There will be one ConvMMN per montage split
        self._cmmn_layers = tuple(ConvMMN(kernel_size=kernel_size, sampling_freq=sampling_freq)
                                  for _ in range(num_montage_splits))
        # Keys are dataset names. Values have length=number of montage splits
        self._channel_splits: Dict[str, Tuple[CHANNELS_IN_MONTAGE_SPLIT, ...]] = dict()

    # ---------------
    # Methods for applying CMMN
    # ---------------
    def _apply_monge_filters(self, region_representations, dataset_indices):
        """
        Method for applying the monge filter to the region representations, where the region representations are as
        expected from the output of the RBP layer

        Note that the convolution with the monge filters will change the input tensor in-place

        Parameters
        ----------
        region_representations: tuple[torch.Tensor, ...]
        dataset_indices: dict[str, tuple[int, ...]]

        Returns
        -------

        """
        # ---------------
        # Input checks
        # ---------------
        # Batch dimension of all montage splits must be the same
        _batch_sizes = set(x.size()[0] for x in region_representations)
        if len(_batch_sizes) != 1:
            raise ValueError(f"Expected all input montage splits to have the same batch size, but received the "
                             f"following batch sizes (N={len(_batch_sizes)}) {_batch_sizes}")

        # The dataset indices must not be overlapping
        all_dataset_indices = set(itertools.chain(*dataset_indices.values()))
        if len(all_dataset_indices) != sum(tuple(len(indices) for indices in dataset_indices.values())):
            _num_non_unique_indices = (sum(tuple(len(indices) for indices in dataset_indices.values())) -
                                       len(all_dataset_indices))
            raise ValueError(f"Expected non-overlapping dataset indices, but {_num_non_unique_indices} had more than "
                             f"one occurrence")

        # The dataset indices have same dimensions as batch size
        if len(all_dataset_indices) != tuple(_batch_sizes)[0]:
            raise ValueError(f"Expected the number of dataset indices to be the same as the batch size, but found "
                             f"{len(all_dataset_indices)} and {tuple(_batch_sizes)[0]}")

        # ---------------
        # Apply filters
        # ---------------
        # Loop through all montage splits outputs
        for cmmn_layer, ms_output in zip(self._cmmn_layers, region_representations):
            # Apply dataset specific filters
            for dataset_name, indices in dataset_indices.items():
                # Convolve only the subjects of the current indices (this changes region_representations in-place)
                # todo: a little un-intuitive/sub-optimal code going on here..
                ms_output[list(indices)] = cmmn_layer({dataset_name: ms_output[list(indices)]})[dataset_name]

        return region_representations

    def __call__(self, region_representations, dataset_indices):
        return self._apply_monge_filters(region_representations, dataset_indices)

    # ---------------
    # Methods for computing PSD, specific to RBP
    # ---------------
    def _compute_representative_psds_per_dataset_and_region(self, data, *, channel_systems: Dict[str, ChannelSystem],
                                                            sampling_freq=None):
        """
        This method can be used to compute representative PSDs per dataset, per montage split, per region

        Parameters
        ----------
        data : dict[str, numpy.ndarray]
        channel_systems : Dict[str, ChannelSystem]
        sampling_freq: float, optional

        Returns
        -------
        dict[str, dict[int, dict[RegionID, numpy.ndarray]]]
            The dict will look like this: {dataset_name: {montage split number: {RegionID: representative PSD}}}, where
            the representative PSD will have shape=(frequencies,)

        """
        # The dict will look like this: {dataset_name: {montage split number: {RegionID: representative PSD}}}
        representative_psds: Dict[str, Dict[int, Dict[RegionID, numpy.ndarray]]] = dict()  # type: ignore[type-arg]
        for dataset_name, eeg_data in data.items():
            representative_psds[dataset_name] = dict()

            # Quick dimension check, the EEG data should be 3D
            if eeg_data.ndim not in (3, 4):
                raise ValueError(f"Expected the provided EEG data to be 3D or 4D, but found {eeg_data.ndim} dimensions "
                                 f"in the '{dataset_name}' dataset")

            # Loop through the montage splits (or actually, the already computed mappings from Region ID to the channels
            # within the region of the given dataset, to be precise)
            montage_splits = self._channel_splits[dataset_name]
            for i, (montage_split, cmmn_layer) in enumerate(zip(montage_splits, self._cmmn_layers)):
                representative_psds[dataset_name][i] = dict()

                # Loop through all regions
                for region_id, channels in montage_split.items():
                    # Extract the indices of the channels within the region
                    allowed_indices = channel_names_to_indices(
                        ch_names=channels, channel_name_to_index=channel_systems[dataset_name].channel_name_to_index
                    )

                    # Get the sampling frequency from CMMN layer if None is provided
                    sampling_freq = cmmn_layer.sampling_freq if sampling_freq is None else sampling_freq
                    assert sampling_freq is not None, ("No sampling frequency was found, neither as input, nor in the "
                                                       "CMMN layer")

                    # Compute a representative PSD for the region
                    region_psd = _compute_representative_region_psd(
                        x=eeg_data[..., allowed_indices, :], kernel_size=cmmn_layer.kernel_size,
                        sampling_freq=sampling_freq
                    )

                    # Store it
                    representative_psds[dataset_name][i][region_id] = region_psd

        return representative_psds

    @staticmethod
    def _stack_representative_psd_region(psds):
        """
        Convenience method for stacking the PSDs computed per montage split and region for a single dataset. Makes it a
        little easier to work with

        Parameters
        ----------
        psds : dict[int, dict[RegionID, numpy.ndarray]]
            Output of _compute_representative_psds_per_dataset_and_region()[dataset_name] for any dataset

        Returns
        -------
        list[numpy.ndarray]
            Each element is a concatenation of the representative PSDs of the regions in the montage split. The length
            of this list should be equal to the number of montage splits. Each element should have
            shape=(num_region_representations, frquencies), see Examples.

        Examples
        --------
        >>> my_psds = {0: {RegionID(0): numpy.random.random(size=(65,)), RegionID(1): numpy.random.random(size=(65,))},
        ...            1: {RegionID(0): numpy.random.random(size=(65,)), RegionID(1): numpy.random.random(size=(65,)),
        ...                RegionID(2): numpy.random.random(size=(65,)), RegionID(3): numpy.random.random(size=(65,))},
        ...            2: {RegionID("A"): numpy.random.random(size=(65,)),
        ...                RegionID("B"): numpy.random.random(size=(65,)),
        ...                RegionID("C"): numpy.random.random(size=(65,))},
        ...            4: {RegionID("Q"): numpy.random.random(size=(65,))}}
        >>> my_outputs = RBPConvMMN._stack_representative_psd_region(my_psds)
        >>> tuple(out_.shape for out_ in my_outputs)  # type: ignore[attr-defined]
        ((2, 65), (4, 65), (3, 65), (1, 65))
        """
        # ------------
        # Stack the region PSDs to a multivariate PSD of the montage split
        # ------------
        montage_psds: List[numpy.ndarray] = []  # type: ignore[type-arg]
        for montage_split_psds in psds.values():
            montage_psds.append(numpy.concatenate([numpy.expand_dims(region_psd, axis=0)
                                                   for region_psd in montage_split_psds.values()]))

        return montage_psds

    # ---------------
    # Methods for fitting
    # ---------------
    def update_channel_splits(self, channel_splits):
        """
        Method for updating the channel splits. If existing keys, this method will override the previous channel splits

        Parameters
        ----------
        channel_splits : dict[str, tuple[CHANNELS_IN_MONTAGE_SPLIT, ...]]

        Returns
        -------
        None
        """
        for dataset_name, ch_splits in channel_splits.items():
            self._channel_splits[dataset_name] = ch_splits

        # ---------------
        # Channel splits checks
        # ---------------
        # All values in channel_splits should have equal number of montage splits
        _num_montage_splits = set(len(split) for split in self._channel_splits.values())
        if len(_num_montage_splits) != 1:
            raise ValueError(f"Expected number of montage splits to be the same for all datasets, but received "
                             f"{len(_num_montage_splits)}")

        # All region IDs should be the same (including order)
        _region_ids = []
        for splits in self._channel_splits.values():
            _region_ids.append(tuple(tuple(split.keys()) for split in splits))
        assert len(set(_region_ids)) == 1, f"Inconsistency in Region IDs: {set(_region_ids)}"

    def fit_psd_barycenters(self, data, *, channel_systems: Dict[str, ChannelSystem], sampling_freq=None):
        # --------------
        # Compute the representative PSDs for all datasets and regions
        # --------------
        region_psds = self._compute_representative_psds_per_dataset_and_region(data, channel_systems=channel_systems,
                                                                               sampling_freq=sampling_freq)

        # --------------
        # Compute barycenters of all regions in all montage splits by aggregation
        # --------------
        # Get the data to a more convenient format
        montage_psds_barycenters = {dataset_name: self._stack_representative_psd_region(psds)
                                    for dataset_name, psds in region_psds.items()}

        # Loop through all montage splits
        for cmmn_layer, montage_splits in zip(self._cmmn_layers, zip(*montage_psds_barycenters.values())):
            # Get the data for the current montage split, for all datasets
            montage_split_psds = {dataset_name: dataset_montage_split
                                  for dataset_name, dataset_montage_split
                                  in zip(montage_psds_barycenters, montage_splits)}

            # Fit the PSD barycenters of the CMMN layer of the current montage split
            cmmn_layer.fit_psd_barycenters(psds=montage_split_psds, sampling_freq=sampling_freq)

    def fit_monge_filters(self, data, *, channel_systems: Dict[str, ChannelSystem]):
        """Fit the CMMN monge filters for all"""
        # --------------
        # Checks
        # --------------
        # Check if PSD barycenters has been fit for all CMMN layers (all montage splits)
        if not all(cmmn_layer.has_fit_barycenter for cmmn_layer in self._cmmn_layers):
            raise RuntimeError("Expected all CMMN layers to be fit prior to fitting the monge filter of the montage "
                               "splits, but that was not case")

        # --------------
        # Fit the monge filters
        # --------------
        # Compute representative PSDs per dataset and region. Using the sampling frequency as registered in the CMMN
        # layers
        region_psds = self._compute_representative_psds_per_dataset_and_region(data, channel_systems=channel_systems)

        # Get the data to a more convenient format
        montage_psds_barycenters = {dataset_name: self._stack_representative_psd_region(psds)
                                    for dataset_name, psds in region_psds.items()}

        # Loop through all montage splits
        for cmmn_layer, montage_splits in zip(self._cmmn_layers, zip(*montage_psds_barycenters.values())):
            # Get the data for the current montage split, for all datasets
            montage_split_psds = {dataset_name: dataset_montage_split
                                  for dataset_name, dataset_montage_split
                                  in zip(montage_psds_barycenters, montage_splits)}

            # Fit the monge filter of the CMMN layer of the current montage split
            cmmn_layer.fit_monge_filters(data=montage_split_psds, is_psds=True)

    # ---------------
    # Properties
    # ---------------
    @property
    def fitted_channel_systems(self):
        """Get the channel systems which has fitted monge filters"""
        # Loop through all montage splits
        fitted_channel_systems: Tuple[str, ...] = tuple()
        for cmmn_layer in self._cmmn_layers:
            if fitted_channel_systems:
                if fitted_channel_systems != cmmn_layer.fitted_monge_filters:
                    raise RuntimeError("Expected all monge filters to be fit on the same channel systems, but this was "
                                       "not the case")
            else:
                fitted_channel_systems = cmmn_layer.fitted_monge_filters
        return fitted_channel_systems


# ---------------
# Functions for computing PSDs
# ---------------
def _compute_single_source_psd(x, *, sampling_freq, nperseg):
    r"""
    Compute the PSDs for a single source domain. If averaged afterward, it will be \hat{\textbf{p}}_k in the
    original paper

    Parameters
    ----------
    x : numpy.ndarray
        The EEG data of a single dataset. Should have shape=(num_subjects, channels, time_steps)
    sampling_freq : float
    nperseg : int
        This is called 'filter_size' in the original implementation by the authors, and is the value 'F' in the
        original paper (at least, the pre-print which is the only currently available version)

    Returns
    -------
    numpy.ndarray
        The PSDs of all subjects and channels

    Examples
    --------
    >>> my_data = numpy.random.normal(0, 1, size=(10, 32, 2000))
    >>> my_psds = _compute_single_source_psd(my_data, sampling_freq=10, nperseg=128)
    >>> my_psds.shape
    (10, 32, 65)
    >>> my_data = numpy.random.normal(0, 1, size=(10, 5, 32, 2000))
    >>> my_psds = _compute_single_source_psd(my_data, sampling_freq=10, nperseg=128)
    >>> my_psds.shape
    (10, 5, 32, 65)
    """
    if x.ndim not in (3, 4):
        raise ValueError(f"Expected input data to be 3D or 4D, but found {x.ndim} dimensions")
    return signal.welch(x=x, axis=-1, fs=sampling_freq, nperseg=nperseg)[-1]


def _aggregate_subject_psds(psds):
    """
    Aggregate the PSDs computed on a single dataset.

    Parameters
    ----------
    psds : numpy.ndarray
        Should have shape=(num_subjects, channels, frequencies)

    Returns
    -------
    numpy.ndarray
        Will have shape=(channels, frequencies)

    Examples
    --------
    >>> my_psds = numpy.random.normal(0, 1, size=(10, 32, 65))
    >>> _aggregate_subject_psds(my_psds).shape
    (32, 65)
    """
    if psds.ndim == 3:
        return numpy.mean(psds, axis=0)
    elif psds.ndim == 4:
        return numpy.mean(psds, axis=(0, 1))
    else:
        raise ValueError(f"Expected PSDs to be 3D or 4D, but found {psds.ndim}D")


def _compute_representative_psd(x, *, sampling_freq, kernel_size):
    """
    Method for computing a representative PSD, given EEG data

    Parameters
    ----------
    x : numpy.ndarray
        Expected shape=(num_subjects, channels, time_steps) or (num_subjects, EEG epochs, channels, time_steps)
    sampling_freq : float
    kernel_size : int

    Returns
    -------
    numpy.ndarray
        Output will be 2D or 3D, shape=(channels, frequencies) or shape=(channels, frequencies)

    Examples
    --------
    >>> _compute_representative_psd(numpy.random.normal(size=(10, 19, 3000)), sampling_freq=100, kernel_size=128).shape
    (19, 65)
    >>> _compute_representative_psd(numpy.random.normal(size=(10, 7, 19, 3000)), sampling_freq=100,
    ...                             kernel_size=128).shape
    (19, 65)
    """
    # Compute PSDs and aggregate them
    return _aggregate_subject_psds(_compute_single_source_psd(x=x, sampling_freq=sampling_freq,
                                                              nperseg=kernel_size))


def _compute_representative_region_psd(x, *, sampling_freq, kernel_size):
    """
    Function which may be used to compute a single representative PSD for a single region, when using RBP

    Parameters
    ----------
    x : numpy.ndarray
        Shape should be 3D or 4D with shape=(subjects, channels within region, time_steps) or
        shape=(subjects, eeg_epochs, channels within region, time_steps)
    sampling_freq : float
    kernel_size : int

    Returns
    -------
    numpy.ndarray
        Output will be 1D, with shape=(frequencies,)

    Examples
    --------
    >>> my_data = numpy.random.normal(size=(10, 6, 3000))
    >>> _compute_representative_region_psd(my_data, sampling_freq=100, kernel_size=128).shape
    (65,)
    >>> my_data = numpy.random.normal(size=(10, 5, 6, 3000))
    >>> _compute_representative_region_psd(my_data, sampling_freq=100, kernel_size=128).shape
    (65,)
    """
    # Compute all PSDs. Will have shape=(subjects, channel, frequencies)
    psds = _compute_single_source_psd(x, sampling_freq=sampling_freq, nperseg=kernel_size)

    # Aggregate the PSDs and return
    return _aggregate_to_region_psd(psds)


def _aggregate_to_region_psd(psds):
    if psds.ndim == 3:
        return numpy.mean(psds, axis=(0, 1))
    elif psds.ndim == 4:
        return numpy.mean(psds, axis=(0, 1, 2))
    else:
        raise ValueError(f"Expected PSDs to be 3D or 4D, but received {psds.ndim}")


def _compute_representative_psds(data, *, sampling_freq, kernel_size):
    """
    Method for computing representative PSDs of multiple datasets

    Parameters
    ----------
    data : dict[str, numpy.ndarray]
        The EEG data from all datasets. The keys are dataset names
    sampling_freq : float
    kernel_size : int

    Returns
    -------
    dict[str, numpy.ndarray]
        The keys are dataset names, the values are PSDs

    Examples
    --------
    >>> my_data = {"d1": numpy.random.normal(0, 1, size=(10, 32, 2000)),
    ...            "d2": numpy.random.normal(0, 1, size=(8, 64, 1500)),
    ...            "d3": numpy.random.normal(0, 1, size=(17, 19, 3000))}
    >>> my_estimated_psds = _compute_representative_psds(my_data, sampling_freq=100, kernel_size=64)
    >>> {name_: psds_.shape for name_, psds_ in my_estimated_psds.items()}  # type: ignore[attr-defined]
    {'d1': (32, 33), 'd2': (64, 33), 'd3': (19, 33)}
    >>> {name_: eeg_.shape for name_, eeg_ in my_data.items()}  # type: ignore[attr-defined]
    {'d1': (10, 32, 2000), 'd2': (8, 64, 1500), 'd3': (17, 19, 3000)}

    Work with multiple EEG epochs as well

    >>> my_data = {"d1": numpy.random.normal(0, 1, size=(10, 4, 32, 2000)),
    ...            "d2": numpy.random.normal(0, 1, size=(8, 7, 64, 1500)),
    ...            "d3": numpy.random.normal(0, 1, size=(17, 5, 19, 3000))}
    >>> my_estimated_psds = _compute_representative_psds(my_data, sampling_freq=100, kernel_size=64)
    >>> {name_: psds_.shape for name_, psds_ in my_estimated_psds.items()}  # type: ignore[attr-defined]
    {'d1': (32, 33), 'd2': (64, 33), 'd3': (19, 33)}
    """
    return {dataset_name: _compute_representative_psd(x=x, sampling_freq=sampling_freq, kernel_size=kernel_size)
            for dataset_name, x in data.items()}


def _compute_psd_barycenters(psds):
    """
    Compute the barycenters (eq. 6 in the paper), given the PSDs from the different datasets

    Parameters
    ----------
    psds: dict[str, numpy.ndarray]
        The PSDs. The numpy arrays should have shape=(channels, frequencies). The channel dimension must be the same
        for all datasets

    Returns
    -------
    numpy.ndarray

    Examples
    --------
    >>> my_psds = {"d1": numpy.array([[4, 5, 3, 7, 7, 3]]), "d2": numpy.array([[6, 7, 1, 0, 3, 8]]),
    ...            "d3": numpy.array([[7, 5, 2, 1, 0, 4]])}
    >>> my_sqrt = numpy.array([[4**0.5 + 6**0.5 + 7**0.5, 5**0.5 + 7**0.5 + 5**0.5, 3**0.5 + 1**0.5 + 2**0.5,
    ...                         7**0.5 + 0**0.5 + 1**0.5, 7**0.5 + 3**0.5 + 0**0.5, 3**0.5 + 8**0.5 + 4**0.5]])
    >>> numpy.array_equal(_compute_psd_barycenters(my_psds), (my_sqrt / 3) ** 2)
    True
    """
    # --------------
    # Input checks
    # --------------
    # Type should be dict
    if not isinstance(psds, dict):
        raise TypeError(f"Expected input to be dict, but found {type(psds)}")

    # Keys should be strings
    if not all(isinstance(dataset_name, str) for dataset_name in psds):
        raise TypeError(f"Expected all keys to be 'str', but found {set(type(key) for key in psds)}")

    # Values should be numpy arrays
    psd_arrays = psds.values()
    if not all(isinstance(psd_array, numpy.ndarray) for psd_array in psd_arrays):
        raise TypeError(f"Expected all values to be numpy arrays, but found {set(type(val) for val in psd_arrays)}")

    # All numpy arrays should have same two dimensions
    if not all(psd_array.ndim == 2 for psd_array in psd_arrays):
        raise ValueError(f"Expected all numpy arrays to be 2D, but found the dimensions "
                         f"{set(psd_array.ndim for psd_array in psd_arrays)}")

    # All shapes should be the same
    if len(set(psd_array.shape for psd_array in psd_arrays)) != 1:
        _unique_shapes = set(psd_array.shape for psd_array in psd_arrays)
        raise ValueError(f"Expected all PSD arrays to have the same shape, but received the following "
                         f"(N={len(_unique_shapes)}) {_unique_shapes}")

    # --------------
    # Compute PSD barycenters using Eq. 6 in the conference paper
    # --------------
    return numpy.mean(numpy.concatenate([numpy.expand_dims(psd ** 0.5, axis=0) for psd in psds.values()],
                                        axis=0), axis=0) ** 2
