import dataclasses
from typing import Dict, Tuple, Any, List

import numpy
import pandas
from scipy import integrate

from elecssl.data.datasets.dataset_base import EEGDatasetBase, MNELoadingError


# -------------------
# Function for computing power
# -------------------
def _compute_band_power_from_psd(psd, f_min, f_max, aggregation_method):
    """
    Function for computing band power of a single EEG, given its PSD

    Parameters
    ----------
    psd : mne.time_frequency.Spectrum
    f_min : float
    f_max : float
    aggregation_method : {"mean", "median"}

    Returns
    -------
    float
    """
    # Integrate between the desired range
    freqs = psd.freqs[(f_min < psd.freqs) & (psd.freqs < f_max)]
    psd_data = numpy.array(psd.get_data())[..., (f_min < psd.freqs) & (psd.freqs < f_max)]

    # The Simpson integration actually returns a numpy array, looks like scipy hasn't updated their type
    # hinting
    power: numpy.ndarray = integrate.simpson(y=psd_data, x=freqs, dx=None, axis=-1)  # type: ignore

    assert power.shape[0] == len(psd.ch_names), \
        (f"Expected Simpson integration to give power per channel, but output dimension was "
         f"{power.shape[0]}, while the number of channels is {len(psd.ch_names)}")

    # Compute the average power across the channels and store it
    if aggregation_method == "mean":
        agg = numpy.mean
    elif aggregation_method == "median":
        agg = numpy.median  # type: ignore[assignment]
    else:
        raise ValueError(f"Could not recognise aggregation method: {aggregation_method}")

    # Return power with sensible unit  todo: what should the unit be?
    return agg(power) * 10**12


def _compute_band_power(eeg, frequency_bands, aggregation_method):
    """
    Function for computing band power for multiple frequency bands, for a single EEG

    Parameters
    ----------
    eeg : mne.io.BaseRaw | mne.Epochs
    frequency_bands : dict[str, tuple[float, float]]
    aggregation_method : str

    Returns
    -------
    dict[str, float]
    """
    # Compute PSD
    psd = eeg.compute_psd()

    # Compute power of bands
    power: Dict[str, float] = dict()
    for band_name, (f_min, f_max) in frequency_bands.items():
        # todo: maybe this can be computed in parallel?
        power[band_name] = _compute_band_power_from_psd(
            psd, f_min=f_min, f_max=f_max, aggregation_method=aggregation_method
        )

    return power


# -------------------
# Computations made on dataset level
# -------------------
@dataclasses.dataclass
class DatasetInfo:
    dataset: EEGDatasetBase
    subjects: Tuple[str, ...]
    kwargs: Dict[str, Any]


def compute_band_powers(datasets, frequency_bands, aggregation_method):
    """
    Function for computing band powers of entire datasets

    Parameters
    ----------
    datasets : tuple[DatasetInfo, ...]
    frequency_bands : dict[str, tuple[float, float]]
    aggregation_method : str

    Returns
    -------
    pandas.DataFrame
    """
    # Initialise what will eventually be converted to a pandas DataFrame
    _f_bands = {band_name: [] for band_name in frequency_bands}  # type: ignore[var-annotated]
    results: Dict[str, List[Any]] = {"Dataset": [], "Subject-ID": [], **_f_bands}

    # Loop though all datasets
    for info in datasets:
        # Loop though all provided subjects for the dataset
        for subject in info.subjects:
            # Load the EEG
            try:
                eeg = info.dataset.load_single_mne_object(subject_id=subject, **info.kwargs)
            except MNELoadingError:
                continue

            # Compute power for all frequency bands of interest
            power = _compute_band_power(eeg=eeg, frequency_bands=frequency_bands, aggregation_method=aggregation_method)

            # Add to results
            for band_name, band_power in power.items():
                results[band_name].append(band_power)

            results["Dataset"].append(type(info.dataset).__name__)
            results["Subject-ID"].append(subject)

    # Convert to pandas DataFrame and return
    return pandas.DataFrame.from_dict(results)
