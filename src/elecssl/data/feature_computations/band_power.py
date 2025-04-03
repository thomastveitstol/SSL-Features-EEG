import concurrent.futures
from concurrent.futures.process import ProcessPoolExecutor
from typing import Dict, Any, List

import mne
import numpy
import pandas
from progressbar import progressbar
from scipy import integrate

from elecssl.data.data_preparation.data_prep_base import run_autoreject
from elecssl.data.datasets.dataset_base import MNELoadingError


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

    # Maybe average across epochs
    if psd_data.ndim == 3:
        psd_data = numpy.mean(psd_data, axis=0)

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

    # Return power with sensible unit
    return agg(power) * 10**12


def _compute_band_power(eeg, frequency_bands, aggregation_method, verbose):
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
    psd = eeg.compute_psd(verbose=verbose)

    # Compute power of bands
    power: Dict[str, float] = dict()
    for band_name, (f_min, f_max) in frequency_bands.items():
        power[band_name] = _compute_band_power_from_psd(
            psd, f_min=f_min, f_max=f_max, aggregation_method=aggregation_method
        )

    return power


def _compute_band_power_single_subject(*, subject, info, crop, band_pass, notch_filter, resample, epochs, min_epochs,
                                       autoreject, average_reference, frequency_bands, verbose, aggregation_method):
    results = {"Subject-ID": subject, "Dataset": type(info.dataset).__name__}

    # Load the EEG
    try:
        eeg = info.dataset.load_single_mne_object(subject_id=subject, **info.kwargs)
    except MNELoadingError:
        return

    # Maybe crop
    if crop is not None:
        eeg.crop(tmin=crop["tmin"], tmax=eeg.n_times / eeg.info["sfreq"] - crop["cut"], verbose=False)

    # Maybe filter
    if band_pass is not None:
        eeg.filter(*band_pass, verbose=False)

    if notch_filter is not None:
        eeg.notch_filter(notch_filter, verbose=False)

    # Maybe resample
    if resample is not None:
        eeg.resample(resample, verbose=False)

    # Maybe epoch
    if epochs is not None:
        eeg = mne.make_fixed_length_epochs(raw=eeg, **epochs)

    # Maybe skip this subject if the number of epochs is insufficient
    if len(eeg) < min_epochs:
        return

    # Maybe run autoreject
    if autoreject is not None:
        eeg = run_autoreject(eeg, **autoreject, autoreject_resample=None)

    # Again, maybe skip this subject if the number of epochs is insufficient after autoreject
    if len(eeg) < min_epochs:
        return

    # Set average reference
    if average_reference:
        eeg.set_eeg_reference(ref_channels="average", verbose=False)

    # Compute power for all frequency bands of interest
    power = _compute_band_power(eeg=eeg, frequency_bands=frequency_bands, verbose=verbose,
                                aggregation_method=aggregation_method)

    # Add to results
    for band_name, band_power in power.items():
        results[band_name] = band_power

    return results


# -------------------
# Computations made on dataset level
# -------------------
def compute_band_powers(datasets, *, frequency_bands, aggregation_method, average_reference, verbose, autoreject,
                        epochs, crop, min_epochs, band_pass, notch_filter, resample, max_workers):
    """
    Function for computing band powers of entire datasets

    Parameters
    ----------
    datasets : tuple[DatasetInfo, ...]
    frequency_bands : dict[str, tuple[float, float]]
    aggregation_method : str
    average_reference : bool
    verbose : bool
    autoreject : dict[str, Any] | None
    epochs : dict[str, Any] | None
    crop : dict[str, Any] | None
    min_epochs : int
    band_pass : tuple[float, float] | None
    notch_filter : float | None
    resample : float 1 None
    max_workers : int

    Returns
    -------
    pandas.DataFrame
    """
    # Quick input check
    if not isinstance(average_reference, bool):
        raise TypeError(f"Expected 'average_reference' to be bool, but found {type(average_reference)}")

    # Initialise what will eventually be converted to a pandas DataFrame
    _f_bands = {band_name: [] for band_name in frequency_bands}  # type: ignore[var-annotated]
    results: Dict[str, List[Any]] = {"Dataset": [], "Subject-ID": [], **_f_bands}

    # Loop though all datasets
    for info in datasets:
        num_subjects = len(info.subjects)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Loop though all provided subjects for the dataset
            future_results = []
            for subject in info.subjects:
                future_results.append(
                    executor.submit(
                        _compute_band_power_single_subject, subject=subject, info=info, crop=crop, band_pass=band_pass,
                        notch_filter=notch_filter, resample=resample, epochs=epochs, min_epochs=min_epochs,
                        autoreject=autoreject, average_reference=average_reference, frequency_bands=frequency_bands,
                        verbose=verbose, aggregation_method=aggregation_method
                    )
                )

            # Collect all results
            for future_result in progressbar(concurrent.futures.as_completed(future_results), max_value=num_subjects,
                                             prefix=f"{type(info.dataset).__name__} ", redirect_stdout=True):
                subject_result = future_result.result()

                if subject_result is not None:
                    # Add the subject's results to the overall results
                    for key, value in subject_result.items():
                        results[key].append(value)

    # Convert to pandas DataFrame and return
    return pandas.DataFrame.from_dict(results)
