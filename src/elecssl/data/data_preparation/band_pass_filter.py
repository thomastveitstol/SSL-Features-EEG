import itertools
import os
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Tuple

import mne
import numpy
import yaml  # type: ignore[import-untyped]
from matplotlib import pyplot

from elecssl.data.data_preparation.data_prep_base import TransformationBase, InsufficientNumEpochsError, run_autoreject
from elecssl.data.datasets.getter import get_channel_system
from elecssl.data.interpolate_datasets import interpolate_single_epochs


class BandPass(TransformationBase):

    _augmentation_name = "band_pass"

    # ---------------
    # Methods related to folders
    # ---------------
    @staticmethod
    def _get_folder_name(*, freq_band, input_length, is_autorejected, resample_multiple, channel_system,
                         interpolation_method):
        """
        Get name of the folder

        Examples
        --------
        >>> my_path = BandPass._get_folder_name(freq_band="delta", is_autorejected=True, resample_multiple=5.5,
        ...                                     input_length=4, channel_system="AChannelSystem",
        ...                                     interpolation_method="spline")
        >>> expected = "data--band_pass-delta--input_length-4s--autoreject-True--sfreq-5.5fmax--interpolation-spline--"
        >>> expected += "ch_system-AChannelSystem"
        >>> str(my_path) == expected
        True
        """
        return Path(f"data--band_pass-{freq_band}--input_length-{input_length}s--autoreject-{is_autorejected}--"
                    f"sfreq-{resample_multiple}fmax--interpolation-{interpolation_method}--ch_system-{channel_system}")

    def _create_folders(self, config, save_to):
        # --------------
        # Create main folder
        # --------------
        try:
            os.mkdir(save_to)
        except FileExistsError as e:
            if not config["IgnoreExistingFolder"]:
                raise e

        # Save the config file
        with open(save_to / f"config_{date.today()}_{datetime.now().strftime('%H%M%S')}.yml", "w") as file:
            yaml.safe_dump(config, file)

        # --------------
        # Create sub folders
        # --------------
        _autoreject_options = ((config["Autoreject"] is not None),)
        _configurations = itertools.product(
            config["FrequencyBands"], _autoreject_options, config["Details"]["resample_multiples"],
            config["Details"]["input_length"], config["Details"]["interpolation_channel_systems"],
            config["Details"]["interpolation_methods"]
        )
        for (freq_band, is_autorejected, resample_multiple, input_length, channel_system,
             interpolation_method) in _configurations:
            # Create folder for a specific version
            folder_name = self._get_folder_name(
                freq_band=freq_band, input_length=input_length, is_autorejected=is_autorejected,
                resample_multiple=resample_multiple, channel_system=channel_system,
                interpolation_method=interpolation_method)
            folder_path = save_to / folder_name
            try:
                os.mkdir(folder_path)
            except FileExistsError as e:
                if not config["IgnoreExistingFolder"]:
                    raise e

            # Create folder for all datasets
            for dataset_name in config["Datasets"]:
                # Ignoring existing folders is meant to avoid having to upload all original data to TSD, and prepare
                # only AI-Mind data on TSD. So can't ignore if the AI-Mind dataset already exist
                os.mkdir(folder_path / dataset_name)

    # ---------------
    # Methods for preprocessing, transforming, and saving
    # ---------------
    @staticmethod
    def _preprocess(raw, *, excluded_channels, time_series_start_secs, band_pass, notch_filter):
        """Preprocessing before transformation (in-place operations can happen and that's fine)"""
        # Remove channels
        if excluded_channels is not None:
            raw = raw.pick(picks="eeg", exclude=excluded_channels)

        # Crop
        if time_series_start_secs is not None:
            raw.crop(tmin=time_series_start_secs)

        # Band-pass filtering
        if band_pass is not None:
            raw.filter(*band_pass, verbose=False)

        # Notch filter
        if notch_filter is not None:
            raw.notch_filter(notch_filter, verbose=False)

        return raw

    def _apply_and_save_single_data(self, raw, subject, config, preprocessing, plot_data, save_data, save_to,
                                    return_rejected_epochs):
        # ---------------
        # Input checks
        # ---------------
        if not isinstance(raw, mne.io.BaseRaw):
            raise TypeError(f"Unexpected type: {type(raw)}")

        # ---------------
        # Pre-processing
        # ---------------
        raw = self._preprocess(raw, **preprocessing)

        # ---------------
        # Preparing and saving data
        # ---------------
        rejected_epochs: Dict[int, Tuple[int, ...]] = dict()
        for epoch_duration in config["Details"]["input_length"]:
            rejected = self._apply_and_save_single_epochs(
                raw.copy(), config=config, subject=subject, epoch_duration=epoch_duration, plot_data=plot_data,
                save_data=save_data, save_to=save_to, return_rejected_epochs=return_rejected_epochs,
                default_band_pass=preprocessing["band_pass"]
            )
            if return_rejected_epochs:
                rejected_epochs[epoch_duration] = rejected
        if return_rejected_epochs:
            return rejected_epochs

    def _apply_and_save_single_epochs(self, raw, *, config, subject, epoch_duration, plot_data, save_data, save_to,
                                      return_rejected_epochs, default_band_pass):
        # Epoch the data
        epochs: mne.Epochs = mne.make_fixed_length_epochs(
            raw, duration=epoch_duration, preload=True, overlap=config["Details"]["epoch_overlap"], verbose=False
        )

        # Raising error if insufficient number of epochs
        num_epochs = config["Details"]["num_epochs"]
        if num_epochs > len(epochs):
            raise InsufficientNumEpochsError

        # Run autoreject
        if config["Autoreject"] is not None:
            epochs, reject_log = run_autoreject(epochs, **config["Autoreject"])
        else:
            reject_log = None

        # Select epochs
        epochs = epochs[:num_epochs]

        # Raising error if insufficient number of epochs
        if num_epochs > len(epochs):
            raise InsufficientNumEpochsError

        # ---------------
        # Loop through and save EEG data for all channel systems and frequency bands
        # ---------------
        interpolation_channel_systems = config["Details"]["interpolation_channel_systems"]
        interpolation_methods = config["Details"]["interpolation_method"]
        for target_channel_system, interpolation_method in itertools.product(interpolation_channel_systems,
                                                                             interpolation_methods):
            # Do interpolation (unless it is the channel system of the subject, in which case we'll skip interpolation)
            if target_channel_system != subject.dataset_name:
                interpolated_epochs = interpolate_single_epochs(
                    source_epochs=epochs.copy(), to_channel_system=get_channel_system(target_channel_system),
                    sampling_freq=epochs.info["sfreq"], method=interpolation_method)
            else:
                interpolated_epochs = epochs.copy()

            # Continue with the band-pass filtering
            for band_name, (l_freq, h_freq) in config["FrequencyBands"].items():
                # Save epochs object
                self._save_eeg_with_specifics(
                    epochs=interpolated_epochs.copy(), band_name=band_name, l_freq=l_freq, h_freq=h_freq,
                    resample_fmax_multiples=config["Details"]["resample_multiples"], subject_id=subject.subject_id,
                    is_autorejected=config["Autoreject"] is not None, plot_data=plot_data,
                    dataset_name=subject.dataset_name, save_data=save_data, epoch_duration=epoch_duration,
                    channel_system=target_channel_system, save_to=save_to, default_band_pass=default_band_pass,
                    interpolation_method=interpolation_method
                )

        if return_rejected_epochs and reject_log is not None:
            # This is what '_apply_drop()' in autoreject does (or, I convert to tuple of built-in ints, for json saving)
            return tuple(int(bad) for bad in numpy.nonzero(reject_log.bad_epochs)[0])

    def _save_eeg_with_specifics(self, epochs: mne.Epochs, *, band_name, l_freq, h_freq, resample_fmax_multiples,
                                 subject_id, is_autorejected, dataset_name: str, plot_data, save_data, epoch_duration,
                                 channel_system, save_to, default_band_pass, interpolation_method):
        """Function for saving EEG data as numpy arrays, which has already been pre-processed to some extent"""
        # Perform band-pass filtering
        epochs.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)

        l_freq = default_band_pass[0] if l_freq is None else l_freq
        h_freq = default_band_pass[1] if h_freq is None else h_freq

        # Loop through all resampling frequencies
        for resample_multiple in resample_fmax_multiples:
            resampled_epochs = epochs.copy()

            if resample_multiple is not None:
                # Calculate the new frequency
                new_freq = resample_multiple * h_freq

                # Perform resampling
                resampled_epochs.resample(new_freq, verbose=False)

            # Re-reference to average
            resampled_epochs.set_eeg_reference(ref_channels="average", verbose=False)

            # Convert to numpy arrays
            data = resampled_epochs.get_data()
            assert data.ndim == 3, (f"Expected the EEG data to have three dimensions (epochs, channels, time steps), "
                                    f"but found shape={data.shape}")

            # Maybe plot the data
            if plot_data:
                print("--------------------------")
                print(f"Band-pass filter: {l_freq, h_freq}")
                print(f"Sampling rate: f_max * {resample_multiple}")
                print(f"Autorejected: {is_autorejected}")
                resampled_epochs.plot(scalings="auto")
                resampled_epochs.compute_psd(verbose=False).plot()

                pyplot.show()

            # Save numpy array
            if save_data:
                _folder_name = self._get_folder_name(
                    freq_band=band_name, is_autorejected=is_autorejected, resample_multiple=resample_multiple,
                    input_length=epoch_duration, channel_system=channel_system,
                    interpolation_method=interpolation_method)
                array_path = save_to / _folder_name / dataset_name / f"{subject_id}.npy"
                numpy.save(array_path, arr=data)
