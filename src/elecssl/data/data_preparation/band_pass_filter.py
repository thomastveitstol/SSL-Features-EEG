import itertools
import os
from pathlib import Path

import autoreject
import mne
import numpy
import yaml  # type: ignore[import-untyped]
from matplotlib import pyplot

from elecssl.data.data_preparation.data_prep_base import TransformationBase, InsufficientNumEpochsError


def _run_autoreject(epochs, *, autoreject_resample, seed, default_num_splits):
    """Function for running autoreject. Operates in-place"""
    if autoreject_resample is not None:
        epochs.resample(autoreject_resample, verbose=False)
    reject = autoreject.AutoReject(verbose=False, random_state=seed, cv=min(default_num_splits, len(epochs)))
    return reject.fit_transform(epochs, return_log=False)


class BandPass(TransformationBase):

    _augmentation_name = "band_pass"

    # ---------------
    # Methods related to folders
    # ---------------
    @staticmethod
    def _get_folder_name(*, freq_band, input_length, is_autorejected, resample_multiple):
        """
        Get name of the folder

        Examples
        --------
        >>> BandPass._get_folder_name(freq_band="delta", is_autorejected=True, resample_multiple=5.5, input_length=4)
        PosixPath('data--band_pass-delta--input_length-4s--autoreject-True--sfreq-5.5fmax')
        """
        return Path(f"data--band_pass-{freq_band}--input_length-{input_length}s--autoreject-{is_autorejected}--"
                    f"sfreq-{resample_multiple}fmax")

    def _create_folders(self, config):
        # --------------
        # Create main folder
        # --------------
        root_path = self._get_path(ocular_state=config["OcularState"])
        os.mkdir(root_path)

        # Save the config file
        with open(root_path / "config.yml", "w") as file:
            yaml.safe_dump(config, file)

        # --------------
        # Create sub folders
        # --------------
        _configurations = itertools.product(
            config["FrequencyBands"], (True, False), config["Details"]["resample_multiples"],
            config["Details"]["input_length"]
        )
        for freq_band, is_autorejected, resample_multiple, input_length in _configurations:
            # Create folder for a specific version
            folder_name = self._get_folder_name(freq_band=freq_band, input_length=input_length,
                                                is_autorejected=is_autorejected, resample_multiple=resample_multiple)
            folder_path = root_path / folder_name
            os.mkdir(folder_path)

            # Create folder for all datasets
            for dataset_name in config["Datasets"]:
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

    def _apply_and_save_single_data(self, raw, subject, config, plot_data, save_data):
        # ---------------
        # Input checks
        # ---------------
        if not isinstance(raw, mne.io.BaseRaw):
            raise TypeError(f"Unexpected type: {type(raw)}")

        # ---------------
        # Pre-processing
        # ---------------
        raw = self._preprocess(raw, **config["InitialPreprocessing"])

        # ---------------
        # Preparing and saving data
        # ---------------
        for epoch_duration in config["Details"]["input_length"]:
            self._apply_and_save_single_epochs(
                raw.copy(), config=config, subject=subject, epoch_duration=epoch_duration, plot_data=plot_data,
                save_data=save_data
            )

    def _apply_and_save_single_epochs(self, raw, *, config, subject, epoch_duration, plot_data, save_data):
        # Epoch the data
        epochs: mne.Epochs = mne.make_fixed_length_epochs(
            raw, duration=epoch_duration, preload=True, overlap=config["Details"]["epoch_overlap"], verbose=False
        )

        # Raising error if insufficient number of epochs
        num_epochs = config["Details"]["num_epochs"]
        if num_epochs > len(epochs):
            raise InsufficientNumEpochsError

        # Run autoreject
        autoreject_epochs = _run_autoreject(epochs.copy(), **config["Autoreject"])

        # Select epochs
        epochs = epochs[:num_epochs]

        if num_epochs > len(autoreject_epochs):
            # Skipping autorejected epochs if insufficient amount
            autoreject_epochs = None
        else:
            autoreject_epochs = autoreject_epochs[:num_epochs]

        # ---------------
        # Loop through and save EEG data for all frequency bands
        # ---------------
        for band_name, (l_freq, h_freq) in config["FrequencyBands"].items():
            # Save non-autorejected
            self._save_eeg_with_specifics(
                epochs=epochs.copy(), band_name=band_name, l_freq=l_freq, h_freq=h_freq,
                resample_fmax_multiples=config["Details"]["resample_multiples"], subject_id=subject.subject_id,
                is_autorejected=False, plot_data=plot_data, dataset_name=subject.dataset_name, save_data=save_data,
                epoch_duration=epoch_duration, ocular_state=config["OcularState"]
            )

            # (Maybe) save with autoreject
            if autoreject_epochs is not None:
                self._save_eeg_with_specifics(
                    epochs=autoreject_epochs.copy(), band_name=band_name, l_freq=l_freq, h_freq=h_freq,
                    resample_fmax_multiples=config["Details"]["resample_multiples"], subject_id=subject.subject_id,
                    is_autorejected=True, plot_data=plot_data, dataset_name=subject.dataset_name, save_data=save_data,
                    epoch_duration=epoch_duration, ocular_state=config["OcularState"]
                )

    def _save_eeg_with_specifics(self, epochs: mne.Epochs, band_name, l_freq, h_freq, resample_fmax_multiples,
                                 subject_id, is_autorejected, dataset_name: str, plot_data, save_data, epoch_duration,
                                 ocular_state):
        """Function for saving EEG data as numpy arrays, which has already been pre-processed to some extent"""
        # Perform band-pass filtering
        epochs.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)

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
                root_path = self._get_path(ocular_state=ocular_state)
                _folder_name = self._get_folder_name(freq_band=band_name, is_autorejected=is_autorejected,
                                                     resample_multiple=resample_multiple, input_length=epoch_duration)
                array_path = root_path / _folder_name / dataset_name / f"{subject_id}.npy"
                numpy.save(array_path, arr=data)
