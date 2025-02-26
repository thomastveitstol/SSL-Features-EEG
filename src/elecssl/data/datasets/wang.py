import contextlib
import io
import os
import pathlib
import warnings

import mne
import numpy
import pandas
from pymatreader import read_mat

from elecssl.data.datasets.dataset_base import EEGDatasetBase, target_method, OcularState, MNELoadingError
from elecssl.data.datasets.utils import sex_to_int


class Wang(EEGDatasetBase):
    """
    The dataset from 'A test-retest resting, and cognitive state EEG dataset during multiple subject-driven states'

    Paper:
        Wang, Y., Duan, W., Dong, D. et al. A test-retest resting, and cognitive state EEG dataset during multiple
        subject-driven states. Sci Data 9, 566 (2022). https://doi.org/10.1038/s41597-022-01607-9
    OpenNeuro:
        Yulin Wang and Wei Duan and Debo Dong and Lihong Ding and Xu Lei (2022). A test-retest resting and cognitive
        state EEG dataset. OpenNeuro. [Dataset] doi: doi:10.18112/openneuro.ds004148.v1.0.1

    Examples
    --------
    >>> Wang().name
    'Wang'
    >>> Wang.get_available_targets(exclude_ssl=True)
    ('age', 'sex')
    >>> len(Wang().get_subject_ids())  # doctest: +SKIP
    60
    >>> Wang().get_subject_ids()[:5]  # doctest: +SKIP
    ('sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05')
    >>> my_channels = tuple(Wang()._get_template_electrode_positions().keys())
    >>> len(my_channels)
    62
    >>> my_channels  # doctest: +NORMALIZE_WHITESPACE
    ('Fp1', 'AF3', 'AF7', 'Fz', 'F1', 'F3', 'F5', 'F7', 'FC1', 'FC3', 'FC5', 'FT7', 'Cz', 'C1', 'C3', 'C5', 'T7', 'CP1',
     'CP3', 'CP5', 'TP7', 'TP9', 'Pz', 'P1', 'P3', 'P5', 'P7', 'PO3', 'PO7', 'Oz', 'O1', 'Fpz', 'Fp2', 'AF4', 'AF8',
     'F2', 'F4', 'F6', 'F8', 'FC2', 'FC4', 'FC6', 'FT8', 'C2', 'C4', 'C6', 'T8', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
     'TP10', 'P2', 'P4', 'P6', 'P8', 'POz', 'PO4', 'PO8', 'O2', 'FCz')
    >>> Wang().channel_name_to_index()  # doctest: +NORMALIZE_WHITESPACE
    {'Fp1': 0, 'AF3': 1, 'AF7': 2, 'Fz': 3, 'F1': 4, 'F3': 5, 'F5': 6, 'F7': 7, 'FC1': 8, 'FC3': 9, 'FC5': 10,
     'FT7': 11, 'Cz': 12, 'C1': 13, 'C3': 14, 'C5': 15, 'T7': 16, 'CP1': 17, 'CP3': 18, 'CP5': 19, 'TP7': 20,
     'TP9': 21, 'Pz': 22, 'P1': 23, 'P3': 24, 'P5': 25, 'P7': 26, 'PO3': 27, 'PO7': 28, 'Oz': 29, 'O1': 30, 'Fpz': 31,
     'Fp2': 32, 'AF4': 33, 'AF8': 34, 'F2': 35, 'F4': 36, 'F6': 37, 'F8': 38, 'FC2': 39, 'FC4': 40, 'FC6': 41,
     'FT8': 42, 'C2': 43, 'C4': 44, 'C6': 45, 'T8': 46, 'CPz': 47, 'CP2': 48, 'CP4': 49, 'CP6': 50, 'TP8': 51,
     'TP10': 52, 'P2': 53, 'P4': 54, 'P6': 55, 'P8': 56, 'POz': 57, 'PO4': 58, 'PO8': 59, 'O2': 60, 'FCz': 61}
    """

    __slots__ = ()

    _montage_name = "standard_1020"  # See 'EEG acquisition' in the original paper
    _ocular_states = (OcularState.EC, OcularState.EO)

    # ----------------
    # Methods for loading
    # ----------------
    def _load_single_raw_mne_object(self, subject_id, *, ocular_state, visit, preload=True):
        # Create path
        if ocular_state == OcularState.EC:
            recording = "eyesclosed"
        elif ocular_state == OcularState.EO:
            recording = "eyesopen"
        else:
            raise ValueError(f"Ocular state not recognised: {ocular_state}")
        subject_path = pathlib.Path(f"{subject_id}/ses-session{visit}/eeg/"
                                    f"{subject_id}_ses-session{visit}_task-{recording}_eeg")
        subject_path = subject_path.with_suffix(".vhdr")
        path = os.path.join(self.get_mne_path(), subject_path)

        # Make MNE raw object
        raw: mne.io.BaseRaw = mne.io.read_raw_brainvision(vhdr_fname=path, preload=preload, verbose=False)

        # Maybe rename channels
        if "Cpz" in raw.info["ch_names"]:
            mne.rename_channels(raw.info, mapping={"Cpz": "CPz"})
        if "FPz" in raw.info["ch_names"]:
            mne.rename_channels(raw.info, mapping={"FPz": "Fpz"})

        # Add FCz as reference
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            # MNE logs (and thus prints) that Fcz is missing positions, but this is fixed in the base method. Therefore
            # redirecting to an unused StringIO object
            with contextlib.redirect_stdout(io.StringIO()):
                raw.add_reference_channels("FCz")

        return raw

    def _load_single_cleaned_mne_object(self, subject_id, *, ocular_state, visit, preload=True):
        """
        Method for loading from derivatives folder

        Parameters
        ----------
        subject_id : str
        ocular_state : OcularState
        visit : int
        preload : bool

        Returns
        -------
        mne.io.eeglab.eeglab.RawEEGLAB

        Examples
        --------
        >>> Wang()._load_single_cleaned_mne_object("sub-28", ocular_state=OcularState.EO, visit=1,
        ...                                        preload=True)  # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
        ...
        elecssl.data.datasets.dataset_base.MNELoadingError: Could not load the data from subject sub-28. See above for
            original traceback
        """
        # Create path
        path_to_cleaned = "derivatives/preprocessed data/preprocessed_data"
        recording = ocular_state.value
        subject_path = pathlib.Path(f"{str(subject_id).zfill(2).replace('-', '')}_{str(visit).zfill(2)}_{recording}")
        subject_path = subject_path.with_suffix(".set")
        path = os.path.join(self.get_mne_path(), path_to_cleaned, subject_path)

        # Make MNE raw object
        try:
            raw = mne.io.read_raw_eeglab(input_fname=path, preload=preload, verbose=False)
        except OSError as e:
            # This happens for subject 28, derivatives, EO
            raise MNELoadingError(f"Could not load the data from subject {subject_id}. See above for original "
                                  f"traceback") from e

        # Maybe rename channels
        if "Cpz" in raw.info["ch_names"]:
            mne.rename_channels(raw.info, mapping={"Cpz": "CPz"})
        if "FPz" in raw.info["ch_names"]:
            mne.rename_channels(raw.info, mapping={"FPz": "Fpz"})

        # Add FCz as reference
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            raw.add_reference_channels("FCz")

        return raw

    @classmethod
    def download(cls):
        # Make directory
        path = cls.get_mne_path()
        os.mkdir(path)

        # Download from OpenNeuro
        import openneuro
        openneuro.download(dataset="ds004148", target_dir=path)  # todo: downloading more than resting state now...

    # ----------------
    # Targets
    # ----------------
    @target_method
    def sex(self, subject_ids):
        # Read the .tsv file
        df = pandas.read_csv(self.get_participants_tsv_path(), sep="\t")

        # Convert to dict
        sub_id_to_age = {name: sex_to_int(sex) for name, sex in zip(df["participant_id"], df["sex"])}

        # Extract the sexes of the subjects, in the same order as the input argument
        return numpy.array([sub_id_to_age[sub_id] for sub_id in subject_ids])

    @target_method
    def age(self, subject_ids):
        # Read the .tsv file
        df = pandas.read_csv(self.get_participants_tsv_path(), sep="\t")

        # Convert to dict
        sub_id_to_age = {name: age for name, age in zip(df["participant_id"], df["age"])}

        # Extract the ages of the subjects, in the same order as the input argument
        return numpy.array([sub_id_to_age[sub_id] for sub_id in subject_ids])

    # ----------------
    # Methods for channel system
    # ----------------
    def _get_electrode_positions(self, subject_id=None):
        # todo: does not contain CPz

        # Create path to .tsv file  todo: hard-coding session 1 and recording
        subject_path = f"{subject_id}/ses-session1/eeg/{subject_id}_ses-session1_electrodes.tsv"
        path = os.path.join(self.get_mne_path(), subject_path)

        # Load .tsv file
        df = pandas.read_csv(path, delimiter="\t")

        # Extract channel names and coordinates
        ch_names = df["name"]
        x_vals = df["x"]
        y_vals = df["y"]
        z_vals = df["z"]

        # Make it a dict and return it
        return {ch_name: (x, y, z) for ch_name, x, y, z in zip(ch_names, x_vals, y_vals, z_vals)}

    def _get_template_electrode_positions(self):
        montage = mne.channels.make_standard_montage(self._montage_name)
        channel_positions = montage.get_positions()["ch_pos"]

        # ---------------
        # Read the channel names
        # ---------------
        # Using the positions from the chanlocs62.mat file in derivatives folder
        path = os.path.join(self.get_mne_path(), "derivatives", "preprocessed data", "chanlocs62.mat")

        # Load the file
        mat_file = read_mat(path)

        # Extract channel names
        channel_names = mat_file["chanlocs"]["labels"]

        # Correct CPz channel name (does not currently match the MNE object)
        cpz_idx = channel_names.index("Cpz")
        channel_names[cpz_idx] = "CPz"

        # Return dict with channel positions, keeping only the ones in the data
        return {ch_name: channel_positions[ch_name] for ch_name in channel_names}

    def channel_name_to_index(self):
        return {ch_name: i for i, ch_name in enumerate(self._get_template_electrode_positions())}
