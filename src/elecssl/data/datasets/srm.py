import os

import mne
import numpy
import openneuro
import pandas

from elecssl.data.datasets.dataset_base import EEGDatasetBase, target_method, OcularState
from elecssl.data.datasets.utils import sex_to_int


class SRM(EEGDatasetBase):
    """
    The SRM dataset

    Paper:
        Hatlestad-Hall, C., Rygvold, T. W., & Andersson, S. (2022). BIDS-structured resting-state electroencephalography
        (EEG) data extracted from an experimental paradigm. Data in Brief, 45, 108647.
        https://doi.org/10.1016/j.dib.2022.108647
    OpenNeuro:
        Christoffer Hatlestad-Hall and Trine Waage Rygvold and Stein Andersson (2022). SRM Resting-state EEG. OpenNeuro.
        [Dataset] doi: doi:10.18112/openneuro.ds003775.v1.2.1

    Examples:
    ----------
    >>> SRM().name
    'SRM'
    >>> SRM.get_available_targets(exclude_ssl=True)
    ('age', 'ravlt_del', 'ravlt_rec', 'ravlt_tot', 'sex')
    >>> SRM().load_targets(subject_ids=("sub-007", "sub-002", "sub-003"), target="age")
    array([37, 29, 62])
    >>> SRM().load_targets(subject_ids=("sub-007", "sub-002", "sub-003"), target="log_age")  # doctest: +ELLIPSIS
    array([3.6..., 3.3..., 4.1...])
    >>> len(SRM().get_subject_ids())
    111
    >>> len(SRM().channel_name_to_index())
    64
    >>> SRM().channel_name_to_index()  # doctest: +NORMALIZE_WHITESPACE
    {'Fp1': 0, 'AF7': 1, 'AF3': 2, 'F1': 3, 'F3': 4, 'F5': 5, 'F7': 6, 'FT7': 7, 'FC5': 8, 'FC3': 9, 'FC1': 10,
     'C1': 11, 'C3': 12, 'C5': 13, 'T7': 14, 'TP7': 15, 'CP5': 16, 'CP3': 17, 'CP1': 18, 'P1': 19, 'P3': 20, 'P5': 21,
     'P7': 22, 'P9': 23, 'PO7': 24, 'PO3': 25, 'O1': 26, 'Iz': 27, 'Oz': 28, 'POz': 29, 'Pz': 30, 'CPz': 31, 'Fpz': 32,
     'Fp2': 33, 'AF8': 34, 'AF4': 35, 'AFz': 36, 'Fz': 37, 'F2': 38, 'F4': 39, 'F6': 40, 'F8': 41, 'FT8': 42, 'FC6': 43,
     'FC4': 44, 'FC2': 45, 'FCz': 46, 'Cz': 47, 'C2': 48, 'C4': 49, 'C6': 50, 'T8': 51, 'TP8': 52, 'CP6': 53, 'CP4': 54,
     'CP2': 55, 'P2': 56, 'P4': 57, 'P6': 58, 'P8': 59, 'P10': 60, 'PO8': 61, 'PO4': 62, 'O2': 63}
    >>> set(type(idx) for idx in SRM().channel_name_to_index().values())  # type: ignore[attr-defined]
    {<class 'int'>}
    >>> SRM().channel_system.montage_name
    'standard_1020'
    """

    __slots__ = ()

    _montage_name = "standard_1020"
    _ocular_states = (OcularState.EC,)

    # ----------------
    # Loading methods
    # ----------------
    def _load_single_raw_mne_object(self, subject_id, *, ocular_state, session, preload=True):
        # Session 't2' is not available for all subjects
        assert session in ("t1", "t2"), f"Expected session to be either 't1' or 't2', but found {session}"

        # Create path
        subject_path = os.path.join(subject_id, f"ses-{session}", "eeg",
                                    f"{subject_id}_ses-{session}_task-resteyesc_eeg.edf")
        path = os.path.join(self.get_mne_path(), subject_path)

        # Make MNE raw object
        return mne.io.read_raw_edf(path, preload=preload, verbose=False)

    def _load_single_cleaned_mne_object(self, subject_id, ocular_state, **kwargs):
        # Load Epochs object
        epochs = self.load_single_cleaned_epochs_object(subject_id, session=kwargs["session"])

        # Concatenate in time
        data = epochs.get_data(copy=True)
        num_epochs, channels, timesteps = data.shape
        data = numpy.reshape(numpy.transpose(data, (1, 0, 2)), (channels, num_epochs * timesteps))

        # Make MNE Raw object
        return mne.io.RawArray(data, info=epochs.info, verbose=False)

    def load_single_cleaned_epochs_object(self, subject_id, session):
        """
        Method for loading cleaned Epochs object of the subject

        Parameters
        ----------
        subject_id : str
            Subject ID
        session : str
            String indicating which session to load from. Must be either 't1' or 't2'. 't2' is not available for only
            some subjects

        Returns
        -------
        mne.Epochs
            The (cleaned) Epochs object of the subject
        """
        # Extract session (should be 't1' or 't2'). 't2' is not available for all subjects
        assert session in ("t1", "t2"), f"Expected session to be either 't1' or 't2', but found {session}"

        # Create path
        subject_path = os.path.join("derivatives", "cleaned_epochs", subject_id, f"ses-{session}", "eeg",
                                    f"{subject_id}_ses-{session}_task-resteyesc_desc-epochs_eeg.set")
        path = os.path.join(self.get_mne_path(), subject_path)

        # Load MNE object and return
        return mne.io.read_epochs_eeglab(input_fname=path, verbose=False)

    @classmethod
    def download(cls):
        # Make directory
        path = cls.get_mne_path()
        os.mkdir(path)

        # Download from OpenNeuro
        openneuro.download(dataset="ds003775", target_dir=path)

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

    @target_method
    def ravlt_tot(self, subject_ids):
        # Read the .tsv file
        df = pandas.read_csv(self.get_participants_tsv_path(), sep="\t")

        # Convert to dict
        sub_id_to_age = {name: age for name, age in zip(df["participant_id"], df["ravlt_tot"])}

        # Extract the ages of the subjects, in the same order as the input argument
        return numpy.array([sub_id_to_age[sub_id] for sub_id in subject_ids])

    @target_method
    def ravlt_del(self, subject_ids):
        # Read the .tsv file
        df = pandas.read_csv(self.get_participants_tsv_path(), sep="\t")

        # Convert to dict
        sub_id_to_age = {name: age for name, age in zip(df["participant_id"], df["ravlt_del"])}

        # Extract the ages of the subjects, in the same order as the input argument
        return numpy.array([sub_id_to_age[sub_id] for sub_id in subject_ids])

    @target_method
    def ravlt_rec(self, subject_ids):
        # Read the .tsv file
        df = pandas.read_csv(self.get_participants_tsv_path(), sep="\t")

        # Convert to dict
        sub_id_to_age = {name: age for name, age in zip(df["participant_id"], df["ravlt_rec"])}

        # Extract the ages of the subjects, in the same order as the input argument
        return numpy.array([sub_id_to_age[sub_id] for sub_id in subject_ids])

    # ----------------
    # Channel system
    # ----------------
    def _get_template_electrode_positions(self):
        # Following the international 10-20 system according to the documentation at open neuro. Thus using MNE default
        montage = mne.channels.make_standard_montage(self._montage_name)
        channel_positions = montage.get_positions()["ch_pos"]

        # ---------------
        # Read the channel names
        # ---------------
        # Create path
        path = os.path.join(self.get_mne_path(), "code", "bidsify-srm-restingstate", "chanlocs",
                            "BioSemi_SRM_template_64_locs.xyz")

        # Make pandas dataframe
        df = pandas.read_table(path, header=None, sep=r"\s+").rename(columns={1: "x", 2: "y", 3: "z",
                                                                                         4: "ch_name"})

        # Extract channel names
        channel_names = tuple(df["ch_name"])

        # Return dict with channel positions, keeping only the ones in the data
        return {ch_name: tuple(pos) for ch_name, pos in channel_positions.items() if ch_name in channel_names}

    def channel_name_to_index(self):
        # Create path
        path = os.path.join(self.get_mne_path(), "code", "bidsify-srm-restingstate", "chanlocs",
                            "BioSemi_SRM_template_64_locs.xyz")

        # Make pandas dataframe
        df = pandas.read_table(path, header=None, sep=r"\s+").rename(columns={0: "idx", 4: "ch_name"})

        # Extract the needed values
        indices = df["idx"].to_numpy() - 1  # Need to subtract 1 due to python 0 zero indexing
        ch_names = tuple(df["ch_name"])

        # Convert to dict and return
        return {ch_name: int(idx) for ch_name, idx in zip(ch_names, indices)}
