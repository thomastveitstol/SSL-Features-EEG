import os
from typing import Literal

import mne
import numpy
import pandas

from elecssl.data.datasets.dataset_base import EEGDatasetBase, OcularState, target_method


class DortmundVital(EEGDatasetBase):
    """
    A dataset which is part of the Dortmund Vital Study, accessible at OpenNeuro (ds005385)

    Paper:
        Getzmann, S., Gajewski, P.D., Schneider, D. et al. Resting-state EEG data before and after cognitive activity
        across the adult lifespan and a 5-year follow-up. Sci Data 11, 988 (2024).
        https://doi.org/10.1038/s41597-024-03797-w
    OpenNeuro:
        Edmund Wascher and Daniel Schneider and Patrick D. Gajewski and Stephan Getzmann (2024). Resting-state EEG data
        before and after cognitive activity across the adult lifespan and a 5-year follow-up. OpenNeuro. [Dataset]
        doi: doi:10.18112/openneuro.ds005385.v1.0.2

    Examples
    --------
    >>> DortmundVital().num_channels
    64
    >>> DortmundVital().age(("sub-001", "sub-003", "sub-002", "sub-007"))  # doctest: +SKIP
    array([60, 44, 67, 24])
    >>> len(DortmundVital().age_availability())  # doctest: +SKIP
    608
    """

    __slots__ = ()

    _channel_names = ('Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4',
                      'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz',
                      'O2', 'PO10', 'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FT9', 'FT7', 'FC3', 'FC4',
                      'FT8', 'FT10', 'C5', 'C1', 'C2', 'C6', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'P5', 'P1', 'P2', 'P6',
                      'PO7', 'PO3', 'POz', 'PO4', 'PO8')
    _montage_name = "standard_1020"
    _ocular_states = (OcularState.EC, OcularState.EO)

    @classmethod
    def download(cls):
        # Make directory
        path = cls.get_mne_path()
        os.mkdir(path)

        # Download from OpenNeuro
        import openneuro
        openneuro.download(dataset="ds005385", target_dir=path)

    # ----------------
    # Loading methods
    # ----------------
    def _load_single_raw_mne_object(self, subject_id, *, ocular_state, session, acquisition: Literal["pre, post"],
                                    preload=True):
        # Create path
        _session_path = f"{subject_id}_ses-{session}_task-{ocular_state.to_pascal_case()}_acq-{acquisition}_eeg.edf"
        path = self.get_mne_path() / subject_id / f"ses-{session}" / "eeg" / _session_path

        # Make MNE raw object
        raw = mne.io.read_raw_edf(path, preload=preload, verbose=False)

        # Drop non-eeg channels
        raw.drop_channels("Status")

        # Reorder channels
        raw.reorder_channels(list(self._channel_names))

        return raw

    # ----------------
    # Channel system
    # ----------------
    def _get_template_electrode_positions(self):
        # Following the standard 10-20 system according to the original article
        montage = mne.channels.make_standard_montage(self._montage_name)
        channel_positions = montage.get_positions()["ch_pos"]

        # Return dict with channel positions, keeping only the ones in the data
        return {ch_name: tuple(pos) for ch_name, pos in channel_positions.items() if ch_name in self._channel_names}

    def channel_name_to_index(self):
        return {ch_name: i for i, ch_name in enumerate(self._channel_names)}

    # ----------------
    # Target methods
    # ----------------
    @target_method
    def age(self, subject_ids):
        # Read the .tsv file
        df = pandas.read_csv(self.get_participants_tsv_path(), sep="\t")

        # Convert to dict
        sub_id_to_age = {name: age for name, age in zip(df["participant_id"], df["age"])}

        # Extract the ages of the subjects, in the same order as the input argument
        return numpy.array([sub_id_to_age[sub_id] for sub_id in subject_ids])

    @age.availability
    def age_availability(self):
        df = pandas.read_csv(self.get_participants_tsv_path(), sep="\t", usecols=["participant_id", "age"])
        return tuple(df["participant_id"][df["age"].notna() & df["age"].notnull()])

    @target_method
    def fake_target(self, subject_ids):
        """Used for debugging"""
        return numpy.array([float("nan")] * len(subject_ids))

    @target_method
    def safe_log_ptau217(self, subject_ids):
        """Required for multi-task learning, but masking is used during the training phase"""
        return numpy.array([float("nan")] * len(subject_ids))
