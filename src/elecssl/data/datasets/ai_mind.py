import os
from pathlib import Path
from typing import Tuple

import mne
import numpy
import pandas

from elecssl.data.datasets.dataset_base import EEGDatasetBase, OcularState, target_method, MNELoadingError
from elecssl.data.paths import get_ai_mind_path, get_ai_mind_cantab_and_sociodemographic_path, \
    get_ai_mind_cantab_and_sociodemographic_ai_dev_path

_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class AIMind(EEGDatasetBase):
    """
    Class for the AI-Mind dataset

    Paper:
        Haraldsen IH, Hatlestad-Hall C, Marra C, Renvall H, Maestú F, Acosta-Hernández J, Alfonsin S, Andersson V,
        Anand A, Ayllón V, Babic A, Belhadi A, Birck C, Bruña R, Caraglia N, Carrarini C, Christensen E, Cicchetti A,
        Daugbjerg S, Di Bidino R, Diaz-Ponce A, Drews A, Giuffrè GM, Georges J, Gil-Gregorio P, Gove D, Govers TM,
        Hallock H, Hietanen M, Holmen L, Hotta J, Kaski S, Khadka R, Kinnunen AS, Koivisto AM, Kulashekhar S, Larsen D,
        Liljeström M, Lind PG, Marcos Dolado A, Marshall S, Merz S, Miraglia F, Montonen J, Mäntynen V, Øksengård AR,
        Olazarán J, Paajanen T, Peña JM, Peña L, Peniche Dl, Perez AS, Radwan M, Ramírez-Toraño F, Rodríguez-Pedrero A,
        Saarinen T, Salas-Carrillo M, Salmelin R, Sousa S, Suyuthi A, Toft M, Toharia P, Tveitstøl T, Tveter M,
        Upreti R, Vermeulen RJ, Vecchio F, Yazidi A and Rossini PM (2024) Intelligent digital tools for screening of
        brain connectivity and dementia risk estimation in people affected by mild cognitive impairment: the AI-Mind
        clinical study protocol. Front. Neurorobot. 17:1289406. doi: 10.3389/fnbot.2023.1289406

    Examples
    --------
    >>> len(AIMind._channel_names)
    126
    """

    __slots__ = ()

    # Note that we are here only interested in the EEG channels
    _channel_names = ("Fp1", "Fpz", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "M1", "T7", "C3",
                      "Cz", "C4", "T8", "M2", "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8", "POz", "O1",
                      "O2", "AF7", "AF3", "AF4", "AF8", "F5", "F1", "F2", "F6", "FC3", "FCz", "FC4", "C5", "C1", "C2",
                      "C6", "CP3", "CP4", "P5", "P1", "P2", "P6", "F9", "PO3", "PO4", "F10", "FT7", "FT8", "TP7", "TP8",
                      "PO7", "PO8", "FT9", "FT10", "TPP9h", "TPP10h", "PO9", "PO10", "P9", "P10", "AFF1", "AFz", "AFF2",
                      "FFC5h", "FFC3h", "FFC4h", "FFC6h", "FCC5h", "FCC3h", "FCC4h", "FCC6h", "CCP5h", "CCP3h", "CCP4h",
                      "CCP6h", "CPP5h", "CPP3h", "CPP4h", "CPP6h", "PPO1", "PPO2", "I1", "Iz", "I2", "AFp3h", "AFp4h",
                      "AFF5h", "AFF6h", "FFT7h", "FFC1h", "FFC2h", "FFT8h", "FTT9h", "FTT7h", "FCC1h", "FCC2h", "FTT8h",
                      "FTT10h", "TTP7h", "CCP1h", "CCP2h", "TTP8h", "TPP7h", "CPP1h", "CPP2h", "TPP8h", "PPO9h",
                      "PPO5h", "PPO6h", "PPO10h", "POO9h", "POO3h", "POO4h", "POO10h", "OI1h", "OI2h")
    _non_eeg_channels = ("CLAV", "VEOGL", "Valid data")
    _ocular_states = (OcularState.EC, OcularState.EO)
    _montage_name = "standard_1005"

    # ----------------
    # Path methods
    # ----------------
    @classmethod
    def get_participants_tsv_path(cls):
        # It will actually be a .csv file
        return cls._get_cantab_and_sociodemographic_path() / "ai-mind_scd-clinical-data_med-sci_2025-03-12.csv"

    @classmethod
    def get_participants_ai_dev_tsv_path(cls):
        # Path for the AI-Dev dataset
        return cls._get_cantab_and_sociodemographic_ai_dev_path() / "ai-mind_scd-clinical-data_ai-dev_2025-03-12.csv"

    @classmethod
    def get_mne_path(cls):
        """The original data is stored in TSD, in a completely different folder than the other external datasets"""
        return get_ai_mind_path()

    @classmethod
    def _get_cantab_and_sociodemographic_path(cls):
        """This is where clinical data variables and CANTAB data is stored"""
        return get_ai_mind_cantab_and_sociodemographic_path()

    @classmethod
    def _get_cantab_and_sociodemographic_ai_dev_path(cls):
        """This is where clinical data variables and CANTAB data is stored"""
        return get_ai_mind_cantab_and_sociodemographic_ai_dev_path()

    # ----------------
    # Loading methods
    # ----------------
    @classmethod
    def _get_subject_ids(cls) -> Tuple[str, ...]:
        # Infer subject IDs from the EEG folders
        return tuple(folder_name for folder_name in os.listdir(cls.get_mne_path())
                     if os.path.isdir(cls.get_mne_path() / folder_name))

    def _get_subject_path(self, *, subject_id, visit, recording, ocular_state):
        """Method for getting the absolute path. I don't know the algorithm for computing the safety character, so
        trying them all instead"""
        path_to_eegs = self.get_mne_path()
        for security_character in _ALPHABET:
            # Try the current safety character
            _visit_folder = f"{subject_id}-{visit}-{security_character}"
            recording_path = Path("sensors") / f"{str(_visit_folder)}_{recording}-{ocular_state.value}_eeg"
            path = (path_to_eegs / subject_id / _visit_folder / recording_path).with_suffix(".fif")

            # If the file exists, return the current path
            if os.path.isfile(path):
                return path

        raise FileNotFoundError(f"Found no file with any security character in the alphabet for {subject_id=}, "
                                f"{ocular_state=}, {visit=}, {recording=}.")

    @staticmethod
    def _get_ocular_state(recording: int):
        return {1: OcularState.EO, 2: OcularState.EC, 3: OcularState.EO, 4: OcularState.EC}[recording]

    def _get_first_available_recording(self, subject_id, *, visit, ocular_state):
        if ocular_state == OcularState.EO:
            for recording in (1, 3):
                try:
                    _ = self._get_subject_path(subject_id=subject_id, visit=visit, recording=recording,
                                               ocular_state=OcularState.EO)
                    return recording
                except FileNotFoundError:
                    pass
        elif ocular_state == OcularState.EC:
            for recording in (2, 4):
                try:
                    _ = self._get_subject_path(subject_id=subject_id, visit=visit, recording=recording,
                                               ocular_state=OcularState.EC)
                    return recording
                except FileNotFoundError:
                    pass
        else:
            raise ValueError(f"Unexpected ocular state: {ocular_state!r}")

        # No recording available
        raise MNELoadingError(f"No recording is available for {subject_id=}, {visit=}, {ocular_state=}")

    def _load_single_raw_mne_object(self, subject_id, *, ocular_state, visit, recording, preload=True):
        """
        Loading raw object

        Parameters
        ----------
        subject_id : str
        ocular_state : OcularState
        visit : int
        recording : int | None
            If None, the first available is used (prioritising 1-EO for eyes open and 2-EC for eyes closed)
        preload : bool

        Returns
        -------
        mne.io.RawArray
        """
        # -------------
        # Input checks
        # -------------
        if recording is not None:
            # Verify the ocular state of recording
            assert self._get_ocular_state(recording) == ocular_state, \
                f"The recording {recording!r} does not match the ocular state {ocular_state!r}"

        # -------------
        # Load object
        # -------------
        # (Maybe) get the first available recording
        if recording is None:
            recording = self._get_first_available_recording(subject_id, visit=visit, ocular_state=ocular_state)

        # Create path
        path = self._get_subject_path(subject_id=subject_id, visit=visit, recording=recording,
                                      ocular_state=self._get_ocular_state(recording))

        # Load MNE object
        epochs = mne.read_epochs(path)

        # -------------
        # Make raw object
        # -------------
        raw = mne.io.RawArray(data=numpy.squeeze(epochs.get_data(), axis=0), info=epochs.info, verbose=False)

        # Remove non EEG channels
        raw = raw.drop_channels(self._non_eeg_channels)

        # Verify that the channels are as expected
        if tuple(raw.ch_names) != self._channel_names:
            raise RuntimeWarning(f"The channel names were not as expected."
                                 f"\nActual: {raw.ch_names}\nExpected: {self._channel_names}")

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
        """Age at the first visit"""
        # Read the .csv file
        df = pandas.read_csv(self.get_participants_tsv_path(), usecols=("participant_id", "age"))

        # Remove safety character to match EEG folder names
        df["participant_id"] = df["participant_id"].apply(_strip_sub_id)

        # Check if all subjects in the csv are unique (should not really be needed here, so I might remove this in the
        # future)
        assert len(df["participant_id"]) == len(set(df["participant_id"]))

        # Convert to dict
        sub_id_to_age = {sub_id: age for sub_id, age in zip(df["participant_id"], df["age"])}

        # Select the ones passed in the subjects list, and return as numpy array
        return numpy.array([sub_id_to_age[sub_id] for sub_id in subject_ids])

    @age.availability
    def age_availability(self):
        """Returns the available subject IDs that have a valid age value"""
        df = pandas.read_csv(self.get_participants_tsv_path(), usecols=("participant_id", "age"))

        # Remove safety character to match EEG folder names
        df["participant_id"] = df["participant_id"].apply(_strip_sub_id)

        # Extract valid ones
        valid_subjects = df["participant_id"][df["age"].notna()]
        return tuple(valid_subjects)

    @target_method
    def ravlt_del_recall(self, subject_ids):
        # Read the .csv file
        df = pandas.read_csv(self.get_participants_tsv_path())

        # Convert to dict
        dict_ = {sub_id: var for sub_id, var in zip(df["participant_id"], df["ravlt_del_recall"])}

        # Select the ones passed in the subjects list, and return as numpy array
        return numpy.array([dict_[sub_id] for sub_id in subject_ids])

    @target_method
    def ptau_217(self, subject_ids):
        # Read the .csv file
        df = pandas.read_csv(self.get_participants_tsv_path())

        # Convert to dict
        dict_ = {sub_id: var for sub_id, var in zip(df["participant_id"], df["ptau217"])}

        # Select the ones passed in the subjects list, and return as numpy array
        return numpy.array([dict_[sub_id] for sub_id in subject_ids])

    @target_method
    def ptau_181(self, subject_ids):
        # Read the .csv file
        df = pandas.read_csv(self.get_participants_tsv_path())

        # Convert to dict
        dict_ = {sub_id: var for sub_id, var in zip(df["participant_id"], df["ptau181"])}

        # Select the ones passed in the subjects list, and return as numpy array
        return numpy.array([dict_[sub_id] for sub_id in subject_ids])

    @target_method
    def fake_target(self, subject_ids):
        """Used for debugging"""
        return numpy.random.normal(loc=12, scale=4, size=(len(subject_ids),))

    @fake_target.availability
    def fake_target_availability(self):
        # Read the .csv file
        df = pandas.read_csv(self.get_participants_ai_dev_tsv_path(), usecols=["participant_id", "ptau217"])
        df["participant_id"] = df["participant_id"].apply(_strip_sub_id)

        # Extract valid ones
        valid_subjects = df["participant_id"][df["ptau217"].notna()]
        return tuple(valid_subjects)

    @target_method
    def safe_log_ptau217(self, subject_ids):
        # Read the .csv file
        df = pandas.read_csv(self.get_participants_ai_dev_tsv_path(), usecols=["participant_id", "ptau217"])
        df["participant_id"] = df["participant_id"].apply(_strip_sub_id)
        df["log_ptau217"] = numpy.log(df["ptau217"])

        # Convert to dict
        dict_ = {sub_id: var for sub_id, var in zip(df["participant_id"], df["log_ptau217"])}

        # Select the ones passed in the subjects list, and return as numpy array
        return numpy.array([dict_[sub_id] for sub_id in subject_ids])

    @safe_log_ptau217.availability
    def safe_log_ptau217_availability(self):
        # Read the .csv file
        df = pandas.read_csv(self.get_participants_ai_dev_tsv_path(), usecols=["participant_id", "ptau217"])
        df["participant_id"] = df["participant_id"].apply(_strip_sub_id)

        # Extract valid ones
        valid_subjects = df["participant_id"][df["ptau217"].notna()]
        return tuple(valid_subjects)

    # ----------------
    # Target methods which is meant to be used with groups metrics in Histories class
    # ----------------
    def apoe_group(self, subject_ids):
        # Read the .csv file
        df = pandas.read_csv(self.get_participants_tsv_path(), encoding="latin-1", sep=";")

        # Convert to dict
        dict_ = {sub_id: self._apoe_to_target(var) for sub_id, var in zip(df["participant_id"], df["apoe"])}

        # Select the ones passed in the subjects list, and return as numpy array
        return numpy.array([dict_[sub_id] for sub_id in subject_ids])

    @staticmethod
    def _apoe_to_target(apoe):
        """Mapping from APOE to numerical value"""
        combinations = ("E2E2", "E2E3", "E2E4", "E3E3", "E3E4", "E4E4")
        mapping = {combination: i for i, combination in enumerate(combinations)}
        return mapping[apoe]


# ----------------
# Small helpful functions
# ----------------
def _strip_sub_id(sub_id):
    """
    Method for removing the safety character of a subject ID, if any

    Parameters
    ----------
    sub_id : str

    Returns
    -------
    str

    Examples
    --------
    >>> _strip_sub_id("1-502-K")
    '1-502'
    >>> _strip_sub_id("1-502")
    '1-502'

    Visit IDs gives error

    >>> _strip_sub_id("1-502-1-K")
    Traceback (most recent call last):
    ...
    ValueError: Unexpected subject ID format of subject id '1-502-1-K'
    """
    split_sub_id = sub_id.split("-")

    if len(split_sub_id) == 2:
        return sub_id
    elif len(split_sub_id) == 3:
        return "-".join(split_sub_id[:-1])
    else:
        raise ValueError(f"Unexpected subject ID format of subject id {sub_id!r}")
