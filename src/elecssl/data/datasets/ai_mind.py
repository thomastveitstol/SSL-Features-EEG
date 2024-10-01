import os
import pickle
from typing import Tuple

import mne
import numpy
import pandas

from elecssl.data.datasets.dataset_base import EEGDatasetBase, OcularState, target_method
from elecssl.data.paths import get_pre_ctad_raw_data_path


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
    >>> len(AIMind().get_subject_ids())
    538
    """

    __slots__ = ()

    # Note that we are here only interested in the EEG channels
    # todo: not very elegant. At least make tests
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
    def get_participants_tsv_path(self):
        # It will actually be a .csv file. Maybe I'll fix it at some point...
        return self.get_mne_path().parent / "ai_scd_crf_corr_apoe_ptau217_ptau181_data_2024-09-16.csv"

    @classmethod
    def get_mne_path(cls):
        """The original data is stored in TSD, in a completely different folder than the other external datasets"""
        return get_pre_ctad_raw_data_path()

    # ----------------
    # Loading methods
    # ----------------
    def _get_subject_ids(self) -> Tuple[str, ...]:
        # As ctad is approaching, I'm just hard-coding up this...
        # todo
        # Get path to all the pickle files
        path_to_pickle_files = get_pre_ctad_raw_data_path() / "1-continuous"

        # Get all pickle file names. Also, remove.pickle and safety character (e.g., 1-111)
        file_names = {file_name[:5] for file_name in os.listdir(path_to_pickle_files)}

        # Manually remove 4-257 because not all files exist
        file_names.remove("4-257")

        # Get subjects from csv file
        subject_ids = tuple(pandas.read_csv(self.get_participants_tsv_path(), sep=";",
                                            encoding="latin-1")["participant_id"])

        # Return only the ones with associated pickle files
        return tuple(sub_id for sub_id in subject_ids if sub_id[:5] in file_names)

    def _get_subject_path(self, *, set_name, subject_id, visit, recording, ocular_state):
        """Method for getting the absolute path. I don't know the algorithm for computing the safety character, so
        trying them all instead"""
        for security_character in _ALPHABET:
            # Try the current safety character
            path = (self.get_mne_path() / set_name / f"{subject_id[:-2]}-{visit}-{security_character}_{recording}-"
                                                     f"{ocular_state.value}").with_suffix(".pickle")

            # If the file exists, return the current path
            if os.path.isfile(path):
                return path

        raise FileNotFoundError(f"Found no file with any security character in the alphabet for {subject_id=}, "
                                f"{ocular_state=}, {visit=}, {set_name=}, {recording=}")

    def _load_single_raw_mne_object(self, subject_id, *, ocular_state, visit, recording, set_name):
        # -------------
        # Load object
        # -------------
        # Create path
        # todo: check the difference between subject id and visit id. I'll need the safety character
        path = self._get_subject_path(set_name=set_name, subject_id=subject_id, visit=visit, recording=recording,
                                      ocular_state=ocular_state)

        # Load pickle file object
        with open(path, "rb") as file:
            epochs: mne.Epochs = pickle.load(file)

        # Make a type check
        if not isinstance(epochs, mne.Epochs):
            raise TypeError(f"Unexpected type of loaded data: {type(epochs)}")

        # -------------
        # Make raw object
        # -------------
        raw = mne.io.RawArray(data=numpy.squeeze(epochs.get_data(), axis=0), info=epochs.info, verbose=False)

        # Remove non EEG channels
        raw = raw.drop_channels(self._non_eeg_channels)

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
    # todo: about time to make reading from .csv/.tsv file the standard
    # ----------------
    @target_method
    def age(self, subject_ids):
        """Age at the first visit"""
        # Read the .csv file
        df = pandas.read_csv(self.get_participants_tsv_path())

        # Check if all subjects in the csv are unique (should not really be needed here, so I might remove this in the
        # future)
        assert len(df["participant_id"]) == len(set(df["participant_id"]))

        # Convert to dict
        sub_id_to_age = {sub_id: age for sub_id, age in zip(df["participant_id"], df["age"])}

        # Select the ones passed in the subjects list, and return as numpy array
        return numpy.array([sub_id_to_age[sub_id] for sub_id in subject_ids])

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
        df = pandas.read_csv(self.get_participants_tsv_path(), encoding="latin-1", sep=";")

        # Convert to dict
        dict_ = {sub_id: var for sub_id, var in zip(df["participant_id"], df["ptau181"])}

        # Select the ones passed in the subjects list, and return as numpy array
        return numpy.array([dict_[sub_id] for sub_id in subject_ids])

    # ----------------
    # Target methods which is meant to be used with groups metrics in Histories class.
    # todo: stop using target_method decorator for this...
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
