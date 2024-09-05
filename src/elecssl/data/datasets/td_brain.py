import os
from typing import Tuple, List, Dict

import mne
import numpy
import pandas

from elecssl.data.datasets.dataset_base import EEGDatasetBase, target_method, OcularState
from elecssl.data.datasets.utils import sex_to_int
from elecssl.data.paths import get_td_brain_raw_data_storage_path


class TDBRAIN(EEGDatasetBase):
    """
    The TDBRAIN dataset

    Paper:
        van Dijk, H., van Wingen, G., Denys, D. et al. The two decades brainclinics research archive for insights in
        neurophysiology (TDBRAIN) database. Sci Data 9, 333 (2022). https://doi.org/10.1038/s41597-022-01409-z
    Link: www.brainclinics.com/resources

    Participant 'sub-19703068' has an error in the .eeg file in the .vhdr file. The problem is using '=' instead of "-".
    It was fixed manually.

    The .vhdr file of sub-19703550 was also fixed manually, as there were two instances where 3 was replaced by 6 in the
    ID

    Examples
    --------
    >>> len(TDBRAIN().get_subject_ids())
    1273
    >>> TDBRAIN().get_subject_ids()[:3]
    ('sub-19681349', 'sub-19681385', 'sub-19684666')
    >>> # The discrepancy in sex with the .tsv file is due to differences in integer to sex mapping
    >>> TDBRAIN().sex(TDBRAIN().get_subject_ids()[:3])
    array([1, 0, 0])
    >>> TDBRAIN().num_channels
    26
    """

    # Extracting channel names from Table 3 in the paper
    _channel_names = ("Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC3", "FCz", "FC4", "T7", "C3", "Cz", "C4", "T8",
                      "CP3", "CPz", "CP4", "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2")
    _montage_name = "standard_1020"  # 10-10 according to the paper
    _ocular_states = (OcularState.EC, OcularState.EO)

    @classmethod
    def get_mne_path(cls):
        """The data was also used in the CDL project, and it's a lot of data, so I don't want to copy it. The data must
        also be downloaded anyway"""
        return get_td_brain_raw_data_storage_path()

    def _get_subject_ids(self) -> Tuple[str, ...]:
        """Get the subject IDs available. Have to override due to (1) a minor variation in column name, and (2) repeated
        subject IDs"""
        # Get the subject IDs from participants file
        participants = tuple(pandas.read_csv(self.get_participants_tsv_path(), sep="\t")["participants_ID"])

        # Keep only a unique set. This implementation is reproducible
        uniques: List[str] = []
        for participant in participants:
            # Requiring the participant ID to be a string effectively removes a nan participant
            if participant not in uniques and isinstance(participant, str):
                uniques.append(participant)

        return tuple(uniques)

    def get_participants_tsv_path(self):
        return os.path.join(self.get_mne_path(), "TDBRAIN_participants_V2_data", "TDBRAIN_participants_V2.tsv")

    def _load_single_raw_mne_object(self, subject_id, *, ocular_state, preload=True):
        # Create path. We will use the first available one
        for session in ("ses-1", "ses-2", "ses-3"):
            subject_path = f"{subject_id}/{session}/eeg/{subject_id}_{session}_task-rest{ocular_state.value}_eeg.vhdr"
            path = os.path.join(self.get_mne_path(), subject_path)

            if os.path.isfile(path):
                break

        else:  # This will only run if we did not break after the for-loop
            raise ValueError(f"No sessions found for subject {subject_id}")

        # Make MNE raw object
        raw: mne.io.BaseRaw = mne.io.read_raw_brainvision(vhdr_fname=path, preload=preload, verbose=False)

        # Drop non-eeg channels
        raw.drop_channels(("VPVA", "VNVB", "HPHL", "HNHR", "Erbs", "OrbOcc", "Mass"))

        return raw

    # ----------------
    # Target methods
    # ----------------
    @target_method
    def sex(self, subject_ids):
        # Read the .tsv file
        df = pandas.read_csv(self.get_participants_tsv_path(), sep="\t")

        # Convert to sex dict. Duplicates in the .tsv file should not be a problem as the kay/value pair will just be
        # overridden. Need to remove the one nan value, thus requiring string participant IDs
        sex = {sub_id: sex_to_int(_int_to_sex(gender)) for sub_id, gender in zip(df["participants_ID"], df["gender"])
               if isinstance(sub_id, str)}

        # Select the ones passed in the subjects list, and return as numpy array
        return numpy.array([sex[sub_id] for sub_id in subject_ids])

    @target_method
    def age(self, subject_ids):
        """
        Examples
        --------
        >>> TDBRAIN().age(("sub-19681349", "sub-19681385", "sub-19684666"))
        array([51.59, 49.96, 47.05])
        >>> my_subjects = TDBRAIN().get_subject_ids()
        >>> all(isinstance(my_age, float) for my_age in TDBRAIN().age(my_subjects))  # type: ignore[attr-defined]
        True

        There are some missing data it seems...

        >>> sum(tuple(numpy.isnan(my_age) for my_age in TDBRAIN().age(my_subjects)))  # type: ignore[attr-defined]
        np.int64(17)
        """
        # Read the .tsv file
        df = pandas.read_csv(self.get_participants_tsv_path(), sep="\t")

        # Convert to age dict. Since we are using the first visit possible, we will use the same for age
        age_mapping: Dict[str, float] = dict()
        sess_ids: Dict[str, int] = dict()
        for sub_id, age, session in zip(df["participants_ID"], df["age"], df["sessID"]):
            # There are some nan values to ignore
            if not isinstance(sub_id, str):
                continue

            # Add the age to the mapping
            subject_age = float(age.replace(",", ".")) if isinstance(age, str) else age
            if sub_id in age_mapping:
                # If the current session is the newest, update the age. Otherwise, do nothing
                if session < sess_ids[sub_id]:
                    age_mapping[sub_id] = subject_age
                elif session > sess_ids[sub_id]:
                    pass
                else:
                    raise ValueError
            else:
                age_mapping[sub_id] = subject_age
                sess_ids[sub_id] = session

        # Select the ones passed in the subjects list, and return as numpy array
        return numpy.array([age_mapping[sub_id] for sub_id in subject_ids])

    @target_method
    def indication(self, subject_ids):
        """This is mostly designed for use with groups metrics in Histories class"""
        # Read the .tsv file
        df = pandas.read_csv(self.get_participants_tsv_path(), sep="\t")

        # Convert to dict
        status_mapping = {}
        for sub_id, status in zip(df["participants_ID"], df["indication"]):
            status_mapping[sub_id] = self._status_to_target(status)

        return numpy.array([status_mapping[sub_id] for sub_id in subject_ids])

    @target_method
    def formal_status(self, subject_ids):
        """This is mostly designed for use with groups metrics in Histories class"""
        # Read the .tsv file
        df = pandas.read_csv(self.get_participants_tsv_path(), sep="\t")

        # Convert to dict
        status_mapping = {}
        for sub_id, status in zip(df["participants_ID"], df["formal_status"]):
            status_mapping[sub_id] = self._status_to_target(status)

        return numpy.array([status_mapping[sub_id] for sub_id in subject_ids])

    @staticmethod
    def _status_to_target(status):
        """Mapping from diagnostic status to numerical value"""
        # I'll follow Table 1. todo: Are they certain about healthy? in that case, I need to add it to formal Dx as well
        # todo: sometimes, there are several diagnosis per subject, s.a., ADHD / ASPERGER, how to handle that?
        #  and what to do with the replication subjects?
        mapping = {"MDD": 0, "ADHD": 1, "SMC": 2, "OCD": 3, "Tinnitus": 4, "Insomnia": 5, "Parkinson": 6, "Burnout": 7,
                   "Dyslexia": 8, "Chronic Pain": 9, "UNKNOWN": 10, "Healthy": 11, "Other": 12, "REPLICATION": 13}
        if status in mapping:
            return mapping[status]

        return len(mapping)  # todo: this must be changed in the future

    # ----------------
    # Channel system
    # ----------------
    def _get_template_electrode_positions(self):
        montage = mne.channels.make_standard_montage(self._montage_name)
        channel_positions = montage.get_positions()["ch_pos"]

        # Return dict with channel positions, keeping only the ones in the data
        return {ch_name: tuple(pos) for ch_name, pos in channel_positions.items() if ch_name in self._channel_names}

    def channel_name_to_index(self):
        return {ch_name: i for i, ch_name in enumerate(self._channel_names)}


# -------------
# Functions
# -------------
def _int_to_sex(integer):
    # This mapping was taken from the TDBRIAN_participants_V2.json file
    if integer == 0:
        return "female"
    elif integer == 1:
        return "male"
    else:
        raise ValueError(f"Unexpected integer sex: {integer}")
