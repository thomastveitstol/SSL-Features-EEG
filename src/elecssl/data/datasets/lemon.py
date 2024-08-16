import os
import warnings
from typing import Tuple

import boto3
import numpy
import pandas
from botocore import UNSIGNED
from botocore.client import Config
import mne

from elecssl.data.datasets.dataset_base import EEGDatasetBase, target_method, OcularState
from elecssl.data.datasets.utils import sex_to_int


class LEMON(EEGDatasetBase):
    """
    Dataset from 'A mind-brain-body dataset of MRI, EEG, cognition, emotion, and peripheral physiology in young and old
    adults'

    Paper:
        Babayan, A., Erbey, M., Kumral, D. et al. A mind-brain-body dataset of MRI, EEG, cognition, emotion, and
        peripheral physiology in young and old adults. Sci Data 6, 180308 (2019). https://doi.org/10.1038/sdata.2018.308

    Examples:
    ----------
    >>> LEMON().name
    'LEMON'
    >>> len(LEMON().get_subject_ids()), LEMON().get_subject_ids()[:4]
    (203, ('sub-032301', 'sub-032302', 'sub-032303', 'sub-032304'))
    >>> len(LEMON()._channel_names)  # There are 61 because the 62nd is VEOG (eye electrode), see original article
    61
    """

    __slots__ = ()

    _channel_names = ("Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4",
                      "T8", "CP5", "CP1", "CP2", "CP6", "AFz", "P7", "P3", "Pz", "P4", "P8", "PO9", "O1", "Oz", "O2",
                      "PO10", "AF7", "AF3", "AF4", "AF8", "F5", "F1", "F2", "F6", "FT7", "FC3", "FC4", "FT8", "C5",
                      "C1", "C2", "C6", "TP7", "CP3", "CPz", "CP4", "TP8", "P5", "P1", "P2", "P6", "PO7", "PO3", "POz",
                      "PO4", "PO8")
    _montage_name = "standard_1020"
    _ocular_states = (OcularState.EC, OcularState.EO)

    # ----------------
    # Loading methods
    # ----------------
    def get_participants_tsv_path(self):
        # Doesn't matter if we specify EC or EO
        return self.get_mne_path() / "EC" / "Participants_MPILMBB_LEMON.csv"

    def _get_subject_ids(self) -> Tuple[str, ...]:
        # Get the subject IDs from participants file
        participants = pandas.read_csv(self.get_participants_tsv_path())["ID"]

        # Keep only the ones in the downloaded EEG data
        _eeg_availables = self.get_mne_path() / "EC"  # todo: would it matter if we wrote EO?
        return tuple(participant for participant in participants if participant in _eeg_availables)

    def _load_single_raw_mne_object(self, subject_id, *, ocular_state, interpolation_method, preload=True):
        # -------------
        # Load object
        # -------------
        # Create path
        path = (self.get_mne_path() / ocular_state.value / subject_id / subject_id).with_suffix(".set")

        # Load MNE object
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", category=RuntimeWarning)
            raw = mne.io.read_raw_eeglab(path, preload=preload, verbose=False)

        # If no interpolation method is used, just return the object
        if interpolation_method is None:
            return raw

        # -------------
        # Interpolation
        # -------------
        # Get missing channels
        missing_channels = tuple(set(self._channel_names) - set(raw.ch_names))
        if not missing_channels:
            return raw

        # Create info objects
        info = mne.create_info(ch_names=raw.ch_names + list(missing_channels), sfreq=raw.info["sfreq"], ch_types="eeg",
                               verbose=False)

        # Create numpy array
        data = raw.get_data()
        data = numpy.concatenate((data, numpy.zeros(shape=(len(missing_channels), data.shape[-1]))), axis=0)

        # Create raw object
        raw = mne.io.RawArray(data=data, info=info, verbose=False)

        # Reorder channels
        raw.reorder_channels(list(self._channel_names))

        # Set the montage
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", category=RuntimeWarning)

            raw.set_montage(
                mne.channels.make_dig_montage(ch_pos=self.channel_system.electrode_positions), verbose=False
            )

        # Set the zero-filled channels to bads, and interpolate
        raw.info["bads"] = list(missing_channels)
        raw.interpolate_bads(verbose=False, method={"eeg": interpolation_method})

        return raw

    @classmethod
    def download(cls):
        # Make root directory
        root_dir = cls.get_mne_path()
        os.mkdir(root_dir)

        # Download EEG for both ocular states
        for ocular_state in (OcularState.EC, OcularState.EO):
            to_path = root_dir / ocular_state.value

            # Make directory
            os.mkdir(to_path)

            # Download
            cls._download(to_path, ocular_state=ocular_state)

    @classmethod
    def _download(cls, to_path, ocular_state: OcularState):
        """
        Method for downloading the MPI Lemon dataset, eyes closed EEG data only

        Created by Mats Tveter and Thomas Tveitstøl

        Returns
        -------
        None
        """
        # MPI Lemon specifications
        bucket = 'fcp-indi'
        prefix = "data/Projects/INDI/MPI-LEMON/EEG_MPILMBB_LEMON/EEG_Preprocessed"

        s3client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

        # Creating buckets needed for downloading the files
        s3_resource = boto3.resource('s3', config=Config(signature_version=UNSIGNED))
        s3_bucket = s3_resource.Bucket(bucket)

        # Paginator is need because the amount of files exceeds the boto3.client possible maxkeys
        paginator = s3client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

        # Looping through the content of the bucket
        for page in pages:
            for obj in page['Contents']:
                # Get the path of the .set or .fdt path
                # e.g. data/Projects/.../EEG_Preprocessed/sub-032514/sub-032514_EO.set
                file_path = obj['Key']

                # Download EEG data, ensuring correct ocular state
                if f"_{ocular_state.value}.set" == file_path[-7:] or f"_{ocular_state.value}.fdt" in file_path[-7:]:
                    # Get subject ID and file type from the folder name
                    subject_id = file_path.split("/")[-2]
                    file_type = file_path.split(".")[-1]  # either .set or .fdt

                    # (Maybe) make folder. The .set and .fdt of a single subject must be placed in the same folder (a
                    # requirement from MNE when loading)
                    path = os.path.join(to_path, subject_id)
                    if not os.path.isdir(path):
                        os.mkdir(path)

                    # Download
                    s3_bucket.download_file(file_path, os.path.join(path, f"{subject_id}.{file_type}"))  # type: ignore

        # Participants file
        s3_bucket.download_file("data/Projects/INDI/MPI-LEMON/Participants_MPILMBB_LEMON.csv",
                                os.path.join(to_path, "Participants_MPILMBB_LEMON.csv"))

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
        df = pandas.read_csv(self.get_participants_tsv_path())

        # Setting age to the mean of lower and upper bound of the interval
        age_intervals = df["Age"]
        lower = numpy.array([int(age_interval.split("-")[0]) for age_interval in age_intervals])
        upper = numpy.array([int(age_interval.split("-")[1]) for age_interval in age_intervals])

        mean_age = (upper + lower) / 2

        # Convert to dict
        sub_id_to_age = {name: age for name, age in zip(df["ID"], mean_age)}

        # Extract the ages of the subjects, in the same order as the input argument
        return numpy.array([sub_id_to_age[sub_id] for sub_id in subject_ids])

    @target_method
    def sex(self, subject_ids):
        # Read the .tsv file
        df = pandas.read_csv(self.get_participants_tsv_path())

        # Convert to sex dict
        sex = {sub_id: sex_to_int(_int_to_sex(gender)) for sub_id, gender
               in zip(df["ID"], df["Gender_ 1=female_2=male"])}

        # Select the ones passed in the subjects list, and return as numpy array
        return numpy.array([sex[sub_id] for sub_id in subject_ids])


# -------------
# Functions
# -------------
def _int_to_sex(integer):
    if integer == 1:
        return "female"
    elif integer == 2:
        return "male"
    else:
        raise ValueError
