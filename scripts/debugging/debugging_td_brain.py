"""
Script for finding errors in the TDBRAIN dataset.

Manual fixes were made for sub-19703068 and sub-19703550 for the CDL project (eyes closed).

The script gave the following output (neglecting progressbar):

LookupError (sub-19687321): unknown encoding: UTF-19687321_ses-1_task-restEO_eeg.eeg
FileNotFoundError (sub-19696913): [Errno 2] No such file or directory: '/media/thomas/AI-Mind - Anonymised data/CDLDatasets/TDBrain/sub-19696913/ses-1/eeg/sub-1969913_ses-1_task-restEO_eeg.vmrk'
FileNotFoundError (sub-19703550): [Errno 2] No such file or directory: '/media/thomas/AI-Mind - Anonymised data/CDLDatasets/TDBrain/sub-19703550/ses-1/eeg/sub-19706550_ses-1_task-restEO_eeg.vmrk'
FileNotFoundError (sub-19721668): [Errno 2] No such file or directory: '/media/thomas/AI-Mind - Anonymised data/CDLDatasets/TDBrain/sub-19721668/ses-1/eeg/sub-819721668_ses-1_task-restEO_eeg.vmrk'

Therefore, I manually made the following fixes:
    - sub-19687321:
        Replaced (in vhdr, under [Common Infos]):
            'Codepage=UTF-19687321_ses-1_task-restEO_eeg.eeg'
        with:
            'Codepage=UTF-8
             DataFile=sub-19687321_ses-1_task-restEO_eeg.eeg'
    - sub-19696913:
        The subject ID number was wrong in the .vhdr file. Replaced '1969913' with '19696913' in DataFile and MarkerFile
        under [Common Infos]
    - sub-19703550:
        The subject ID number was wrong in the .vhdr file. Replaced '19706550' with '19703550' in DataFile and
        MarkerFile under [Common Infos]
    - sub-19721668:
        The subject ID number was wrong in the .vhdr file. Replaced '819721668' with '19721668' in DataFile and
        MarkerFile under [Common Infos]
"""
from progressbar import progressbar

from elecssl.data.datasets.dataset_base import OcularState
from elecssl.data.datasets.td_brain import TDBRAIN


def main():
    # Loop through all subjects
    errors = []
    for subject_id in progressbar(TDBRAIN().get_subject_ids(), prefix="Subject ", redirect_stdout=True):
        try:
            _ = TDBRAIN().load_single_mne_object(subject_id=subject_id, ocular_state=OcularState.EO,
                                                 derivatives=False, preload=False)
        except (LookupError, FileNotFoundError) as e:
            errors.append(f"{type(e).__name__} ({subject_id}): {e}")

    # Print all errors
    for error in errors:
        print(error)


if __name__ == "__main__":
    main()
