"""
I had some problems with saving some data for the LEMON dataset, so this script tries to find potential bugs.

Nothing was found
"""
from progressbar import progressbar

from elecssl.data.datasets.dataset_base import OcularState
from elecssl.data.datasets.lemon import LEMON


def main():
    # Loop through all subjects
    errors = []
    for subject_id in progressbar(LEMON().get_subject_ids(), prefix="Subject ", redirect_stdout=True):
        for ocular_state in (OcularState.EO, OcularState.EC):
            _ = LEMON().load_single_mne_object(
                subject_id=subject_id, ocular_state=ocular_state, derivatives=False, preload=True,
                interpolation_method=None
            )

    # Print all errors
    for error in errors:
        print(error)


if __name__ == "__main__":
    main()
