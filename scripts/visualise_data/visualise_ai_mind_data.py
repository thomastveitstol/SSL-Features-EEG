"""
Script for checking out the AI-Mind data
"""
import io
import itertools
from contextlib import redirect_stdout

import mne
from matplotlib import pyplot

from elecssl.data.datasets.ai_mind import AIMind
from elecssl.data.datasets.dataset_base import OcularState


def _get_ocular_state(recording):
    """Function for getting the expected ocular state of a recording"""
    if recording in (1, 3):
        ocular_state = OcularState.EO
    elif recording in (2, 4):
        ocular_state = OcularState.EC
    else:
        raise ValueError(f"Unexpected recording: {recording}")
    return ocular_state


def _plot_single_eeg(subject_id, *, recording, visit):
    ocular_state = _get_ocular_state(recording)
    derivatives = False  # the version is specified differently

    # ----------------
    # Load data
    # ----------------
    eeg = AIMind().load_single_mne_object(
        subject_id=subject_id, recording=recording, ocular_state=ocular_state, derivatives=derivatives, visit=visit
    )

    eeg.filter(1, 45)
    eeg.notch_filter(50)
    # ----------------
    # Plot data
    # ----------------
    epochs = mne.make_fixed_length_epochs(eeg, duration=5, preload=True, verbose=False)
    epochs.compute_psd(fmin=0, fmax=80, verbose=False).plot()

    eeg.plot(duration=5)

    pyplot.show()


def main():
    visits = (1,)
    recordings = (1, 2, 3, 4)

    # ----------------
    # Hyperparameters
    # ----------------
    subjects = set(AIMind().get_subject_ids())

    for subject_name, visit, recording in itertools.product(subjects, visits, recordings):
        print(f"{subject_name}-{visit}: {subject_name}-{visit}_{recording}-{_get_ocular_state(recording).value}")
        try:
            with redirect_stdout(io.StringIO()):
                _plot_single_eeg(subject_name, recording=recording, visit=visit)
        except FileNotFoundError:
            print("File not found...")


if __name__ == "__main__":
    main()
