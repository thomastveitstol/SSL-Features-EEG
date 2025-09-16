"""
Script for visually inspecting EEG data (including PSD)
"""
import io
import itertools
from contextlib import redirect_stdout

import mne
from matplotlib import pyplot

from elecssl.data.datasets.dataset_base import OcularState
from elecssl.data.datasets.dortmund_vital import DortmundVital


def _plot_single_eeg(subject_id, *, session, acquisition, ocular_state):
    # ----------------
    # Load data
    # ----------------
    eeg = DortmundVital().load_single_mne_object(
        subject_id=subject_id, session=session, ocular_state=ocular_state, acquisition=acquisition, derivatives=False
    )
    # Crop
    eeg.crop(30, eeg.n_times / eeg.info["sfreq"] - 10)

    # Filter
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
    acquisition = "pre"
    session = 1
    ocular_states = (OcularState.EO, OcularState.EC)

    # ----------------
    # Hyperparameters
    # ----------------
    subjects = set(DortmundVital().get_subject_ids())

    for subject_name, ocular_state in itertools.product(subjects, ocular_states):
        print(f"{subject_name}-{ocular_state.value}")
        with redirect_stdout(io.StringIO()):
            _plot_single_eeg(subject_name, acquisition=acquisition, ocular_state=ocular_state, session=session)


if __name__ == "__main__":
    main()
