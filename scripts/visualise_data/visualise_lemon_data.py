"""
Script for visually inspecting EEG data (including PSD)
"""
import io
import itertools
from contextlib import redirect_stdout

from matplotlib import pyplot

from elecssl.data.datasets.dataset_base import OcularState
from elecssl.data.datasets.lemon import LEMON


def _plot_single_eeg(subject_id, *, interpolation_method, ocular_state):
    # ----------------
    # Load data
    # ----------------
    eeg = LEMON().load_single_mne_object(
        subject_id=subject_id, ocular_state=ocular_state, interpolation_method=interpolation_method, derivatives=False
    )
    # Crop
    eeg.crop(30, eeg.n_times / eeg.info["sfreq"] - 10)

    # Filter
    eeg.filter(1, 45)
    eeg.notch_filter(50)

    # ----------------
    # Plot data
    # ----------------
    eeg.plot()
    eeg.compute_psd().plot()

    pyplot.show()


def main():
    interpolation_method = "MNE"
    ocular_states = (OcularState.EO, OcularState.EC)

    # ----------------
    # Hyperparameters
    # ----------------
    subjects = set(LEMON().get_subject_ids())

    for subject_name, ocular_state in itertools.product(subjects, ocular_states):
        print(f"{subject_name}-{ocular_state.value}")
        with redirect_stdout(io.StringIO()):
            _plot_single_eeg(subject_name, ocular_state=ocular_state, interpolation_method=interpolation_method)


if __name__ == "__main__":
    main()
