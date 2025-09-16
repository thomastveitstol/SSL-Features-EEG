"""
Script I made after notch filter failed for some subjects in the LEMON dataset. I was because the sampling frequency is
not the same for all.

However, it doesn't look like a notch filter is needed for LEMON
"""
import mne
from matplotlib import pyplot

from elecssl.data.datasets.dataset_base import OcularState
from elecssl.data.datasets.lemon import LEMON


def main():
    # Select subject and load data
    for sub_id in LEMON().get_subject_ids():
        eeg = mne.make_fixed_length_epochs(
            LEMON().load_single_mne_object(subject_id=sub_id, ocular_state=OcularState.EC, interpolation_method="MNE",
                                           preload=True),
            duration=5, preload=True, verbose=False)

        # Apply notch filter
        print(eeg.info["sfreq"])
        # eeg.notch_filter(50)
        # eeg.resample(250)

        # Plot
        # eeg.plot()
        eeg.compute_psd(fmin=0, fmax=80, verbose=False).plot()

        pyplot.show()


if __name__ == "__main__":
    main()
