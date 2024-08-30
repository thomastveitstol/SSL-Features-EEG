from matplotlib import pyplot

from elecssl.data.datasets.dataset_base import OcularState
from elecssl.data.datasets.lemon import LEMON


def main():
    # Select subject and load data
    sub_id = LEMON().get_subject_ids()[30]
    eeg = LEMON().load_single_mne_object(subject_id=sub_id, ocular_state=OcularState.EC, interpolation_method="MNE")

    # Apply notch filter
    print(eeg.info["sfreq"])
    # eeg.notch_filter(50)

    # Plot
    eeg.plot()
    eeg.compute_psd().plot()

    pyplot.show()


if __name__ == "__main__":
    main()
