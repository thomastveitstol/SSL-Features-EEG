"""
Script for getting to know the data. Channel names and stuff

Dortmund Vital sub-342 looks bad without any cleaning.
"""
from matplotlib import pyplot
from mne import make_fixed_length_epochs

from elecssl.data.data_preparation.data_prep_base import run_autoreject
from elecssl.data.datasets.dataset_base import OcularState
from elecssl.data.datasets.dortmund_vital import DortmundVital


def main():
    # -------------
    # Choices
    # -------------
    subject = "sub-342"
    ocular_state = OcularState.EO
    acquisition = "pre"
    session = 1

    duration = 4
    use_epochs = True
    resample = None  # 180
    apply_autoreject = True

    preload = True

    if isinstance(subject, int):
        subject_id = DortmundVital().get_subject_ids()[subject]
    else:
        subject_id = subject

    # -------------
    # Get data
    # -------------
    eeg = DortmundVital().load_single_mne_object(
        subject_id=subject_id, ocular_state=ocular_state, session=session, acquisition=acquisition, preload=preload
    )
    # Crop
    eeg.crop(30, eeg.n_times / eeg.info["sfreq"] - 10)

    eeg.filter(1, 45, verbose=False)
    eeg.notch_filter(50, verbose=False)

    if resample is not None:
        eeg.resample(resample, verbose=False)

    if use_epochs:
        eeg = make_fixed_length_epochs(raw=eeg, duration=duration, preload=True, verbose=False)

        if apply_autoreject:
            eeg = run_autoreject(eeg, autoreject_resample=None, seed=1, default_num_splits=10)

    # -------------
    # Plot
    # -------------
    eeg.plot()
    psd = eeg.compute_psd(verbose=False)
    print(psd.get_data().shape)
    psd.plot()

    pyplot.show()


if __name__ == "__main__":
    main()
