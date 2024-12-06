"""
Script for getting to know the data. Channel names and stuff
"""
from autoreject import autoreject
from matplotlib import pyplot
from mne import make_fixed_length_epochs

from elecssl.data.datasets.dataset_base import OcularState
from elecssl.data.datasets.dortmund_vital import DortmundVital


def main():
    # -------------
    # Choices
    # -------------
    subject = "sub-052"
    ocular_state = OcularState.EO
    acquisition = "pre"
    session = 1

    duration = 4
    use_epochs = True
    autoreject_resample = None
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
    eeg.filter(1, 45, verbose=False)
    eeg.notch_filter(50, verbose=False)

    if use_epochs:
        eeg = make_fixed_length_epochs(raw=eeg, duration=duration, preload=True, verbose=False)

        if apply_autoreject:
            if autoreject_resample is not None:
                eeg.resample(autoreject_resample, verbose=False)
            reject = autoreject.AutoReject(verbose=False)
            eeg = reject.fit_transform(eeg, return_log=False)

    # -------------
    # Plot
    # -------------
    eeg.plot()
    eeg.compute_psd(verbose=False).plot(verbose=False)

    pyplot.show()


if __name__ == "__main__":
    main()
