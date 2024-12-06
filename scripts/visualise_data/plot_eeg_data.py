"""
Script for getting to know the data. Channel names and stuff
"""
from matplotlib import pyplot

from elecssl.data.datasets.dataset_base import OcularState
from elecssl.data.datasets.dortmund_vital import DortmundVital


def main():
    # -------------
    # Choices
    # -------------
    subject = 4
    ocular_state = OcularState.EC
    acquisition = "pre"
    session = 1

    preload = True

    subject_id = DortmundVital().get_subject_ids()[subject]

    # -------------
    # Get data
    # -------------
    raw = DortmundVital().load_single_mne_object(
        subject_id=subject_id, ocular_state=ocular_state, session=session, acquisition=acquisition, preload=preload
    )

    raw.filter(1, 45, verbose=False)
    raw.notch_filter(50, verbose=False)

    # -------------
    # Plot
    # -------------
    raw.plot()
    raw.compute_psd(verbose=False).plot()

    pyplot.show()


if __name__ == "__main__":
    main()
