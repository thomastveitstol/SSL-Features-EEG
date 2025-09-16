import os

import pytest

from elecssl.data.datasets.ai_mind import AIMind
from elecssl.data.datasets.dataset_base import get_channel_name_order, OcularState, EEGDatasetBase
from elecssl.data.datasets.dortmund_vital import DortmundVital
from elecssl.data.datasets.lemon import LEMON


def _generate_dataset_params(datasets_to_test, *, skip_in_tox):
    params = []
    for dataset, multiple_kwargs in datasets_to_test:
        marks = []

        name = type(dataset).__name__

        # Skip if dataset directory doesn't exist
        path = dataset.get_mne_path()
        if not os.path.isdir(path):
            marks.append(pytest.mark.skip(reason=f"Dataset {name!r} not available at {type(dataset).get_mne_path()}"))

        # Skip these in tox
        if name in skip_in_tox and "TOX_ENV_NAME" in os.environ:
            marks.append(pytest.mark.skipif(True, reason=f"Too time consuming to run {name!r} in tox"))

        params.append(pytest.param(dataset, multiple_kwargs, id=name, marks=marks))

    return params


# -----------
# Tests
# -----------
@pytest.mark.parametrize("dataset, multiple_kwargs", _generate_dataset_params([
    (DortmundVital(), ({"derivatives": False, "acquisition": "pre", "ocular_state": OcularState.EO, "session": 1},
                       {"derivatives": False, "acquisition": "pre", "ocular_state": OcularState.EC, "session": 1})),
    (AIMind(), ({"derivatives": False, "visit": 1, "ocular_state": OcularState.EO, "recording": 1},
                {"derivatives": False, "visit": 1, "ocular_state": OcularState.EC, "recording": 2},
                {"derivatives": False, "visit": 1, "ocular_state": OcularState.EO, "recording": 3},
                {"derivatives": False, "visit": 1, "ocular_state": OcularState.EC, "recording": 4})),
    (LEMON(), ({"interpolation_method": "MNE", "derivatives": False, "ocular_state": OcularState.EO},
               {"interpolation_method": "MNE", "derivatives": False, "ocular_state": OcularState.EC}),
     )], skip_in_tox=("AIMind", "LEMON")
))
def test_channel_names_ordering(dataset, multiple_kwargs):
    """Test if the channel names are always the same per dataset (equal and correctly ordered). Lemon is tested
    separately, as interpolation must be performed"""
    assert isinstance(dataset, EEGDatasetBase)

    # Get the channel names
    expected_channel_names = get_channel_name_order(dataset.channel_name_to_index())

    # Loop through different configurations
    for kwargs in multiple_kwargs:
        # Loop through all subjects
        for i, subject_id in enumerate(dataset.get_subject_ids()):
            # Load the EEG object
            try:
                raw = dataset.load_single_mne_object(subject_id=subject_id, **kwargs, preload=False)
            except FileNotFoundError:
                continue

            # Test if the channel names are as expected
            assert tuple(raw.ch_names) == expected_channel_names, \
                (f"The loaded channel names did not match the expected ones for the dataset {dataset.name!r}, subject "
                 f"{subject_id!r} ({i}-th subject):\nActual: {tuple(raw.ch_names)}\nExpected: {expected_channel_names}")
