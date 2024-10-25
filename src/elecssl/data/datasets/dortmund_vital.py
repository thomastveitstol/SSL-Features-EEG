import os

import openneuro

from elecssl.data.datasets.dataset_base import EEGDatasetBase, OcularState


class DortmundVital(EEGDatasetBase):
    """
    A dataset which is part of the Dortmund Vital Study, accessible at OpenNeuro (ds005385)

    Paper:
        Getzmann, S., Gajewski, P.D., Schneider, D. et al. Resting-state EEG data before and after cognitive activity
        across the adult lifespan and a 5-year follow-up. Sci Data 11, 988 (2024).
        https://doi.org/10.1038/s41597-024-03797-w
    OpenNeuro:
        Edmund Wascher and Daniel Schneider and Patrick D. Gajewski and Stephan Getzmann (2024). Resting-state EEG data
        before and after cognitive activity across the adult lifespan and a 5-year follow-up. OpenNeuro. [Dataset]
        doi: doi:10.18112/openneuro.ds005385.v1.0.2
    """

    __slots__ = ()

    _ocular_states = (OcularState.EC, OcularState.EO)

    @classmethod
    def download(cls):
        # Make directory
        path = cls.get_mne_path()
        os.mkdir(path)

        # Download from OpenNeuro
        openneuro.download(dataset="ds005385", target_dir=path)

    # ----------------
    # Loading methods
    # ----------------
    def _load_single_raw_mne_object(self, *args, **kwargs):
        raise NotImplementedError

    # ----------------
    # Channel system
    # ----------------
    def channel_name_to_index(self):
        raise NotImplementedError
