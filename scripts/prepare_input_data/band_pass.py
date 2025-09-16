import os
from pathlib import Path

from elecssl.data.data_preparation.band_pass_filter import BandPass
from elecssl.data.paths import get_numpy_data_storage_path


def main():
    # ---------------
    # Make paths
    # ---------------
    save_to = get_numpy_data_storage_path()
    # config_path = Path(os.path.dirname(__file__)) / "config_files" / "band_pass.yml"
    config_path = Path(os.path.dirname(__file__)) / "config_files" / "rerun_band_pass.yml"

    # ---------------
    # Loop through the ocular states
    # ---------------
    BandPass().prepare_data_for_experiments(config_path=config_path, save_to=save_to)


if __name__ == "__main__":
    main()
