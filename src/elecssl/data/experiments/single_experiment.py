import os
import traceback

import torch


class SSLExperiment:
    """
    Class for running a single experiment. The model is trained on an SSL task while relationships to other variables
    (specified in the config file, such as age or a cognitive test score)
    """

    def __init__(self, config, pre_processing_config, results_path, device=None):
        """

        Parameters
        ----------
        config : dict[str, Any]
        pre_processing_config : dict[str, Any]
        results_path : pathlib.Path
        device
        """
        # Create path
        os.mkdir(results_path)

        # Store attributes
        self._config = config
        self._pre_processing_config = pre_processing_config
        self._results_path = results_path
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device

    # -------------
    # Dunder methods for context manager (using the 'with' statement). See this video from mCoding for more information
    # on context managers https://www.youtube.com/watch?v=LBJlGwJ899Y&t=640s
    # -------------
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """This will execute when exiting the with statement. It will NOT execute if the run was killed by the operating
        system, which can happen if too much data is loaded into memory"""
        # If everything was as it should, just exit
        if exc_val is None:
            return None

        # Otherwise, document the error received in a text file
        with open((self._results_path / exc_type.__name__).with_suffix(".txt"), "w") as file:
            file.write("Traceback (most recent call last):\n")
            traceback.print_tb(exc_tb, file=file)
            file.write(f"{exc_type.__name__}: {exc_val}")

    # -------------
    # Main method for running the cross validation experiment
    # -------------
    def run_experiment(self):
        raise NotImplementedError
