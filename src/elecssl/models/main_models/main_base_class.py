from pathlib import Path

import torch.nn as nn


class MainModuleBase(nn.Module):
    """
    Base class for all models which will used. For example, it combines spatial method (interpolation/RBP), domain
    discriminator training, and the DL architecture itself.
    """

    @classmethod
    def load_model(cls, path: Path) -> "MainModuleBase":
        raise NotImplementedError

    @classmethod
    def save_model(cls, name: str, path: Path):
        raise NotImplementedError
