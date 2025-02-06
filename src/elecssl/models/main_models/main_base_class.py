from pathlib import Path

import torch
import torch.nn as nn


class MainModuleBase(nn.Module):
    """
    Base class for all models which will used. For example, it combines spatial method (interpolation/RBP), domain
    discriminator training, and the DL architecture itself.
    """

    @classmethod
    def load_model(cls, name: str, path: Path) -> "MainModuleBase":
        model = torch.load((path / name).with_suffix(".pt"))
        if not isinstance(model, cls):
            raise ModuleLoadError(f"Expected the loaded module to be from the same class as attempted loaded from "
                                  f"({cls.__name__}), but got {type(model)}")
        return model

    def save_model(self, name: str, path: Path):
        # todo: Sub-optimal to use this saving
        torch.save(self, (path / name).with_suffix(".pt"))


# ----------
# Errors
# ----------
class ModuleLoadError(Exception):
    ...
