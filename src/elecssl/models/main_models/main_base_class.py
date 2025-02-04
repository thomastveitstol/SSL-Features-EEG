from pathlib import Path

import torch.nn as nn


class MainModuleBase(nn.Module):

    def load_model(self, path: Path) -> "MainModuleBase":
        raise NotImplementedError

    def save_model(self, name: str, path: Path):
        raise NotImplementedError