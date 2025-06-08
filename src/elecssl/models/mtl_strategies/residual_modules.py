from torch import nn


class ResidualHead(nn.Module):
    """
    Module used for making predictions from the residuals

    Examples
    --------
    >>> hasattr(ResidualHead, "shared"), hasattr(ResidualHead(), "shared")
    (True, True)
    >>> ResidualHead.shared, ResidualHead().shared
    (False, False)
    """
    shared = False  # This is supposed to be used with 'shared_parameters' to avoid setting the parameters of these
    # models to 'shared across tasks'. Because they are supposed to be task-specific

    def __init__(self):
        super().__init__()
        self._head = nn.Linear(1, 1, bias=True)

    def forward(self, residuals):
        return self._head(residuals)
