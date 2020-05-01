from typing import List
from torch import nn


class _BaseNetwork:
    """Superclass for uniform representation string for easier logging and debugging"""

    def __repr__(self):
        ret = ""
        for p in ["strategy", "optim", "coverage", "scopes", "width", "damp", "freeze", "self_connection", "pad"]:
            if p in self.__dict__.keys():
                ret += str(self.__dict__[p])
            ret += ","
        return f"{self.__class__.__name__},{ret[:-1]}"


class _LateralConnectivityBase(nn.Module):
    """Improved Superclass ensuring equal structure and init of future baselines"""

    def __init__(self, scopes: List[int], width: List[int], damp: List[float], strategy: str, optim: str,
                 self_connection: bool = False, pad: str = "circular"):
        super().__init__()
        assert strategy in ["CLC", "SSLC", "CLC-G", "SSLC-G"]
        assert optim in ["adaptive", "frozen", "parametric"]
        assert len(scopes) == len(width) == len(damp)
        assert pad in ["circular", "zeros"]
        self.strategy = strategy
        self.scopes = scopes
        self.width = width
        self.damp = damp
        self.coverage = len(scopes)
        self.freeze = optim == "frozen"
        self.optim = optim
        self.self_connection = self_connection
        self.pad = pad
        self.is_circular = pad == "circular"

    def forward(self, x):
        raise NotImplementedError
