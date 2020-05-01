from typing import List
from torch import nn

from model.inhibition_layer import ParametricInhibition, ConvergedInhibition, ConvergedFrozenInhibition, \
    SingleShotInhibition, SingleShotGaussian, ConvergedGaussian


class _BaseNetwork:
    """Superclass for uniform representation string for easier logging and debugging"""

    def __repr__(self):
        ret = ""
        for p in ["strategy", "optim", "coverage", "scopes", "widths", "damps", "freeze", "self_connection", "pad"]:
            if p in self.__dict__.keys():
                ret += str(self.__dict__[p])
            ret += ","
        return f"{self.__class__.__name__},{ret[:-1]}"


class _LateralConnectivityBase(_BaseNetwork, nn.Module):
    """Improved Superclass ensuring equal structure and init of future baselines"""

    def __init__(self, scopes: List[int], widths: List[int], damps: List[float], strategy: str, optim: str,
                 self_connection: bool = False, pad: str = "circular"):
        super().__init__()
        assert strategy in ["CLC", "SSLC", "CLC-G", "SSLC-G"]
        assert optim in ["adaptive", "frozen", "parametric"]
        assert len(scopes) == len(widths) == len(damps)
        assert pad in ["circular", "zeros"]
        self.strategy = strategy
        self.scopes = scopes
        self.widths = widths
        self.damps = damps
        self.coverage = len(scopes)
        self.freeze = optim == "frozen"
        self.optim = optim
        self.self_connection = self_connection
        self.pad = pad
        self.is_circular = pad == "circular"

        self.logger = None

    def lateral_connect_layer_type(self, num_layer: int = 1, in_channels=None):
        """
        returns an LC layer determined by strategy and optim,
        CLC-G and SSLC-G do not care about optim, they will always be frozen,
        SSLC has no parametric optim

        :param num_layer:       the number of the LC layer, starting at 1
        :param in_channels:     obligatory for frozen optimisations

        :return:                the LC layer
        """
        idx = num_layer - 1
        if self.strategy == "CLC":
            if self.optim == "adaptive":
                return ConvergedInhibition(scope=self.scopes[idx],
                                           ricker_width=self.widths[idx],
                                           damp=self.damps[idx])
            elif self.optim == "frozen":
                assert in_channels is not None, "in_channels is required for frozen optimisation"
                return ConvergedFrozenInhibition(scope=self.scopes[idx],
                                                 ricker_width=self.widths[idx],
                                                 damp=self.damps[idx],
                                                 in_channels=in_channels)
            elif self.optim == "parametric":
                return ParametricInhibition(scope=self.scopes[idx],
                                            initial_ricker_width=self.widths[idx],
                                            initial_damp=self.damps[idx],
                                            in_channels=in_channels)
        elif self.strategy == "SSLC":
            if self.optim == "adaptive":
                return SingleShotInhibition(scope=self.scopes[idx],
                                            ricker_width=self.widths[idx],
                                            damp=self.damps[idx],
                                            learn_weights=True)
            elif self.optim == "frozen":
                return SingleShotInhibition(scope=self.scopes[idx],
                                            ricker_width=self.widths[idx],
                                            damp=self.damps[idx])
        elif self.strategy == "CLC-G":
            assert in_channels is not None, "in_channels is required for frozen optimisation"
            return ConvergedGaussian(scope=self.scopes[idx],
                                     ricker_width=self.widths[idx],
                                     damp=self.damps[idx],
                                     in_channels=in_channels)
        elif self.strategy == "SSLC-G":
            return SingleShotGaussian(scope=self.scopes[idx],
                                      width=self.widths[idx],
                                      damp=self.damps[idx])

        raise AttributeError("lateral connectivity type not supported")

    def forward(self, x):
        raise NotImplementedError


if __name__ == '__main__':
    tests = [_LateralConnectivityBase(scopes=[27], widths=[3], damps=[0.1], strategy="CLC", optim="frozen"),
             _LateralConnectivityBase(scopes=[27], widths=[3], damps=[0.1], strategy="CLC", optim="adaptive"),
             _LateralConnectivityBase(scopes=[27], widths=[3], damps=[0.1], strategy="CLC", optim="parametric"),
             _LateralConnectivityBase(scopes=[27], widths=[3], damps=[0.1], strategy="SSLC", optim="frozen"),
             _LateralConnectivityBase(scopes=[27], widths=[3], damps=[0.1], strategy="SSLC", optim="adaptive"),
             _LateralConnectivityBase(scopes=[27], widths=[3], damps=[0.1], strategy="CLC-G", optim="frozen"),
             _LateralConnectivityBase(scopes=[27], widths=[3], damps=[0.1], strategy="SSLC-G", optim="frozen")]

    for model in tests:
        # in channels only used for frozen, but passed for test anyway
        print(model, f"\t", model.lateral_connect_layer_type(in_channels=64))
