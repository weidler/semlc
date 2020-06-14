from typing import List
from torch import nn

from model.inhibition_layer import ParametricInhibition, ConvergedInhibition, ConvergedFrozenInhibition, \
    SingleShotInhibition, SingleShotGaussian, ConvergedGaussian


class BaseNetwork(nn.Module):
    """Improved Superclass ensuring equal structure and init of future baselines"""

    def __repr__(self):
        ret = ""
        for p in ["strategy", "optim", "coverage", "scopes", "widths", "damps", "freeze", "self_connection", "pad"]:
            if p in self.__dict__.keys():
                ret += str(self.__dict__[p])
            ret += ","
        return f"{self.__class__.__name__},{ret[:-1]}"

    def __init__(self, scopes: List[int] = None, widths: List[int] = None, damps: List[float] = None, strategy: str = None,
                 optim: str = None, self_connection: bool = False, pad: str = "circular"):
        super().__init__()

        self.strategy = strategy
        self.scopes = scopes
        self.widths = widths
        self.damps = damps
        self.freeze = optim == "frozen"
        self.optim = optim
        self.self_connection = self_connection
        self.pad = pad
        self.is_circular = pad == "circular"

        self.logger = None

        self.is_lc = False

        if all(v is None for v in [scopes, widths, damps, strategy, optim]):
            pass
        elif None in [scopes, widths, damps, strategy, optim] and not all(v is None for v in [scopes, widths, damps, strategy, optim]):
            raise ValueError(f"Provided incomplete information to build LC Network.")
        else:
            assert strategy in ["CLC", "SSLC", "CLC-G", "SSLC-G"]
            assert optim in ["adaptive", "frozen", "parametric"]
            assert len(scopes) == len(widths) == len(damps)
            assert pad in ["circular", "zeros"]

            self.coverage = len(scopes)

            self.is_lc = True

    def lateral_connect_layer_type(self, num_layer: int = 1, in_channels=None):
        """
        returns an LC layer determined by strategy and optim,
        CLC-G and SSLC-G do not care about optim, they will always be frozen,
        SSLC has no parametric optim

        :param num_layer:       the number of the LC layer, starting at 1
        :param in_channels:     obligatory for frozen optimisations

        :return:                the LC layer
        """
        if not self.is_lc:
            raise AttributeError("Network does not allow LC layers.")

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
    tests = [BaseNetwork(scopes=[27], widths=[3], damps=[0.1], strategy="CLC", optim="frozen"),
             BaseNetwork(scopes=[27], widths=[3], damps=[0.1], strategy="CLC", optim="adaptive"),
             BaseNetwork(scopes=[27], widths=[3], damps=[0.1], strategy="CLC", optim="parametric"),
             BaseNetwork(scopes=[27], widths=[3], damps=[0.1], strategy="SSLC", optim="frozen"),
             BaseNetwork(scopes=[27], widths=[3], damps=[0.1], strategy="SSLC", optim="adaptive"),
             BaseNetwork(scopes=[27], widths=[3], damps=[0.1], strategy="CLC-G", optim="frozen"),
             BaseNetwork(scopes=[27], widths=[3], damps=[0.1], strategy="SSLC-G", optim="frozen"),
             BaseNetwork()]

    for model in tests:
        # in channels only used for frozen, but passed for test anyway
        print(model, f"\t", model.lateral_connect_layer_type(in_channels=64))
