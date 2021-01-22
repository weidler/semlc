from typing import List
from torch import nn

from layers.semantic_layers import ParametricSemLC, ConvergedSemLC, ConvergedFrozenSemLC, \
    SingleShotSemLC, ConvergedGaussianSemLC


class BaseNetwork(nn.Module):
    """Improved Superclass ensuring equal structure and init of future baselines"""

    def __repr__(self):
        ret = ""
        for p in ["strategy", "optim", "coverage", "scopes", "widths", "damps", "freeze", "self_connection", "pad"]:
            if p in self.__dict__.keys():
                ret += str(self.__dict__[p])
            ret += ","
        return f"{self.__class__.__name__},{ret[:-1]}"

    def __init__(self, widths: List[int] = None, damps: List[float] = None, strategy: str = None, optim: str = None,
                 self_connection: bool = False, pad: str = "circular"):
        super().__init__()

        self.strategy = strategy
        self.widths = widths
        self.damps = damps
        self.freeze = optim == "frozen"
        self.optim = optim
        self.self_connection = self_connection
        self.pad = pad
        self.is_circular = pad == "circular"

        self.logger = None

        self.is_lc = False

        if all(v is None for v in [widths, damps, strategy, optim]):
            pass
        elif None in [widths, damps, strategy, optim] and not all(v is None for v in [widths, damps, strategy, optim]):
            raise ValueError(f"Provided incomplete information to build LC Network.")
        else:
            assert strategy in ["CLC", "SSLC", "CLC-G"]
            assert optim in ["adaptive", "frozen", "parametric"]
            assert pad in ["circular", "zeros"]

            self.coverage = len(widths)

            self.is_lc = True

    def lateral_connect_layer_type(self, num_layer: int = 1, in_channels=None):
        """
        returns an LC layers determined by strategy and optim,
        CLC-G and SSLC-G do not care about optim, they will always be frozen,
        SSLC has no parametric optim

        :param num_layer:       the number of the LC layers, starting at 1
        :param in_channels:     obligatory for frozen optimisations

        :return:                the LC layers
        """
        if not self.is_lc:
            raise AttributeError("Network does not allow LC layers.")

        idx = num_layer - 1
        if self.strategy == "CLC":
            if self.optim == "adaptive":
                return ConvergedSemLC(in_channels=in_channels, ricker_width=self.widths[idx], damp=self.damps[idx])
            elif self.optim == "frozen":
                assert in_channels is not None, "in_channels is required for frozen optimisation"
                return ConvergedFrozenSemLC(in_channels=in_channels, ricker_width=self.widths[idx],
                                            damp=self.damps[idx])
            elif self.optim == "parametric":
                return ParametricSemLC(in_channels=in_channels, ricker_width=self.widths[idx],
                                       initial_damp=self.damps[idx])
        elif self.strategy == "SSLC":
            if self.optim == "adaptive":
                return SingleShotSemLC(in_channels=in_channels, ricker_width=self.widths[idx], damp=self.damps[idx], learn_weights=True)
            elif self.optim == "frozen":
                return SingleShotSemLC(in_channels=in_channels, ricker_width=self.widths[idx], damp=self.damps[idx])
        elif self.strategy == "CLC-G":
            assert in_channels is not None, "in_channels is required for frozen optimisation"
            return ConvergedGaussianSemLC(in_channels=in_channels, ricker_width=self.widths[idx], damp=self.damps[idx])

        raise AttributeError("lateral connectivity type not supported")

    def forward(self, x):
        raise NotImplementedError

    def serialize_meta(self):
        return {
            "network_type": self.__class__.__name__,
            "input_channels": self.input_channels,
            "input_width": self.input_width,
            "input_height": self.input_height,
            "is_lateral": self.is_lateral,
            "lateral_type": self.lateral_type,
            "complex_cells": self.is_complex
        }


if __name__ == '__main__':
    tests = [BaseNetwork(widths=[3], damps=[0.1], strategy="CLC", optim="frozen"),
             BaseNetwork(widths=[3], damps=[0.1], strategy="CLC", optim="adaptive"),
             BaseNetwork(widths=[3], damps=[0.1], strategy="CLC", optim="parametric"),
             BaseNetwork(widths=[3], damps=[0.1], strategy="SSLC", optim="frozen"),
             BaseNetwork(widths=[3], damps=[0.1], strategy="SSLC", optim="adaptive"),
             BaseNetwork(widths=[3], damps=[0.1], strategy="CLC-G", optim="frozen"),
             BaseNetwork(widths=[3], damps=[0.1], strategy="SSLC-G", optim="frozen"),
             BaseNetwork()]

    for model in tests:
        # in channels only used for frozen, but passed for test anyway
        print(model, f"\t", model.lateral_connect_layer_type(in_channels=64))
