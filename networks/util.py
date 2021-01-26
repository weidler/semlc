import functools
from typing import Tuple, Callable, Union

from torch import nn
from torchsummary import torchsummary

from layers.semantic_layers import SemLC, AdaptiveSemLC, ParametricSemLC, SingleShotSemLC, GaussianSemLC, LRN, CMapLRN
from networks import CapsNet, BaseNetwork, Shallow
from networks import Simple
from networks.alexnet import AlexNet

AVAILABLE_NETWORKS = ["simple", "shallow", "alexnet", "capsnet"]


def prepare_lc_builder(setting: str, ricker_width: float, ricker_damp: float) -> Union[None, Callable[..., BaseNetwork]]:
    """Return a partial function of the semantic lateral connectivity layer requested by name."""

    if setting is None or setting == "none":
        return None

    setting = setting.strip().lower()

    # SEMLC
    if setting in ["semlc"]:
        return functools.partial(SemLC, ricker_width=ricker_width, ricker_damp=ricker_damp)
    elif setting in ["adaptive-semlc"]:
        return functools.partial(AdaptiveSemLC, ricker_width=ricker_width, ricker_damp=ricker_damp)
    elif setting in ["parametric-semlc"]:
        return functools.partial(ParametricSemLC, ricker_width=ricker_width, ricker_damp=ricker_damp)
    elif setting in ["singleshot-semlc"]:
        return functools.partial(SingleShotSemLC, ricker_width=ricker_width, ricker_damp=ricker_damp)

    # COMPETITORS
    elif setting in ["lrn"]:
        return functools.partial(LRN)
    elif setting in ["cmap-lrn"]:
        return functools.partial(CMapLRN)
    elif setting in ["gaussian-semlc"]:
        return functools.partial(GaussianSemLC, ricker_width=ricker_width, ricker_damp=ricker_damp)
    else:
        raise NotImplementedError("LC layer construction for given layer settings not implemented.")


def build_network(network: str, input_shape: Tuple[int, int, int], n_classes: int = None, lc: Callable = None,
                  init_std: float = None) -> BaseNetwork:
    network = network.lower()

    assert n_classes is not None, "The number of classes in the classification must be declared."

    if network in ["simple", Simple.__name__.lower()]:
        model = Simple(input_shape=input_shape, n_classes=n_classes, lateral_layer_function=lc,
                       **(dict(conv_one_init_std=init_std) if init_std is not None else dict()))
    elif network in ["shallow", Shallow.__name__.lower()]:
        model = Shallow(input_shape=input_shape, n_classes=n_classes, lateral_layer_function=lc)
    elif network in ["alexnet", AlexNet.__name__.lower()]:
        model = AlexNet(input_shape=input_shape, n_classes=n_classes, lateral_layer_function=lc)
    elif network in ["capsnet", CapsNet.__name__.lower()]:
        model = CapsNet(input_shape=input_shape, n_classes=n_classes, lateral_layer_function=lc)
    else:
        raise NotImplementedError(f"Requested network '{network}' does not exist.")

    return model


if __name__ == '__main__':
    lateral_connectivity_function = prepare_lc_builder("semlc", 3, 0.2)

    sh_net = build_network("shallow", input_shape=(3, 32, 32), n_classes=10, lc=lateral_connectivity_function)
    sb_net = build_network("simple", input_shape=(3, 32, 32), n_classes=10, lc=lateral_connectivity_function)
    alex_net = build_network("alexnet", input_shape=(3, 32, 32), n_classes=10, lc=lateral_connectivity_function)

    torchsummary.summary(sh_net, sh_net.input_shape, device="cpu")
    print("\n\n")
    torchsummary.summary(sb_net, sb_net.input_shape, device="cpu")
    print("\n\n")
    torchsummary.summary(alex_net, alex_net.input_shape, device="cpu")
