from typing import Tuple, Callable

from torchsummary import torchsummary

from layers.util import prepare_lc_builder
from networks import CapsNet, BaseNetwork, Shallow, CORnetS, CORnetZ, AlexNet, Simple

AVAILABLE_NETWORKS = ["simple", "shallow", "alexnet", "capsnet", "cornet-s", "cornet-z"]


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
    elif network in ["cornet-s", CORnetS.__name__.lower()]:
        model = CORnetS(input_shape=input_shape, n_classes=n_classes, lateral_layer_function=lc)
    elif network in ["cornet-z", CORnetZ.__name__.lower()]:
        model = CORnetZ(input_shape=input_shape, n_classes=n_classes, lateral_layer_function=lc)
    else:
        raise NotImplementedError(f"Requested network '{network}' does not exist.")

    return model


if __name__ == '__main__':
    lateral_connectivity_function = prepare_lc_builder("semlc", (3, 5), 2, 0.2)

    sh_net = build_network("shallow", input_shape=(3, 32, 32), n_classes=10, lc=lateral_connectivity_function)
    sb_net = build_network("simple", input_shape=(3, 32, 32), n_classes=10, lc=lateral_connectivity_function)
    alex_net = build_network("alexnet", input_shape=(3, 32, 32), n_classes=10, lc=lateral_connectivity_function)

    torchsummary.summary(sh_net, sh_net.input_shape, device="cpu")
    print("\n\n")
    torchsummary.summary(sb_net, sb_net.input_shape, device="cpu")
    print("\n\n")
    torchsummary.summary(alex_net, alex_net.input_shape, device="cpu")
