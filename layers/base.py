from typing import Tuple, Any, List

from torch import nn, Tensor


class BaseSemLCLayer(nn.Module):

    def __init__(self, hooked_conv: nn.Conv2d, ricker_width: float, ricker_damp: float):
        super().__init__()

        self.hooked_conv = hooked_conv
        self.in_channels = self.hooked_conv.out_channels
        self.out_channels = self.hooked_conv.out_channels

        self.is_compiled = False
        self.input_height, self.input_width = None, None
        self.activations_shape = (None, None, None)

        self.ricker_width = ricker_width
        self.ricker_damp = ricker_damp

        self.gabor_filters = None

    def __repr__(self):
        return f"{self.__class__.__name__}[w={self.ricker_width}; d={self.ricker_damp}; s={self.in_channels}]"

    def compile(self, spatial_input_size: Tuple[int, int]):
        self.input_height, self.input_width = spatial_input_size[-2:]
        self.activations_shape = (self.in_channels, self.input_height, self.input_width)

        self.is_compiled = True

    def _forward_unimplemented(self, *input: Any) -> None:
        raise Exception("Not implemented.")

    def sort_filters_in_layer(self, layer: int = 0):
        """ Sorts the filters in a given layers according to the two_opt TSP algorithm.

        :param layer: the number of the layers

        :return: the sorted filters
        """
        filters = self.get_filters_from_layer(layer)
        from utilities.filter_ordering import two_opt
        sorted_filters: List[Tensor] = two_opt(filters)
        return sorted_filters

    def get_filters_from_layer(self, layer: int = 0):
        """ Returns the filters from the given layers
        :param layer:           the layers

        :return:                the tensor of filters
        """
        return self.features[layer].weight.data.numpy()
