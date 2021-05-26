from typing import Tuple, Any, List

from torch import nn, Tensor


class BaseSemLCLayer(nn.Module):

    def __init__(self, hooked_conv: nn.Conv2d, widths: Tuple[float, float], ratio: float, damping: float, rings: int = 1):
        super().__init__()

        self.hooked_conv = hooked_conv
        self.in_channels = self.hooked_conv.out_channels
        self.out_channels = self.hooked_conv.out_channels

        # rings
        self.n_rings = rings
        self.ring_size = self.in_channels / self.n_rings
        assert self.ring_size.is_integer(), "The number of channels must be a multiple of the number of rings."
        self.ring_size = int(self.ring_size)  # convert to integer to use as index

        # allocate rings
        self.rings = [
            self.hooked_conv.weight[i * self.ring_size:(i + 1) * self.ring_size, ...] for i in range(self.n_rings)
        ]

        print([r.shape for r in self.rings])

        self.is_compiled = False
        self.input_height, self.input_width = None, None
        self.activations_shape = (None, None, None)

        self.widths = widths
        self.damping = damping
        self.ratio = ratio

        self.gabor_filters = None

    def __repr__(self):
        return f"{self.__class__.__name__}[w={self.widths}; r={self.ratio}; d={self.damping}; s={self.in_channels}]"

    @property
    def name(self):
        return self.__class__.__name__

    def compile(self, spatial_input_size: Tuple[int, int]):
        self.input_height, self.input_width = spatial_input_size[-2:]
        self.activations_shape = (self.in_channels, self.input_height, self.input_width)

        self.is_compiled = True

    def _forward_unimplemented(self, *input: Any) -> None:
        raise Exception("Not implemented.")

    def sort_filters_in_layer(self, layer: int = 0):
        """Sorts the filters_per_group in a given layers according to the two_opt TSP algorithm.

        :param layer: the number of the layers

        :return: the sorted filters_per_group
        """
        filters = self.get_filters_from_layer(layer)
        from utilities.filter_ordering import two_opt
        sorted_filters: List[Tensor] = two_opt(filters)
        return sorted_filters

    def get_filters_from_layer(self, layer: int = 0):
        """ Returns the filters_per_group from the given layers
        :param layer:           the layers

        :return:                the tensor of filters_per_group
        """
        return self.features[layer].weight.data.numpy()
