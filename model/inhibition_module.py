from abc import ABC
from typing import List

from torch import Tensor


class InhibitionModule(ABC):

    def sort_filters_in_layer(self, layer: int = 0):
        """
        sorts the filters in a given layer according to the two_opt TSP alogrithm
        :param layer: the number of the layer
        :return: the sorted filters
        """
        filters = self.get_filters_from_layer(layer)
        from util.filter_ordering import two_opt
        sorted_filters: List[Tensor] = two_opt(filters)
        return sorted_filters

    def get_filters_from_layer(self, layer: int = 0):
        return self.features[layer].weight.data.numpy()

    def __repr__(self):
        params = []
        for p in ["scope", "width", "damp", "self_connection", "is_circular"]:
            if p in self.__dict__.keys():
                params.append(f"{p}={str(self.__dict__[p])}")
        return f"{self.__class__.__name__}({', '.join(params)})"
