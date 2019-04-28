from abc import ABC, abstractmethod


class InhibitionModule(ABC):

    @abstractmethod
    def get_layers_for_visualization(self):
        pass


