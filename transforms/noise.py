import torch

import numpy


class GaussianNoise:

    def __call__(self, pic: torch.Tensor):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to have noise applied to.
        Returns:
            Tensor: Image with noise.
        """

        return pic.apply_(self._noise)

    def _noise(self, x):
        return max(0, min(1, x + numpy.random.normal(0, 0.03)))

    def __repr__(self):
        return self.__class__.__name__ + '()'
