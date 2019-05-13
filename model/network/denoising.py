import torch

from torch import nn


# DENOISING
from model.inhibition_layer import Inhibition
from model.network.classification import BaseClassificationCNN


class ConvolutionalEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=5),
            nn.ReLU(True)
        )

    def forward(self, sample):
        return self.conv(sample)


class ConvolutionalDecoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, kernel_size=5),
            nn.Sigmoid()
        )

    def forward(self, sample):
        out = self.conv(sample)

        return out


class ConvolutionalEncoderWithInhibition(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5),
            Inhibition(7),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=5),
            nn.ReLU(True)
        )

    def forward(self, sample):
        out = self.conv(sample)

        return out


class BaseDenoisingCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = ConvolutionalEncoder()
        self.decoder = ConvolutionalDecoder()

    def forward(self, image_tensor):
        encoding = self.encoder(image_tensor)
        decoding = self.decoder(encoding)

        return decoding


class InhibitionDenoisingCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = ConvolutionalEncoderWithInhibition()
        self.decoder = ConvolutionalDecoder()

    def forward(self, image_tensor):
        encoding = self.encoder(image_tensor)
        decoding = self.decoder(encoding)

        return decoding