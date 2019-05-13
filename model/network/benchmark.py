import torch
import torch.nn.functional as F

from torch import nn


# DENOISING


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


class BaseDenoisingCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = ConvolutionalEncoder()
        self.decoder = ConvolutionalDecoder()

    def forward(self, image_tensor):
        encoding = self.encoder(image_tensor)
        decoding = self.decoder(encoding)

        return decoding


# CLASSIFICATION


class BaseClassificationCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.classifier(x)

        return x


if __name__ == "__main__":
    net = BaseClassificationCNN()
    net(torch.ones([1, 3, 32, 32]))
