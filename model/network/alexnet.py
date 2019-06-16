from torch import nn
#from torchvision.models import AlexNet

from model.inhibition_layer import Inhibition, RecurrentInhibition


class AlexNetInhibition(nn.Module):

    def __init__(self, inhibition_strategy: str = "once", learn_inhibition_weights=False, inhibition_depth=1,
                 num_classes=10):
        super().__init__()
        self.inhibition_strategy = inhibition_strategy

        assert self.inhibition_strategy in ["once", "recurrent"]

        counter = 1
        self.features = nn.Sequential()
        self.features.add_module("conv_1", nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2))
        if counter <= inhibition_depth:
            self.features.add_module("inhib_{}".format(counter),
                                     Inhibition(5, learn_weights=learn_inhibition_weights)
                                     if self.inhibition_strategy == "once"
                                     else RecurrentInhibition(5, learn_weights=learn_inhibition_weights))
            counter+=1
        self.features.add_module("relu_1",nn.ReLU(inplace=True))
        self.features.add_module("maxpool_1", nn.MaxPool2d(kernel_size=3, stride=2))
        self.features.add_module("conv_2", nn.Conv2d(96, 256, kernel_size=5, padding=2))
        if counter <= inhibition_depth:
            self.features.add_module("inhib_{}".format(counter),
                                     Inhibition(5, learn_weights=learn_inhibition_weights)
                                     if self.inhibition_strategy == "once"
                                     else RecurrentInhibition(5, learn_weights=learn_inhibition_weights))
            counter += 1
        self.features.add_module("relu_2", nn.ReLU(inplace=True))
        self.features.add_module("maxpool_2", nn.MaxPool2d(kernel_size=3, stride=2))
        self.features.add_module("conv_3", nn.Conv2d(256, 384, kernel_size=3, padding=1))
        if counter <= inhibition_depth:
            self.features.add_module("inhib_{}".format(counter),
                                     Inhibition(5, learn_weights=learn_inhibition_weights)
                                     if self.inhibition_strategy == "once"
                                     else RecurrentInhibition(5, learn_weights=learn_inhibition_weights))
            counter += 1
        self.features.add_module("relu_3", nn.ReLU(inplace=True))
        self.features.add_module("conv_4", nn.Conv2d(384, 384, kernel_size=3, padding=1))
        if counter <= inhibition_depth:
            self.features.add_module("inhib_{}".format(counter),
                                     Inhibition(5, learn_weights=learn_inhibition_weights)
                                     if self.inhibition_strategy == "once"
                                     else RecurrentInhibition(5, learn_weights=learn_inhibition_weights))
            counter += 1
        self.features.add_module("relu_4", nn.ReLU(inplace=True))
        self.features.add_module("conv_5", nn.Conv2d(384, 256, kernel_size=3, padding=1))
        if counter <= inhibition_depth:
            self.features.add_module("inhib_{}".format(counter),
                                     Inhibition(5, learn_weights=learn_inhibition_weights)
                                     if self.inhibition_strategy == "once"
                                     else RecurrentInhibition(5, learn_weights=learn_inhibition_weights))
            counter += 1
        self.features.add_module("relu_5", nn.ReLU(inplace=True))
        self.features.add_module("maxpool_3", nn.MaxPool2d(kernel_size=3, stride=2))

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)

        return x


class AlexNet(nn.Module):

    def __init__(self, num_classes=10):

        super().__init__()

        self.features = nn.Sequential()
        self.features.add_module("conv_1", nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2))
        self.features.add_module("relu_1", nn.ReLU(inplace=True))
        self.features.add_module("maxpool_1", nn.MaxPool2d(kernel_size=3))
        self.features.add_module("conv_2", nn.Conv2d(96, 256, kernel_size=5, padding=2))
        self.features.add_module("relu_2", nn.ReLU(inplace=True))
        self.features.add_module("maxpool_2", nn.MaxPool2d(kernel_size=3))
        self.features.add_module("conv_3", nn.Conv2d(256, 384, kernel_size=3, padding=1))
        self.features.add_module("relu_3", nn.ReLU(inplace=True))
        self.features.add_module("conv_4", nn.Conv2d(384, 384, kernel_size=3, padding=1))
        self.features.add_module("relu_4", nn.ReLU(inplace=True))
        self.features.add_module("conv_5", nn.Conv2d(384, 256, kernel_size=3, padding=1))
        self.features.add_module("relu_5", nn.ReLU(inplace=True))
        self.features.add_module("maxpool_3", nn.MaxPool2d(kernel_size=3))

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)

        return x


if __name__ == '__main__':
    net = AlexNetInhibition(inhibition_depth=2)
    alex = AlexNet()
    print(net.features)
    print(alex.features)
