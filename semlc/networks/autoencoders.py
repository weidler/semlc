import os

import torch
import torchvision
from torch import nn, optim
from torch.autograd import Variable
from torchvision import transforms

from networks import BaseNetwork
from networks.util import prepare_lc_builder
from utilities import show_image


class SimpleAutoEncoder(BaseNetwork):
    def __init__(self, input_shape, lateral_layer_function):
        super(SimpleAutoEncoder, self).__init__(input_shape, lateral_layer_function)
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.conv_one = nn.Conv2d(3, 12, 4, stride=2, padding=1)
        conv_one_out_size = self.conv_one(self.generate_random_input()).shape

        if self.is_lateral:
            self.lateral_layer = self.lateral_layer_function(self.conv_one)
            self.lateral_layer.compile(conv_one_out_size[-2:])

        self.encoder = nn.Sequential(
            self.conv_one,  # [batch, 12, 16, 16]
            *((self.lateral_layer,) if self.is_lateral else ()),
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.Conv2d(24, 32, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.Conv2d(32, 32, 2, stride=2, padding=1),  # [batch, 32, 2, 2]
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 2, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),  # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class ShallowAutoEncoder(SimpleAutoEncoder):
    def __init__(self, input_shape, lateral_layer_function):
        super(ShallowAutoEncoder, self).__init__(input_shape, lateral_layer_function)
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.conv_one = nn.Conv2d(3, 64, 4, stride=2, padding=1)
        conv_one_out_size = self.conv_one(self.generate_random_input()).shape

        if self.is_lateral:
            self.lateral_layer = self.lateral_layer_function(self.conv_one)
            self.lateral_layer.compile(conv_one_out_size[-2:])

        self.encoder = nn.Sequential(
            self.conv_one,  # [batch, 12, 16, 16]
            *((self.lateral_layer,) if self.is_lateral else ()),
            nn.ReLU(),
            nn.Conv2d(64, 64, 2, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),  # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )


if __name__ == '__main__':
    def get_torch_vars(x):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)

    lc = prepare_lc_builder("semlc", 0.5, 0.2)
    ae = SimpleAutoEncoder(input_shape=(3, 32, 32), lateral_layer_function=lc)
    ae.to("cuda")

    # Load data
    transform = transforms.Compose([transforms.ToTensor(), ])
    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Define an optimizer and criterion
    criterion = nn.MSELoss()
    optimizer = optim.Adam(ae.parameters())

    for epoch in range(500):
        running_loss = 0.0
        for i, (inputs, _) in enumerate(trainloader, 0):
            inputs = get_torch_vars(inputs)

            encoded, outputs = ae(inputs)
            loss = criterion(outputs, inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.data
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.6f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print("\nGaude! Consummatum est.")
    print('Saving Model...')
    if not os.path.exists('weights'):
        os.mkdir('weights')
    torch.save(ae.state_dict(), "weights/ae.pkl")

    print("Inspecting...")
    iterator = iter(testloader)
    images, labels = iterator.next()
    show_image(torchvision.utils.make_grid(images))


    images = Variable(images.cuda())

    decoded_imgs = ae(images)[1]
    show_image(torchvision.utils.make_grid(decoded_imgs.data))