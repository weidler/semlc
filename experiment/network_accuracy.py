import torch

import torchvision
from torchvision import transforms

from util.eval import accuracy_with_confidence, accuracy
from model.network.alexnet_paper import InhibitionNetwork

use_cuda = False
if torch.cuda.is_available():
    use_cuda = True
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

print(f"USE CUDA: {use_cuda}.")

transform = transforms.Compose([transforms.RandomCrop(24),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

test_set = torchvision.datasets.CIFAR10("../data/cifar10/", train=False, download=True, transform=transform)

nets = [InhibitionNetwork(scope=[63],
                          width=6,
                          damp=0.12,
                          inhibition_depth=0,
                          inhibition_strategy="once",
                          logdir="test") for i in range(1, 11)]

for i, net in enumerate(nets, 1):
    net.load_state_dict(torch.load(f"../saved_models/baseline/baseline_{i}/ConvNet11_179.model"))

print(accuracy(nets[1], test_set, 1))
print(accuracy_with_confidence(nets, test_set, 128, 0.95))

