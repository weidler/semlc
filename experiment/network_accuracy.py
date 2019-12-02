import random

import numpy
import torch
import torchvision
from torchvision import transforms
import pandas as pd
from tqdm import tqdm

from model.network.alexnet_cifar import SingleShotInhibitionNetwork, BaselineCMap, Baseline, ConvergedInhibitionNetwork, \
    ParametricInhibitionNetwork
from util.eval import accuracy_with_confidence

# Random seeding is very important, since without the random cropping may be different
torch.manual_seed(12311)
numpy.random.seed(12311)
random.seed(12311)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

use_cuda = False
if torch.cuda.is_available():
    use_cuda = True
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

print(f"USE CUDA: {use_cuda}.")

keychain = "../output/keychain.txt"
model_path = "../output/"

strategies = ["baseline", "cmap", "ss", "ss_freeze", "converged", "converged_self", "converged_freeze",
              "converged_freeze_self", "parametric", "parametric_self", "vgg19",
              "vgg19_inhib", "vgg19_inhib_self"]

df = pd.read_csv(keychain, sep="\t", names=['id', 'group', 'model', 'datetime'])

# SET UP NETS AND SETTINGS

num_nets = 30

all_nets = {
    'converged_freeze': [ConvergedInhibitionNetwork([45], 3, 0.2, freeze=True) for i in range(1, num_nets + 1)],
    'converged': [ConvergedInhibitionNetwork([27], 3, 0.1, freeze=False) for i in range(1, num_nets + 1)],
    'ss': [SingleShotInhibitionNetwork([63], 8, 0.2, freeze=False) for i in range(1, num_nets + 1)],
    'ss_freeze': [SingleShotInhibitionNetwork([27], 3, 0.1, freeze=True) for i in range(1, num_nets + 1)],
    'parametric': [ParametricInhibitionNetwork([45], 3, 0.2) for i in range(1, num_nets + 1)],
    'baseline': [Baseline() for i in range(1, num_nets + 1)]
}

random_transform_test = True

# LOAD TEST DATA
if random_transform_test:
    transform = transforms.Compose([transforms.RandomCrop(24),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
else:
    transform = transforms.Compose([transforms.CenterCrop(24),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

test_set = torchvision.datasets.CIFAR10("../data/cifar10/", train=False, download=True, transform=transform)


# EVALUATE

for strategy in strategies:
    if strategy in all_nets.keys():
        filenames = df[df['group'].str.match(rf'{strategy}_\d\d?')]['id']
        for i, row in tqdm(enumerate(filenames), disable=True):
            filename = f"{model_path}{row}_best.model"
            print(f"Loading {filename}")
            all_nets[strategy][i].load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))

        print(strategy, accuracy_with_confidence(all_nets[strategy], test_set, 128, 0.95))


# CENTER CROPPING
# converged_freeze  (83.42433333333334, 0.1323670165061982, (83.29196631682714, 83.55670034983953))
# converged         (83.364, 0.12344272430805966, (83.24055727569194, 83.48744272430807))
# ss_freeze         (83.29899999999999, 0.13855937444158548, (83.16044062555841, 83.43755937444158))
# ss                (83.22, 0.13332364466227511, (83.08667635533773, 83.35332364466227))
# parametric        (83.31866666666667, 0.13095761588471813, (83.18770905078195, 83.4496242825514))
# baseline          (83.33766666666666, 0.10752667459086476, (83.2301399920758, 83.44519334125752))
