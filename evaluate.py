"""Tests the accuracy on a given model or an average over a set of models"""

import argparse
import random

import numpy
import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from main import get_network
from util.eval import accuracies_from_list, accuracy_from_data_loader

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

keychain = "./output/keychain.txt"
model_path = "./output/"

df = pd.read_csv(keychain, sep="\t", names=['id', 'group', 'model', 'datetime'])

old_strategies = ["baseline", "cmap", "vgg19", "vgg19_inhib", "vgg19_inhib_self"]
strategies = ["CLC", "SSLC", "CLC-G", "SSLC-G"] + old_strategies
optims = ["adaptive", "frozen", "parametric"]

parser = argparse.ArgumentParser(usage='\nEXAMPLE: \n$ main.py CLC frozen\n\noptionally evaluate HP optimisation '
                                       'using hp_params.json (index 23 in this example)\n'
                                       '$ main.py CLC frozen -p 23\n\nor all 50 HP opts at once (sequentially)\n'
                                       '$ main.py CLC frozen -pa 50\n\noptionally overwrite default params\n'
                                       '$ main.py CLC frozen -c 3 -s 1,3,5 -w 2,3,4 -d 0.5,0.2,0.3\n')
parser.add_argument("strategy", type=str, choices=strategies)
parser.add_argument("optim", type=str, choices=optims)
parser.add_argument("-s", "--scopes", dest="scopes", type=str, help="overwrite default scopes")
parser.add_argument("-w", "--widths", dest="widths", type=str, help="overwrite default widths")
parser.add_argument("-d", "--damps", dest="damps", type=str, help="overwrite default damps")
parser.add_argument("-c", "--cov", dest="coverage", type=int, help="coverage, default=1", default=1)
parser.add_argument("-p", "--hpopt", type=str, help="hp optimisation with given index")
parser.add_argument("-pa", "--hpoptall", type=str, help="all (given) n hp opts")
args = parser.parse_args()

strategy = args.strategy
optim = args.optim


def evaluate_exp(filenames, strategy, optim, args):
    # print(filenames)
    print(f"loading {len(filenames)} models for {strategy} {optim}...")

    accuracies = []
    for i, row in tqdm(enumerate(filenames), disable=True):
        filename = f"{model_path}{row}_best.model"
        network = get_network(strategy, optim, args)
        network.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
        data_loader = DataLoader(test_set, batch_size=128,
                                 shuffle=False, num_workers=0)
        acc = accuracy_from_data_loader(network, data_loader)
        print(acc)
        accuracies.append(acc)
    print(f"\nLoaded {len(filenames)} files for strategy {strategy} {optim}.")
    if len(filenames):
        print(accuracies)
        print(f"{strategy}{'[wA]' if random_transform_test else ''}: {accuracies_from_list(accuracies)}")


# test on both the augmented and non-augmented test set
for random_transform_test in [False, True]:
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

    test_set = torchvision.datasets.CIFAR10("./data/cifar10/", train=False, download=True, transform=transform)

    if args.hpoptall:
        for i in range(int(args.hpoptall)):
            filenames = df[df['group'].str.match(rf'{strategy}_{optim}_hp_{i}')]['id']
            args.hpopt = f"{i}"
            evaluate_exp(filenames, strategy, optim, args)
    elif args.hpopt:
        filenames = df[df['group'].str.match(rf'{strategy}_{optim}_hp_{args.hpopt}')]['id']
        evaluate_exp(filenames, strategy, optim, args)
    else:
        filenames = df[df['group'].str.match(rf'{strategy}_{optim}_\d\d?')]['id']
        evaluate_exp(filenames, strategy, optim, args)
