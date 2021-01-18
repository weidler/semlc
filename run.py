"""A script for all main experiments that allows running multiple experiments of the same strategy."""

import json

import torch
import torchvision
from torch import nn
from torch.utils.data import Subset
from torchvision import transforms

from model.network.VGG import vgg19, vgg19_inhib
from model.network.alexnet_cifar import BaselineCMap, Baseline, AlexNetLC
from util.eval import accuracy
from util.ourlogging import Logger
from util.train import train

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

with open('default_config.json') as f:
    CONFIG = json.load(f).get('CONFIG', {})


def get_hp_params(args):
    if args.hpopt:
        with open('hp_params.json') as hp:
            conf = json.load(hp).get(args.hpopt)

        CONFIG[args.strategy][args.optim]['widths'] = conf['widths']
        CONFIG[args.strategy][args.optim]['damps'] = conf['damps']


def get_config(args):
    # check for HP optmisation
    get_hp_params(args)

    # optionally overwrite config
    if args.widths:
        widths = [int(x) for x in args.widths.split(',')]
        assert len(widths) == args.coverage, \
            f"number of widths ({len(widths)}) does not match coverage {args.coverage}"
        CONFIG[args.strategy][args.optim]['widths'] = widths
    if args.damps:
        damps = [float(x) for x in args.damps.split(',')]
        assert len(damps) == args.coverage, \
            f"number of damps ({len(damps)}) does not match coverage {args.coverage}"
        CONFIG[args.strategy][args.optim]['damps'] = damps
    else:
        assert args.coverage == len(CONFIG[args.strategy][args.optim]['widths']) == len(
            CONFIG[args.strategy][args.optim]['damps']), \
            f"coverage ({args.coverage}) does not match configuration. Please check coverage param " \
            f"and change or overwrite config with arguments -s, -w, -d."

    return CONFIG


def get_params(args, param):
    assert param in ['widths', 'damps'], f"invalid param {param}"
    parameter = CONFIG[args.strategy][args.optim][param]

    return parameter


def build_network(strategy, optim, args):
    if strategy == "baseline":
        network = Baseline()
    elif strategy == "cmap":
        network = BaselineCMap()
    elif strategy == "vgg19":
        network = vgg19()
    elif strategy == "vgg19_inhib":
        network = vgg19_inhib()
    elif strategy == "vgg19_inhib_self":
        network = vgg19_inhib(self_connection=True)
    else:
        get_config(args)
        widths = get_params(args, 'widths')
        damps = get_params(args, 'damps')
        print('CONFIG:', widths, damps)

        network = AlexNetLC(widths, damps, strategy=strategy, optim=optim)

    return network


def run(args):
    strategy = args.strategy
    optim = args.optim
    iterations = args.i

    crop = 32 if "vgg" in strategy else 24
    padding = 4 if "vgg" in strategy else None

    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    print(f"USE CUDA: {use_cuda}.")

    # transformation
    transform = transforms.Compose([transforms.RandomCrop(crop, padding),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # load data
    trainval_set = torchvision.datasets.CIFAR10("./data/cifar10/", train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10("./data/cifar10/", train=False, download=True, transform=transform)

    for i in range(0, iterations):
        val_indices = list(range(int((i % 10) * len(trainval_set) / 10), int(((i % 10) + 1) * len(trainval_set) / 10)))
        train_indices = list(filter(lambda x: x not in val_indices, list(range(len(trainval_set)))))
        val_set = Subset(trainval_set, indices=val_indices)
        train_set = Subset(trainval_set, indices=train_indices)

        network = build_network(strategy, optim, args)

        print(network)
        print(network.features)

        if use_cuda:
            network.cuda()

        experiment_code = f"{strategy}_{optim}_hp_{args.hpopt}" if args.hpopt else f"{strategy}_{optim}_{i}"
        logger = Logger(network, experiment_code=experiment_code)

        train(net=network,
              num_epoch=180,
              train_set=train_set,
              batch_size=128,
              criterion=nn.CrossEntropyLoss(),
              logger=logger,
              val_set=val_set,
              learn_rate=0.001,
              verbose=False)

        network.eval()
        logger.log(f"\nFinal Test Accuracy: {accuracy(network, test_set, 128)}")


if __name__ == '__main__':
    import argparse

    old_strategies = ["baseline", "cmap", "vgg19", "vgg19_inhib", "vgg19_inhib_self"]
    strategies = ["CLC", "SSLC", "CLC-G", "SSLC-G"] + old_strategies
    optims = ["adaptive", "frozen", "parametric"]

    parser = argparse.ArgumentParser(usage='\nEXAMPLE: \n$ run.py CLC frozen\n\noptionally do HP optimisation '
                                           'using hp_params.json \n(index 23 in this example)\n'
                                           '$ run.py CLC frozen -p 23\n\noptionally overwrite default params\n'
                                           '$ run.py CLC frozen -c 3 -s 1,3,5 -w 2,3,4 -d 0.5,0.2,0.3\n')
    parser.add_argument("strategy", type=str, choices=strategies)
    parser.add_argument("optim", type=str, choices=optims)
    parser.add_argument("-w", "--widths", dest="widths", type=str, help="overwrite default widths")
    parser.add_argument("-d", "--damps", dest="damps", type=str, help="overwrite default damps")
    parser.add_argument("-c", "--cov", dest="coverage", type=int, help="coverage, default=1", default=1)
    parser.add_argument("-i", type=int, default=1, help="the number of iterations, default=1")
    parser.add_argument("-p", "--hpopt", type=str, help="hp optimisation with given index")
    args = parser.parse_args()

    run(args)
