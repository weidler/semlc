"""A script for all main experiments that allows running multiple experiments of the same group."""

import json

import torch
import torchvision
from torch import nn
from torch.utils.data import Subset, random_split, DataLoader
from torchvision import transforms

from evaluate import evaluate_on
from networks.util import build_network, AVAILABLE_NETWORKS, prepare_lc_builder
from utilities.data import get_number_of_classes, get_training_dataset, AVAILABLE_DATASETS, load_test_set
from utilities.eval import accuracy
from utilities.log import ExperimentLogger
from utilities.train import train_model

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def run(args):
    # (de-)activate GPU utilization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.force_device is not None and args.force_device != "":
        if args.force_device in ["gpu", "cuda"]:
            args.force_device = "cuda"
        device = torch.device(args.force_device)
    print(f"Optimizing on device '{device}'")

    # load data
    force_crop = None
    train_data = get_training_dataset(args.data, force_crop=force_crop)

    for i in range(0, args.i):
        train_set, validation_set = random_split(train_data, [int(len(train_data) * 0.9),
                                                              len(train_data) - int(len(train_data) * 0.9)])

        train_set_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2, )
        validation_set_loader = DataLoader(validation_set, batch_size=128, shuffle=True, num_workers=2)
        image_channels, image_width, image_height = next(iter(train_set_loader))[0].shape[1:]
        n_classes = get_number_of_classes(train_data)

        lc_layer_function = None
        if args.strategy != "none":
            lc_layer_function = prepare_lc_builder(args.strategy, args.widths, args.damps)
        network = build_network(args.network, input_shape=(image_channels, image_height, image_width),
                                n_classes=n_classes, lc=lc_layer_function, init_std=args.init_std)
        network.to(device)

        if args.auto_group:
            args.group = "-".join(list(map(lambda s: s.lower(), [
                network.__class__.__name__,
                args.data,
                *([args.group] if args.group != "none" else []),
            ])))
        logger_args = dict(group=args.group) if args.group is not None else dict()
        logger = ExperimentLogger(network, train_data, **logger_args)

        print(f"Model of type '{network.__class__.__name__}'{f' with lateral connections {network.lateral_layer} ' if network.is_lateral else ''} "
              f"created with id {logger.id} in group {args.group}."
              f"\n\nStarting Training on {train_data.__class__.__name__} with {len(train_set)} samples distributed over {len(train_set_loader)} batches."
              f"\nOptimizing for {args.epochs} epochs and validating on {len(validation_set)} samples every epoch.")

        train_model(model=network,
                    train_set_loader=train_set_loader,
                    val_set_loader=validation_set_loader,
                    n_epochs=args.epochs,
                    logger=logger,
                    device=device)

        test_data = load_test_set(image_channels, image_height, image_width, args.data)
        evaluate_on(network, test_data, model_dir=logger.model_dir)

        print("\nGaude! Consummatum est.\n\n")


if __name__ == '__main__':
    import argparse

    strategies = ["none", "lrn", "cmap-lrn", "gaussian-semlc", "semlc", "adaptive-semlc", "parametric-semlc", "singleshot-semlc"]

    parser = argparse.ArgumentParser()
    parser.add_argument("network", type=str, choices=AVAILABLE_NETWORKS)
    parser.add_argument("strategy", type=str, choices=strategies)
    parser.add_argument("--data", type=str, default="cifar10", choices=AVAILABLE_DATASETS, help="dataset to use")
    parser.add_argument("-w", "--widths", dest="widths", type=int, help="overwrite default widths", default=3)
    parser.add_argument("-d", "--damps", dest="damps", type=float, help="overwrite default damps", default=0.2)
    parser.add_argument("-c", "--cov", dest="coverage", type=int, help="coverage, default=1", default=1)
    parser.add_argument("-e", "--epochs", type=int, default=180, help="Number of epochs per model.")
    parser.add_argument("--init-std", type=float, help="std for weight initialization")

    parser.add_argument("-i", type=int, default=1, help="the number of iterations, default=1")
    parser.add_argument("-p", "--hpopt", type=str, help="hp optimisation with given index")
    parser.add_argument("--group", type=str, default=None, help="A group identifier, just for organizing.")
    parser.add_argument("--auto-group", action="store_true", help="Construct group name automatically based on parameters.")
    parser.add_argument("--force-device", type=str, choices=["cuda", "gpu", "cpu"])

    run(parser.parse_args())
