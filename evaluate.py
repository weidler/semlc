import argparse
import json
import os
from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from config import CONFIG
from core.statistics import acc
from core.transform import make_test_transform_composition
from networks.util import build_network
from utilities.data import get_number_of_classes, AVAILABLE_DATASETS, get_dataset_class
from utilities.evaluation import evaluate_classification


def load_test_set(image_channels: int, image_height: int, image_width: int, dataset: str):
    """Return all available test dataset variants for given dataset."""
    dataset_class = get_dataset_class(dataset)

    return {"default": dataset_class(root=CONFIG.DATA_DIR, train=False, download=True,
                                     transform=make_test_transform_composition((image_width, image_height),
                                                                               image_channels))}


def evaluate_on(model: nn.Module, data: Dict[str, Dataset], model_dir: str, batch_size: int = 128, device: str = None):
    # turn on evaluation mode of model
    model.eval()
    device = device if device is not None else next(model.parameters()).device

    # evaluate all datasets
    evaluation_results = {}
    for test_setting in data.keys():
        test_data_loader = DataLoader(data[test_setting], batch_size=batch_size, shuffle=False, num_workers=2)

        # evaluate
        correct, total, loss = evaluate_classification(model, test_data_loader, criterion=nn.CrossEntropyLoss(),
                                                       device=device)
        # calculate metrics
        total_accuracy = acc(correct.sum(), total.sum())
        category_wise_accuracy = acc(correct, total)
        balanced_total_accuracy = category_wise_accuracy.mean()
        print(f"[{test_setting}] Accuracy of the network on the 10000 test images: {round(total_accuracy.item(), 2)}")

        # store results
        evaluation_results[test_setting] = dict(
            total=total_accuracy.item(),
            balanced=balanced_total_accuracy.item(),
            categories=category_wise_accuracy.tolist(),
        )

    # load potentially existing evaluation
    existing_data = {}
    if os.path.isfile(f"{model_dir}/evaluation.json"):
        with open(f"{model_dir}/evaluation.json", "r") as f:
            existing_data = json.load(f)
    existing_data.update(evaluation_results)

    # write results
    with open(f"{model_dir}/evaluation.json", "w") as f:
        json.dump(existing_data, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("id", type=str)
    parser.add_argument("dataset", nargs='?', type=str, choices=AVAILABLE_DATASETS, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-settings", nargs="*", choices=CONFIG.AVAILABLE_TEST_SETTINGS,
                        default=CONFIG.AVAILABLE_TEST_SETTINGS)
    parser.add_argument("--force-device", type=str, default=None)

    args = parser.parse_args()

    # SET DEVICE
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.force_device is not None:
        device = torch.device(args.force_device)
    print(f"Using device '{device}'")

    # LOAD
    model_id = args.id.strip()
    model_dir = os.path.join(CONFIG.MODEL_DIR, str(model_id))

    with open(f"{model_dir}/meta.json") as f:
        meta = json.load(f)

    if args.dataset is None:
        args.dataset = meta.get("dataset").get("name")
    args.dataset = args.dataset.lower()

    # PREPARE DATA
    image_width, image_height = (32, 32)
    if args.dataset == "cifar10":
        image_width, image_height = (32, 32)
    elif args.dataset == "mnist":
        image_width, image_height = (28, 28)
    elif args.dataset == "fashionmnist":
        image_width, image_height = (28, 28)

    image_channels = meta["input_channels"]
    test_data = load_test_sets(image_channels, image_height, image_width, args.dataset)

    # MAKE MODEL
    n_classes = get_number_of_classes(test_data["default"])
    model = build_network(meta["network_type"],
                          task="classification",
                          input_shape=(meta["input_channels"], image_height, image_width),
                          n_classes=n_classes, lc=meta.get("lateral_type"), complex_cells=meta.get("complex_cells"))
    model.load_state_dict(torch.load(f"{model_dir}/best.parameters"))
    print(f"Model loaded with id {model_id}.")
    model.to(device)

    # EVALUATE
    evaluate_on(model, test_data, model_dir, batch_size=args.batch_size)
