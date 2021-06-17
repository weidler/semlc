import json
import os
import socket
import time
from typing import Union

import numpy
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset

from config import CONFIG
from networks import BaseNetwork
from utilities.data.imagenet import DALITorchLoader


class ExperimentLogger:

    def __init__(self, model: BaseNetwork, dataloader: Union[DALITorchLoader, DataLoader], group: str = "ungrouped"):
        self.model = model
        self.dataloader = dataloader
        self.group = group

        self._generate_id()
        while os.path.isdir(self.model_dir):  # wait until time-based id is free
            self._generate_id()
        os.makedirs(self.model_dir)

        # make and initialize log file
        self.log_file = os.path.join(self.model_dir, "train.log")
        with open(self.log_file, "w") as f:
            json.dump(self._init_log_file_template(), f)

        # write experiment info
        with open(f"{self.model_dir}/meta.json", "w") as f:
            json.dump(self._make_meta_dict(), f)

    def _generate_id(self):
        self.id = int(round(time.time() * 1000000))
        self.model_dir = os.path.join(CONFIG.MODEL_DIR, str(self.id))

    @staticmethod
    def _init_log_file_template() -> dict:
        return dict(
            epoch=[],
            val_acc=[],
            train_loss=[],
            val_loss=[],
        )

    def _make_meta_dict(self):
        dataset_descriptor = {}
        if isinstance(self.dataloader, VisionDataset):
            dataset_descriptor = dict(
                name=self.dataloader.__class__.__name__,
                n_classes=len(self.dataloader.classes) if hasattr(self.dataloader, "classes") else len(
                    numpy.unique(self.dataloader.labels)),
                classes=self.dataloader.classes if hasattr(self.dataloader, "classes") else numpy.unique(
                    self.dataloader.labels).tolist(),
                transform=[str(t) for t in self.dataloader.transform.transforms]
            )
        elif isinstance(self.dataloader, DALITorchLoader):
            dataset_descriptor = dict(
                name=self.dataloader.dataset_name,
                n_classes=self.dataloader.n_classes,
                classes=["unknown"],
                transform=["unknown"]
            )

        return dict(
            **self.model.serialize_meta(),
            id=self.id,
            group=self.group,
            host=socket.gethostname(),
            dataset=dataset_descriptor,
            ended_naturally=False
        )

    def log(self, epoch: int, train_loss: float, val_loss: float, val_acc: float) -> None:
        """Log ongoing statistics."""
        with open(self.log_file, "r") as f:
            current = json.load(f)

        current["epoch"].append(epoch)
        current["train_loss"].append(train_loss)
        current["val_loss"].append(val_loss)
        current["val_acc"].append(val_acc)

        with open(self.log_file, "w") as f:
            json.dump(current, f)

    def finalize(self, time_taken: float):
        """Finalize the training.

        Adjusts some meta info to match the fact that training ended naturally. Also saves the last parameter state.

        Args:
            time_taken: time passed since beginning and end of training (in seconds)
        """
        # torch.save(self.layers.state_dict(), f"{self.model_dir}/final.parameters")

        with open(self.log_file, "r") as f:
            log = json.load(f)

        with open(f"{self.model_dir}/meta.json", "r") as f:
            meta = json.load(f)

        meta["ended_naturally"] = True
        meta["train_time"] = time_taken
        meta["time_per_epoch"] = time_taken / max(log["epoch"])

        with open(f"{self.model_dir}/meta.json", "w") as f:
            json.dump(meta, f)
