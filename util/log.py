import json
import os
import socket
import time

import numpy
from torchvision.datasets import VisionDataset

from config import CONFIG
from networks import BaseNetwork


class ExperimentLogger:

    def __init__(self, model: BaseNetwork, dataset: VisionDataset, group: str = "ungrouped"):
        self.model = model
        self.dataset = dataset
        self.group = group
        self.transform = self.dataset.transform

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
        return dict(
            **self.model.serialize_meta(),
            id=self.id,
            group=self.group,
            host=socket.gethostname(),
            dataset=dict(
                name=self.dataset.__class__.__name__,
                n_classes=len(self.dataset.classes) if hasattr(self.dataset, "classes") else len(
                    numpy.unique(self.dataset.labels)),
                classes=self.dataset.classes if hasattr(self.dataset, "classes") else numpy.unique(
                    self.dataset.labels).tolist(),
                tranform=[str(t) for t in self.dataset.transform.transforms]
            ),
            ended_naturally=False
        )

    def save(self):
        """
        Save if current conditions meeting settings.
        """
        raise NotImplementedError()

    def log(self, epoch: int, train_loss: float, val_loss: float, val_acc: float) -> None:
        """
        Log ongoing statistics.
        """
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
        print("Finalizing training session.")
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
