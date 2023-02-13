from tqdm import tqdm
import numpy as np
from pathlib import Path
from neuro_ml.dataset.abstract import AbstractDataset
from zenlog import log
import torch


class TwoShiftedNeuronsToWeightsDataset(AbstractDataset):
    IS_GEOMETRIC = False

    def __init__(
        self,
        filenames,
        dataset_params,
        is_classifier,
    ) -> None:
        self.X = []
        self.y = []

        for filename in tqdm(
            filenames,
            unit="files",
            desc=f"Loading dataset",
            leave=False,
            colour="#52D1DC",
        ):
            raw_data = np.load(filename, allow_pickle=True)
            x = raw_data["X"]
            y = raw_data["W_0"]
            for i in range(len(raw_data["X"])):
                self.X.append(torch.tensor(x[i]).float())
                self.y.append(self.to_binary(
                    y[i]) if is_classifier else torch.tensor(y[i]).float())

    def __len__(self):
        return len(self.X)

    def to_binary(self, y):
        return 1. if y > 0 else 0.

    def __getitem__(self, idx):
        return self.X[idx], (), self.y[idx]
