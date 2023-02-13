from dataclasses import dataclass

import torch
import torch.nn as nn

from neuro_ml.dataset import TwoShiftedNeuronsToWeightsDataset


@dataclass
class SimpleParams:
    n_shifts: int = 10


class Simple(torch.nn.Module):
    DATASET = TwoShiftedNeuronsToWeightsDataset
    NAME = "simple"

    def __init__(self, params: SimpleParams):
        super().__init__()
        self.n_shifts = params.n_shifts
        self.fc1 = nn.Linear(
            params.n_shifts,
            10*params.n_shifts,
        )
        self.fc2 = nn.Linear(
            10*params.n_shifts, params.n_shifts
        )
        self.fc3 = nn.Linear(
                params.n_shifts, 1
        )


    def forward(self, x, _):
        x = self.fc1(x).relu()
        x = self.fc2(x).relu()
        x = self.fc3(x)
        return x.squeeze()

    def save(self, filename):
        torch.save(self, f"models/{self.NAME}/{filename}")
