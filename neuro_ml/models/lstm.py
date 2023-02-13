from dataclasses import dataclass

import torch
import torch.nn as nn

from neuro_ml.dataset import TimeSeriesToWeightsDataset


@dataclass
class LSTMParams:
    timestep_bin_length: int
    n_nodes: int
    num_layers: int = 10
    has_batch: bool = True


class LSTM(torch.nn.Module):
    loss = torch.nn.MSELoss()
    DATASET = TimeSeriesToWeightsDataset
    NAME = "lstm"

    def __init__(self, params: LSTMParams):
        assert params.has_batch, "LSTM is only implemented with batch"
        super().__init__()
        self.lstm = nn.LSTM(
            params.timestep_bin_length,
            params.timestep_bin_length,
            num_layers=params.num_layers,
            dropout=0.1,
            batch_first=True,
        )
        self.fc1 = nn.Linear(
            params.timestep_bin_length * params.n_nodes,
            params.timestep_bin_length * params.n_nodes,
        )
        self.fc2 = nn.Linear(
            params.timestep_bin_length * params.n_nodes, params.n_nodes * params.n_nodes
        )

    def forward(self, x, _):
        x, _ = self.lstm(x)
        x = x.flatten(start_dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def save(self, filename):
        torch.save(self, f"models/{self.NAME}/{filename}")
