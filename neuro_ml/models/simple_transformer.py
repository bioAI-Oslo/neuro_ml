import torch
import torch.nn as nn
from dataclasses import dataclass

from neuro_ml.dataset import TimeSeriesToWeightsDataset


@dataclass
class SimpleTransformerPrams:
    d_model: int
    n_head: int


class SimpleTransformer(nn.Module):
    loss = torch.nn.MSELoss()
    DATASET = TimeSeriesToWeightsDataset

    def __init__(self, params: SimpleTransformerPrams):
        super().__init__()

        self.transformer = nn.Transformer(
            d_model=params.d_model, nhead=params.n_head, batch_first=True
        ).encoder

        self.fc1 = nn.Linear(20000, 2000)
        self.fc2 = nn.Linear(2000, 400)

    def forward(self, src, _):
        x = self.transformer(src)
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x
