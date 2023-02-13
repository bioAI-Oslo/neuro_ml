import torch
from dataclasses import dataclass
from torch_geometric.nn import GCNConv, Linear
from neuro_ml.dataset import TimeSeriesAndEdgeIndicesToWeightsDataset


@dataclass
class GCNParams:
    timestep_bin_length: int
    n_neurons: int


class GCN(torch.nn.Module):
    DATASET = TimeSeriesAndEdgeIndicesToWeightsDataset

    def __init__(
        self,
        params: GCNParams
    ):
        super(GCN, self).__init__()
        self.gcn = GCNConv(params.timestep_bin_length,
                           params.timestep_bin_length)
        self.fc1 = Linear(params.timestep_bin_length, params.n_neurons)
        self.fc2 = Linear(params.n_neurons**2, params.n_neurons**2)
        self.n_neurons = params.n_neurons
        self.timestep_bin_length = params.timestep_bin_length

    def forward(self, x, edge_index):
        h = self.gcn(x, edge_index)
        h = h.reshape(-1, self.n_neurons, self.timestep_bin_length)
        x = self.fc1(h).relu()
        x = x.reshape(-1, self.n_neurons**2)
        y_hat = self.fc2(x)
        return y_hat.flatten()
