import torch
from torch.nn import LeakyReLU, Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from dataclasses import dataclass
from neuro_ml.dataset import TimeSeriesAndEdgeIndicesToWeightsDataset

@dataclass
class EdgeRegressorParams:
    n_shifts: int = 10
    n_neurons: int = 20

class EdgeRegressor(MessagePassing):
    """
    Calculates x_i' using the formula:
    x_i' = MLP_2(||_{j \in \mathcal{N}(i) MLP_1(1/N ||_{t=1}^M x_i * P^t x_j))
    """

    DATASET = TimeSeriesAndEdgeIndicesToWeightsDataset
    NAME = "edge_regressor"

    def __init__(self, params):
        super().__init__()
        self.n_shifts = params.n_shifts # M, number of time steps for which we consider the influence of i  on j forward in time
        self.n_neurons = params.n_neurons # N, number of neurons in the network
        self.selection_matrix = (torch.eye(self.n_neurons)
                .repeat_interleave(self.n_neurons, dim=0)
            ) # Matrix that selects the j-th neuron in the MLP_1 output
        self.mlp1 = Seq(
            Linear(params.n_shifts, params.n_neurons),
            ReLU(),
            Linear(params.n_neurons, 10*params.n_neurons),
            ReLU(),
            Linear(10*params.n_neurons, params.n_neurons),
            ReLU(),
            Linear(params.n_neurons, 1)
            ) # First MLP
        self.mlp2 = Seq(
            Linear(params.n_neurons, 10*params.n_neurons),
            ReLU(),
            Linear(10*params.n_neurons, params.n_neurons)
        ) # Second MLP

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]
        batch_size = int(x_i.shape[0] / self.n_neurons**2)

        # Calculate the influence of i on j forward in time
        inner_products = [torch.sum(x_i*self.shift(x_j, -(t+1)), dim=1).unsqueeze(dim=1) for t in range(self.n_shifts)]

        # Concatenate the inner products to get vectors of length M
        tmp = torch.cat(inner_products, dim=1)


        tmp = tmp / (x_i.shape[1]/100) # Normalize the inner products to get co-firings per 100 time steps

        batch_selection_matrix = self.selection_matrix.repeat(batch_size, 1) # Repeat the selection matrix for the entire batch

        return self.mlp1(tmp)*batch_selection_matrix # Apply the first MLP to the entire batch

    def update(self, inputs):
        return self.mlp2(inputs) # Apply the second MLP

    def shift(self, x, n):
        # Shifts the time series x by n time steps to the left
        result = torch.zeros_like(x)
        if n < 0:
            result[:, :n] = x[:, -n:]
        elif n > 0:
            result[:, -n:] = x[:, :n]
        else:
            result = x
        return result

    def save(self, filename):
        # Saves the model to a file
        torch.save(self.state_dict(), f"models/{self.NAME}/{filename}")
