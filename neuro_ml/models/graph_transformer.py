import torch
import torch.nn as nn
import numpy as np
from zenlog import log
from dataclasses import dataclass
from scipy.sparse.linalg import eigs
from torch_geometric.nn import LayerNorm
from torch_geometric.nn import GATConv, Linear
from neuro_ml.dataset import TimeSeriesAndEdgeIndicesToWeightsDataset
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix


@dataclass
class GraphTransformerParams:
    timestep_bin_length: int
    n_neurons: int
    n_latent_features: int
    n_eigen_values: int
    n_heads: int
    n_edges: int
    batch_size: int
    device: torch.device
    num_layers: int = 3


class GraphTransformer(torch.nn.Module):
    loss = torch.nn.L1Loss()
    DATASET = TimeSeriesAndEdgeIndicesToWeightsDataset

    def __init__(self, params: GraphTransformerParams):
        super(GraphTransformer, self).__init__()
        self.timestep_bin_length = params.timestep_bin_length
        self.n_neurons = params.n_neurons
        self.batch_size = params.batch_size
        self.n_eigen_values = params.n_eigen_values
        self.n_latent_features = params.n_latent_features
        self.n_edges = params.n_edges
        self.device = params.device

        self.A = Linear(
            self.timestep_bin_length, self.n_latent_features
        )  # projection of feature vector x_i into d dimensions
        self.C = Linear(
            self.n_eigen_values, self.n_latent_features
        )  # Projection of eigen values into d dimensions


        self.transformers = nn.ModuleList(
            [
                GraphTransformerLayer(
                    self.n_latent_features, self.n_latent_features, params.n_heads
                ).to(self.device)
                for _ in range(params.num_layers)
            ]
        )

        self.Q = Linear(params.n_neurons**2, params.n_neurons**2)
        self.P = Linear(params.n_neurons**2, params.n_neurons**2)

    def forward(self, x, edge_index):
        batch_samples = int(edge_index.size(1) / self.n_edges)
        eigen_values = self.get_graph_laplacian(
            edge_index, k=self.n_eigen_values, n_nodes=self.n_neurons
        )
        eigen_values = (
            eigen_values.view(-1, self.n_eigen_values)
            .repeat_interleave(self.n_neurons, dim=0)
            .to(self.device)
        )

        x = self.A(x)
        e = self.C(eigen_values)
        x = x + e

        for transformer in self.transformers:
            x = transformer(x, edge_index)

        x = x.reshape(batch_samples, self.n_neurons, self.n_latent_features)

        x = x @ x.transpose(1, 2) / self.n_neurons**2
        x = x.reshape(batch_samples, self.n_neurons**2)

        x = self.Q(x).relu()
        y_bar = self.P(x)


        return y_bar.flatten()

    def get_graph_laplacian(self, edge_index, n_nodes, k):
        batch_size = int(edge_index.size(1) / self.n_edges)
        eigen_values = torch.zeros(k * batch_size)
        step = int(edge_index.size(1) / batch_size)
        for i in range(batch_size):
            normalized_edge_index = edge_index[
                :, step * i : step * (i + 1)
            ] - n_nodes * i * torch.ones((2, step)).int().to(self.device)
            laplacian_edge_index, laplacian_edge_weight = get_laplacian(
                normalized_edge_index, normalization="sym"
            )
            L = to_scipy_sparse_matrix(
                laplacian_edge_index, laplacian_edge_weight, n_nodes
            )
            eigen_values[k * i : k * (i + 1)] = torch.tensor(
                np.real(eigs(L, k=k, which="SM", return_eigenvectors=False))
            )
        return eigen_values


class GraphTransformerLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, concat=True, heads=1):
        super().__init__()
        self.attention = GATConv(
            in_channels, out_channels, heads=heads, concat=concat, add_self_loops=False
        )
        self.O = Linear(heads * out_channels, out_channels)
        self.norm = LayerNorm(out_channels)
        self.W1 = Linear(out_channels, 2 * out_channels)
        self.W2 = Linear(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        h_bar = self.O(self.attention(x, edge_index))
        h_bar_bar = self.norm(h_bar)
        h_bar_bar_bar = self.W2(self.W1(h_bar_bar).relu())
        h = self.norm(h_bar_bar_bar)
        return h
