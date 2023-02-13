import torch
from torch_geometric.nn import GCNConv
from torch_geometric.utils.negative_sampling import edge_index_to_vector, negative_sampling
from dataclasses import dataclass
import torch.nn.functional as F
from neuro_ml.dataset import LinkClassificationWithGraphsDataset

@dataclass
class ClassifierGCNParams:
    timestep_bin_length: int
    n_neurons: int

class ClassifierGCN(torch.nn.Module):
    DATASET = LinkClassificationWithGraphsDataset

    def __init__(self, params: ClassifierGCNParams):
        super(ClassifierGCN, self).__init__()
        self.latent_features = int(params.timestep_bin_length / 2)
        self.conv1 = GCNConv(params.timestep_bin_length, self.latent_features, add_self_loops=False)
        self.fc1 = torch.nn.Linear(params.n_neurons**2, 2*params.n_neurons**2)
        self.fc2 = torch.nn.Linear(2*params.n_neurons**2, params.n_neurons**2)
        self.timestep_bin_length = params.timestep_bin_length
        self.n_neurons = params.n_neurons

    def forward(self, x, edge_index):
        z = self.conv1(x, edge_index).relu()
        z = z.reshape(-1, self.n_neurons, self.latent_features)
        z = (z @ z.transpose(1, 2))
        z = z.reshape(-1, self.n_neurons**2)
        z = self.fc1(z).relu()
        y_hat = self.fc2(z).sigmoid()
        return y_hat.flatten()

    def loss(self, input, target):
        return F.binary_cross_entropy(input, target)
