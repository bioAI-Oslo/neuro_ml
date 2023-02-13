import torch
from torch.nn import LeakyReLU, Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing
from dataclasses import dataclass
from neuro_ml.dataset import TimeSeriesAndEdgeIndicesToWeightsDataset
from neuro_ml.models import EdgeRegressor

@dataclass
class EdgeClassifierParams:
    n_shifts: int = 10
    n_neurons: int = 20
    n_classes: int = 2

class EdgeClassifier(torch.nn.Module):
    DATASET = TimeSeriesAndEdgeIndicesToWeightsDataset
    NAME = "edge_classifier"

    def __init__(self, params):
        super().__init__()
        self.n_neurons = params.n_neurons
        self.n_classes = params.n_classes
        self.edge_embedding = EdgeRegressor(params)
        self.classifier = Seq(
                Linear(params.n_neurons, params.n_neurons),
                ReLU(),
                Linear(params.n_neurons, params.n_classes*params.n_neurons)
        )

    def forward(self, x, other_inputs):
        x = self.edge_embedding(x, other_inputs)
        x = self.classifier(x)
        return x.reshape(20, 20, 3).softmax(dim=2)
    
    def save(self, filename):
        torch.save(self.state_dict(), f"models/{self.NAME}/{filename}")

