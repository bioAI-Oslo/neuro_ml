import torch
from tqdm import tqdm
from neuro_ml.dataset.abstract import AbstractDataset
from torch_geometric.utils import train_test_split_edges

class LinkClassificationWithGraphsDataset(AbstractDataset):
    IS_GEOMETRIC = True

    def __init__(
        self, filenames, n_neurons, timestep_bin_length, n_total_timesteps
    ) -> None:
        super().__init__(filenames, n_neurons, timestep_bin_length, n_total_timesteps)

        self._create_fully_connected_edge_index(n_neurons)
        self._get_link_labels()

    def __len__(self):
        return len(self.X)

    def _get_link_labels(self):
        self.link_labels = []
        for w_0 in tqdm(
            self.w_0,
            desc="Creating link lables",
            leave=False,
            colour="#432810",
        ):
            zeros = torch.zeros(len(w_0))
            ones = torch.ones(len(w_0))
            link_labels = torch.where(w_0  == 0, zeros, ones)
            self.link_labels.append(link_labels)

    def __getitem__(self, idx):
        return {
            "inputs": {"X": self.X[idx], "edge_index": self.edge_index[idx]},
            "y": self.link_labels[idx],
        }
