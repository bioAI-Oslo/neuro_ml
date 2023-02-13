import math
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from torch_geometric.data import Data


class AbstractDataset(Dataset):
    def __init__(
        self,
        filenames,
        dataset_params,
        model_is_classifier,
    ) -> None:
        super().__init__()

        self._load_x_and_y(
            filenames,
            dataset_params,
            model_is_classifier,
        )

    def __len__(self):
        raise NotImplementedError(
            "Length of dataset is not implemented for abstract dataset"
        )

    def __getitem__(self, idx):
        raise NotImplementedError(
            "Getting item is not implemented for abstract dataset"
        )

    def _load_x_and_y(
        self,
        filenames,
        dataset_params,
        model_is_classifier,
    ):
        """
        Load the dataset
        """
        self.X = []
        self.y = []

        for filename in tqdm(
            filenames,
            unit="files",
            desc=f"Loading dataset",
            leave=False,
            colour="#52D1DC",
        ):
            # Convert sparse X to dense
            raw_data = np.load(filename, allow_pickle=True)

            raw_x = torch.tensor(raw_data["X_sparse"]).T
            sparse_x = torch.sparse_coo_tensor(raw_x, torch.ones(raw_x.shape[1]), size=(
                dataset_params.n_neurons, dataset_params.n_timesteps))

            X = sparse_x.to_dense()

            # If model is a classifier, one-hot encode the weight matrix
            y = (
                self.one_hot(torch.tensor(raw_data["W_0"]))
                if model_is_classifier
                else torch.tensor(raw_data["W_0"])
            )

            # Cut X into windows of length timestep_bin_length and append
            for i in range(
                math.floor(
                    dataset_params.n_timesteps / dataset_params.timestep_bin_length
                )
            ):
                x_slice = X[
                    :,
                    dataset_params.timestep_bin_length
                    * i: dataset_params.timestep_bin_length
                    * (i + 1),
                ]
                if x_slice.any():
                    self.X.append(x_slice.float())
                    self.y.append(y.float())

    def _create_edge_indices(self, n_neurons):
        """
        For each simulation in the dataset create an edge index based on the non-zero elements of W_0
        """
        self.edge_index = []

        for y in tqdm(
            self.y,
            desc="Creating edge indices",
            leave=False,
            colour="#432818",
        ):
            y = y.reshape(n_neurons, n_neurons)
            edge_index = torch.nonzero(y)

            self.edge_index.append(edge_index.T)

    def _create_fully_connected_edge_index(self, n_neurons):
        """
        For each simulation in the dataset create a fully connected edge index
        """
        self.edge_index = []
        for y in tqdm(
            self.y,
            desc="Creating edge indices",
            leave=False,
            colour="#432818",
        ):
            edge_index = torch.ones(n_neurons, n_neurons)
            self.edge_index.append(edge_index.nonzero().T)

    def create_geometric_data(self):
        """
        Create a list of torch_geometric.data.Data objects from the dataset
        """
        data = []
        for i in range(len(self)):
            inputs, y = self[i].values()
            data_i = Data(inputs["X"], inputs["edge_index"], y=y)
            data.append(data_i)
        return data

    def to_binary(self, y):
        """
        Create a binary representation of the weight matrix
        """
        zeros = torch.zeros(len(y))
        ones = torch.ones(len(y))
        y = torch.where(y == 0, zeros, ones)
        return y

    def one_hot(self, y):
        """
        Create a one-hot representation of the weight matrix
        """
        return F.one_hot((y.sign() + 1).to(torch.int64))
