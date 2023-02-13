from neuro_ml.dataset.abstract import AbstractDataset


class TimeSeriesAndEdgeIndicesToWeightsDataset(AbstractDataset):
    IS_GEOMETRIC = True

    def __init__(
        self,
        filenames,
        dataset_params,
        is_classifier,
    ) -> None:
        super().__init__(
            filenames,
            dataset_params,
            is_classifier,
        )

        self._create_fully_connected_edge_index(dataset_params.n_neurons)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            "inputs": {"X": self.X[idx], "edge_index": self.edge_index[idx]},
            "y": self.y[idx],
        }
