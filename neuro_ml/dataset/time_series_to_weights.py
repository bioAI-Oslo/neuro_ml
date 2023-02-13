from tqdm import tqdm
from pathlib import Path
from neuro_ml.dataset.abstract import AbstractDataset
from zenlog import log


class TimeSeriesToWeightsDataset(AbstractDataset):
    IS_GEOMETRIC = False

    def __init__(
        self,
        filenames,
        dataset_params,
        is_classifier,
    ) -> None:
        super().__init__(filenames, dataset_params, is_classifier)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], (), self.y[idx]
