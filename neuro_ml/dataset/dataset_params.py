from dataclasses import dataclass
from enum import Enum


class SimulationEnum(Enum):
    mikkel = "mikkel"
    rust = "rust"
    rust_dense_separated_cluster = "rust_dense_separated_cluster"
    python_dense = "python_dense"
    python_dense_separated_cluster = "python_dense_separated_cluster"
    python_dense_no_inhibatory_params = "python_no_inhibatory_params"
    inner_products = "inner_products"


@dataclass
class DatasetParams:
    n_neurons: int
    n_timesteps: int
    timestep_bin_length: int
    simulation_enum: SimulationEnum
    number_of_files: int

    @property
    def foldername(self):
        return f"{self.simulation_enum.name}_{self.n_neurons}_neurons_{self.n_timesteps}_timesteps"
