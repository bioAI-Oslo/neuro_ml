import json
import os
from pathlib import Path
import hashlib
import numpy as np
from os import times
from tqdm import tqdm
import matplotlib.pyplot as plt
from numpy.random import default_rng
from neuro_ml.config import dataset_path


from mikkel_sim.generator import (
    construct_connectivity_filters,
    construct_connectivity_matrix,
    dales_law_transform,
    simulate,
)
from tqdm.std import time


def construct(params, rng=None):
    rng = default_rng() if rng is None else rng
    W_0 = construct_connectivity_matrix(params)  # Change this to give two populations
    W_0 = dales_law_transform(W_0)
    W, excit_idx, inhib_idx = construct_connectivity_filters(W_0, params)

    return W, W_0, excit_idx, inhib_idx


def generate_data(n_neurons, n_sims, n_steps, root_data_path):
    data_path = root_data_path / Path(f"mikkel_{n_neurons}_neurons_{n_steps}_timesteps")
    assert int(n_neurons / 2) == n_neurons / 2
    n_neurons = int(n_neurons / 2)

    data_path.mkdir(parents=True, exist_ok=True)

    print(
        f"Creating dataset with {n_neurons*2} neurons, {n_sims} sims, {n_steps} steps"
    )

    params = {
        "const": 5,
        "n_neurons": n_neurons,
        "dt": 1e-3,
        "ref_scale": 10,
        "abs_ref_scale": 3,
        "spike_scale": 5,
        "abs_ref_strength": -100,
        "rel_ref_strength": -30,
        "alpha": 0.2,
        "glorot_normal": {"mu": 0, "sigma": 5},
        "n_timesteps": n_steps,
    }

    filenames = []

    for seed in range(n_sims):
        print(f"Seed {seed}", end="\r")
        rng = default_rng(seed)
        W, W_0, excit_idx, inhib_idx = construct(params, rng=rng)

        result = simulate(
            W=W,
            W_0=W_0,
            params=params,
            pbar=True,
        )

        dataset_params = params.copy()
        dataset_params["n_sims"] = n_sims
        dataset_params["seed"] = seed
        dataset_str = json.dumps(dataset_params, sort_keys=True)
        fname = f"{seed}.npz"
        filenames.append(fname)

        np.savez(
            data_path / fname,
            X_sparse=result,
            W_0=W_0,
        )
        params["filenames"] = filenames
        with open(f"{data_path}/simulation-params.json", "w") as f:
            json.dump(params, f)
