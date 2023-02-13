from logging import critical
from neuro_ml.models.edge_regressor import EdgeRegressor, EdgeRegressorParams
from neuro_ml.models.edge_classifier import EdgeClassifier, EdgeClassifierParams
import pandas as pd
import torchmetrics
import matplotlib.animation as animation
import random
import torch
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

def load_data(file):
    params_match = re.search(r".*_(\d+)_neurons_(\d+)_timesteps.*", str(file))

    n_neurons = int(params_match.group(1))
    n_timesteps = int(params_match.group(2))

    data = np.load(file, allow_pickle=True)

    X_sparse = data["X_sparse"].T
    w0 = data["W_0"]
    sparse_x = torch.sparse_coo_tensor(X_sparse, torch.ones(X_sparse.shape[1]), size = (n_neurons, n_timesteps))

    X = sparse_x.to_dense()

    return X, w0

def plot_bar_charts(X, n_bins):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))

    #  axes[0].title("Firings per neuron")
    axes[0].bar(range(1, X.shape[0] + 1), np.sum(X, axis=1))
    #  plt.show()

    n_bins = n_bins
    axes[1].plot(
        range(1, n_bins + 1),
        np.sum(np.reshape(np.sum(X, axis=0), (n_bins, -1)), axis=1),
    )
    plt.show()

def plot_confusion_matrix(y, y_hat):
    y = y.argmax(dim=2).flatten()
    y_hat = y_hat.reshape(400, -1)
    confmat = torchmetrics.functional.confusion_matrix(y_hat, y, num_classes=3)
    plt.figure()
    df_cm = pd.DataFrame(np.array(confmat), index=["Negative", "Zero", "Positive"], columns=["Negative", "Zero", "Positive"])
    sns.heatmap(df_cm, annot=True, cbar=False)
    plt.xlabel("True class")
    plt.ylabel("Predicted class")
    plt.show()

def plot_classifications(y, y_hat):
    y = y.argmax(dim=2)
    y_hat = y_hat.argmax(dim=2)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_figheight(4.5)
    fig.set_figwidth(12)

    sns.heatmap(y, ax=ax1)
    sns.heatmap(y_hat, ax=ax2)
    plt.show()


def plot_heat_maps(y, y_hat):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_figheight(4.5)
    fig.set_figwidth(12)

    sns.heatmap(y, ax=ax1, vmin = -5, vmax = 5)
    sns.heatmap(y_hat, ax=ax2, vmin = -5, vmax = 5)
    ax1.set_title("$y$")
    ax1.tick_params(axis="x")
    ax1.tick_params(axis="y")

    ax2.set_title(r"$\hat{y}$")
    ax2.tick_params(axis="x")
    ax2.tick_params(axis="y")

    plt.show()

def animate_heat_map(y, y_hats):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_figheight(4.5)
    fig.set_figwidth(12)

    def init():
        ax1.set_title("$y$")
        # ax1.tick_params(axis="x")
        # ax1.tick_params(axis="y")

        ax2.set_title(r"$\hat{y}$")
        # ax2.tick_params(axis="x")
        # ax2.tick_params(axis="y")

        sns.heatmap(y, ax=ax1, vmin = -5, vmax = 5, cbar=False, xticklabels=False, yticklabels=False)
        sns.heatmap(y_hats[0], ax=ax2, vmin = -5, vmax = 5, cbar=False, xticklabels=False, yticklabels=False)

    def animate(i):
        ax2.clear()
        if i < 9:
            ax2.set_title(f"T=1000")
        else:
            ax2.set_title(f"T={(1000*(i+1) // 10000) * 10000}")
        sns.heatmap(y_hats[i], ax=ax2, vmin=-5, vmax=5, cbar=False, xticklabels=False, yticklabels=False)

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(y_hats), repeat=False)

    FFwriter = animation.FFMpegWriter(fps=20)
    anim.save('report/animations/learning_anim.mp4', writer = FFwriter)

    plt.show()

def prepare_heat_anim(file):
    X, w0 = load_data(file)
    y = torch.tensor(w0)
    n_neurons = X.shape[0]
    edge_index = torch.ones(20, 20).nonzero().T
    n_timesteps = range(1000, 100_000+1, 1000)

    params = EdgeRegressorParams(n_shifts=10)
    m = EdgeRegressor(params)
    m.load_state_dict(torch.load(f"models/edge_regressor/{n_neurons}_neurons_100000_timesteps_70.pt", map_location = torch.device("cpu")))
    y_hats = []
    m.eval()
    for n_timestep in n_timesteps:
        with torch.no_grad():
            y_hats.append(m(X[:, :n_timestep], edge_index))

    return y, y_hats

def classify(file):
    X, w0 = load_data(file)
    y = torch.tensor(w0)
    y = torch.nn.functional.one_hot((y.sign() + 1).to(torch.int64))
    edge_index = torch.ones(20, 20).nonzero().T

    params = EdgeClassifierParams(n_shifts=20, n_classes=3)
    m = EdgeClassifier(params)
    m.load_state_dict(torch.load(f"models/edge_classifier/20_neurons_100000_timesteps_40.pt", map_location = torch.device("cpu")))
    m.eval()
    with torch.no_grad():
        y_hat = m(X, edge_index)
    return y, y_hat

def regress(file):
    X, w0 = load_data(file)
    y = torch.tensor(w0)
    edge_index = torch.ones(20, 20).nonzero().T

    params = EdgeRegressorParams(n_shifts=10)
    m = EdgeRegressor(params)
    m.load_state_dict(torch.load(f"models/edge_regressor/20_neurons_100000_timesteps_70.pt", map_location = torch.device("cpu")))
    m.eval()
    with torch.no_grad():
        y_hat = m(X, edge_index)

    return y, y_hat

def plot_timestep_dependence(file):
    X, w0 = load_data(file)
    y = torch.tensor(w0)
    edge_index = torch.ones(20, 20).nonzero().T

    params = EdgeRegressorParams(n_shifts=10)
    m = EdgeRegressor(params)
    m.load_state_dict(torch.load(f"models/edge_regressor/20_neurons_100000_timesteps_70.pt", map_location = torch.device("cpu")))
    criterion = torch.nn.MSELoss()
    m.eval()
    loss = []
    with torch.no_grad():
        for i in range(10000, 100_000+1, 1000):
            y_hat = m(X[:, :i], edge_index)
            loss.append(criterion(y, y_hat).item())

    plt.plot(range(10000, 100_000+1, 1000), loss)
    plt.xlabel("Timesteps")
    plt.ylabel("MSE loss")
    plt.savefig("report/plots/timestep_loss.png")
    plt.show()


dataset_path = (
    Path("plot_data")
)


directories = list(dataset_path.iterdir())
# for i, path in enumerate(directories):
    # print(f"{i+1}) {path}")

# option = input("\nSelect option: ")

# directory = directories[int(option) - 1]
directory = directories[4]

while True:
    file = random.choice(list(directory.iterdir()))
    if file.suffix == ".json":
        print(f"Skipping json file {file}")
        continue

    # y, y_hat = classify(file)
    # y, y_hat = regress(file)
    # plot_heat_maps(y, y_hat)

    # plot_timestep_dependence(file)

    # plot_classifications(y, y_hat)
    # plot_confusion_matrix(y, y_hat)

    y, y_hats = prepare_heat_anim(file)
    animate_heat_map(y, y_hats)
