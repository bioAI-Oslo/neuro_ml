# Neuro ml summer project

## Install

You need to install pytorch, preferably with CUDA. In addition, install `requirements.txt`

```console
$ pip install -r requirements.txt
```

## Usage

### Simulation code

```console
$ python -m mikkel_sim
```

```
usage: __main__.py [-h] [-d N] [-s NUM_OF_STEPS] [-n NUM_NEURONS]

Create simulated neuron datasets

options:
  -h, --help            show this help message and exit
  -d N, --num-data N    Generate d datasets (default: 100)
  -s NUM_OF_STEPS, --num-of-steps NUM_OF_STEPS
                        Number of time steps for the simulation (default: 100000)
  -n NUM_NEURONS, --num-neurons NUM_NEURONS
                        Use n neurons in simulation (default: 20)
```

### Train/test model

Make sure to specify the correct arguments in `neuro_ml/__main__.py`, inside `if __name__ == "__main__"`

```console
$ python -m neuro_ml 
```

### Technical improvements
- X is very sparse so we could use a sparse representation for more efficient memory use

### General improvements
- Predict edge features instead of node features. This is more clear conceptually since the weights are associated with edges and is supported by edge_updater and edge_update in the
MessagePassing base class
- Only use the inner MLP and make it more powerful (or use a transformer/lstm instead)

### Testing
- Dependence on M (number of time steps that we calculate the co-firing rates for)

