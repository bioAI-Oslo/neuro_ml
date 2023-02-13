import argparse
from pathlib import Path
from mikkel_sim.make_dataset import (
    generate_data,
)


def main():
    parser = argparse.ArgumentParser(
        description="Create simulated neuron datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-d",
        "--num-data",
        type=int,
        metavar="N",
        default=100,
        help="Generate d datasets",
    )

    parser.add_argument(
        "-s",
        "--num-of-steps",
        type=int,
        default=int(100_000),
        help="Number of time steps for the simulation",
    )

    parser.add_argument(
        "-n", "--num-neurons", type=int, default=20, help="Use n neurons in simulation"
    )

    parser.add_argument(
        "-p", "--path", default="data", help="The root path for the data"
    )

    args = parser.parse_args()

    root_data_path = Path(args.path)

    generate_data(args.num_neurons, args.num_data, args.num_of_steps, root_data_path)


if __name__ == "__main__":
    main()
