import argparse
import sys

import matplotlib.pyplot as plt

import cats_dog
import cnn_mnist
import mlp_mnist

sys.path.insert(0, "../")
from utils.plot_learning_curves import plot_inits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epoch", help="Choose the number of epoch",
        default=1, type=int
    )
    parser.add_argument(
        "--lr", help="Choose the learning rate",
        default=0.01, type=float
    )
    parser.add_argument(
        "--batch", help="Choose batchsize",
        default=256, type=int
    )
    parser.add_argument(
        "--init", help="Choose the init mode",
        default="glorot", choices=["glorot", "uniform", "normal"],
        type=str
    )
    args = parser.parse_args()
    args.init = "glorot"
    _, _, loss_train_glorot, _ = mlp_mnist.main(args)
    args.init = "uniform"
    _, _, loss_train_uniform, _ = mlp_mnist.main(args)
    args.init = "normal"
    _, _, loss_train_normal, _ = mlp_mnist.main(args)
    datas = {
        "glorot": loss_train_glorot,
        "uniform": loss_train_uniform,
        "normal": loss_train_normal
    }
    plot_inits(datas)
    plt.show()
