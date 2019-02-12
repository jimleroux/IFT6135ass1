import argparse
import mlp_mnist
import cnn_mnist
import cats_dog
import sys
sys.path.insert(0, "../")
from utils.plot_learning_curves import plot_inits

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epoch", help="Choose the number of epoch",
        default=10, type=int
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
    