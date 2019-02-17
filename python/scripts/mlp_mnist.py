import argparse
import os
import sys
sys.path.insert(0, "../")

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from datasets.dataset import Dataset, data_split, import_mnist
from models.mlp import NeuralNetwork
from utils.plot_functions import plot_grad_check
from utils.model_save_load import load, save

def main(args):
    epoch = args.epoch
    lr = args.lr
    batch = args.batch
    grad_check = args.grad_check
    init = args.init
    np.random.seed(10)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(), transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
  
    train, valid = import_mnist(transform)
    mlp = NeuralNetwork(init=init)
    if grad_check:
        plot_grad_check(mlp, train)
    out = mlp.train(
        train, valid, num_epoch=epoch, lr=lr, batchsize=batch)
    return out

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
        default="glorot", choices=["glorot", "zero", "normal"],
        type=str
    )
    parser.add_argument(
        "--grad_check", help="Specify to grad check.",
        action="store_true"
    )
    args = parser.parse_args()
    main(args)
