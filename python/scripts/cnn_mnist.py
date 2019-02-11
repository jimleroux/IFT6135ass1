import argparse
import os
import sys
sys.path.insert(0, "../models")
sys.path.insert(0, "../datasets")

import torch
import torchvision
import torchvision.transforms as transforms


from dataset import Dataset
from dataset import data_split
from cnn import ConvNet
from dataset import import_mnist

def main(args):
    epoch = args.epoch
    lr = args.lr
    batch = args.batch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.manual_seed(10)

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(), transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    train, valid = import_mnist(transform)
    cnn = ConvNet("mnist").to(device)
    out = cnn.train_(
        train, valid, device, num_epoch=epoch, lr=lr, batchsize=batch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epoch", help="Choose the number of epoch",
        default=70, type=int)
    parser.add_argument(
        "--lr", help="Choose the learning rate",
        default=0.01, type=float)
    parser.add_argument(
        "--batch", help="Choose batchsize",
        default=64, type=int)
    args = parser.parse_args()
    main(args)