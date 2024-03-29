import argparse
import os
import sys
sys.path.insert(0, "../")

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from datasets.dataset import Dataset, data_split, import_mnist
from models.cnn import ConvNet


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
    model_parameters = filter(lambda p: p.requires_grad, cnn.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("CNN has {} parameters".format(params))
    out = cnn.train_(
        train, valid, device, num_epoch=epoch, lr=lr, batchsize=batch)
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
    args = parser.parse_args()
    main(args)
