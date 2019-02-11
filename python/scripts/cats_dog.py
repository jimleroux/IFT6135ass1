import argparse
import os
import sys
sys.path.insert(0, "../")

import torch
import torchvision
import torchvision.transforms as transforms

from datasets.dataset import Dataset, data_split
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
    
    cat_dog_data = Dataset(transforms=transform)
    train, valid = data_split(cat_dog_data)
    cnn_cd = ConvNet("cat_and_dogs").to(device)
    out = cnn_cd.train_(
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
        type=int)
    args = parser.parse_args()
    main(args)
