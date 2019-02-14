import argparse
import os
import sys
sys.path.insert(0, "../")

import torch
import torchvision
import torchvision.transforms as transforms

from datasets.dataset import Dataset, data_split, TestDataset
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
    submission = TestDataset()
    train, valid = data_split(cat_dog_data)
    cnn = ConvNet("cat_and_dogs").to(device)
    out = cnn.train_(
        train, valid, device, num_epoch=epoch, lr=lr, batchsize=batch)
    cnn.prediction(submission, batchsize)
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epoch", help="Choose the number of epoch.",
        default=70, type=int
    )
    parser.add_argument(
        "--lr", help="Choose the learning rate.",
        default=0.1, type=float
    )
    parser.add_argument(
        "--batch", help="Choose batchsize.",
        default=256, type=int
    )
    args = parser.parse_args()
    main(args)
