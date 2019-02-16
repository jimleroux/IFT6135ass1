import argparse
import os
import sys
sys.path.insert(0, "../")

import torch
import torchvision
import torchvision.transforms as transforms

from datasets.dataset import Dataset, data_split, TestDataset
from models.cnn import ConvNet
import matplotlib.pyplot as plt

class meanstd:
    def __call__(self, im):
        means = torch.sum(im, dim=(1,2),keepdim=True)/im.shape[1]**2
        std = torch.sqrt(torch.sum((im-means)**2,dim=(1,2),keepdim=True)/im.shape[1]**2)
        return ((im-means)/std)/((im-means)/std).max()

def main(args):
    epoch = args.epoch
    lr = args.lr
    batch = args.batch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.manual_seed(10)

    transform = transforms.Compose([
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        #meanstd()
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    normalize = transforms.Compose([
        transforms.ToTensor(),
        #meanstd()
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    cat_dog_data = Dataset(transforms=transform)
    submission = TestDataset(transforms=normalize)
    train, valid = data_split(cat_dog_data)
    cnn = ConvNet("cat_and_dogs").to(device)
    out = cnn.train_(
        train, valid, submission, device, num_epoch=epoch, lr=lr, batchsize=batch)
    #pred = cnn.prediction(submission, batch, device)
    #dprint(pred)
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
