# -*- coding: utf-8 -*-

"""
Kaggle competition for assignement 1 of IFT6135.
"""

__authors__ = "Jimmy Leroux"
__version__ = "1.0"
__maintainer__ = "Jimmy Leroux"
__studentid__ = "1024610"

import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from cnn import ConvNet
from mlp import NN
if torch.cuda.is_available():
    import cupy as np
else:
    import numpy as np

CURRENT_DIR = os.getcwd()


class dataset(torch.utils.data.dataset.Dataset):

    def __init__(self, transforms=None):
        os.chdir("../dataset/trainset/")
        self.data = torchvision.datasets.ImageFolder(
            os.getcwd(), transform=transforms)
        os.chdir(CURRENT_DIR)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", help="Choose mlp cnndc or cnnmnist, default is both",
        type=str)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.manual_seed(10)
    plt.style.use('ggplot')
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    plt.rc('axes', labelsize=15)

    # Define the transformations.
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(), transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    # Create, load and split the datas.
    cat_dog_data = dataset(transforms=transform)
    split = [
        int(0.8*len(cat_dog_data)),
        len(cat_dog_data)-int(0.8*len(cat_dog_data))
        ]
    cd_train, cd_valid = torch.utils.data.dataset.random_split(
                cat_dog_data, split)

    mnist_train = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)
    
    # Create the models
    cnn_cd = ConvNet("cat_and_dog").to(device)
    neural_network = NN()
    cnn_mnist = ConvNet("mnist").to(device)

    if args.model is None:
        acc_train, acc_test = neural_network.train(mnist_train, mnist_test)
        out_dc = cnn_cd.train_(cd_train, cd_valid, device, num_epoch=70)
        out_mnist = cnn_mnist.train_(mnist_train, mnist_test, device)

    elif args.model == "mlp":
        print("MLP training:\n")
        acc_train, acc_test = neural_network.train(mnist_train, mnist_test)
    elif args.model == "cnndc":
        print("CNN training:\n")
        out_dc = cnn_cd.train_(cd_train, cd_valid, device, num_epoch=70)
    elif args.model == "cnnmnist":
        print("CNN training:\n")
        out_mnist = cnn_mnist.train_(mnist_train, mnist_test, device)

    # plt.figure()
    # plt.plot(range(1,n_epoch+1),e_train, 'sk-', label='Train')
    # plt.plot(range(1, n_epoch+1),e_valid, 'sr-', label='Valid')
    # plt.xlabel('Epoch')
    # plt.ylabel('Error')
    # plt.legend(fontsize=25)
    # plt.show()
