import os

import torch
import torchvision


class Dataset(torch.utils.data.dataset.Dataset):
    """
    Class implementing a pytorch dataset. We need to define the __getitem__
    and __len__.
    """
    def __init__(self, transforms=None):
        self.data = torchvision.datasets.ImageFolder(
            "../../dataset/trainset/", transform=transforms)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

def import_mnist(transform):
    mnist_train = torchvision.datasets.MNIST(root='./data', train=True, 
                                             download=True, transform=transform)
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False,
                                            download=True, transform=transform)
    return mnist_train, mnist_test

def data_split(dataset, ratio=0.8):
    split = [
        int(ratio*len(dataset)),len(dataset)-int(ratio*len(dataset))
        ]
    train, valid = torch.utils.data.dataset.random_split(
        dataset, split)
    return train, valid
