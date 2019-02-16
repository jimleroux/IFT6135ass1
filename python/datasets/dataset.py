import os
import sys

import torch
import torchvision


class Dataset(torch.utils.data.dataset.Dataset):
    """
    Class implementing a pytorch dataset. We need to define the __getitem__
    and __len__. The dataset is automaticly labeled by the folder name.
    """
    def __init__(self, transforms=None):
        self.data = torchvision.datasets.ImageFolder(
            "../../dataset/trainset/", transform=transforms)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

class TestDataset(torch.utils.data.dataset.Dataset):
    """
    Class implementing a pytorch dataset. We need to define the __getitem__
    and __len__. The dataset is automaticly labeled by the folder name.
    """
    def __init__(self, transforms=None):
        self.data = torchvision.datasets.ImageFolder(
            "../../dataset/testset/", transform=transforms)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def find_classes(dir):
    """
    Finds the class folders in a dataset.
    Args:
        dir (string): Root directory path.
    Returns:
        tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
    Ensures:
        No class is a subdirectory of another.
    """
    if sys.version_info >= (3, 5):
        # Faster and available in Python 3.5 and above
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    else:
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images

class DatasetFolderWithPath(torch.utils.data.dataset.Dataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None):
        extensions = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
        classes, class_to_idx = find_classes(root)
        samples = make_dataset(
            root, class_to_idx, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

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
