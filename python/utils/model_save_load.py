import os

import torch


def load(name):
    CURRENT_PATH = os.getcwd()
    os.chdir("../models/saved_models/")
    model = torch.load(os.getcwd()+name)
    return model

def save(model, name):
    CURRENT_PATH = os.getcwd()
    os.chdir("../models/saved_models/")
    torch.save(model, os.getcwd()+name)
