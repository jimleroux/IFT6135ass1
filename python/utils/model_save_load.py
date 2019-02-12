import os

import torch


def load():
    CURRENT_PATH = os.getcwd()
    os.chdir("../models/saved_models/")
    model = torch.load(os.getcwd())
    return model

def save(model):
    CURRENT_PATH = os.getcwd()
    os.chdir("../models/saved_models/")
    torch.save(model, os.getcwd())
