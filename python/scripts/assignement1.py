import argparse
import sys
sys.path.insert(0, "../")

#import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import torch

import cats_dog
import cnn_mnist
import mlp_mnist

from utils.plot_functions import plot_loss


if __name__ == "__main__":
    np.random.seed(10)
    cp.random.seed(10)
    torch.manual_seed(10)   
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
        default=64, type=int
    )
    parser.add_argument(
        "--init", help="Choose the init mode",
        default="glorot", choices=["glorot", "uniform", "normal"],
        type=str
    )
    parser.add_argument(
        "--loading", help="Load a model or not",
        action="store_true"
    )
    parser.add_argument(
        "--grad_check", help="Specify to grad check.",
        action="store_true"
    )
    parser.add_argument(
        "--mlp_mnist", help="Specify to train.",
        action="store_true"
    )
    parser.add_argument(
        "--mlp_mnist_97", help="Specify to train and plot a 97% mlp.",
        action="store_true"
    )
    parser.add_argument(
        "--cnn_mnist", help="Specify to train.",
        action="store_true"
    )
    parser.add_argument(
        "--cnn_kaggle", help="Specify to train.",
        action="store_true"
    )
    parser.add_argument(
        "--plot_inits", help="Specify to plot inits graph.",
        action="store_true"
    )
    parser.add_argument(
        "--plot_kaggle", help="Specify to plot graphs for cnn_kaggle",
        action="store_true"
    )
    args = parser.parse_args()

    if args.mlp_mnist:
        args.init = "glorot"
        err_train, err_valid, loss_train_glorot, _ = mlp_mnist.main(args)
        if args.plot_inits:
            args.init = "uniform"
            args.grad_check = False
            _, _, loss_train_uniform, _ = mlp_mnist.main(args)
            args.init = "normal"
            _, _, loss_train_normal, _ = mlp_mnist.main(args)
            datas = {
                "glorot": loss_train_glorot,
                "uniform": loss_train_uniform,
                "normal": loss_train_normal
            }
            plot_loss(datas, graphname="initgraph", ylabel="Mean loss")
    if args.mlp_mnist_97:
        temp = args.epoch
        args.epoch = 50
        err_train, err_valid, _, _ = mlp_mnist.main(args)
        args.epoch = temp
        datas = {
            "Train": err_train,
            "Test": err_valid
        }
        plot_loss(datas, graphname="mlp97", ylabel="Error")
    if args.cnn_mnist:
        loss_train_cnn, _, err_train_cnn, err_valid_cnn = cnn_mnist.main(args)
        if args.mlp_mnist:
            datas = {
                "MLP train": err_train,
                "MLP valid": err_valid,
                "CNN train": err_train_cnn,
                "CNN valid": err_valid_cnn
            }
            plot_loss(datas, graphname="mlpvscnn", ylabel="Error")
    
    if args.cnn_kaggle:
        loss_train_kag, loss_valid_kag, err_train_kag, err_valid_kag  = cats_dog.main(args)
        if args.plot_kaggle:
            datas_loss = {
                "Train_loss": loss_train_kag,
                "Valid_loss": loss_valid_kag,
            }
            datas_err = {
                "Error_train": err_train_kag,
                "Error_valid": err_train_kag
            }
            plot_loss(datas_loss, graphname="cnn Kaggle Loss",  ylabel="Loss")
            plot_loss(datas_err, graphname="cnn Kaggle Error", ylabel="Error")
    
    plt.show()
