import os

import matplotlib.pyplot as plt
import cupy as cp


def plot_loss(datas, graphname, ylabel):
    plt.style.use('ggplot')
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('axes', labelsize=12)
    plt.figure()
    for name, data in datas.items():
        plt.plot(range(1,len(data)+1), data, label=name, marker="s")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig("../../graphs/"+graphname+".png", dpi=500)

def plot_grad_check(model, data):
    plt.style.use('ggplot')
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('axes', labelsize=12)
    N = {k*10**i for i in range(5) for k in [1,5]}
    inputs = model.transform_input(data[0][0].view(1, 784))
    labels = model.onehot(data[0][1].view(1, 1))
    maxes = []
    for n in sorted(N):
        epsilon = 1. / n
        grad_test, grad = model.grad_check(inputs, labels, epsilon)
        maxes.append(cp.max(cp.abs(grad_test-grad)))
    
    plt.figure()
    plt.loglog(sorted(N), maxes, marker="s")
    plt.xlabel("N")
    plt.ylabel(r"$\max_{1 \leq i \leq p} |\nabla^N_i"
        + r"-\frac{\partial L}{\partial \theta_i}|$")
    plt.tight_layout()
    plt.savefig("../../graphs/gradcheck.png", dpi=500)