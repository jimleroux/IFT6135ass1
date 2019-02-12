import os

import matplotlib.pyplot as plt


def plot_inits(datas):
    plt.style.use('ggplot')
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('axes', labelsize=12)
    plt.figure()
    for name, data in datas.items():
        plt.plot(range(1,len(data)+1), data, label=name, marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Mean loss")
    plt.legend()
    plt.savefig("../../graphs/initgraph.png", dpi=500)
