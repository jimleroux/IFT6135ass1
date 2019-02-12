import matplotlib.pyplot as plt


def plot_inits(**datas):
    plt.style.use('ggplot')
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('axes', labelsize=12)
    plt.figure()
    for name, data in datas.items():
        plt.plot(data, label=name, marker="s")
    plt.legend()
    