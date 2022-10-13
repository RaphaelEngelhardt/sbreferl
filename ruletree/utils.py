"""Collection of (mainly plotting) utilities in the rule learning context"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def describe_samples(samplesfile, outfile=None, show=False):
    """
    Computes and returns descriptive statistics of input samplesfile
    :param samplesfile: text file containing the samples
    :param outfile: optional filename for plots of histogram
    :param show: boolean default False, if True shows histogram
    :return: pandas DataFrame containing descriptive statistics of samplesfile
    """

    dataset = pd.read_feather(samplesfile)
    dataset.drop(columns='episode_return', inplace=True)
    dataset.hist()

    plt.tight_layout()
    if outfile is None:
        outfile = '.'.join(samplesfile.split('.')[:-1]) + '_hist.pdf'
    plt.savefig(outfile)
    if show:
        plt.show()
    plt.close()
    
    stat = pd.DataFrame({'median': dataset.median(),
                         'mean': dataset.mean(),
                         'std': dataset.std()})
    print(f'{samplesfile} contains {len(dataset)} samples.')
    print(stat)

    return stat


def plot_learning(hist, outfile=None, show=True):
    """
    Produces and optionally saves and/or displays a plot showing the
    progress of training
    :param dict hist: As returned by deep RL learning methods
    :param outfile: Filename for saving produced plot. If None plot is
                    not saved
    :param bool show: Specifies whether plot should be shown or not
    """

    y = hist['episode_rewards']
    x = np.arange(1, len(y)+1)

    sns.set_style("darkgrid")
    sns.lineplot(x=x, y=y)
    plt.xlabel("Training episodes")
    plt.ylabel("Return")

    if outfile is not None:
        plt.savefig(outfile)
    if show:
        plt.show()
    plt.close()
