# Code Reference: https://github.com/OptMLGroup/DeepBeerInventory-RL.
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import *


# plotting
def plotting(plt, data, colori, pltLabel):
    # plt.hold(True)

    for i in range(np.shape(data)[0]):
        plt.subplot(4, 5, 5 * i + 1)
        plt.plot(np.transpose(data[i])[0, :], np.transpose(data[i])[1, :], colori, label=pltLabel)
        plt.xlabel('Time')
        plt.ylabel('IL')
        plt.grid(True)

        plt.subplot(4, 5, 5 * i + 2)
        plt.plot(np.transpose(data[i])[0, :], np.transpose(data[i])[2, :], colori, label=pltLabel)
        plt.xlabel('Time')
        plt.ylabel('OO')
        plt.grid(True)

        plt.subplot(4, 5, 5 * i + 3)
        plt.plot(np.transpose(data[i])[0, :], np.transpose(data[i])[3, :], colori, label=pltLabel)
        plt.xlabel('Time')
        plt.ylabel('a')
        plt.grid(True)

        plt.subplot(4, 5, 5 * i + 4)
        plt.plot(np.transpose(data[i])[0, :], np.transpose(data[i])[5, :], colori, label=pltLabel)
        plt.xlabel('Time')
        plt.ylabel('OUTL')
        plt.grid(True)

        plt.subplot(4, 5, 5 * i + 5)
        plt.plot(np.transpose(data[i])[0, :], -1 * np.transpose(data[i])[4, :], colori, label=pltLabel)
        plt.xlabel('Time')
        plt.ylabel('r')
        plt.grid(True)

    return plt


def savePlot(players, curGame, Rsltdnn, RsltFrmu, RsltOptm, RsltRnd, config, m):
    #add title to plot
    if config.if_titled_figure:
        plt.suptitle(
            "sum OurPolicy=" + str(round(sum(Rsltdnn), 2)) + "; sum Strm=" + str(round(sum(RsltFrmu), 2)) +
            "; sum BS=" + str(round(sum(RsltOptm), 2)) + "; sum Rnd=" + str(round(sum(RsltRnd), 2)) + "\n" +
            "Ag OurPolicy=" + str([round(Rsltdnn[i], 2) for i in range(config.NoAgent)]) + "; Ag Strm=" +
            str([round(RsltFrmu[i], 2) for i in range(config.NoAgent)]) + "; Ag BS=" +
            str([round(RsltOptm[i], 2) for i in range(config.NoAgent)]) + "; Ag Rnd=" +
            str([round(RsltRnd[i], 2) for i in range(config.NoAgent)]),
            fontsize=12
        )

    #insert legend to the figure
    legend = plt.legend(bbox_to_anchor=(-1.4, -.165, 1., -.102), shadow=True, ncol=4)

    # configures spaces between subplots
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.5, hspace=.5)
    # save the figure
    path = os.path.join(config.figure_dir, 'saved_figures/')
    if not os.path.exists(path):
        os.mkdir(path)
    plt.savefig(path + str(curGame) + '-' + str(m) + '.png', format='png')
    print("figure" + str(curGame) + ".png saved in folder \"saved_figures\"")
    plt.close(curGame)
