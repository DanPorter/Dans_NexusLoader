"""
File for automatically plotting scans
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from . import functions_general as fg
from . import functions_plotting as fp


def scanplot(xvals, yvals, yerrors=None, title=None, xlabel=None, ylabel=None, legend=None):
    """
    Create a nice matplotlib plot
    """

    fmt = '-o'

    xvals = np.array(xvals)
    yvals = np.array(yvals).reshape(1,-1)
    if yerrors is not None:
        yerrors = np.array(yerrors).reshape(1,-1)
    ysets = yvals.shape[0]

    plt.figure(figsize=[8,6], dpi=90)
    for n in range(ysets):
        if yerrors is None:
            plt.plot(xvals, yvals[n], fmt, lw=2, ms=8)
        else:
            plt.errorbar(xvals, yvals[n], yerrors[n], fmt=fmt, lw=2, ms=8)
    if title is not None:
        plt.title(title, fontsize=12)
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=18)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=18)
    if legend is not None:
        plt.legend(legend, loc=0, frameon=False, prop={'size':16,'family':'serif'})

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.setp(plt.gca().spines.values(), linewidth=2)
    if plt.gca().get_yaxis().get_scale() != 'log':
        plt.ticklabel_format(useOffset=False)
        plt.ticklabel_format(style='sci',scilimits=(-3,3))

