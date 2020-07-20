"""
File for automatically plotting scans
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from . import functions_general as fg
from . import functions_plotting as fp


__version__ = 0.3
__date__ = '15/07/20'


def plotline(axis, xvals, yvals, yerrors=None, fmt='-o', **kwargs):
    """
    Create a plot object on the given axis
    :param axis: plt.axis object, e.g. plt.gca()
    :param xvals: array : values for the x-axis
    :param yvals: array : values for the y-axis
    :param yerrors: None or array : values for y-axis errors
    :param fmt: line format as in plt.plot
    :return: [pl] list of matplotlib Line2D objects
    """

    if 'lw' not in kwargs.keys():
        kwargs['lw'] = 2
    if 'ms' not in kwargs.keys():
        kwargs['ms'] = 10

    if yerrors is None:
        ln = axis.plot(xvals, yvals, fmt, **kwargs)
        lines = ln
    else:
        ln, xbar, ybar = axis.errorbar(xvals, yvals, yerrors, fmt=fmt, **kwargs)
        lines = [ln] + list(xbar) + list(ybar)
    return lines


def scanplot(xvals, yvals, yerrors=None, title=None, xlabel=None, ylabel=None, legend=None, **kwargs):
    """
    Create a nice matplotlib plot
    returns axis object
     if yvals = None, don't plot
    """

    plt.figure(figsize=[8, 6], dpi=90)
    ax = plt.subplot(111)
    if yvals is not None:
        plotline(ax, xvals, yvals, yerrors, **kwargs)

    if title is not None:
        plt.title(title, fontsize=12)
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=18)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=18)
    if legend is not None:
        plt.legend(legend, loc=0, frameon=False, prop={'size': 16, 'family': 'serif'})

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.setp(plt.gca().spines.values(), linewidth=2)
    if plt.gca().get_yaxis().get_scale() != 'log':
        plt.ticklabel_format(useOffset=False)
        plt.ticklabel_format(style='sci',scilimits=(-3, 3))
    return ax


def scansplot(xvals, yvals, yerrors=None, title=None, xlabel=None, ylabel=None, legend=None, colours=None, **kwargs):
    """
    Create a nice matplotlib plot of multiple scans
    xvals, yvals, yerrors, legend and colours must be either None or lists of the same length
    returns axis object
    """

    plt.figure(figsize=[8, 6], dpi=90)
    ax = plt.subplot(111)
    for n in range(len(yvals)):
        if colours:
            kwargs['c'] = colours[n]
        if yerrors is None:
            plotline(ax, xvals[n], yvals[n], **kwargs)
        else:
            plotline(ax, xvals[n], yvals[n], yerrors[n], **kwargs)

    if title is not None:
        plt.title(title, fontsize=12)
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=18)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=18)
    if legend is not None:
        plt.legend(legend, loc=0, frameon=False, prop={'size': 16, 'family': 'serif'})

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.setp(plt.gca().spines.values(), linewidth=2)
    if plt.gca().get_yaxis().get_scale() != 'log':
        plt.ticklabel_format(useOffset=False)
        plt.ticklabel_format(style='sci',scilimits=(-3, 3))
    return ax


def imageplot(image_data, title=None, cmap=None, clim=None):
    """
    Plot image data as a matplotlib figure
    :param image_data: [n,m] numpy array
    :param title: title to display above image
    :param cmap: colormap
    :param clim: colour limits
    :return: None
    """

    plt.figure(figsize=[8, 6], dpi=90)
    plt.imshow(image_data, cmap)

    if clim is not None:
        plt.clim(clim)
    if title is not None:
        plt.title(title, fontsize=12)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.setp(plt.gca().spines.values(), linewidth=2)
