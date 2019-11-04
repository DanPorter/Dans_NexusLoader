"""
Assortment of useful plotting functions using matplotlib
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from . import functions_general as fg


'----------------------------Plot manipulation--------------------------'


def labels(ttl=None, xvar=None, yvar=None, zvar=None, legend=False, size='Normal', font='Times New Roman'):
    """
    Add formatted labels to current plot, also increases the tick size
    :param ttl: title
    :param xvar: x label
    :param yvar: y label
    :param zvar: z label (3D plots only)
    :param legend: False/ True, adds default legend to plot
    :param size: 'Normal' or 'Big'
    :param font: str font name, 'Times New Roman'
    :return: None
    """

    if size.lower() in ['big', 'large', 'xxl', 'xl']:
        tik = 30
        tit = 32
        lab = 35
        leg = 25
    else:
        # Normal
        tik = 18
        tit = 20
        lab = 22
        leg = 18

    plt.xticks(fontsize=tik, fontname=font)
    plt.yticks(fontsize=tik, fontname=font)
    plt.setp(plt.gca().spines.values(), linewidth=2)
    if plt.gca().get_yaxis().get_scale() != 'log':
        plt.ticklabel_format(useOffset=False)
        plt.ticklabel_format(style='sci', scilimits=(-3, 3))

    if ttl is not None:
        plt.gca().set_title(ttl, fontsize=tit, fontweight='bold', fontname=font)

    if xvar is not None:
        plt.gca().set_xlabel(xvar, fontsize=lab, fontname=font)

    if yvar is not None:
        plt.gca().set_ylabel(yvar, fontsize=lab, fontname=font)

    if zvar is not None:
        # Don't think this works, use ax.set_zaxis
        plt.gca().set_zlabel(zvar, fontsize=lab, fontname=font)

    if legend:
        plt.legend(loc=0, frameon=False, prop={'size': leg, 'family': 'serif'})


def saveplot(name, dpi=None, figure=None):
    """
    Saves current figure as a png in the home directory
    :param name: filename, including or expluding directory and or extension
    :param dpi: image resolution, higher means larger image size, default=matplotlib default
    :param figure: figure number, default = plt.gcf()
    :return: None

    E.G.
    ---select figure to save by clicking on it---
    saveplot('test')
    E.G.
    saveplot('c:\somedir\apicture.jpg', dpi=600, figure=3)
    """

    if type(name) is int:
        name = str(name)

    if figure is None:
        gcf = plt.gcf()
    else:
        gcf = plt.figure(figure)

    dir = os.path.dirname(name)
    file, ext = os.path.basename(name)

    if len(dir) == 0:
        dir = os.path.expanduser('~')

    if len(ext) == 0:
        ext = '.png'

    savefile = os.path.join(dir, file + ext)
    gcf.savefig(savefile, dpi=dpi)
    print('Saved Figure {} as {}'.format(gcf.number, savefile))


def newplot(*args, **kwargs):
    """
    Shortcut to creating a simple plot
    E.G.
      x = np.arange(-5,5,0.1)
      y = x**2
      newplot(x,y,'r-',lw=2,label='Line')
    """

    if 'linewidth' and 'lw' not in kwargs.keys():
        kwargs['linewidth'] = 2

    plt.figure(figsize=[12, 12])
    plt.plot(*args, **kwargs)

    plt.setp(plt.gca().spines.values(), linewidth=2)
    plt.xticks(fontsize=25, fontname='Times New Roman')
    plt.yticks(fontsize=25, fontname='Times New Roman')
    plt.ticklabel_format(useOffset=False)
    plt.ticklabel_format(style='sci', scilimits=(-3, 3))


def multiplot(xvals, yvals=None, datarange=None, cmap='jet', labels=None, marker=None):
    """
    Shortcut to creating a simple multiplot with either colorbar or legend
    E.G.
      x = np.arange(-5,5,0.1)
      ys = [x**2, 1+x**2, 2+x**2, 3+x**2, 4+x**2]
      datarange = [0,1,2,3,4]
      multiplot(x, ys, datarange, cmap='winter')
    OR:
      x = np.arange(-5,5,0.1)
      ys = [x**2, 1+x**2, 2+x**2, 3+x**2, 4+x**2]
      labels = ['x*x','2+x*x','3+x*x','4+x*x']
      multiplot(x, ys, labels=labels)
    """

    if yvals is None:
        yvals = xvals
        xvals = []
    yvals = np.asarray(yvals)
    xvals = np.asarray(xvals)

    if datarange is None:
        datarange = range(len(yvals))
    datarange = np.asarray(datarange, dtype=np.float)

    cm = plt.get_cmap(cmap)
    colrange = (datarange - datarange.min()) / (datarange.max() - datarange.min())

    if marker is None:
        marker = ''
    linearg = '-' + marker

    plt.figure(figsize=[12, 12])
    for n in range(len(datarange)):
        col = cm(colrange[n])
        if len(xvals) == 0:
            plt.plot(yvals[n], linearg, lw=2, color=col)
        elif len(xvals.shape) == 1:
            plt.plot(xvals, yvals[n], linearg, lw=2, color=col)
        else:
            plt.plot(xvals[n], yvals[n], linearg, lw=2, color=col)

    plt.setp(plt.gca().spines.values(), linewidth=2)
    plt.xticks(fontsize=25, fontname='Times New Roman')
    plt.yticks(fontsize=25, fontname='Times New Roman')
    plt.ticklabel_format(useOffset=False)
    plt.ticklabel_format(style='sci', scilimits=(-3, 3))

    if labels is None:
        # Add Colorbar
        sm = plt.cm.ScalarMappable(cmap=cm)
        sm.set_array(datarange)
        cbar = plt.colorbar(sm)
        # cbar.set_label('variation [unit]', fontsize=24, fontweight='bold', fontname='Times New Roman')
    else:
        # Add legend
        plt.legend(labels, loc=0, frameon=False, prop={'size': 20, 'family': 'serif'})


def newplot3(*args, **kwargs):
    """
    Shortcut to creating a simple 3D plot
    Automatically tiles 1 dimensional x and y arrays to match 2D z array,
    assuming z.shape = (len(x),len(y))

    E.G.
      newplot3([1,2,3,4],[9,8,7],[[2,4,6],[8,10,12],[14,16,18],[20,22,24]],'-o')
    """

    if 'linewidth' and 'lw' not in kwargs.keys():
        kwargs['linewidth'] = 2

    fig = plt.figure(figsize=[12, 12])
    ax = fig.add_subplot(111, projection='3d')

    x = np.asarray(args[0], dtype=np.float)
    y = np.asarray(args[1], dtype=np.float)
    z = np.asarray(args[2], dtype=np.float)

    if z.ndim == 2:
        if x.ndim < 2:
            x = np.tile(x, z.shape[1]).reshape(z.T.shape).T
        if y.ndim < 2:
            y = np.tile(y, z.shape[0]).reshape(z.shape)

        # Plot each array independently
        for n in range(len(z)):
            ax.plot(x[n], y[n], z[n], *args[3:], **kwargs)
    else:
        ax.plot(*args, **kwargs)


def sliderplot(YY, X=None, slidervals=None, *args, **kwargs):
    """
    Shortcut to creating a simple 2D plot with a slider to go through a third dimension
    YY = [nxm]: y axis data (initially plots Y[0,:])
     X = [n] or [nxm]:  x axis data (can be 1D or 2D, either same length or shape as Y)
     slidervals = None or [m]: Values to give in the slider

    E.G.
      sliderplot([1,2,3],[[2,4,6],[8,10,12],[14,16,18],[20,22,24]],slidervals=[3,6,9,12])
    """

    if 'linewidth' and 'lw' not in kwargs.keys():
        kwargs['linewidth'] = 2

    fig = plt.figure(figsize=[12, 12])

    X = np.asarray(X, dtype=np.float)
    Y = np.asarray(YY, dtype=np.float)
    if slidervals is None:
        slidervals = range(Y.shape[0])
    slidervals = np.asarray(slidervals, dtype=np.float)

    if X.ndim < 2:
        X = np.tile(X, Y.shape[0]).reshape(Y.shape)

    plotline, = plt.plot(X[0, :], Y[0, :], *args, **kwargs)
    plt.axis([X.min(), X.max(), Y.min(), Y.max()])
    plt.subplots_adjust(bottom=0.2)
    ax = plt.gca()

    " Create slider on plot"
    axsldr = plt.axes([0.15, 0.05, 0.65, 0.03], axisbg='lightgoldenrodyellow')

    sldr = plt.Slider(axsldr, '', 0, len(slidervals) - 1)
    txt = axsldr.set_xlabel('{} [{}]'.format(slidervals[0], 0), fontsize=18)

    plt.sca(ax)

    " Slider update function"

    def update(val):
        "Update function for pilatus image"
        pno = int(np.floor(sldr.val))
        plotline.set_xdata(X[pno, :])
        plotline.set_ydata(Y[pno, :])
        txt.set_text('{} [{}]'.format(slidervals[pno], pno))
        plt.draw()
        plt.gcf().canvas.draw()
        # fig1.canvas.draw()

    sldr.on_changed(update)


def sliderplot2D(ZZZ, XX=None, YY=None, slidervals=None, *args, **kwargs):
    """
    Shortcut to creating an image plot with a slider to go through a third dimension
    ZZZ = [nxmxo]: z axis data
     XX = [nxm] or [n]:  x axis data
     YY = [nxm] or [m]: y axis data
     slidervals = None or [o]: Values to give in the slider

    if XX and/or YY have a single dimension, the 2D values are generated via meshgrid

    E.G.
      sliderplot([1,2,3],[[2,4,6],[8,10,12],[14,16,18],[20,22,24]],slidervals=[3,6,9,12])
    """

    if 'linewidth' and 'lw' not in kwargs.keys():
        kwargs['linewidth'] = 2

    fig = plt.figure(figsize=[12, 12])

    ZZZ = np.asarray(ZZZ, dtype=np.float)

    if slidervals is None:
        slidervals = range(ZZZ.shape[2])
    slidervals = np.asarray(slidervals, dtype=np.float)

    if XX is None:
        XX = range(ZZZ.shape[1])
    if YY is None:
        YY = range(ZZZ.shape[0])
    XX = np.asarray(XX, dtype=np.float)
    YY = np.asarray(YY, dtype=np.float)
    if XX.ndim < 2:
        XX, YY = np.meshgrid(XX, YY)

    p = plt.pcolormesh(XX, YY, ZZZ[:, :, 0])
    # p.set_clim(cax)

    plt.subplots_adjust(bottom=0.2)
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.autoscale(tight=True)

    " Create slider on plot"
    axsldr = plt.axes([0.15, 0.05, 0.65, 0.03], axisbg='lightgoldenrodyellow')

    sldr = plt.Slider(axsldr, '', 0, len(slidervals) - 1)
    txt = axsldr.set_xlabel('{} [{}]'.format(slidervals[0], 0), fontsize=18)

    plt.sca(ax)

    " Slider update function"

    def update(val):
        "Update function for pilatus image"
        pno = int(np.round(sldr.val))
        p.set_array(ZZZ[:-1, :-1, pno].ravel())
        txt.set_text('{} [{}]'.format(slidervals[pno], pno))
        plt.draw()
        plt.gcf().canvas.draw()
        # fig1.canvas.draw()

    sldr.on_changed(update)

