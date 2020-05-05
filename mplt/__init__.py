"""Set of niceties wrapping matplotlib
"""
__author__ = 'Craig Stringham'
__version__ = '2.0.7'
# in order to pass through and un-overloaded functions to pyplot
import sys
from pathlib import Path
import logging

import numpy as np

from matplotlib.pyplot import *
import matplotlib.pyplot as plt
from matplotlib.path import Path as mpPath

logger = logging.getLogger(__name__)


## modify  matplotlib's rcParams so that it cycles through linestyles as well
linecolors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
rcParams['axes.prop_cycle'] = (
    "cycler('linestyle', ['-', '--', ':']) * cycler('color', {})".format(linecolors))
rcParams['image.aspect'] = 'auto'


def myfig(figname=None, showtitle=True, clearfig=True, **kwargs):
    """Create figures optionally tracked by name with the name set as the title
    """
    if figname is None:
        figname = np.random.randint(10000)
    if clearfig:
        plt.close(figname)
    fig = plt.figure(figname, **kwargs)
    if showtitle:
        plt.suptitle(figname)
    return fig


def myshow():
    """Smart checking for plotting depending on whether running in a notebook,
    interactively, or from a script"""
    if plt.get_backend().find('inline') > 0:
        # don't call show when run from notebook
        return
    try:
        # don't block when running interactively
        sys.ps1
        show(False)
    except AttributeError:
        # not running interactively
        show(True)


def plot3(*args, **kwargs):
    """plot a 3 dimensional line"""
    ax = gca(projection='3d')
    ax.plot(*args, **kwargs)
    return ax


def format_coord(x, y, X, extent=None):
    """Set the format of the coordinates that are read when hovering
    over on a plot.
    """
    numrows, numcols = X.shape
    if extent is not None:
        col = int((x - extent[0]) / (extent[1] - extent[0]) * numcols + 0.5)
        row = int((y - extent[3]) / (extent[2] - extent[3]) * numrows + 0.5)
    else:
        col = int(x + 0.5)
        row = int(y + 0.5)
    if col >= 0 and col < numcols and row >= 0 and row < numrows:
        z = X[row, col]
        return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x, y, z)
    else:
        return 'x=%1.4f, y=%1.4f, z=E!' % (x, y)


def imshow(*args, **kwargs):
    """overload matplotlib's imshow to plot the coordinates with the mouse
    """
    ax = plt.imshow(*args, **kwargs)
    ax.format_coord = lambda x, y: format_coord(x, y, args[0])
    return ax


def pcolor(*args, **kwargs):
    """overload matplotlib's pcolor to plot the coordinates with the mouse
    """
    plt.pcolor(*args, **kwargs)
    ax = gca()
    ax.format_coord = lambda x, y: format_coord(x, y, args[0])


def pcolormesh(*args, **kwargs):
    """overload matplotlib's pcolormesh to plot the coordinates with the mouse
    """
    ax = plt.pcolormesh(*args, **kwargs)
    ax.format_coord = lambda x, y: format_coord(x, y, args[0])
    return ax


def imenhance(x, percentiles=[5, 95]):
    """clip an image the the given percentile levels
    """
    isf = np.isfinite(x)
    (vmin, vmax) = np.percentile(x[isf], percentiles)
    y = x
    y[x < vmin] = vmin
    y[x > vmax] = vmax
    return y


def imshowe(x, percentiles=[5, 95], **kwargs):
    """plot an image with the max and min set to the given percentiles
    """
    isf = np.isfinite(x)
    if np.any(isf):
        (vmin, vmax) = np.percentile(x[isf], percentiles)
    else:
        logger.warning('No finite numbers in array')
        vmin = 0
        vmax = 1
    if 'percentiles' in kwargs:
        del(kwargs['percentiles'])
    return imshow(x, vmin=vmin, vmax=vmax, **kwargs)


def im2(X, Y, xtitle, ytitle, **kwargs):
    ax1 = plt.subplot(121)
    title(xtitle)
    imshow(X, **kwargs)
    colorbar()
    #ax1.format_coord= lambda x,y:format_coord(x,y,X)
    ax2 = plt.subplot(122, sharex=ax1, sharey=ax1)
    title(ytitle)
    imshow(Y, **kwargs)
    colorbar()
    #ax2.format_coord= lambda x,y:format_coord(x,y,Y)
    plt.subplot(121)
    return ax1, ax2


def pcolor2(x, y, xtitle, ytitle, **kwargs):
    ax = plt.subplot(121)
    title(xtitle)
    pcolor(x, **kwargs)
    colorbar()
    ax2 = plt.subplot(122, sharex=ax, sharey=ax)
    title(ytitle)
    pcolor(y, **kwargs)
    colorbar()
    return ax, ax2


def imimshow(x, **kwargs):
    """Display real and imaginary parts of an image with linked axes
    """
    if not np.any(np.iscomplex(x)):  # and x.dtype != np.complex64:
        imshow(x, **kwargs)
    else:
        im2(x.real, x.imag, 'real', 'imag', **kwargs)


def mpimshow(x, **kwargs):
    """Display magnitude and phase an image with linked axes
    """
    if x.size < 1:
        logger.debug('Warning empty array supplied')
        return
    if not np.any(np.iscomplex(x)):
        return imshow(x, **kwargs)
    else:
        return im2(np.abs(x), angle_offset(x), 'Magnitude', 'Phase', **kwargs)


def dbpimshow(x, **kwargs):
    """Display magnitude in deciblels and phase an image with linked axes
    """
    if np.all(np.isreal(x)) and x.dtype != np.complex64:
        imshow(x, **kwargs)
    else:
        im2(20 * np.log10(np.abs(x)), angle_offset(x),
            'Magnitude(dB)', 'Phase', **kwargs)


def impcolor(x, **kwargs):
    """Display real and imaginary parts of an image with linked axes
    using pcolor
    """
    if np.all(np.isreal(x)) and x.dtype != np.complex64:
        pcolor(x, **kwargs)
    else:
        pcolor2(x.real, x.imag, 'real', 'imag', **kwargs)


def mppcolor(x, **kwargs):
    """Display magnitude and phase of an image with linked axes
    using pcolor
    """
    if np.all(np.isreal(x)) and x.dtype != np.complex64:
        pcolor(x, **kwargs)
    else:
        pcolor2(np.abs(x), angle_offset(x), 'Magnitude', 'Phase', **kwargs)


def implot(x, y=None, **kwargs):
    """Plot a complex signal with real and imaginary in different
    subplots but with linked x axes
    """
    if y is None:
        y = x
        x = np.arange(x.shape[0])
    ax = subplot(211)
    plot(x, y.real, **kwargs)
    title('real')
    setp(ax.get_xticklabels(), visible=False)    
    subplot(212, sharex=ax, sharey=ax)
    plot(x, y.imag, **kwargs)
    title('imag')


def mpplot(x, y=None, **kwargs):
    """Plot a complex signal with magnitude and phase in different
    subplots but with linked x axes
    """
    unwrap = kwargs.pop('unwrap', False)
    if y is None:
        y = x
        x = np.arange(x.shape[0])
    ax = subplot(211)
    plot(x, np.abs(y), **kwargs)
    title('magnitude')

    setp(ax.get_xticklabels(), visible=False)
    ax2 = subplot(212, sharex=ax)
    ang = np.angle(y)
    if unwrap:
        ang = np.unwrap(ang, axis=0)
    plot(x, ang, **kwargs)
    title('phase')
    return ax, ax2


def dbpplot(x, y=None, **kwargs):
    """Plot a complex signal with magnitude in dB and phase in different
    subplots but with linked x axes
    """
    unwrap = kwargs.pop('unwrap', False)
    if y is None:
        y = x
        x = np.arange(x.shape[0])
    ax = subplot(211)
    plot(x, 20 * np.log10(np.abs(y)), **kwargs)
    title('magnitude (dB)')
    setp(ax.get_xticklabels(), visible=False)
    subplot(212, sharex=ax)
    ang = np.angle(y)
    if unwrap:
        ang = np.unwrap(ang, axis=0)
    plot(x, ang, **kwargs)
    title('phase')


def plotyy(*args, **kwargs):
    """Plot two lines on the same subplot but with separate y axes on the
    left and right

    modified from http://matplotlib.org/examples/api/two_scales.html"""
    ax1 = gca()
    if len(args) == 4:
        t1, s1, t2, s2 = args
    if len(args) == 3:
        t, s1, s2 = args
        t1 = t
        t2 = t
    elif len(args) == 2:
        s1, s2 = args
        t1 = np.arange(len(s1))
        t2 = np.arange(len(s2))
    else:
        raise Exception(
            'I don''t know how to handle {} arguments'.format(len(args)))

    color0 = kwargs.pop('color0', 'b')
    color1 = kwargs.pop('color1', 'r')

    ax1.plot(t1, s1, color=color0, **kwargs)
    # Make the y-axis label and tick labels match the line color.
    ax1.set_ylabel('f(x)', color=color0)
    ax1.set_ylim(np.percentile(s1, [1, 99]))
    for tl in ax1.get_yticklabels():
        tl.set_color(color0)
    ax2 = ax1.twinx()
    ax2.plot(t2, s2, linestyle=':', color=color1)
    ax2.set_ylabel('g(x)', color=color1)
    ax2.set_ylim(np.percentile(s2, [1, 99]))
    ax2.ticklabel_format(useOffset=False)
    for tl in ax2.get_yticklabels():
        tl.set_color(color1)
    return ax1, ax2


def imshow_overlay(z1, z2, **kwargs):
    """Overlay one image ontop of another using alpha settings
    """
    ax = gca()
    alpha = 0.5
    if 'alpha' in kwargs:
        alpha = kwargs['alpha']
        del(kwargs['alpha'])
    ax.imshow(z1, cmap=cm.gray, **kwargs)
    ax.imshow(z2, alpha=alpha, cmap=cm.jet, **kwargs)


def multiimage(*args, **kwargs):
    """multiimage plots multiple images in one figure.
    """
    colorbar = kwargs.pop('colorbar', False)
    altpref = kwargs.pop('altpref', False)
    layout = kwargs.pop('layout', None)
    subtitles = kwargs.pop('subtitles', None)
    labels = kwargs.pop('labels', None)
    subtitles = subtitles or labels
    pscale = kwargs.pop('pscale', None)
    noticks = kwargs.pop('noticks', None)
    noxticks = kwargs.pop('noxticks', None)

    numplot = len(args)
    prefsizes = [(1, 1), (1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (2, 3),
                 (3, 3), (3, 3), (3, 3), (3, 4), (3, 4), (3, 4),
                 (4, 4), (4, 4), (4, 4), (4, 4)]
    if args[0].shape[0] < 1.5 * args[0].shape[1]:
        prefsizes = [(y, x) for x, y in prefsizes]

    if numplot > len(prefsizes):
        w = np.ceil(np.sqrt(numplot))
        h = (w - 1)
        if 2 * h < numplot:
            h += 1

    elif layout is not None:
        (w, h) = layout
    else:
        (w, h) = prefsizes[numplot]

    if altpref:
        h, w = (w, h)

    for count, img in enumerate(args):
        if count == 0:
            ax1 = subplot(w, h, 1)
        else:
            subplot(w, h, count + 1, sharex=ax1, sharey=ax1)
        if pscale is not None:
            kwargs['vmin'], kwargs['vmax'] = np.percentile(
                img.flatten(), pscale)
        imshow(img, **kwargs)
        if subtitles is not None:
            title(subtitles[count])
        if colorbar:
            plt.colorbar()

        if noticks or noxticks:
            ax1.get_xaxis().set_visible(False)
            ax1.xaxis.set_ticklabels([])
        if noticks:
            ax1.get_yaxis().set_visible(False)
            ax1.yaxis.set_ticklabels([])


def cubeimage(cube, **kwargs):
    """given a 3d array display each plane """
    arglist = [cube[0, :, :]]
    for k in range(cube.shape[0] - 1):
        arglist.append(cube[k + 1, :, :])
    multiimage(*arglist, **kwargs)


def fancy_hist(data):
    """ from http://stackoverflow.com/a/6353051/1840190 """
    from matplotlib.ticker import FormatStrFormatter
    ax = gca()
    counts, bins, patches = ax.hist(
        data, 40, facecolor='yellow', edgecolor='gray')
    # Set the ticks to be at the edges of the bins.
    ax.set_xticks(bins)
    # Set the xaxis's tick labels to be formatted with 1 decimal place...
    ax.xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))

    # Change the colors of bars at the edges...
    twentyfifth, seventyfifth = np.percentile(data, [25, 75])
    for patch, rightside, leftside in zip(patches, bins[1:], bins[:-1]):
        if rightside < twentyfifth:
            patch.set_facecolor('green')
        elif leftside > seventyfifth:
            patch.set_facecolor('red')

    # Label the raw counts and the percentages below the x-axis...
    bin_centers = 0.5 * np.diff(bins) + bins[:-1]
    for count, x in zip(counts, bin_centers):
        # Label the raw counts
        ax.annotate(str(count), xy=(x, 0), xycoords=('data', 'axes fraction'),
                    xytext=(0, -18), textcoords='offset points', va='top', ha='center')

        # Label the percentages
        percent = '%0.0f%%' % (100 * float(count) / counts.sum())
        ax.annotate(percent, xy=(x, 0), xycoords=('data', 'axes fraction'),
                    xytext=(0, -32), textcoords='offset points', va='top', ha='center')

    # Give ourselves some more room at the bottom of the plot
    subplots_adjust(bottom=0.15)


def add_color_bar(fig, ax, im, frac=.08):
    axp = ax.get_position()
    # left bottom width height
    cax = fig.add_axes(
        [axp.x1, axp.y0, frac * (axp.x1 - axp.x0), axp.y1 - axp.y0])
    fig.colorbar(im, cax=cax)


def saveall(outputdir=None, extension='.png'):
    """Save all of the open figures to the given directory.
    If the pickle format is used (extension='.pickle') a script to load and
    plot the images is also generated.

    args:
    outputdir (str): directory to save the output image files
    extension (str): image extension defining the image type
    """
    outputdir = Path() if outputdir is None else Path(outputdir)
    outputdir.mkdir(exist_ok=True)
    pickleFig = False
    if extension.find('.pickle') == 0:
        pickleFig = True

    if pickleFig:
        figscript = '#!/usr/bin/env python\n'
        figscript += 'import matplotlib.pyplot as plt\nimport pickle\n'

    for f in plt.get_fignums():
        fig = plt.figure(f)
        figname = fig.get_label().replace(' ', '_')
        figname = 'FIG_' + figname.replace('/', '_-')
        if pickleFig:
            import pickle
            picklename = '{}.pickle'.format(figname)
            with open(outputdir / picklename, 'wb') as fout:
                pickle.dump(fig, fout)
            figscript += 'ax = pickle.load(open(r''{}'',''rb''))\n'.format(
                picklename)
        else:
            plt.savefig(outputdir / '{}{}'.format(figname, extension))
    if pickleFig:
        figscript += 'plt.show()\n'
        figscriptfile = open(outputdir / 'plotfigs.py', 'w')
        figscriptfile.write(figscript)
        figscriptfile.close()


def polyContains(polyBounds, points, plot=False):
    """find the points that are contained in the bounding polygon"""
    boxcodes = [mpPath.MOVETO]
    polyBounds = np.append(polyBounds, polyBounds[0]).reshape((-1, 2))
    for k in range(len(polyBounds) - 1):
        boxcodes.append(mpPath.LINETO)
    boxcodes[-1] = mpPath.CLOSEPOLY
    bbPoly = mpPath(polyBounds, boxcodes)

    if plot:  # debugging patch

        import matplotlib as mpl
        fig = myfig('debug patch')
        ax = fig.add_subplot(111)
        patch = mpl.patches.PathPatch(
            bbPoly, facecolor='orange', lw=2, alpha=0.5)
        ax.add_patch(patch)
        fpa = np.asarray(points)
        scatter(fpa[:, 0], fpa[:, 1])
        myshow()

    withinbox = bbPoly.contains_points(points)
    return withinbox

## various utilities
def mbin(bin_edges):
    """Return the middle of bins"""
    return bin_edges[:-1] + .5 * np.diff(bin_edges)


def angle_offset(img):
    """Calculate the angle, but wrap around the mean angle"""
    m = np.nanmean(img)
    m /= abs(m)
    out = np.angle(img * m.conj()) + np.angle(m)
    return out
