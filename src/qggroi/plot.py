"""Functions for plotting QGG data.

Typical workflow is for these functions to take the required data as input and
return a matplotlib Figure. The user can then choose to display the figure
interactively (with matplotlib.pyplot.show()) or save the figure (handled by
the included save_figure method).
"""
from typing import Callable, NamedTuple, Optional
from math import sqrt

import pandas
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

from qggroi.roi import ROI


class MultiPlot(NamedTuple):
    fig: Figure
    ax_main: Axes
    ax_left: Axes
    ax_bottom: Axes
    ax_cbar: Axes


def plot_image_and_profiles(data: np.ndarray) -> Figure:
    fig, ax_main, ax_xprofile, ax_yprofile, ax_cbar = image_profile_multiplot(data)
    im = ax_main.imshow(data, origin='lower', aspect='auto')
    ax_xprofile.step(*profile(data, 'x'), color='black')
    ax_yprofile.step(*profile(data, 'y'), color='black')
    cbar = plt.colorbar(im, cax=ax_cbar)
    cbar.set_label('Counts', ha='right', y=1.0)
    return fig


def plot_roi(roi: ROI, data: np.ndarray) -> Figure:
    fig, ax_main, ax_xprofile, ax_yprofile, ax_cbar = image_profile_multiplot(data)
    im = ax_main.imshow(data, origin='lower', aspect='auto')
    roi_rectangle = Rectangle(
        (roi.x_bounds[0], roi.y_bounds[0]),
        roi.x_bounds[1] - roi.x_bounds[0],
        roi.y_bounds[1] - roi.y_bounds[0],
        edgecolor='r',
        facecolor='none'
    )
    ax_main.add_patch(roi_rectangle)
    ax_xprofile.step(*profile(data, 'x', roi.y_bounds), color='black')
    ax_xprofile.axvspan(roi.x_bounds[0], roi.x_bounds[1], facecolor=(1, 0, 0, .5))
    ax_yprofile.step(*profile(data, 'y'), color='black')
    ax_yprofile.axhspan(roi.y_bounds[0], roi.y_bounds[1], facecolor=(1, 0, 0, .5))
    cbar = plt.colorbar(im, cax=ax_cbar)
    cbar.set_label('Counts', ha='right', y=1.0)
    ax_xprofile.set_xlabel('$x$ (within $y$ region)', ha='right', x=1.0)
    return fig


def plot_image(data: np.ndarray) -> Figure:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_axes([.1, .1, .8, .85])
    img = ax.imshow(data, origin='lower', aspect='auto')
    cbar = plt.colorbar(img)
    cbar.set_label('Counts', ha='right', y=1.0)
    ax.set_xlabel('$x$', ha='right', x=1.0)
    ax.set_ylabel('$y$', ha='right', y=1.0)
    return fig


def plot_step(
    x: np.ndarray | list,
    y: np.ndarray | list,
    xlabel: str = '$x$',
    ylabel: str = '$y$',
    **kwargs
) -> Figure:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_axes([.1, .1, .8, .85])
    ax.step(x, y, **kwargs)
    ax.set_xlabel(xlabel, ha='right', x=1.0)
    ax.set_ylabel('Counts', ha='right', y=1.0)
    return fig


def plot_histogram(data: np.ndarray, **kwargs) -> Optional[Figure]:
    """Plots input data as a histogram using plt.step.

    Takes only height data, indexes are used for the other axis.
    """
    x = np.arange(len(data))
    return plot_step(x, data, ylabel='Counts', **kwargs)


def plot_all_frames(df: pandas.DataFrame, plot_function: Callable) -> None:
    for _, row in df.iterrows():
        plot_function(row['top'], row.timestamp.replace(' ', '_'), 'top')
        plot_function(row['bottom'], row.timestamp.replace(' ', '_'), 'bottom')


def image_profile_multiplot(data: np.ndarray) -> MultiPlot:
    """Creates a figure with 4 axes for plotting an imshow with projections.

    Does some set preconfiguration for labels, limits, etc.
    """
    fig = plt.figure(figsize=(8, 6))
    grid = GridSpec(
        2, 3,
        width_ratios=[1, 4, .2],
        height_ratios=[4, 1],
        hspace=0.0,
        wspace=0.0,
        left=.1,
        bottom=.1,
        right=.9,
        top=.95,
    )
    # set up axes
    ax_xprofile = fig.add_subplot(grid[1, 1])
    ax_yprofile = fig.add_subplot(grid[0, 0])
    ax_main = fig.add_subplot(grid[0, 1], sharex=ax_xprofile, sharey=ax_yprofile)
    ax_cbar = fig.add_subplot(grid[0, 2])
    # handle labels
    ax_main.tick_params(labelbottom=False, labelleft=False)
    ax_xprofile.set_xlim(0, data.shape[1] - 1)
    ax_yprofile.set_ylim(0, data.shape[0] - 1)
    ax_xprofile.tick_params(labelleft=False)
    ax_yprofile.tick_params(labelbottom=False)
    ax_xprofile.ticklabel_format(style='plain', axis='y')
    ax_yprofile.ticklabel_format(style='plain', axis='x')
    ax_xprofile.set_xlabel(r'$x$', ha='right', x=1.0)
    ax_yprofile.set_ylabel(r'$y$', ha='right', y=1.0)
    return fig, ax_main, ax_xprofile, ax_yprofile, ax_cbar


def plot_per_roi_lissajous(df: pandas.DataFrame, **kwargs) -> list[Figure]:
    rtn = []
    for i in range(len(df['top_roi_integrals'][0])):
        top_ints = [roi_ints[i] for roi_ints in df['top_roi_integrals']]
        bottom_ints = [roi_ints[i] for roi_ints in df['bottom_roi_integrals']]
        fig, ax = plot_poisson_error(bottom_ints, top_ints, whicherr='both', fmt='k.', **kwargs)
        ax.set_xlabel(f'ROI {i+1} bottom detector integral', ha='right', x=1.0)
        ax.set_ylabel(f'ROI {i+1} top detector integral', ha='right', y=1.0)
        rtn.append(fig)
    return rtn


def plot_per_roi_integrals(df: pandas.DataFrame, **kwargs) -> list[Figure]:
    rtn = []
    for i in range(len(df['top_roi_integrals'][0])):
        top_ints = [roi_ints[i] for roi_ints in df['top_roi_integrals']]
        bottom_ints = [roi_ints[i] for roi_ints in df['bottom_roi_integrals']]
        fig, ax = plot_poisson_error(df['timestamp'], top_ints, fmt='x', **kwargs)
        plot_poisson_error(df['timestamp'], bottom_ints, ax=ax, fmt='o', **kwargs)
        ax.set_xlabel('Timestamp', ha='right', x=1.0)
        ax.set_ylabel(f'ROI {i+1} integral', ha='right', y=1.0)
        plt.legend(
            ['Top detector', 'Bottom detector'],
            framealpha=0, fontsize=13,
            # bbox_to_anchor=(.78, .785, .2, .2),
            handleheight=2
        )
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymin + (ymax - ymin) * 1.15)
        rtn.append(fig)
    return rtn


def plot_total_integrals(df: pandas.DataFrame, **kwargs) -> Figure:
    fig, ax = plot_poisson_error(df['timestamp'], df['top_total_integral'], fmt='x', **kwargs)
    plot_poisson_error(df['timestamp'], df['bottom_total_integral'], ax=ax, fmt='o', **kwargs)
    ax.set_xlabel('Timestamp', ha='right', x=1.0)
    ax.set_ylabel('Sum of ROI integrals', ha='right', y=1.0)
    plt.legend(
        ['Top detector', 'Bottom detector'],
        framealpha=0, fontsize=13,
        bbox_to_anchor=(.78, .785, .2, .2),
        handleheight=2
    )
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymin + (ymax - ymin) * 1.15)
    return fig


def plot_poisson_error(x, y, ax: Axes = None, whicherr: str = 'y', **kwargs) -> tuple[Figure, Axes]:
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_axes([.1, .1, .8, .85])
    else:
        fig = None
    xerr = [sqrt(xi) for xi in x] if whicherr in ('x', 'both') else None
    yerr = [sqrt(yi) for yi in y] if whicherr in ('y', 'both') else None
    ax.errorbar(x, y, xerr=xerr, yerr=yerr, **kwargs)
    return fig, ax


def profile(
    data: np.ndarray,
    which: str = 'x',
    bounds: list[int | None] = [None, None]
) -> tuple[np.ndarray, np.ndarray]:
    if which == 'x':
        return np.arange(data.shape[1]), np.mean(data[bounds[0]:bounds[1], :], axis=0)
    else:
        return np.mean(data[:, bounds[0]:bounds[1]], axis=1), np.arange(data.shape[0])


def save_figure(fig: Figure, name: str) -> None:
    fig.savefig('output/' + name)
    print(f'Saved figure {name}')
    plt.close(fig)
