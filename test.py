from math import floor, ceil
from typing import Callable

import sqlite3
import pandas
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

DB_FILE = "extracted.db"
TOP_COL = 'top_det_pd'
BOT_COL = 'bot_det_pd'


def main():
    with sqlite3.connect(DB_FILE) as db:
        df = pandas.read_sql_query('select * from shot_data;', db)
    df['timestamp'] = pandas.to_datetime(df['timestamp'])
    df[TOP_COL] = df[TOP_COL].apply(decode)
    df[BOT_COL] = df[BOT_COL].apply(decode)
    df['difference'] = df.apply(lambda row: row[TOP_COL] - row[BOT_COL], axis=1)

    arr_cols = [TOP_COL, BOT_COL, 'difference']

    df_std = df[arr_cols].stack().groupby(level=1).apply(lambda x: np.std(np.array(x)))
    df_mean = df[arr_cols].stack().groupby(level=1).apply(lambda x: np.mean(np.array(x)))

    rois = locate_rois(df_std[TOP_COL], limit=7)
    for roi in rois:
        # plot_roi(roi, df_std['difference'])
        plot_roi(roi, df[TOP_COL][0])

    df['top_integral'] = df[TOP_COL].apply(lambda x: sum([roi.integral(x) for roi in rois]))
    df['bot_integral'] = df[BOT_COL].apply(lambda x: sum([roi.integral(x) for roi in rois]))
    # df['top_integral'] = df[TOP_COL].apply(lambda x: rois[0].integral(x))
    # df['bot_integral'] = df[BOT_COL].apply(lambda x: rois[0].integral(x))
    df.plot(x='bot_integral', y='top_integral', style='o')

    # roi_one_integrals = []
    # for idata in df[TOP_COL]:
    #     roi_one_integrals.append(sum([roi.integral(idata) for roi in rois]))
    # plt.plot(np.arange(len(roi_one_integrals)), roi_one_integrals)

    # res = detect_peaks(df_std['difference'])
    # print(res)
    # fancy_image_plot(res)

    # for col in [arr_cols[2]]:
    #     examine_regions(df_mean[col])
    #     fancy_image_plot(df_mean[col])
    #     plt.title(col)
    # plot_all_frames(df, save_as_image)

    plt.show()


class ROI:
    def __init__(self, bounds: list[list[int, int]]):
        assert len(bounds) == 2
        assert len(bounds[0]) == 2
        assert len(bounds[1]) == 2
        self.x_bounds = bounds[0]
        self.y_bounds = bounds[1]

    def integral(self, data: np.ndarray) -> float:
        """Calculate the integral of this ROI in the given data array."""
        return np.sum(self.apply_bounds(data))

    def apply_bounds(self, data: np.ndarray) -> np.ndarray:
        return data[self.y_bounds[0] : self.y_bounds[1], self.x_bounds[0] : self.x_bounds[1]]

    def __repr__(self) -> None:
        return repr(self.x_bounds) + repr(self.y_bounds)

    def printint(self, data):
        return f'{repr(self)} {self.integral(data)}'


def locate_rois(data: np.ndarray, limit: int = None) -> list[ROI]:
    """Locate Regions of Interest in data.

    Performs 2D peak search by identifying peaks in a 1D profile of the y-axis, then within each
    peak searching the x distribution for peaks.
    Returns an ROI object containing x- and y-bounds for the ROI.

    Keyword arguments:
    data -- 2D image in which to find ROIs
    limit -- Maximum number of ROIs returned. For limit n, function returns the n ROIs with the
    largest integral (default None, all ROIs are returned)
    """
    rois = []
    y_profile = np.mean(data, axis=1)
    y_peaks = find_peak_1d(y_profile)
    for y_bounds in y_peaks:
        x_profile = np.mean(data[y_bounds[0] : y_bounds[1] + 1], axis=0)
        x_peaks = find_peak_1d(x_profile, relative_prominence=0.1, stretch=1)
        rois += [ROI([x_bounds, y_bounds]) for x_bounds in x_peaks]
    # sort list by size of ROIs so we only return largest <limit> ROIs
    rois.sort(key=lambda roi: roi.integral(data), reverse=True)
    return rois[:limit]


def find_peak_1d(
        data: np.ndarray,
        relative_prominence: float = 0.01,
        stretch: int = 0
) -> list[list[int, int]]:
    """Finds peaks in a 1D distribution, returns edges of each peak found.

    Keyword arguments:
    data -- 1D array of data in which to find peaks
    relative_prominence -- threshold for height of peaks above background, as fraction
    of maximum entry in data (default 0.01)
    stretch -- amount by which to expand each peak boundary
    """
    # find peaks
    absolute_prominence = relative_prominence * np.max(data)
    peak_data = np.concatenate(([0], data, [0]))
    peak_locations, info = find_peaks(peak_data, prominence=absolute_prominence)
    # get widths of peaks
    w = peak_widths(peak_data, peak_locations)
    edges = [
        [max(floor(left_edge - stretch - 1), 0), min(ceil(right_edge + stretch - 1), len(data))]
        for left_edge, right_edge in zip(*w[2:])
    ]
    # merge any overlapping peaks
    if len(edges) == 1:
        return edges
    merged_indices = []
    for i, (bounds, next_bounds) in enumerate(zip(edges[:-1], edges[1:])):
        if bounds[1] >= next_bounds[0]:
            bounds[1] = next_bounds[1]
            merged_indices.append(i + 1)
    # return all non-merged indices
    return [bounds for i, bounds in enumerate(edges) if i not in merged_indices]


def plothist(data):
    fig = plt.figure(figsize=(8, 6))
    plt.step(np.arange(len(data)), data)
    return fig


def examine_regions(data: np.ndarray, title: str = 'Plot') -> None:
    y_regions = [(0, 5), (18, 25), (25, 30), (30, 35)]
    for y in y_regions:
        fancy_image_plot(data[y[0]:y[1], :])
        plt.title(f'{title} Region: y={y}')


def plot_all_frames(df: pandas.DataFrame, plot_function: Callable) -> None:
    for _, row in df.iterrows():
        plot_function(row[TOP_COL], row.timestamp.replace(' ', '_'), 'top')
        plot_function(row[BOT_COL], row.timestamp.replace(' ', '_'), 'bottom')


def save_as_hist(data: np.ndarray, name: str, path: str) -> None:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_axes([.1, .1, .8, .85])
    ax.step(*profile(data, 'y'))
    fig.savefig(f'hist/{path}/{name}.pdf')
    plt.close(fig)


def save_as_image(data: np.ndarray, name: str, path: str) -> None:
    # fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_axes([.1, .1, .8, .85])
    # img = ax.imshow(data)
    # plt.colorbar(img)
    fig = fancy_image_plot(data)
    fig.savefig(f'img/{path}/{name}.pdf')
    plt.close(fig)


def decode(data: bytes) -> np.ndarray:
    data_array = np.frombuffer(data, dtype=np.uint32)
    # enforce assumption that input data is a square 2D array
    size = len(data_array) ** .5
    assert int(size) == size
    size = int(size)
    # return handle_overflow(np.reshape(data_array, (size, size)))
    data_array = np.reshape(data_array, (size, size))
    data_array = handle_overflow(data_array)
    data_array = normalise(data_array)
    return data_array


def normalise(data: np.ndarray) -> np.ndarray:
    return data / data.sum()


def handle_overflow(data: np.ndarray) -> np.ndarray:
    # return data
    new_data = data.astype('int')
    new_data[new_data > 3e9] -= np.iinfo(np.uint32).max
    return new_data


def profile(
    data: np.ndarray,
    which: str = 'x',
    bounds: list[int | None] = [None, None]
) -> tuple[np.ndarray, np.ndarray]:
    if which == 'x':
        return np.arange(data.shape[1]), np.mean(data[bounds[0]:bounds[1], :], axis=0)
    else:
        return np.mean(data[:, bounds[0]:bounds[1]], axis=1), np.arange(data.shape[0])


def fancy_image_plot(data: np.ndarray) -> Figure:
    fig = plt.figure(figsize=(8, 6))
    grid = GridSpec(
        2, 3,
        width_ratios=[1, 4, .2],
        height_ratios=[4, 1],
        hspace=0.0,
        wspace=0.0
    )
    # set up axes
    ax_xprofile = fig.add_subplot(grid[1, 1])
    ax_yprofile = fig.add_subplot(grid[0, 0])
    ax_main = fig.add_subplot(grid[0, 1], sharex=ax_xprofile, sharey=ax_yprofile)
    ax_cbar = fig.add_subplot(grid[0, 2])
    # make plots
    im = ax_main.imshow(data, origin='lower', aspect='auto')
    ax_xprofile.plot(*profile(data, 'x'), color='black')
    ax_yprofile.plot(*profile(data, 'y'), color='black')
    cbar = plt.colorbar(im, cax=ax_cbar)
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
    cbar.set_label('Counts', ha='right', y=1.0)
    return fig


def plot_roi(roi: ROI, data: np.ndarray) -> Figure:
    fig = plt.figure(figsize=(8, 6))
    grid = GridSpec(
        2, 3,
        width_ratios=[1, 4, .2],
        height_ratios=[4, 1],
        hspace=0.0,
        wspace=0.0
    )
    # set up axes
    ax_xprofile = fig.add_subplot(grid[1, 1])
    ax_yprofile = fig.add_subplot(grid[0, 0])
    ax_main = fig.add_subplot(grid[0, 1], sharex=ax_xprofile, sharey=ax_yprofile)
    ax_cbar = fig.add_subplot(grid[0, 2])
    # make plots
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
    ax_yprofile.step(*profile(data, 'y', roi.x_bounds), color='black')
    ax_yprofile.axhspan(roi.y_bounds[0], roi.y_bounds[1], facecolor=(1, 0, 0, .5))
    cbar = plt.colorbar(im, cax=ax_cbar)
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
    cbar.set_label('Counts', ha='right', y=1.0)
    return fig


if __name__ == "__main__":
    main()
