from math import floor, ceil

import numpy as np
import scipy.signal as sp_sig

from qggroi.roi import ROI


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
    peak_locations, info = sp_sig.find_peaks(peak_data, prominence=absolute_prominence)
    # get widths of peaks
    w = sp_sig.peak_widths(peak_data, peak_locations)
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
