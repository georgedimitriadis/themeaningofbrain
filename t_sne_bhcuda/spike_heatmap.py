"""
Helper functions for the manual spike sorting GUI based on the t-sne of spikes

Author:             George Dimitriadis    <george dimitriadis uk>
Version:            0.2.0
"""

import numpy as np
import scipy.stats as stats
import scipy.interpolate as interpolate
import os.path as op
from six import exec_
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import warnings

def peaktopeak(data, voltage_step_size=1e-6, scale_microvolts=1000000, window_size=60):
    """
    Generates the minima, maxima and peak to peak (p2p) numbers (in microvolts) of all the channels of all spikes
    Parameters
    ----------
    data: a channels x time x spikes array
    voltage_step_size: the y digitization of the amplifier
    scale_microvolts
    window_size: the window size (in samples) within which the function searches for maxima, minima and p2p.
    Must be smaller than the size of the time axis in the data

    Returns
    -------
    argmaxima: the time point (in samples) of the maximum of each channel
    argminima: the time point (in samples) of the minimum of each channel
    maxima: the channels' maxima
    minima: the channels' minima
    p2p: the channels' peak to peak voltage difference
    """
    extracellular_avg_volts = np.average(data[:, :, :], axis=2)
    num_time_points = extracellular_avg_volts.shape[1]
    extracellular_avg_microvolts = extracellular_avg_volts * scale_microvolts * voltage_step_size
    num_channels = np.size(extracellular_avg_microvolts, axis=0)
    lower_bound = int(num_time_points / 2.0 - window_size / 2.0)
    upper_bound = int(num_time_points / 2.0 + window_size / 2.0)

    argminima = np.zeros(num_channels)
    for m in range(num_channels):
        argminima[m] = np.argmin(extracellular_avg_microvolts[m][lower_bound:upper_bound])+lower_bound

    argmaxima = np.zeros(num_channels)
    for n in range(num_channels):
        argmaxima[n] = np.argmax(extracellular_avg_microvolts[n][lower_bound:upper_bound])+lower_bound

    maxima = np.zeros(num_channels)
    for p in range(num_channels):
            maxima[p] = np.max(extracellular_avg_microvolts[p][lower_bound:upper_bound])

    minima = np.zeros(num_channels)
    for k in range(num_channels):
        minima[k] = np.min(extracellular_avg_microvolts[k][lower_bound:upper_bound])

    p2p = maxima-minima

    stdv_minima = np.zeros(num_channels)
    stdv_maxima = np.zeros(num_channels)

    stdv = stats.sem(data[:, :, :], axis=2)
    stdv = stdv * voltage_step_size * scale_microvolts

    for b in range(num_channels):
        stdv_minima[b] = stdv[b, int(argminima[b])]
        stdv_maxima[b] = stdv[b, int(argmaxima[b])]

    error = np.sqrt((stdv_minima * stdv_minima) + (stdv_maxima * stdv_maxima))

    return argmaxima, argminima, maxima, minima, p2p, error


def get_probe_geometry_from_prb_file(prb_file):
    """
    Extracts the dictionaries from the .prb probe file

    Parameters
    ----------
    prb_file: the probe geometry file

    Returns
    -------
    shanks: the dictionary of dictionaries in the .prb file
    """
    path = op.realpath(op.expanduser(prb_file))
    assert op.exists(path)
    with open(path, 'r') as f:
        contents = f.read()
    metadata = {}
    exec_(contents, {}, metadata)
    metadata = {k.lower(): v for (k, v) in metadata.items()}
    shanks = metadata['channel_groups']
    return shanks


def plot_topoplot(axis, channel_positions, data, show=True, rotate_90=False, flip_ud=False, flip_lr=False, **kwargs):
    """
    This function interpolates the data between electrodes and plots it into
    the output.

    Parameters
    ----------
    axis: an instance of matplotlib.pyplot axes where you want the heatmap to output.
    channel_positions: a Pandas Series with the positions of the electrodes
    (this is the one shank output of get_probe_geometry_from_prb_file function)
    data: a numpy array containing the data to be interpolate and then displayed.
    show: a boolean variable to assert whether you want the heatmap to be displayed on the screen
    rotate_90: if true then rotate probe figure by 90 degrees by switching the x and y coordinate values
    flip_ud: if true then flip the probe upside down
    flip_lr: if true then flip the probe left to right (this flip happens after the ud flip if both are true)
    kwargs can be:
    - hpos and vpos define the horizontal and vertical position offset of the
      output heatmap, respectively.
    - width and height define the horizontal and vertical scale of the output
      heatmap, respectively.
    - gridscale defines the resolution of the interpolation.
    - interpolation_method defines the method used to interpolate the data between positions in channel_positions.
      Choose from:
      ‘none’, ‘nearest’, ‘bilinear’, ‘bicubic’, ‘spline16’, ‘spline36’,
      ‘hanning’, ‘hamming’, ‘hermite’, ‘kaiser’, ‘quadric’, ‘catrom’,
      ‘gaussian’, ‘bessel’, ‘mitchell’, ‘sinc’, ‘lanczos’
    - zlimits defines the limits of the amplitude of the output heatmap.

    Returns
    -------
    image: the heatmap.
    channels_grid: the grid of electrodes.
    """
    if not kwargs.get('hpos'):
        hpos = 0
    else:
        hpos = kwargs['hpos']
    if not kwargs.get('vpos'):
        vpos = 0
    else:
        vpos = kwargs['vpos']
    if not kwargs.get('width'):
        width = None
    else:
        width = kwargs['width']
    if not kwargs.get('height'):
        height = None
    else:
        height = kwargs['height']
    if not kwargs.get('gridscale'):
        gridscale = 1
    else:
        gridscale = kwargs['gridscale']
    if not kwargs.get('interpolation_method'):
        interpolation_method = "bicubic"
    else:
        interpolation_method = kwargs['interpolation_method']
    if not kwargs.get('zlimits'):
        zlimits = None
    else:
        zlimits = kwargs['zlimits']

    if np.isnan(data).any():
        warnings.warn('The data passed to contain NaN values. \
        These will create unexpected results in the interpolation. \
        Deal with them.')

    channel_positions = channel_positions.sort_index(ascending=[1])
    if not rotate_90:
        channel_positions = np.array([[x, y] for x, y in channel_positions.values])
        if flip_ud:
            channel_positions[:, 1] = np.abs(channel_positions[:, 1] - np.max(channel_positions[:, 1]))
        if flip_lr:
            channel_positions[:, 0] = np.abs(channel_positions[:, 0] - np.max(channel_positions[:, 0]))
    else:
        channel_positions = np.array([[y, x] for x, y in channel_positions.values])
        if flip_ud:
            channel_positions[:, 0] = np.abs(channel_positions[:, 0] - np.max(channel_positions[:, 0]))
        if flip_lr:
            channel_positions[:, 1] = np.abs(channel_positions[:, 1] - np.max(channel_positions[:, 1]))

    all_coordinates = channel_positions

    natural_width = np.max(all_coordinates[:, 0]) - np.min(all_coordinates[:, 0])
    natural_height = np.max(all_coordinates[:, 1]) - np.min(all_coordinates[:, 1])

    if not width and not height:
        x_scaling = 1
        y_scaling = 1
    elif not width and height:
        y_scaling = height/natural_height
        x_scaling = y_scaling
    elif width and not height:
        x_scaling = width/natural_width
        y_scaling = x_scaling
    elif width and height:
        x_scaling = width/natural_width
        y_scaling = height/natural_height

    chan_x = channel_positions[:, 0] * x_scaling + hpos
    chan_y = channel_positions[:, 1] * y_scaling + vpos
    chan_x = np.max(chan_x) - chan_x

    hlim = [np.min(chan_y), np.max(chan_y)]
    vlim = [np.min(chan_x), np.max(chan_x)]

    if interpolation_method is not 'none':
        yi, xi = np.mgrid[hlim[0]:hlim[1]:complex(0, gridscale)*(hlim[1]-hlim[0]),
                          vlim[0]:vlim[1]:complex(0, gridscale)*(vlim[1]-vlim[0])]
    else:
        yi, xi = np.mgrid[hlim[0]:hlim[1]+1, vlim[0]:vlim[1]+1]

    zi = interpolate.griddata((chan_x, chan_y), data, (xi, yi))

    if not zlimits:
        vmin = data.min()
        vmax = data.max()
    else:
        vmin = zlimits[0]
        vmax = zlimits[1]

    cmap = plt.get_cmap("jet")
    image = axis.imshow(zi.T, cmap=cmap, origin=['lower'], vmin=vmin,
                        vmax=vmax, interpolation=interpolation_method,
                        extent=[hlim[0], hlim[1], vlim[0], vlim[1]],
                        aspect='equal')
    channels_grid = axis.scatter(chan_y, chan_x)

    if show:
        cb = plt.colorbar(image)
        plt.show()
    return image, channels_grid


def create_heatmap(data, prb_file, voltage_step_size=1e-6, scale_microvolts=1000000, window_size=60,
                   rotate_90=False, flip_ud=False, flip_lr=False):
    """

    Parameters
    ----------
    data: a channels x time x spikes array
    voltage_step_size: the y digitization of the amplifier
    scale_microvolts
    window_size: the window size (in samples) within which the function searches for maxima, minima and p2p.
    Must be smaller than the size of the time axis in the data
    prb_file: the probe definition file as is used by phy to generate the spikes
    rotate_90: if True rotate the heatmap by 90 degrees
    flip_ud: if True flip the heatmap upside down
    flip_lr: If True flip the heatmap left to right

    Returns
    -------
    final_image: 2d array of int32 of x_size x y_size numbers defining the color of each pixel
    x_size: the pixel number of the heatmap's x axis
    y_size: the pixel number of the heatmap's y axis
    """
    _, _, _, _, p2p, error = peaktopeak(data, voltage_step_size=voltage_step_size,
                                        scale_microvolts=scale_microvolts, window_size=window_size)

    shanks = get_probe_geometry_from_prb_file(prb_file)
    num_of_shanks = len(list(shanks.keys()))
    fig = plt.figure()
    for shank in shanks:
        channel_positions = pd.Series(shanks[shank]['geometry'])
        ax = fig.add_subplot(1, num_of_shanks, shank + 1)
        data = p2p[channel_positions.index]
        image, channels_grid = plot_topoplot(ax, channel_positions, data, show=False, rotate_90=rotate_90,
                                             flip_ud=flip_ud, flip_lr=flip_lr)

        image.write_png('temp.png')  # Required to generate the _rgbacache info
        temp_image = image._rgbacache

        if shank == 0:
            y_dim_pixels = temp_image.shape[0]
            grid_image_spacing = np.zeros((y_dim_pixels, 10, 4))
            grid_image_spacing[:, :, :] = 255
            grid_image = temp_image
        else:
            conc = np.concatenate((grid_image_spacing, temp_image), axis=1)
            grid_image = np.append(grid_image, conc, axis=1)
    plt.close(fig)
    x_size = grid_image.shape[0]
    y_size = grid_image.shape[1]
    final_image = np.empty((x_size, y_size), dtype=np.uint32)
    view = final_image.view(dtype=np.uint8).reshape((x_size, y_size, 4))
    for i in np.arange(4):
        view[:, :, i] = grid_image[:, :, i]

    return final_image, (x_size, y_size)

