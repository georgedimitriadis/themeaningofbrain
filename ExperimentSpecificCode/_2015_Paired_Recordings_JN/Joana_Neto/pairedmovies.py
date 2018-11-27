
import numpy as np
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
from matplotlib.pyplot import sca
from matplotlib.ticker import MultipleLocator
import pandas as pd
import itertools
import warnings
# import Utilities as ut
import matplotlib.animation as animation
from matplotlib.widgets import Button
import mne.filter as filters


plt.rcParams['animation.ffmpeg_path'] ="C:\\Users\\KAMPFF-LAB_ANALYSIS4\\Downloads\\ffmpeg-20160116-git-d7c75a5-win64-static\\ffmpeg-20160116-git-d7c75a5-win64-static\\bin\\ffmpeg.exe"

# Movies 32channels

def polytrode_channels(bad_channels=[]):
    '''
    This function produces a grid with the electrodes positions

    Inputs:
    bad_channels is a list or numpy array with channels you may want to
    disregard.

    Outputs:
    channel_positions is a Pandas Series with the electrodes positions (in
    two dimensions)
    '''
    electrode_coordinate_grid = list(itertools.product(np.arange(0, 22),
                                                       np.arange(0, 3)))

    electrode_amplifier_index_on_grid = np.array([32, 0, 33,
                                                  34, 31, 35,
                                                  24, 36, 7,
                                                  37, 1, 38,
                                                  21, 39, 10,
                                                  40, 30, 41,
                                                  25, 42, 6,
                                                  43, 15, 44,
                                                  20, 45, 11,
                                                  46, 16, 47,
                                                  26, 48, 5,
                                                  49, 14, 50,
                                                  19, 51, 12,
                                                  52, 17, 53,
                                                  27, 54, 4,
                                                  55, 8, 56,
                                                  18, 57, 13,
                                                  58, 23, 59,
                                                  28, 60, 3,
                                                  61, 9, 62,
                                                  29, 63, 2,
                                                  64, 22, 65])

    electrode_amplifier_name_on_grid = np.array(["Int"+str(x) for x in electrode_amplifier_index_on_grid])
    channel_position_indices = [electrode_amplifier_index_on_grid,
                                electrode_amplifier_name_on_grid]
    channel_positions = pd.Series(electrode_coordinate_grid,
                                  channel_position_indices)
    no_channels = [0, 2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29,
                   31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59,
                   61, 63, 65]
    bad_channels = no_channels + bad_channels
    a = np.arange(0, 66)
    b = np.delete(a, bad_channels)
    channel_positions = channel_positions[b]
    return channel_positions


def plot_topoplot(axis, channel_positions, data, show=True, **kwargs):
    '''
    This function interpolates the data between electrodes and plots it into
    the output.

    Inputs:
    axis is an instance of matplotlib.axes where you want the heatmap to be
    output.
    channel_positions is a Pandas Series with the positions of the electrodes
    (this is the output of polytrode_channels function)
    data is a numpy array containing the data to be interpolate and then
    displayed.
    show is a boolean variable to assert whether you want the heatmap to be
    printed on the screen
    kwargs can :
    - hpos and vpos define the horizontal and vertical position offset of the
      output heatmap, respectively.
    - width and height define the horizontal and vertical scale of the output
      heatmap, respectively.
    - gridscale defines the resolution of the interpolation.
    - interpolation_method defines the method used to interpolate the data
      between positions in channel_positions.
      Choose from:
      ‘none’, ‘nearest’, ‘bilinear’, ‘bicubic’, ‘spline16’, ‘spline36’,
      ‘hanning’, ‘hamming’, ‘hermite’, ‘kaiser’, ‘quadric’, ‘catrom’,
      ‘gaussian’, ‘bessel’, ‘mitchell’, ‘sinc’, ‘lanczos’
    - zlimits defines the limits of the amplitude of the output heatmap.

    Outputs:
    image is the heatmap.
    scat is the grid of electrodes.
    '''
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
        gridscale = 10
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
        warnings.warn('The data passed to gdft_plot_topo contain NaN values. \
        These will create unexpected results in the interpolation. \
        Deal with them.')

    channel_positions = channel_positions.sort_index(ascending=[1])
    channel_positions = np.array([[x, y] for x, y in channel_positions.values])
    allCoordinates = channel_positions

    naturalWidth = np.max(allCoordinates[:, 0]) - np.min(allCoordinates[:, 0])
    naturalHeight = np.max(allCoordinates[:, 1]) - np.min(allCoordinates[:, 1])

    if not width and not height:
        xScaling = 1
        yScaling = 1
    elif not width and height:
        yScaling = height/naturalHeight
        xScaling = yScaling
    elif width and not height:
        xScaling = width/naturalWidth
        yScaling = xScaling
    elif width and height:
        xScaling = width/naturalWidth
        yScaling = height/naturalHeight

    chanX = channel_positions[:, 0] * xScaling + hpos
    chanY = channel_positions[:, 1] * yScaling + vpos
    chanX = np.max(chanX) - chanX

    hlim = [np.min(chanY), np.max(chanY)]
    vlim = [np.min(chanX), np.max(chanX)]

    if interpolation_method is not 'none':
        yi, xi = np.mgrid[hlim[0]:hlim[1]:complex(0, gridscale)*(hlim[1]-hlim[0]), vlim[0]:vlim[1]:complex(0, gridscale)*(vlim[1]-vlim[0])]
    else:
        # for no interpolation show one pixel per data point
        yi, xi = np.mgrid[hlim[0]:hlim[1]+1, vlim[0]:vlim[1]+1]

    Zi = interpolate.griddata((chanX, chanY), data, (xi, yi))

    if not zlimits:
        vmin = data.min()
        vmax = data.max()
    else:
        vmin = zlimits[0]
        vmax = zlimits[1]

    cmap = plt.get_cmap("jet")
    image = axis.imshow(Zi.T, cmap=cmap, origin=['lower'], vmin=vmin,
                        vmax=vmax, interpolation=interpolation_method,
                        extent=[hlim[0], hlim[1], vlim[0], vlim[1]],
                        aspect='equal')

    scat = axis.scatter(chanY, chanX)
    if show:
        cb = plt.colorbar(image)
        plt.show()
    return image, scat


def find_closest(array, target):
    # a must be sorted
    idx = array.searchsorted(target)
    idx = np.clip(idx, 1, len(array)-1)
    left = array[idx-1]
    right = array[idx]
    idx -= target - left < right - target
    return idx


def plot_video_topoplot_with_juxta(data, juxtaData, time_axis, channel_positions,
                        times_to_plot=[5.4355, 5.439], time_window=0.000034,
                        time_step=0.000034, sampling_freq=30000,
                        zlimits=[-300, 100],filename='I:\Ephys\Data juxta_extra\data_25_11_72um\paper\\mymovie.avi'):
    '''
    This function creates a video file of time ordered heatmaps.

    Inputs:
    data is a numpy array containing the data to produce all the heatmaps.
    time_axis is a numpy array containing the time points of sampling. (in
    seconds)
    channel_positions is a Pandas Series with the positions of the electrodes
    (this is the output of polytrode_channels function)
    times_to_plot defines the upper and lower bound of the time window to be
    used. (in seconds)
    time_window defines the size of the window to calculate the average. (in
    seconds)
    time_step is the time interval between consecutive heatmaps (in seconds).
    sampling_freq is the sampling frequency, in Hz, at which the data was
    acquired.
    zlimits defines the limits of the magnitude of all heatmaps.
    filename is the path to the output video file.
    '''
    global images, sub_time_indices

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    sample_step = int(time_step * sampling_freq)
    sub_time_indices = np.arange(find_closest(time_axis, times_to_plot[0]),
                                 find_closest(time_axis, times_to_plot[1]))
    sub_time_indices = sub_time_indices[0::sample_step]
    print(sub_time_indices)
    if np.size(channel_positions) <= 64:
        text_y = 8.3
    elif np.size(channel_positions) <= 128:
        text_y = 17
    text_x = -6
    images = []
    for t in sub_time_indices:
        samples = [t, t + (time_window*sampling_freq)]
        print(t)
        data_to_plot = np.mean(data[:, int(samples[0]):int(samples[1])], 1)
        image, scat = plot_topoplot(ax2, channel_positions, data_to_plot,
                                    show=False, interpmethod="quadric",
                                    gridscale=5, zlimits=zlimits)
        ax2.set_yticks([], minor=False)
        ax2.set_xticks([], minor=False)
        stringy = '%.2f' % (time_axis[t]*1000)
        stringy = 't' + '=' + stringy + ' ms'
        txt = plt.text(x=text_x, y=text_y, s=stringy)
        #ax2.text(4,22, '\u00B5V')
        ax2.set_title("JTA voltage at t / \u00B5V")


        if t < np.max(sub_time_indices) - 30 - np.min(sub_time_indices):
            pointsToPlot = juxtaData[ t-15:t-15+30]
        else:
            pointsToPlot = juxtaData[t-15:t-15+30]
        grafico, = ax1.plot(np.arange(-15, 15)/30., pointsToPlot, 'b')
        #ax1.set_yticks([0], minor=False)
        #ax1.set_yticks([0], minor=True)
        #ax1.yaxis.grid(False, which='major')
        #ax1.yaxis.grid(False, which='minor')
        ax1.set_xticks([0])
        ax1.grid(True)
        ax1.set_title("Juxta Signal/ mV")
        images.append([grafico, image, scat, txt])
    FFwriter = animation.FFMpegWriter()
    ani = animation.ArtistAnimation(fig, images, interval=1000, blit=True,
                                    repeat_delay=1000)
    plt.colorbar(mappable=image)

    if filename is not None:
        ani.save(filename, writer=FFwriter, fps=0.5, bitrate=5000, dpi=300,
                 extra_args=['h264'])
    plt.show()






rootDir=r'E:\Data\2014-11-13'
videoFilename = "2015_11_13_Pair1.avi"

voltage_step_size = 0.195e-6
scale_uV = 1000000
scale_mV = 1000

alignamplifier=all_cells_ivm_filtered_data['1']
juxtaData = all_cells_patch_data['1']
juxtaData = np.average(juxtaData,axis=-1) * scale_mV
data = np.mean(alignamplifier,axis=2)
data = data * scale_uV * voltage_step_size
time_axis = np.arange(-(np.shape(data)[1]/30000.0)/2, (np.shape(data)[1]/30000.0)/2,1/30000.0 )

plot_video_topoplot_with_juxta(data,juxtaData, time_axis, polytrode_channels(),
                               times_to_plot=[-0.001, 0.001], zlimits=[np.min(data), np.max(data) * 1.1],
                               filename= os.path.join(rootDir,videoFilename))




# Movies 128channels

def polytrode_channels128(bad_channels=[]):
    '''
    This function produces a grid with the electrodes positions

    Inputs:
    bad_channels is a list or numpy array with channels you may want to
    disregard.

    Outputs:
    channel_positions is a Pandas Series with the electrodes positions (in
    two dimensions)
    '''

    electrode_coordinate_grid = list(itertools.product(np.arange(0, 32),
                                                       np.arange(0, 4)))

    r1 = np.array([103,	101, 99,	97,	95,	93,	91,	89,	87,	70,	66,	84,	82,	80,	108,	110,	47,	45,	43,	41,	1, 61,	57,
                    36,	34,	32,	30,	28,	26,	24,	22,	20])
    r2 = np.array([106, 104, 115, 117, 119, 121, 123, 125, 127, 71, 67, 74, 76, 78, 114, 112,
                   49, 51, 53, 55, 2, 62, 58, 4, 6, 8, 10, 12, 14, 21, 19, 16])
    r3 = np.array([102, 100, 98, 96, 94, 92, 90, 88, 86, 72, 68, 65, 85, 83, 81, 111, 46, 44, 42, 40, 38, 63, 59,
                   39, 37, 35, 33, 31, 29, 27, 25, 23])
    r4 = np.array([109, 107, 105, 116, 118, 120, 122, 124, 126, 73, 69, 64, 75, 77, 79, 113,
                   48, 50, 52, 54, 56, 0, 60, 3, 5, 7, 9, 11, 13, 15, 18,-1])

    all_electrodes_concat = np.concatenate((r1, r2, r3, r4))
    all_electrodes = all_electrodes_concat.reshape((4, 32))
    all_electrodes = np.flipud(all_electrodes.T)
    electrode_amplifier_index_on_grid = np.reshape(all_electrodes, 128)

    electrode_amplifier_name_on_grid = np.array(["Int"+str(x) for x in electrode_amplifier_index_on_grid])
    channel_position_indices = [electrode_amplifier_index_on_grid,
                                electrode_amplifier_name_on_grid]
    channel_positions = pd.Series(electrode_coordinate_grid,
                                  channel_position_indices)

    return channel_positions


def plot_topoplot128(axis, channel_positions, data, show=True, **kwargs):
    '''
    This function interpolates the data between electrodes and plots it into
    the output.

    Inputs:
    axis is an instance of matplotlib.axes where you want the heatmap to be
    output.
    channel_positions is a Pandas Series with the positions of the electrodes
    (this is the output of polytrode_channels function)
    data is a numpy array containing the data to be interpolate and then
    displayed.
    show is a boolean variable to assert whether you want the heatmap to be
    printed on the screen
    kwargs can :
    - hpos and vpos define the horizontal and vertical position offset of the
      output heatmap, respectively.
    - width and height define the horizontal and vertical scale of the output
      heatmap, respectively.
    - gridscale defines the resolution of the interpolation.
    - interpolation_method defines the method used to interpolate the data
      between positions in channel_positions.
      Choose from:
      ‘none’, ‘nearest’, ‘bilinear’, ‘bicubic’, ‘spline16’, ‘spline36’,
      ‘hanning’, ‘hamming’, ‘hermite’, ‘kaiser’, ‘quadric’, ‘catrom’,
      ‘gaussian’, ‘bessel’, ‘mitchell’, ‘sinc’, ‘lanczos’
    - zlimits defines the limits of the amplitude of the output heatmap.

    Outputs:
    image is the heatmap.
    scat is the grid of electrodes.
    '''
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
        gridscale = 5
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
        warnings.warn('The data passed to gdft_plot_topo contain NaN values. \
        These will create unexpected results in the interpolation. \
        Deal with them.')

     channel_positions = polytrode_channels128(bad_channels=[])

    channel_positions = channel_positions.sort_index(ascending=[1])
    channel_positions = np.array([[x, y] for x, y in channel_positions.values])
    allCoordinates = channel_positions

    naturalWidth = np.max(allCoordinates[:, 0]) - np.min(allCoordinates[:, 0])
    naturalHeight = np.max(allCoordinates[:, 1]) - np.min(allCoordinates[:, 1])

    if not width and not height:
        xScaling = 1
        yScaling = 1
    elif not width and height:
        yScaling = height/naturalHeight
        xScaling = yScaling
    elif width and not height:
        xScaling = width/naturalWidth
        yScaling = xScaling
    elif width and height:
        xScaling = width/naturalWidth
        yScaling = height/naturalHeight

    chanX = channel_positions[:, 0] * xScaling + hpos
    chanY = channel_positions[:, 1] * yScaling + vpos
    chanX = np.max(chanX) - chanX

    hlim = [np.min(chanY), np.max(chanY)]
    vlim = [np.min(chanX), np.max(chanX)]

    if interpolation_method is not 'none':
        yi, xi = np.mgrid[hlim[0]:hlim[1]:complex(0, gridscale)*(hlim[1]-hlim[0]), vlim[0]:vlim[1]:complex(0, gridscale)*(vlim[1]-vlim[0])]
    else:
        # for no interpolation show one pixel per data point
        yi, xi = np.mgrid[hlim[0]:hlim[1]+1, vlim[0]:vlim[1]+1]

    Zi = interpolate.griddata((chanX, chanY), data, (xi, yi))

    if not zlimits:
        vmin = data.min()
        vmax = data.max()
    else:
        vmin = zlimits[0]
        vmax = zlimits[1]

    cmap = plt.get_cmap("jet")
    image = axis.imshow(Zi.T, cmap=cmap, origin=['lower'], vmin=vmin,
                        vmax=vmax, interpolation=interpolation_method,
                        extent=[hlim[0], hlim[1], vlim[0], vlim[1]],
                        aspect='equal')

    scat = axis.scatter(chanY, chanX)
    if show:
        plt.colorbar(image)
        plt.show()
    return image, scat



def plot_video_topoplot_with_juxta128(data, juxtaData, time_axis, channel_positions,
                        times_to_plot=[5.4355, 5.439], time_window=0.000034,
                        time_step=0.000034, sampling_freq=30000,
                        zlimits=[-300, 100],filename='I:\Ephys\Data juxta_extra\data_25_11_72um\paper\\mymovie.avi'):
    '''
    This function creates a video file of time ordered heatmaps.

    Inputs:
    data is a numpy array containing the data to produce all the heatmaps.
    time_axis is a numpy array containing the time points of sampling. (in
    seconds)
    channel_positions is a Pandas Series with the positions of the electrodes
    (this is the output of polytrode_channels function)
    times_to_plot defines the upper and lower bound of the time window to be
    used. (in seconds)
    time_window defines the size of the window to calculate the average. (in
    seconds)
    time_step is the time interval between consecutive heatmaps (in seconds).
    sampling_freq is the sampling frequency, in Hz, at which the data was
    acquired.
    zlimits defines the limits of the magnitude of all heatmaps.
    filename is the path to the output video file.
    '''
    global images, sub_time_indices

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    sample_step = int(time_step * sampling_freq)
    sub_time_indices = np.arange(find_closest(time_axis, times_to_plot[0]),
                                 find_closest(time_axis, times_to_plot[1]))
    sub_time_indices = sub_time_indices[0::sample_step]
    print(sub_time_indices)
    if np.size(channel_positions) <= 64:
        text_y = 8.3
    elif np.size(channel_positions) <= 128:
        text_y = 17
    text_x = -8
    images = []
    for t in sub_time_indices:
        samples = [t, t + (time_window*sampling_freq)]
        print(t)
        data_to_plot = np.mean(data[:, int(samples[0]):int(samples[1])], 1)
        image, scat = plot_topoplot128(ax2, channel_positions, data_to_plot,
                                    show=False, interpmethod="quadric",
                                    gridscale=5, zlimits=zlimits)
        ax2.set_yticks([], minor=False)
        ax2.set_xticks([], minor=False)
        stringy = '%.2f' % (time_axis[t]*1000)
        stringy = 't' + '=' + stringy + ' ms'
        txt = plt.text(x=text_x, y=text_y, s=stringy)
        #ax2.text(4,22, '\u00B5V')
        ax2.set_title("JTA voltage at t / \u00B5V")

        if t < np.max(sub_time_indices) - 30 - np.min(sub_time_indices):
            pointsToPlot = juxtaData[ t-15:t-15+30]
        else:
            pointsToPlot = juxtaData[t-15:t-15+30]
        grafico, = ax1.plot(np.arange(-15, 15)/30., pointsToPlot, 'b')
        #ax1.set_yticks([0], minor=False)
        #ax1.set_yticks([0], minor=True)
        #ax1.yaxis.grid(False, which='major')
        #ax1.yaxis.grid(False, which='minor')
        ax1.set_xticks([0])
        ax1.grid(True)
        ax1.set_title("Juxta Signal/ mV")
        images.append([grafico, image, scat, txt])
    FFwriter = animation.FFMpegWriter()
    ani = animation.ArtistAnimation(fig, images, interval=1000, blit=True,
                                    repeat_delay=1000)
    plt.colorbar(mappable=image)

    if filename is not None:
        ani.save(filename, writer=FFwriter, fps=0.5, bitrate=5000, dpi=300,
                 extra_args=['h264'])
    plt.show()


rootDir=r'D:\Protocols\PairedRecordings\Neuroseeker128\Data\2015-09-09'
videoFilename = "2015_09_09_Pair6.0.avi"

voltage_step_size = 0.195e-6
scale_uV = 1000000
scale_mV = 1000
#sampling_freq = 30000
#low_pass_freq = 5000

alignamplifier=all_cells_ivm_filtered_data['6']
juxtaData = all_cells_patch_data['6']
juxtaData = np.average(juxtaData,axis=-1) * scale_mV
#iir_params = {'order': 3, 'ftype': 'butter', 'padlen': 0}
#juxtaData = filters.low_pass_filter(juxtaData, sampling_freq, low_pass_freq, method='iir', iir_params=iir_params)
data = np.mean(alignamplifier,axis=2)
data = data * scale_uV * voltage_step_size
time_axis = np.arange(-(np.shape(data)[1]/30000.0)/2, (np.shape(data)[1]/30000.0)/2,1/30000.0 )

plot_video_topoplot_with_juxta128(data,juxtaData, time_axis, polytrode_channels128(),
                               times_to_plot=[-0.001, 0.001], zlimits=[np.min(data), np.max(data) * 1.1],
                               filename= os.path.join(rootDir,videoFilename))


