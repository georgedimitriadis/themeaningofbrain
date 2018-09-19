import Layouts.Probes.klustakwik_prb_generator as prb_gen
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate as interpolate
import scipy.stats as stats

import Layouts.Probes.klustakwik_prb_generator as prb_gen

# import Utilities as ut

# P2P, MIN and MAX 128channels and 32channels

voltage_step_size = 0.195e-6
scale_uV = 1000000
scale_ms = 1000
#voltage_step_size = 1
#scale_uV = 1


def peaktopeak(all_cells_ivm_filtered_data, windowSize=60):

    for i in np.arange(0, len(good_cells)):
        extra_average_V = np.average(all_cells_ivm_filtered_data[good_cells[i]][:,:,:],axis=2)
        NumSamples=extra_average_V.shape[1]
        extra_average_microVolts = extra_average_V * scale_uV * voltage_step_size
        NumSites=np.size(extra_average_microVolts,axis = 0)
        lowerBound=int(NumSamples/2.0-windowSize/2.0)
        upperBound=int(NumSamples/2.0+windowSize/2.0)

        #shift=(upperBound-lowerBound)
        #if shift%2 != 0:
        #    shift += 1
        #shift /= 2

        argminima = np.zeros(NumSites)
        for m in range(NumSites):
            argminima[m] = np.argmin(extra_average_microVolts[m][lowerBound:upperBound])+lowerBound
        #argminima = argminima/30 #convert to ms

        argmaxima = np.zeros(NumSites)
        for n in range(NumSites):
            argmaxima[n] = np.argmax(extra_average_microVolts[n][lowerBound:upperBound])+lowerBound
        #argmaxima = argmaxima/30 #convert to ms

        maxima = np.zeros(NumSites)
        for p in range(NumSites):
                maxima[p] = np.max(extra_average_microVolts[p][lowerBound:upperBound])

        minima = np.zeros(NumSites)
        for k in range(NumSites):
            minima[k] = np.min(extra_average_microVolts[k][lowerBound:upperBound])

        p2p = maxima-minima

        stdv_minimos = np.zeros(NumSites)
        stdv_maximos = np.zeros(NumSites)

        stdv = stats.sem(all_cells_ivm_filtered_data[good_cells[i]][:,:,:], axis=2)
        stdv = stdv * voltage_step_size * scale_uV

        for b in range(NumSites):
           stdv_minimos[b]= stdv[b, argminima[b]]
           stdv_maximos[b]= stdv[b, argmaxima[b]]

        error =  np.sqrt((stdv_minimos * stdv_minimos)+ (stdv_maximos*stdv_maximos))


        np.save(os.path.join(analysis_folder,'stdv_minimos_EXTRA_Cell'+ good_cells[i] + '.npy'), stdv_minimos)
        np.save(os.path.join(analysis_folder,'stdv_maximos_EXTRA_Cell'+ good_cells[i] + '.npy'), stdv_maximos)
        np.save(os.path.join(analysis_folder,'stdv_average_EXTRA_Cell'+ good_cells[i] + '.npy'), stdv)
        np.save(os.path.join(analysis_folder,'error_EXTRA_Cell'+ good_cells[i] + '.npy'), error)
        np.save(os.path.join(analysis_folder,'p2p_EXTRA_Cell'+ good_cells[i] + '.npy'), p2p)
        np.save(os.path.join(analysis_folder,'minima_EXTRA_Cell'+ good_cells[i] + '.npy'), minima)
        np.save(os.path.join(analysis_folder,'maxima_EXTRA_Cell'+ good_cells[i] + '.npy'), maxima)
        np.save(os.path.join(analysis_folder,'argmaxima_EXTRA_Cell'+ good_cells[i] + '.npy'), argmaxima)
        np.save(os.path.join(analysis_folder,'argminima_EXTRA_Cell'+ good_cells[i] + '.npy'), argminima)

    return argmaxima, argminima, maxima, minima, p2p

# Value of amplitudes, ERROR (NO ORDER)

filename = r'E:\Data\2014-11-25\Analysis'
amplitudes = np.load(os.path.join(filename + '\p2p_EXTRA_cell4.npy'))
error = np.load(os.path.join(filename +  '\error_EXTRA_cell4.npy'))

min_ampl= amplitudes.min()
max_ampl= amplitudes.max()
channel = amplitudes.argmax()
print(min_ampl)
print(max_ampl)
print(channel)

closest_channel = 2
print(amplitudes[closest_channel])
print(error[closest_channel])



# Heatmapp 32channels for p2p amplitudes


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

#how to plt hetampp?
#Method1

fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
image, scat = plot_topoplot(ax1,polytrode_channels(),amplitudes,show=False, interpmethod="quadric", gridscale=5, zlimits=[np.min(amplitudes), np.max(amplitudes)])
ax1.set_yticks([], minor=False)
ax1.set_xticks([], minor=False)
plt.colorbar(mappable=image, ticks = [np.min(amplitudes), 0, np.max(amplitudes)])
plt.show()

#Method2
# or plot heatmapp like this  but we need the amplitudes by specific order:










# Heatmapp 128channels


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

# How to plot heatmapp?





# Value of amplitudes, ERROR (NO ORDER)

filename = r'D:\Protocols\PairedRecordings\Neuroseeker128\Data\2015-08-26\Analysis'
amplitudes = np.load(os.path.join(filename + '\p2p_EXTRA_cell1.npy'))
error = np.load(os.path.join(filename +  '\error_EXTRA_cell1.npy'))

min_ampl= amplitudes.min()
max_ampl= amplitudes.max()
channel = amplitudes.argmax()
print(min_ampl)
print(max_ampl)
print(channel)





#Method1

fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
image, scat = plot_topoplot128(ax1,polytrode_channels128(),amplitudes,show=False, interpmethod="quadric", gridscale=5, zlimits=[np.min(amplitudes), np.max(amplitudes)])
ax1.set_yticks([], minor=False)
ax1.set_xticks([], minor=False)
plt.colorbar(mappable=image, ticks = [np.min(amplitudes), 0, np.max(amplitudes)])
plt.show()

#Method2
# or plot heatmapp like this  but we need the amplitudes by specific order:

def create_all_electrodes(filename=None, bad_channels=None):

    r1 = np.array([103,	101, 99,	97,	95,	93,	91,	89,	87,	70,	66,	84,	82,	80,	108,	110,	47,	45,	43,	41,	1,61,	57,
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

    if filename is not None:
        prb_gen.generate_prb_file(filename=filename, all_electrodes_array=all_electrodes, channel_number=128)

    return all_electrodes


all_electrodes = create_all_electrodes()

def heatmapp_amplituide(all_cells_ivm_filtered_data, all_electrodes, good_cells_number = 0, windowSize=60):

    extra_average = np.average(all_cells_ivm_filtered_data[good_cells[good_cells_number]][:,:,:],axis=2)
    extra_average_microVolts = extra_average * scale_uV * voltage_step_size
    orderedSites = all_electrodes.reshape(1,128)
    NumSamples=extra_average.shape[1]
    NumSites=np.size(extra_average_microVolts,axis = 0)
    lowerBound=int(NumSamples/2.0-windowSize/2.0)
    upperBound=int(NumSamples/2.0+windowSize/2.0)
    amplitude = np.zeros(128)
    for j in np.arange(128):
        amplitude[j] = np.max(extra_average_microVolts[orderedSites[0][j],lowerBound:upperBound])-np.min(extra_average_microVolts[orderedSites[0][j],lowerBound:upperBound])
    return amplitude

amplitude_order = heatmapp_amplituide(all_cells_ivm_filtered_data, all_electrodes, good_cells_number = 0, windowSize=60)

B = np.copy(amplitude_order)
orderedSites = all_electrodes.reshape(1,128)
Bmod = np.reshape(B,(32,4))
fig, ax = plt.subplots()
plt.axis('off')
im = ax.imshow(Bmod, cmap=plt.get_cmap('jet'),vmin = np.min(B),vmax= np.max(B))
cb = fig.colorbar(im,ticks = [np.min(B), 0,np.max(B)])
cb.ax.tick_params(labelsize = 20)
plt.show()



