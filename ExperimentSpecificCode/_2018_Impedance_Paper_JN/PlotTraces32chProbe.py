import itertools
import os
from math import atan2, degrees

import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import mne.filter as filters
import numpy as np
import pandas as pd
import scipy.ndimage.measurements as sp_m
import scipy.signal as signal

import BrainDataAnalysis.Graphics.ploting_functions as pf
import IO.ephys as ephys
import Layouts.Probes.klustakwik_prb_generator as prb_gen


#File names-------------------------------------------------------------------------------------------------------------

#Neuronexus 5
#recording 2014-11-25  Pair3.0
analysis_folder = r'E:\Paper Impedance\Neuronexus5\Noise\Recording_2014_11_25\pair3.0'
raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier2014-11-25T23_00_08.bin')


#recording 2014-11-25  Pair3.0 after applying CAR
analysis_folder = r'E:\Paper Impedance\Neuronexus5\Noise\Recording_2014_11_25\pair3.0'
raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier2014-11-25T23_00_08_bonsaiCAR0Hz_allchs.bin')
amp_dtype = np.int16


#recording 2014-11-25  Pair2.0
analysis_folder = r'E:\Paper Impedance\Neuronexus5\Noise\Recording_2014_11_25\pair2.0'
raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier2014-11-25T22_44_57.bin')


#recording 2014-11-25  Pair1.0
analysis_folder = r'E:\Paper Impedance\Neuronexus5\Noise\Recording_2014_11_25\pair1.0'
raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier2014-11-25T21_27_13.bin')


#recording 2014-11-13  Pair1.0
analysis_folder = r'E:\Paper Impedance\Neuronexus5\Noise\Recording_2014_11_13\Pair1.0'
raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier2014-11-13T19_01_55.bin')


#recording 2014-11-13  Pair3.0
analysis_folder = r'E:\Paper Impedance\Neuronexus5\Noise\Recording_2014_11_13\pair3.0'
raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier2014-11-13T18_48_11.bin')


#Neuronexus 6 recording 2017-02-02  rec3
analysis_folder = r'E:\Paper Impedance\Neuronexus6\Noise\rec3'
raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier2017-02-02T15_49_35.bin')


#recording 2017-02-02  rec2
analysis_folder = r'E:\Paper Impedance\Neuronexus6\Noise\rec2'
raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier2017-02-02T15_03_44.bin')


#recording 2017-02-02  rec1
analysis_folder = r'E:\Paper Impedance\Neuronexus6\Noise\rec1'
raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier2017-02-02T14_38_11.bin')


#recording 2017-02-02  rec4
analysis_folder = r'E:\Paper Impedance\Neuronexus6\Noise\rec4'
raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier2017-02-02T16_57_16.bin')


#recording 2017-02-02  rec5
analysis_folder = r'E:\Paper Impedance\Neuronexus6\Noise\rec5'
raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier2017-02-02T17_18_46.bin')


#LoriProbe in vivo
analysis_folder = r'F:\DataKilosort\32chprobe\loriprobe\2016_11_05-12_46'
raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier_s162016-11-07T16_18_02.bin')




#Figure2: cluster 147 and 8 examples from recording 23_00_08 and 15_49_35, respectively---------------------------------

#Time spikes for each cluster-------------------------------------------------------------------------------------------
#times=spike_times[kilosort_units[147][:]]
times = spike_times[kilosort_units[8][:]]

#Plot spike times in a window-------------------------------------------------------------------------------------------
start = int(times.shape[0]/2)
#start = int(times.shape[0]/7)
windowstart = times[start]

window_size_secs = 1
filtered_data_type = np.float64
#sampling_freq = 30000
sampling_freq = 20000

high_pass_freq = 250
window_size = int(window_size_secs * sampling_freq)
ms_scale = sampling_freq /1000

plt.figure()
dif = windowstart - times
args = np.argwhere(dif < window_size/2)

for arg in args[:,0]:
    if arg > int(times.shape[0]/2):
        triggerline(window_size_secs*1000/2 + dif[arg][0]/ms_scale, alpha=0.05,color= 'b')
    else:
        triggerline(window_size_secs*1000/2 - dif[arg][0]/ms_scale, alpha=0.05,color='b')



#Open Data--------------------------------------------------------------------------------------------------------------
amp_dtype = np.uint16
Probe_y_digitization = 32768
num_ivm_channels = 32
voltage_step_size = 0.195e-6
scale_uV = 1000000
scale_ms = 1000


#High pass filter-------------------------------------------------------------------------------------------------------
def highpass(data, BUTTER_ORDER=3, F_HIGH=14250, sampleFreq=30000.0, passFreq=100.0):
    b, a = signal.butter(BUTTER_ORDER, (passFreq / (sampleFreq / 2), F_HIGH / (sampleFreq / 2)), 'pass')
    return signal.filtfilt(b, a, data)



#Get the high passed data for the current time window------------------------------------------------------------------
raw_data_ivm = ephys.load_raw_data(raw_data_file_ivm, numchannels=num_ivm_channels, dtype=amp_dtype)
#temp_unfiltered = raw_data_ivm.dataMatrix[ :, window * window_size : (window + 1) * window_size ]
temp_unfiltered = raw_data_ivm.dataMatrix[ :, windowstart - window_size/2 : windowstart + window_size/2 ]
temp_unfiltered = temp_unfiltered.astype(filtered_data_type)
temp_filtered = highpass(temp_unfiltered, F_HIGH=(sampling_freq / 2) * 0.95, sampleFreq=sampling_freq,
                         passFreq=high_pass_freq)


temp_filtered_uV = temp_filtered * scale_uV * voltage_step_size



#Get the raw data for the current time window---------------------------------------------------------------------------
def baseline_correct_basis(data, beginSample=0, endSample=-1):
    if np.size(np.shape(data)) > 1:
        baseline = np.mean(data[:, [beginSample, endSample]], 1)
    else:
        baseline = np.mean(data[beginSample:endSample])
    return np.transpose(np.transpose(data) - np.transpose(baseline))


temp_unfiltered_uV = temp_unfiltered * scale_uV * voltage_step_size

temp_unfiltered_uV_baseline = baseline_correct_basis(temp_unfiltered_uV, beginSample=0, endSample=-1)



#Label channels in the plot --------------------------------------------------------------------------------------------
def labelLine(line, x, label=None, align=True, **kwargs):
    ax = line.get_axes()
    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if (x < xdata[ 0 ]) or (x > xdata[ -1 ]):
        print('x label location is outside data range!')
        return

    # Find corresponding y co-ordinate and angle of the
    ip = 1
    for i in range(len(xdata)):
        if x < xdata[ i ]:
            ip = i
            break

    y = ydata[ ip - 1 ] + (ydata[ ip ] - ydata[ ip - 1 ]) * (x - xdata[ ip - 1 ]) / (xdata[ ip ] - xdata[ ip - 1 ])

    if not label:
        label = line.get_label()

    if align:
        # Compute the slope
        dx = xdata[ ip ] - xdata[ ip - 1 ]
        dy = ydata[ ip ] - ydata[ ip - 1 ]
        ang = degrees(atan2(dy, dx))

        # Transform to screen co-ordinates
        pt = np.array([ x, y ]).reshape((1, 2))
        trans_angle = ax.transData.transform_angles(np.array((ang,)), pt)[ 0 ]

    else:
        trans_angle = 0

    # Set a bunch of keyword arguments
    if 'color' not in kwargs:
        kwargs[ 'color' ] = line.get_color()

    if ('horizontalalignment' not in kwargs) and ('ha' not in kwargs):
        kwargs[ 'ha' ] = 'center'

    if ('verticalalignment' not in kwargs) and ('va' not in kwargs):
        kwargs[ 'va' ] = 'center'

    if 'backgroundcolor' not in kwargs:
        kwargs[ 'backgroundcolor' ] = ax.get_axis_bgcolor()

    if 'clip_on' not in kwargs:
        kwargs[ 'clip_on' ] = True

    if 'zorder' not in kwargs:
        kwargs[ 'zorder' ] = 2.5

    ax.text(x, y, label, rotation=trans_angle, **kwargs)


def labelLines(lines, align=True, xvals=None, **kwargs):
    ax = lines[ 0 ].get_axes()
    labLines = [ ]
    labels = [ ]

    # Take only the lines which have labels other than the default ones
    for line in lines:
        label = line.get_label()
        if "_line" not in label:
            labLines.append(line)
            labels.append(label)

    if xvals is None:
        xmin, xmax = ax.get_xlim()
        xvals = np.linspace(xmin, xmax, len(labLines) + 2)[ 1:-1 ]

    for line, x, label in zip(labLines, xvals, labels):
        labelLine(line, x, label, align, **kwargs)


# Plot a line at time x------------------------------------------------------------------------------------------------
def triggerline(x, **kwargs):
    if x is not None:
        ylim = plt.ylim()
        plt.vlines(x,ylim[0],ylim[1],colors='b', alpha=0.4)



# Plot figure 2 E,F-----------------------------------------------------------------------------------------------------
def plot_average_extra(data, yoffset=1):
    #sites_order_geometry= [16,26,5,14]
    #sites_order_geometry = [15, 11, 16, 5]
    sites_order_geometry = [15, 20, 26, 16, 11, 5]
    num_samples = data.shape[1]
    sample_axis= np.arange(0,num_samples)
    time_axis= sample_axis/sampling_freq
    plt.figure()
    for m in np.arange(np.shape(sites_order_geometry)[0]):
        plt.plot(time_axis*scale_ms, data[sites_order_geometry[m],:].T - yoffset * m, color='k', label =str(sites_order_geometry[m]))
        plt.ylabel('Voltage (\u00B5V)', fontsize=20)
        plt.xlabel('Time (ms)',fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
    labelLines(plt.gca().get_lines(), align=False, fontsize=10)

#plot high-pass filtered data
plt.figure(1)
plot_average_extra(temp_filtered_uV, yoffset=500)
for arg in args[:,0]:
    if arg > int(times.shape[0]/2):
        triggerline(window_size_secs*1000/2 + dif[arg][0]/ms_scale, alpha=0.05,color= 'b')
    else:
        triggerline(window_size_secs*1000/2 - dif[arg][0]/ms_scale, alpha=0.05,color='b')

#plot raw data
plt.figure(2)
plot_average_extra(temp_unfiltered_uV_baseline, yoffset=1000)
for arg in args[:,0]:
    if arg > int(times.shape[0]/2):
        triggerline(window_size_secs*1000/2 + dif[arg][0]/ms_scale, alpha=0.05,colors= 'r')
    else:
        triggerline(window_size_secs*1000/2 - dif[arg][0]/ms_scale, alpha=0.05, colors='r')
















#Code not in use--------------------------------------------------------------------------------------------------------

# Plot the spread out h.p. data-----------------------------------------------------------------------------------------
#my method 1
def plot_average_extra(data, yoffset=1):

    origin_cmap= plt.get_cmap('hsv')
    shrunk_cmap = shiftedColorMap(origin_cmap, start=0.6, midpoint=0.7, stop=1, name='shrunk')
    cm=shrunk_cmap
    cNorm=colors.Normalize(vmin=0,vmax=32)
    scalarMap= cmx.ScalarMappable(norm=cNorm,cmap=cm)
    sites_order_geometry= [0,31,24,7,1,21,10,30,25,6,15,20,11,16,26,5,14,19,12,17,27,4,8,18,13,23,28,3,9,29,2,22]
    num_samples = data.shape[1]
    sample_axis= np.arange(0,num_samples)
    time_axis= sample_axis/sampling_freq
    plt.figure()
    for m in np.arange(np.shape(data)[0]):
        colorVal=scalarMap.to_rgba(np.shape(data)[0]-m)
        plt.plot(time_axis*scale_ms,data[sites_order_geometry[m],:].T - yoffset * m, color=colorVal,label =str(sites_order_geometry[m]))
        plt.ylabel('Voltage (\u00B5V)', fontsize=20)
        plt.xlabel('Time (ms)',fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
    labelLines(plt.gca().get_lines(), align=False, fontsize=10)


plt.figure(1)
plot_average_extra(temp_filtered_uV, yoffset=400)



plt.figure(2)
plot_average_extra(temp_unfiltered_uV_baseline, yoffset=600)



# Plot the spread out h.p. data
#my method 2

col_spacing = 20
row_spacing = 25

new_data = np.zeros(shape=(np.shape(temp_filtered_uV)[0], np.shape(temp_filtered)[1]))

num_of_rows = np.shape(electrode_structure)[ 0 ]
num_of_cols = np.shape(electrode_structure)[ 1 ]

# stdv = np.average(np.std(data_baseline, axis=1), axis=0)

stdv = np.average(np.std(temp_filtered_uV, axis=1), axis=0)

col_spacing = col_spacing * stdv
row_spacing = row_spacing * stdv



for r in np.arange(0, num_of_rows):
    for c in np.arange(0, num_of_cols):
            if electrode_structure[r,c] != -1:
                new_data[ electrode_structure[ r, c ], : ] = temp_filtered_uV[ electrode_structure[ r, c ], : ] - \
                                                             r * col_spacing - \
                                                             c * row_spacing
            else:
                print(str([r,c]))



plt.figure(1)
num_samples = new_data.shape[1]
sample_axis= np.arange(0,num_samples)
time_axis= sample_axis/sampling_freq
for i in np.arange(num_ivm_channels):
    plt.plot(time_axis*scale_ms, new_data[i,:].T, label =str([i]))
labelLines(plt.gca().get_lines(), align=False, fontsize=14)




# Plot the spread out h.p. data
#George method
temp_filtered_spread = pf.spread_data(temp_filtered, electrode_structure, col_spacing=25, row_spacing=15)
plt.figure(1)
plt.plot(new_data[:,:].T)


# Find thresholds
stdvs = np.median(np.abs(temp_filtered_uV) / 0.6745, axis=1)

large_thresholds = np.zeros(np.shape(temp_filtered))
small_thresholds = np.zeros(np.shape(temp_filtered))
for c in range(num_ivm_channels):
    large_thresholds[ c, : ] = 7 * stdvs[ c ]
    small_thresholds[ c, : ] = 2 * stdvs[ c ]

# Generate thresholded array of -1 (if negative threshold is passed), +1 (if possitive threshold is passed)
# and 0 otherwise
threshold_crossing_regions = np.zeros(np.shape(temp_filtered))
threshold_crossing_regions[ temp_filtered < -large_thresholds ] = -1
threshold_crossing_regions[ temp_filtered > large_thresholds ] = 1

# Put the thresholded data on the 2D probe with time the 3rd dimension

electrode_structure, channel_positions = create_32channels_nn_prb(bad_channels=[-1])



on_probe_shape = tuple(np.concatenate((np.shape(electrode_structure), [ np.shape(threshold_crossing_regions)[ 1 ]])))
threshold_crossing_regions_on_probe = np.zeros(on_probe_shape)

for r in range(23):
    for c in range(3):
        threshold_crossing_regions_on_probe[ r, c, : ] = threshold_crossing_regions[ electrode_structure[ r, c ], : ]

labels, num_of_features = sp_m.label(threshold_crossing_regions_on_probe)
object_slices = sp_m.find_objects(labels)


for i in range(num_of_features):
    start_channel = object_slices[ i ][ 0 ].start * 4 + object_slices[ i ][ 1 ].start
    end_channel = object_slices[ i ][ 0 ].stop * 4 + object_slices[ i ][ 1 ].stop
    channels_in_each_label.append([ start_channel, end_channel ])
    print(channels_in_each_label[ i ])


# Plot the thresholded data spread out
#my method
col_spacing = 20
row_spacing = 25

new_data = np.zeros(shape=(np.shape(threshold_crossing_regions)[0], np.shape(threshold_crossing_regions)[1]))

num_of_rows = np.shape(electrode_structure)[ 0 ]
num_of_cols = np.shape(electrode_structure)[ 1 ]

# stdv = np.average(np.std(data_baseline, axis=1), axis=0)

stdv = np.average(np.std(threshold_crossing_regions, axis=1), axis=0)

col_spacing = col_spacing * stdv
row_spacing = row_spacing * stdv



for r in np.arange(0, num_of_rows):
    for c in np.arange(0, num_of_cols):
            if electrode_structure[r,c] != -1:
                new_data[ electrode_structure[ r, c ], : ] = threshold_crossing_regions[ electrode_structure[ r, c ], : ] - \
                                                             r * col_spacing - \
                                                             c * row_spacing
            else:
                print(str([r,c]))


plt.figure(2)
for i in np.arange(num_ivm_channels):
    plt.plot(new_data[i,:].T, label =str([i]))
labelLines(plt.gca().get_lines(), align=False, fontsize=14)

# Plot the thresholded data spread out
#george metodh
spread_threshold_crossing_regions = pf.spread_data(threshold_crossing_regions, electrode_structure, col_spacing=50,


                                                   row_spacing=2)
plt.figure(2)
plt.plot(spread_threshold_crossing_regions.T)

# Plot the thresholded data on the probe one time point at a time
#george metodh
pf.scan_through_image_stack(threshold_crossing_regions_on_probe)



# Trying to lower noise level in neuronexus 5

window = 0
window_size_secs = 10
filtered_data_type = np.float64
high_pass_freq = 500
window_size = int(window_size_secs * sampling_freq)

amp_dtype = np.uint16
Probe_y_digitization = 32768
num_ivm_channels = 32
voltage_step_size = 0.195e-6
scale_uV = 1000000
scale_ms = 1000
ch = 0

def highpass(data, BUTTER_ORDER=3, F_HIGH=14250, sampleFreq=30000.0, passFreq=100.0):
    b, a = signal.butter(BUTTER_ORDER, (passFreq / (sampleFreq / 2), F_HIGH / (sampleFreq / 2)), 'pass')
    return signal.filtfilt(b, a, data)


# Neuronexus 6 recording 2017-02-02  rec3
analysis_folder = r'E:\Paper Impedance\Neuronexus6\Noise\rec3'
raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier2017-02-02T15_49_35.bin')
sampling_freq = 20000

# Get the high passed data for the current time window
raw_data_ivm = ephys.load_raw_data(raw_data_file_ivm, numchannels=num_ivm_channels, dtype=amp_dtype)
temp_unfiltered = raw_data_ivm.dataMatrix[ :, window * window_size:(window + 1) * window_size ]
temp_unfiltered = temp_unfiltered.astype(filtered_data_type)
temp_filtered = highpass(temp_unfiltered, F_HIGH=(sampling_freq / 2) * 0.95, sampleFreq=sampling_freq,
                         passFreq=high_pass_freq)

# temp_filtered = filters.high_pass_filter(temp_unfiltered,sampling_freq, high_pass_freq, method='iir',iir_params=iir_params)

temp_filtered_uV = temp_filtered * scale_uV * voltage_step_size

num_samples = temp_filtered_uV.shape[1]
sample_axis = np.arange(0, num_samples)
time_axis = sample_axis/sampling_freq
yoffset = 0
plt.figure()
plt.plot(time_axis*scale_ms,temp_filtered_uV[ch,:].T, color='r', label= str(ch) + '_Neuronexus6_Rec3')



# Neuronexus 5
#recording 2014-11-25  Pair3.0
analysis_folder = r'E:\Paper Impedance\Neuronexus5\Noise\Recording_2014_11_25\pair3.0'
raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier2014-11-25T23_00_08.bin')
sampling_freq = 30000


# Neuronexus 5
#recording 2014-11-25  Pair3.0
analysis_folder = r'E:\Paper Impedance\Neuronexus5\Noise\Recording_2014_11_25\pair3.0'
raw_data_file_ivm = os.path.join(analysis_folder + '\\' + 'amplifier2014-11-25T23_00_08_bonsaiCAR1Hz_allchs.bin')
sampling_freq = 30000
amp_dtype = np.int16

# Get the high passed data for the current time window
raw_data_ivm = ephys.load_raw_data(raw_data_file_ivm, numchannels=num_ivm_channels, dtype=amp_dtype)
temp_unfiltered = raw_data_ivm.dataMatrix[ :, window * window_size:(window + 1) * window_size ]
temp_unfiltered = temp_unfiltered.astype(filtered_data_type)
temp_filtered = highpass(temp_unfiltered, F_HIGH=(sampling_freq/ 2) * 0.95, sampleFreq=sampling_freq,
                         passFreq=high_pass_freq)


temp_filtered_uV = temp_filtered * scale_uV * voltage_step_size

#Comon reference average--------------------------------------------------------------------------------

def common_average_reference(array):
    return np.median(array, axis=0)


def chunkwise_common_average_ref(array):

    block_median = common_average_reference(array)
    denoised = array - block_median

    return denoised


#LOW-PASS
low_pass_freq = 5000
iir_params = {'order': 3, 'ftype': 'butter', 'padlen': 0}

temp_filtered_band = filters.low_pass_filter(temp_filtered, sampling_freq, low_pass_freq, method='iir', iir_params=iir_params)

# temp_filtered = filters.high_pass_filter(temp_unfiltered,sampling_freq, high_pass_freq, method='iir',iir_params=iir_params)

temp_filtered_uV = temp_filtered_band * scale_uV * voltage_step_size

plt.plot(time_axis*scale_ms,temp_filtered_uV[ch,:].T - yoffset, color='r',label= str(ch) + '_Neuronexus5_Pair3.0')
plt.ylabel('Voltage (\u00B5V)', fontsize=20)
plt.xlabel('Time (ms)',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(frameon=False)



# Colormap from Pylyb

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


# Plot a line at time x

def triggerline(x):
    if x is not None:
        ylim = plt.ylim()
        plt.vlines(x,ylim[0],ylim[1])


def create_32channels_nn_prb(filename=None, bad_channels=None):
    """
    This function produces a grid with the electrodes positions of
    Neuronexus A1x32-Poly3-5mm-25s-177-CM32 silicone probe
    Parameters
    ----------
    filename -- the filename (with path) of the .prb file for klustakwik
    bad_channels -- a list or numpy array with channels you may want to disregard
    Returns
    -------
    all_electrodes --
    channel_positions -- a Pandas Series with the electrodes positions (in
    two dimensions)
    """

    electrode_amplifier_index_on_grid = np.array([ -1, 0, -1,
                                                   -1, -1, -1,
                                                   -1, 31, -1,
                                                   24, -1, 7,
                                                   -1, 1, -1,
                                                   21, -1, 10,
                                                   -1, 30, -1,
                                                   25, -1, 6,
                                                   -1, 15, -1,
                                                   20, -1, 11,
                                                   -1, 16, -1,
                                                   26, -1, 5,
                                                   -1, 14, -1,
                                                   19, -1, 12,
                                                   -1, 17, -1,
                                                   27, -1, 4,
                                                   -1, 8, -1,
                                                   18, -1, 13,
                                                   -1, 23, -1,
                                                   28, -1, 3,
                                                   -1, 9, -1,
                                                   29, -1, 2,
                                                   -1, 22, -1])

    all_electrodes = electrode_amplifier_index_on_grid.reshape((23, 3))
    if filename is not None:
        prb_gen.generate_prb_file(filename=filename, all_electrodes_array=all_electrodes, channel_number=32,
                                  steps_r=3, steps_c=3)

    electrode_coordinate_grid = list(itertools.product(np.arange(1, 24), np.arange(1, 4)))
    #electrode_coordinate_grid = [ tuple(reversed(x)) for x in electrode_coordinate_grid ]
    electrode_amplifier_name_on_grid = np.array([ "Int" + str(x) for x in electrode_amplifier_index_on_grid ])
    indices_arrays = [ electrode_amplifier_index_on_grid.tolist(), electrode_amplifier_name_on_grid.tolist() ]
    indices_tuples = list(zip(*indices_arrays))
    channel_position_indices = pd.MultiIndex.from_tuples(indices_tuples, names=[ 'Numbers', 'Strings' ])
    channel_positions = pd.Series(electrode_coordinate_grid, index=channel_position_indices)
    channel_positions.columns = [ 'Positions' ]
    channel_positions = channel_positions.reset_index(level=None, drop=False, name='Positions')
    if bad_channels is not None:
        channel_positions = channel_positions[~channel_positions.Numbers.isin(bad_channels) ]

    return all_electrodes, channel_positions

#--------------------------------------------------------------------------------------------------------


# electrode_structure is length x width #of electrodes with the number of the intan data in each element
def spread_data(data, electrode_structure, col_spacing=3, row_spacing=0.5):
    # data_baseline = (data.T - np.median(data, axis=1).T).T

    new_data = np.zeros(shape=(np.shape(data)[0], np.shape(data)[1]))

    num_of_rows = np.shape(electrode_structure)[ 0 ]
    num_of_cols = np.shape(electrode_structure)[ 1 ]

    # stdv = np.average(np.std(data_baseline, axis=1), axis=0)

    stdv = np.average(np.std(data, axis=1), axis=0)

    col_spacing = col_spacing * stdv
    row_spacing = row_spacing * stdv


    for r in np.arange(0, num_of_rows):
        for c in np.arange(0, num_of_cols):
                new_data[ electrode_structure[ r, c ], : ] = data[ electrode_structure[ r, c ], : ] - \
                                                             r * col_spacing - \
                                                             c * row_spacing
    return new_data



# ----------------------------------------------------------------
# Label line with line2D label data
def labelLine(line, x, label=None, align=True, **kwargs):
    ax = line.get_axes()
    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if (x < xdata[ 0 ]) or (x > xdata[ -1 ]):
        print('x label location is outside data range!')
        return

    # Find corresponding y co-ordinate and angle of the
    ip = 1
    for i in range(len(xdata)):
        if x < xdata[ i ]:
            ip = i
            break

    y = ydata[ ip - 1 ] + (ydata[ ip ] - ydata[ ip - 1 ]) * (x - xdata[ ip - 1 ]) / (xdata[ ip ] - xdata[ ip - 1 ])

    if not label:
        label = line.get_label()

    if align:
        # Compute the slope
        dx = xdata[ ip ] - xdata[ ip - 1 ]
        dy = ydata[ ip ] - ydata[ ip - 1 ]
        ang = degrees(atan2(dy, dx))

        # Transform to screen co-ordinates
        pt = np.array([ x, y ]).reshape((1, 2))
        trans_angle = ax.transData.transform_angles(np.array((ang,)), pt)[ 0 ]

    else:
        trans_angle = 0

    # Set a bunch of keyword arguments
    if 'color' not in kwargs:
        kwargs[ 'color' ] = line.get_color()

    if ('horizontalalignment' not in kwargs) and ('ha' not in kwargs):
        kwargs[ 'ha' ] = 'center'

    if ('verticalalignment' not in kwargs) and ('va' not in kwargs):
        kwargs[ 'va' ] = 'center'

    if 'backgroundcolor' not in kwargs:
        kwargs[ 'backgroundcolor' ] = ax.get_axis_bgcolor()

    if 'clip_on' not in kwargs:
        kwargs[ 'clip_on' ] = True

    if 'zorder' not in kwargs:
        kwargs[ 'zorder' ] = 2.5

    ax.text(x, y, label, rotation=trans_angle, **kwargs)


def labelLines(lines, align=True, xvals=None, **kwargs):
    ax = lines[ 0 ].get_axes()
    labLines = [ ]
    labels = [ ]

    # Take only the lines which have labels other than the default ones
    for line in lines:
        label = line.get_label()
        if "_line" not in label:
            labLines.append(line)
            labels.append(label)

    if xvals is None:
        xmin, xmax = ax.get_xlim()
        xvals = np.linspace(xmin, xmax, len(labLines) + 2)[ 1:-1 ]

    for line, x, label in zip(labLines, xvals, labels):
        labelLine(line, x, label, align, **kwargs)

