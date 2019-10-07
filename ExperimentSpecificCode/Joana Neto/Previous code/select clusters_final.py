import IO.ephys as ephys
import matplotlib.colors as colors
import matplotlib.cm as cmx
import os
import scipy.signal as signal
import scipy.stats as stats

import numpy as np
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import warnings
# import Utilities as ut
import matplotlib.animation as animation

base_folder = r'Z:\j\Neuroseeker256ch_probe6'

date = {1: '2017-02-08', 2:''}

all_recordings_capture_times = {1: {'1': '21_38_55', '2': '20_04_54', '3': '15_34_04'},
                                2: {'1':'','2':''}}

recordings_cluster = {1: ['1','2','3'],
                      2:['1','2']}

cluster_indices_good = {1: {'1': [46, 186, 252, 202], '2': [27, 53, 12, 42, 44], '3': [19, 6, 37, 98, 228, 64]},
                        2:{'1': [], '2': []}}

#----------------------------------------------------------------------------------------------------------------------

surgery = 1
pick_recording = '1' #we need to select the recording


recordings = recordings_cluster[surgery]
clusterSelected = cluster_indices_good[surgery][pick_recording]


date = date[surgery]
data_folder = os.path.join(base_folder + '\\' + date, 'Data')
analysis_folder = os.path.join(base_folder + '\\' + date, 'Analysis')

recordings_capture_times = all_recordings_capture_times[surgery]

#----------------------------------------------------------------------------------------------------------------------

num_of_points_in_spike_trig_ivm = 128
num_of_points_for_padding = 2 * num_of_points_in_spike_trig_ivm


inter_spike_time_distance = 30

num_ivm_channels = 256
amp_dtype = np.uint16

sampling_freq = 20000
high_pass_freq = 500
filtered_data_type = np.float64

types_of_data_to_load = {'t': 'ivm_data_filtered_cell{}.dat',
                         'f': 'ivm_data_filtered_cell{}',
                         'c': 'ivm_data_filtered_continous_cell{}.dat',
                         'k': 'ivm_data_filtered_klusta_spikes_cell{}.dat',
                         'm': 'ivm_data_raw_cell{}.dat',
                         }

types_of_sorting = {'s': 'goodcluster_{}_'}


#Filter for extracellular recording
def highpass(data,BUTTER_ORDER=3, F_HIGH=14250,sampleFreq=30000.0,passFreq=500.0):
    b, a = signal.butter(BUTTER_ORDER,(passFreq/(sampleFreq/2), F_HIGH/(sampleFreq/2)),'pass')
    return signal.filtfilt(b,a,data)


# Generate the (channels x time_points x spikes) high passed extracellular recordings datasets for all cells
all_cells_ivm_filtered_data = {}
data_to_load = 't'
cluster_to_load = 's'
passFreq = high_pass_freq


raw_data_file_ivm = os.path.join(data_folder + '\\' + 'amplifier' + date + 'T' + all_recordings_capture_times[surgery][pick_recording] + '.bin')
raw_data_ivm = ephys.load_raw_data(raw_data_file_ivm, numchannels=num_ivm_channels, dtype=amp_dtype)

spike_template_folder = os.path.join(base_folder + '\\' + date + '\\' + 'Datakilosort' +  '\\' + 'nfilt256'+ '\\' + 'amplifier' + date + 'T' + all_recordings_capture_times[surgery][pick_recording])
spike_templates = np.load(os.path.join(spike_template_folder + '\\' + 'spike_clusters.npy'))
spike_templates = np.reshape(spike_templates, ((len(spike_templates)),1))
spike_times = np.load(os.path.join(spike_template_folder + '\\' + 'spike_times.npy'))


kilosort_units = {}
for i in np.arange(len(spike_templates)):
    cluster = spike_templates[i][0]
    if cluster in kilosort_units:
        kilosort_units[cluster] = np.append(kilosort_units[cluster], i)
    else:
        kilosort_units[cluster] = i

cluster_info = pd.DataFrame(columns=['Cluster', 'Num_of_Spikes', 'Spike_Indices'])
cluster_info = cluster_info.set_index('Cluster')
cluster_info['Spike_Indices'] = cluster_info['Spike_Indices'].astype(list)

for g in kilosort_units.keys():
    if np.size(kilosort_units[g]) == 1:
        kilosort_units[g] = [kilosort_units[g]]
    cluster_name = str(g)
    cluster_info.set_value(cluster_name, 'Num_of_Spikes', len(kilosort_units[g]))
    cluster_info.set_value(cluster_name, 'Spike_Indices', kilosort_units[g])


for clus in np.arange(0, len(clusterSelected)):
    num_of_spikes = len(kilosort_units[clusterSelected[clus]])

    shape_of_filt_spike_trig_ivm = ((num_ivm_channels,
                                 num_of_points_in_spike_trig_ivm,
                                 num_of_spikes))

    ivm_data_filtered = np.memmap(os.path.join(analysis_folder, types_of_sorting[cluster_to_load].format(clusterSelected[clus]) + types_of_data_to_load[data_to_load].format(pick_recording)),
                                               dtype=filtered_data_type,
                                               mode='w+',
                                               shape=shape_of_filt_spike_trig_ivm)

    for spike in np.arange(0, num_of_spikes):
        trigger_point = spike_times[kilosort_units[clusterSelected[clus]][spike]]
        start_point = int(trigger_point - (num_of_points_in_spike_trig_ivm + num_of_points_for_padding)/2)
        if start_point < 0:
            break
        end_point = int(trigger_point + (num_of_points_in_spike_trig_ivm + num_of_points_for_padding)/2)
        if end_point > raw_data_ivm.shape()[1]:
            break
        temp_unfiltered = raw_data_ivm.dataMatrix[:, start_point:end_point]
        temp_unfiltered = temp_unfiltered.astype(filtered_data_type)
        temp_filtered = highpass(temp_unfiltered)
        temp_filtered = temp_filtered[:, int(num_of_points_for_padding / 2):-int(num_of_points_for_padding / 2)]
        ivm_data_filtered[:, :, spike] = temp_filtered
del ivm_data_filtered


# Load the extracellular recording cut data from the .dat files on hard disk onto memmaped arrays

all_cells_ivm_filtered_data = {}
data_to_load = 't'
cluster_to_load = 's'

for clus in np.arange(0, len(clusterSelected)):
    num_of_spikes = len(kilosort_units[clusterSelected[clus]])
    if data_to_load == 't':
        shape_of_filt_spike_trig_ivm = (num_ivm_channels,
                                    num_of_points_in_spike_trig_ivm,
                                    num_of_spikes)
        time_axis = np.arange(-num_of_points_in_spike_trig_ivm/(2*sampling_freq),
                  num_of_points_in_spike_trig_ivm/(2*sampling_freq),
                  1/sampling_freq)

    ivm_data_filtered = np.memmap(os.path.join(analysis_folder, types_of_sorting[cluster_to_load].format(clusterSelected[clus]) + types_of_data_to_load[data_to_load].format(pick_recording)),
                                dtype=filtered_data_type,
                                mode='r',
                                shape=shape_of_filt_spike_trig_ivm)

    all_cells_ivm_filtered_data[clusterSelected[clus]] = ivm_data_filtered


#PLOT
#----------------------------------------------------------------------------------------------------------------------

#Autocorrelagram for each cluster--------------------------------------------------------------------------------------
def crosscorrelate_spike_trains(spike_times_train_1, spike_times_train_2, lag=None):
    if spike_times_train_1.size < spike_times_train_2.size:
        if lag is None:
            lag = np.ceil(10 * np.mean(np.diff(spike_times_train_1)))
        reverse = False
    else:
        if lag is None:
            lag = np.ceil(20 * np.mean(np.diff(spike_times_train_2)))
        spike_times_train_1, spike_times_train_2 = spike_times_train_2, spike_times_train_1
        reverse = True

    differences = np.array([])
    for k in np.arange(0, spike_times_train_1.size):
        differences = np.append(differences, spike_times_train_1[k] - spike_times_train_2[np.nonzero(
            (spike_times_train_2 > spike_times_train_1[k] - lag)
             & (spike_times_train_2 < spike_times_train_1[k] + lag)
             & (spike_times_train_2 != spike_times_train_1[k]))])
    if reverse is True:
        differences = -differences
    norm = np.sqrt(spike_times_train_1.size * spike_times_train_2.size)
    return differences, norm


lag = 50

for pick_cluster in clusterSelected:
    times_cluster = np.reshape(spike_times[kilosort_units[pick_cluster]], (spike_times[kilosort_units[pick_cluster]].shape[0]))
    diffs, norm = crosscorrelate_spike_trains((times_cluster/30).astype(np.int64),(times_cluster/30).astype(np.int64),lag=50)
    #hist, edges = np.histogram(diffs)
    plt.figure()
    plt.hist(diffs, bins=101, normed=True, range=(-50,50), align ='mid') #if normed = True, the probability density function at the bin, normalized such that the integral over the range is 1.
    plt.xlim(-lag,lag)


# Colormap from Pylyb--------------------------------------------------------------------------------------------------

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

# Plot a line at time x------------------------------------------------------------------------------------------------

def triggerline(x):
    if x is not None:
        ylim = plt.ylim()
        plt.vlines(x,ylim[0],ylim[1])


# Return mapping to plot in space---------------------------------------------------------------------------------------
def create_256channels_neuroseeker_prb(bad_channels=None):
    '''
     This function produces a grid with the electrodes positions for the 256 channel probe

     Inputs:
     bad_channels is a list or numpy array with channels you may want to
     disregard.

     Outputs:
     channel_positions is a Pandas Series with the electrodes positions (in
     two dimensions)
     '''


    electrode_amplifier_index_on_grid = np.genfromtxt(r'Z:\j\Neuroseeker256ch_probe6\2017-02-08\chanmap.csv', delimiter=",")
    all_electrodes = electrode_amplifier_index_on_grid.astype(np.int16)

    return all_electrodes


# Plot 256 channels averages overlaid
all_electrodes = create_256channels_neuroseeker_prb()
voltage_step_size = 0.195e-6
scale_uV = 1000000
scale_ms = 1000

def plot_average_extra(all_cells_ivm_filtered_data, electrode_structure=all_electrodes, yoffset=1):
    origin_cmap= plt.get_cmap('hsv')
    shrunk_cmap = shiftedColorMap(origin_cmap, start=0.6, midpoint=0.7, stop=1, name='shrunk')
    cm = shrunk_cmap
    cNorm = colors.Normalize(vmin=0,vmax=255)
    scalarMap = cmx.ScalarMappable(norm=cNorm,cmap=cm)
    subplot_number_array = electrode_structure.reshape(1,255)
    for clus in np.arange(0, len(clusterSelected)):
        num_of_spikes = len(kilosort_units[clusterSelected[clus]])
        extra_average_V = np.average(all_cells_ivm_filtered_data[clusterSelected[clus]][:,:,:],axis=2)
        num_samples=extra_average_V.shape[1]
        sample_axis= np.arange(-(num_samples/2),(num_samples/2))
        time_axis= sample_axis/sampling_freq
        extra_average_microVolts = extra_average_V * scale_uV * voltage_step_size
        plt.figure()
        for m in np.arange(np.shape(extra_average_microVolts)[0]-1):
            colorVal = scalarMap.to_rgba(np.shape(extra_average_microVolts)[0]-m)
            plt.plot(time_axis * scale_ms, extra_average_microVolts[subplot_number_array[:,m],:].T, color=colorVal)
            plt.xlim(-2, 2) #window 4ms
            plt.ylim(np.min(extra_average_microVolts)-yoffset,np.max(extra_average_microVolts)+yoffset )
            plt.ylabel('Voltage (\u00B5V)', fontsize=20)
            plt.xlabel('Time (ms)',fontsize=20)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            triggerline(0)


# Plot 256 channels averages in space
voltage_step_size = 0.195e-6
scale_uV = 1000000
scale_ms = 1000

def plot_average_extra_geometry(all_cells_ivm_filtered_data, electrode_structure=all_electrodes):
    origin_cmap= plt.get_cmap('hsv')
    shrunk_cmap = shiftedColorMap(origin_cmap, start=0.6, midpoint=0.7, stop=1, name='shrunk')
    cm=shrunk_cmap
    cNorm=colors.Normalize(vmin=0,vmax=255)
    scalarMap= cmx.ScalarMappable(norm=cNorm,cmap=cm)
    subplot_number_array = electrode_structure.reshape(1,255)
    for clus in np.arange(0, len(clusterSelected)):
        plt.figure(clus+1)
        num_of_spikes = len(kilosort_units[clusterSelected[clus]])
        extra_average_V = np.average(all_cells_ivm_filtered_data[clusterSelected[clus]][:,:,:],axis=2)
        num_samples=extra_average_V.shape[1]
        sample_axis= np.arange(-(num_samples/2),(num_samples/2))
        time_axis= sample_axis/sampling_freq
        extra_average_microVolts = extra_average_V * scale_uV * voltage_step_size
        for i in np.arange(1, subplot_number_array.shape[1]+1):
            plt.subplot(17,15,i)
            colorVal = scalarMap.to_rgba(subplot_number_array.shape[1]-i)
            plt.plot(time_axis*scale_ms, extra_average_microVolts[np.int(subplot_number_array[:,i-1]),:],color=colorVal)
            plt.ylim(np.min(np.min(extra_average_microVolts, axis=1)), np.max(np.max(extra_average_microVolts, axis=1)))
            plt.xlim(-2, 2)
            plt.axis("OFF")

#--------------------------------------------------------------------------------------------------------------------
# Each cluster P2P, MIN, MAX 256channels

voltage_step_size = 0.195e-6
scale_uV = 1000000
scale_ms = 1000
#voltage_step_size = 1
#scale_uV = 1


def peaktopeak(all_cells_ivm_filtered_data, windowSize=60):

    for clus in np.arange(0, len(clusterSelected)):
        extra_average_V= np.average(all_cells_ivm_filtered_data[clusterSelected[clus]][:,:,:],axis=2) * voltage_step_size
        NumSamples=extra_average_V.shape[1]
        extra_average_microVolts = extra_average_V * scale_uV
        NumSites=np.size(extra_average_microVolts,axis = 0)
        lowerBound=int(NumSamples/2.0-windowSize/2.0)
        upperBound=int(NumSamples/2.0+windowSize/2.0)

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

        stdv = stats.sem(all_cells_ivm_filtered_data[clusterSelected[clus]][:,:,:],axis=2)
        stdv = stdv * voltage_step_size * scale_uV

        for b in range(NumSites):
           stdv_minimos[b]= stdv[b, argminima[b]]
           stdv_maximos[b]= stdv[b, argmaxima[b]]

        error =  np.sqrt((stdv_minimos * stdv_minimos)+ (stdv_maximos*stdv_maximos))


        np.save(os.path.join(analysis_folder,'stdv_minimos_EXTRA_Cluster'+ str(clusterSelected[clus]) + '.npy'), stdv_minimos)
        np.save(os.path.join(analysis_folder,'stdv_maximos_EXTRA_Cluster'+ str(clusterSelected[clus])  + '.npy'), stdv_maximos)
        np.save(os.path.join(analysis_folder,'stdv_average_EXTRA_Cluster'+ str(clusterSelected[clus])+ '.npy'), stdv)
        np.save(os.path.join(analysis_folder,'error_EXTRA_Cluster'+ str(clusterSelected[clus])+ '.npy'), error)
        np.save(os.path.join(analysis_folder,'p2p_EXTRA_Cluster'+ str(clusterSelected[clus]) + '.npy'), p2p)
        np.save(os.path.join(analysis_folder,'minima_EXTRA_Cluster'+ str(clusterSelected[clus]) + '.npy'), minima)
        np.save(os.path.join(analysis_folder,'maxima_EXTRA_Cluster'+ str(clusterSelected[clus])+ '.npy'), maxima)
        np.save(os.path.join(analysis_folder,'argmaxima_EXTRA_Cluster'+ str(clusterSelected[clus]) + '.npy'), argmaxima)
        np.save(os.path.join(analysis_folder,'argminima_EXTRA_Cluster'+ str(clusterSelected[clus]) + '.npy'), argminima)

    return argmaxima, argminima, maxima, minima, p2p

#Print the P2P. MIN, MAX------------------------------------------------------------------------------------------------

for pick_cluster in clusterSelected:
    filename_p2p = os.path.join(analysis_folder + '\p2p_EXTRA_Cluster' + str(pick_cluster) + '.npy')
    filename_error= os.path.join(analysis_folder +  '\error_EXTRA_Cluster' + str(pick_cluster) +'.npy')
    amplitudes = np.load(filename_p2p)
    error = np.load(filename_error)
    min_ampl= amplitudes.min()
    max_ampl= amplitudes.max()
    channel = amplitudes.argmax()

    print('#------------------------------------------------------')
    print('recording_'+ str(pick_recording))
    print('cluster_'+ str(pick_cluster))
    print('min_p2p ' + str(min_ampl))
    print('max_p2p ' + str(max_ampl))
    print('maxp2p_channel ' + str(channel))
    select_indices_biggestAmp = np.where(amplitudes >= (amplitudes.max()/2))
    print('Channels_atleasthalfofmaxp2p ' + str(select_indices_biggestAmp))
    Amp_select_indices = amplitudes[select_indices_biggestAmp]
    print('Amplitude_atleasthalfofmaxp2p ' + str(Amp_select_indices))
    print('#------------------------------------------------------')

#Plot Heatmap and movies------------------------------------------------------------------------------------------------

def polytrode_256channels(bad_channels=[]):
    '''
     This function produces a grid with the electrodes positions for the 256 channel probe

     Inputs:
     bad_channels is a list or numpy array with channels you may want to
     disregard.

     Outputs:
     channel_positions is a Pandas Series with the electrodes positions (in
     two dimensions)
     '''

    electrode_coordinate_grid = list(itertools.product(np.arange(0, 18),
                                                       np.arange(0, 15)))

    electrode_amplifier_index_on_grid = np.genfromtxt(r'Z:\j\Neuroseeker256ch_probe6\2017-02-08\chanmap256.csv', delimiter=",")

    electrode_amplifier_index_on_grid = electrode_amplifier_index_on_grid.astype(np.int16)


    reshaped = np.reshape(electrode_amplifier_index_on_grid,np.shape(electrode_amplifier_index_on_grid)[0]*np.shape(electrode_amplifier_index_on_grid)[1])

    electrode_amplifier_index_on_grid = reshaped
    electrode_amplifier_name_on_grid = np.array(["Int"+str(x) for x in electrode_amplifier_index_on_grid])
    indices_arrays = [electrode_amplifier_index_on_grid.tolist(), electrode_amplifier_name_on_grid.tolist()]
    indices_tuples = list(zip(*indices_arrays))

    channel_position_indices = pd.MultiIndex.from_tuples(indices_tuples, names=['Numbers', 'Strings'])
    channel_positions = pd.Series(electrode_coordinate_grid, index = channel_position_indices)

    channel_positions.columns=['Positions']
    channel_positions = channel_positions.reset_index(level=None, drop=False, name='Positions')
    if bad_channels is not None:
        channel_positions = channel_positions[~channel_positions.Numbers.isin(bad_channels)]
    return channel_positions


def plot_topoplot256(axis, channel_positions, data, show=True, **kwargs):
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

    channel_positions = channel_positions.sort('Numbers', ascending=True)
    channel_positions = np.array([[x, y] for x, y in channel_positions.Positions])
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

#Method1 by using plot_topoplot

for pick_cluster in clusterSelected:
    filename_p2p = os.path.join(analysis_folder + '\p2p_EXTRA_Cluster' + str(pick_cluster) + '.npy')
    filename_error= os.path.join(analysis_folder +  '\error_EXTRA_Cluster' + str(pick_cluster) +'.npy')
    amplitudes = np.load(filename_p2p)
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    channel_positions=polytrode_256channels(bad_channels=[-1])
    image, scat = plot_topoplot256(ax1, channel_positions, amplitudes, show=False, interpmethod="quadric", gridscale=5, zlimits=[np.min(amplitudes), np.max(amplitudes)])
    ax1.set_yticks([], minor=False)
    ax1.set_xticks([], minor=False)
    plt.colorbar(mappable=image, ticks = [np.min(amplitudes), 0, np.max(amplitudes)])
    plt.show()




def plot_video_topoplot_with_juxta256(data, juxtaData, time_axis, channel_positions,
                        times_to_plot=[5.4355, 5.439], time_window=0.000034,
                        time_step=0.000034, sampling_freq=20000,
                        zlimits=[-300, 100],filename=r'Z:\j\Neuroseeker256ch_probe6\2017-02-08\Analysis\mymovie.avi'):
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
        image, scat = plot_topoplot256(ax2, channel_positions, data_to_plot,
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
    ani = animation.ArtistAnimation(fig, images, interval=1000, blit=True,repeat_delay=1000)
    plt.colorbar(mappable=image)

    if filename is not None:
        ani.save(filename, writer=FFwriter, fps=0.5, bitrate=5000, dpi=300,
                 extra_args=['h264'])
    plt.show()

#Method by using plot_video_topoplot_with_juxta256


voltage_step_size = 0.195e-6
scale_uV = 1000000
scale_mV = 1000
sampling_freq = 20000

for pick_cluster in clusterSelected:
    rootDir = r'Z:\j\Neuroseeker256ch_probe6\2017-02-08\Analysis'
    videoFilename = "cluster" + str(pick_cluster) + ".avi"
    extra_average_microVolts = np.average(all_cells_ivm_filtered_data[clusterSelected[clus]][:,:,:],axis=2) * voltage_step_size* scale_uV
    num_samples = extra_average_microVolts.shape[1]
    sample_axis = np.arange(-(num_samples/2),(num_samples/2))
    time_axis = sample_axis/sampling_freq
    #juxta
    filename_p2p = os.path.join(analysis_folder + '\p2p_EXTRA_Cluster' + str(pick_cluster) + '.npy')
    amplitudes = np.load(filename_p2p)
    channel = amplitudes.argmax()
#alignamplifier=all_cells_ivm_filtered_data['6']
    juxtaData = extra_average_microVolts[channel,:]
    juxtaData = np.average(juxtaData,axis=-1) * scale_mV
    #iir_params = {'order': 3, 'ftype': 'butter', 'padlen': 0}
    #juxtaData = filters.low_pass_filter(juxtaData, sampling_freq, low_pass_freq, method='iir', iir_params=iir_params)
    #data = np.mean(alignamplifier,axis=2)
    #data = data * scale_uV * voltage_step_size
    #time_axis = np.arange(-(np.shape(data)[1]/30000.0)/2, (np.shape(data)[1]/30000.0)/2,1/30000.0 )

    plot_video_topoplot_with_juxta128(extra_average_microVolts,juxtaData, time_axis, polytrode_channels256(),
                                times_to_plot=[-0.001, 0.001], zlimits=[np.min(extra_average_microVolts), np.max(extra_average_microVolts) * 1.1],
                                filename= os.path.join(rootDir,videoFilename))



























for pick_cluster in clusterSelected:
    filename_p2p = os.path.join(analysis_folder + '\p2p_EXTRA_Cluster' + str(pick_cluster) + '.npy')
    filename_error= os.path.join(analysis_folder +  '\error_EXTRA_Cluster' + str(pick_cluster) +'.npy')
    amplitudes = np.load(filename_p2p)
    extra_average = np.average(all_cells_ivm_filtered_data[all_cells_ivm_filtered_data[clusterSelected[clus]]][:,:,:],axis=2)
    extra_average_microVolts = extra_average * scale_uV * voltage_step_size
    orderedSites = all_electrodes.reshape(1,np.shape(all_electrodes)[0]*np.shape(all_electrodes)[1])
    NumSamples=extra_average.shape[1]
    NumSites=np.size(extra_average_microVolts,axis = 0)
    lowerBound=int(NumSamples/2.0-windowSize/2.0)
    upperBound=int(NumSamples/2.0+windowSize/2.0)
    amplitude = np.zeros(np.shape(all_electrodes)[0]*np.shape(all_electrodes)[1])
    for j in np.arange(np.shape(all_electrodes)[0]*np.shape(all_electrodes)[1]):
        amplitude[j] = np.max(extra_average_microVolts[orderedSites[0][j],lowerBound:upperBound])-np.min(extra_average_microVolts[orderedSites[0][j],lowerBound:upperBound])


amplitude_order = heatmapp_amplitude(all_cells_ivm_filtered_data, all_electrodes, good_cells_number = 0, windowSize=60)

B = np.copy(amplitude_order)
orderedSites = all_electrodes.reshape(1,128)
Bmod = np.reshape(B,(32,4))
fig, ax = plt.subplots()
plt.axis('off')
im = ax.imshow(Bmod, cmap=plt.get_cmap('jet'),vmin = np.min(B),vmax= np.max(B))
cb = fig.colorbar(im,ticks = [np.min(B), 0,np.max(B)])
cb.ax.tick_params(labelsize = 20)
plt.show()



#-----------------------------------------------------------------------------------------------------------------








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

#how to plt hetampp?---------------------------------------------------------------------------------------------------
#Method1

fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
image, scat = plot_topoplot(ax1,polytrode_channels(),amplitudes,show=False, interpmethod="quadric", gridscale=5, zlimits=[np.min(amplitudes), np.max(amplitudes)])
ax1.set_yticks([], minor=False)
ax1.set_xticks([], minor=False)
plt.colorbar(mappable=image, ticks = [np.min(amplitudes), 0, np.max(amplitudes)])
plt.show()




#----------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------
# Heatmapp 32channels for p2p amplitudes
