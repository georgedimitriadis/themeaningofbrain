
import os
import BrainDataAnalysis.timelocked_analysis_functions as tf
import IO.ephys as ephys
import scipy.signal as signal
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import warnings
import numpy as np
import Layouts.Probes.klustakwik_prb_generator as prb_gen
import itertools
import numpy as np
import matplotlib.colors as colors
import matplotlib.cm as cmx
import scipy.signal as signal

import os
import IO.ephys as ephys
import mne.filter as filters
import matplotlib.pyplot as plt
import random
from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import warnings
import numpy as np
import Layouts.Probes.klustakwik_prb_generator as prb_gen
import pandas as pd
import itertools
import numpy as np
import matplotlib as mpl
from scipy import stats
import matplotlib.colors as colors
import matplotlib.cm as cmx
import os
import scipy.signal as signal
import matplotlib.colors as mcolors
import math
import scipy.stats as stats

import numpy as np
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
from matplotlib.pyplot import sca
from matplotlib.ticker import MultipleLocator
import pandas as pd
import itertools
import warnings
import matplotlib.gridspec as gridspec
# import Utilities as ut
import matplotlib.animation as animation
from matplotlib.widgets import Button
import mne.filter as filters

from ShuttlingAnalysis.paper.preprocess import sessionlabel

base_folder = r'E:\Data_32ch'

#dates = {99: '2015-09-09', 98: '2015-09-04', 97: '2015-09-03', 96: '2015-08-28', 94: '2015-08-26', 93: '2015-08-21'}

dates = {89: '2014-03-20', 90: '2014-03-26', 91: '2014-10-17', 92: '2014-11-13', 93: '2014-11-25', 94:'2014-05-20'}

#dates = {89: '2014-02-19', 90: '2013-12-20', 91: '2014-02-14', 92: '2014-10-10', 93: '2015-04-24'}


all_cell_capture_times = {94:{'1': '20_45_05'},
                          93: {'1': '21_27_13', '1_1': '22_09_28', '2': '22_44_57', '3': '23_00_08', '4': '22_31_54', '5': '00_00_00'},
                          92: {'1': '19_01_55', '2': '18_29_27', '3': '18_48_11', '4': '22_23_43', '5': '15_35_31', '7': '18_05_50'},
                          91: {'1': '16_46_02', '1_1': '17_12_27', '2': '18_19_09'},
                          90: {'1': '05_01_42', '2': '05_11_53', '3': '05_28_47'},
                          89: {'1': '20_21_41', '2': '21_04_07', '3': '21_19_51'}}

#all_cell_capture_times = {93: {'1': '15_24_49'},
 #                         92: {'1': '17_30_04', '2': '21_22_28', '3': '19_38_35','4': '20_06_33'},
  #                        91: {'1': '18_43_25'},
   #                       90: {'1': '02_41_29'},
    #                      89: {'1': '01_16_39'}}

all_spike_thresholds = {93: {'1': 0.8e-3, '1_1': 1e-3, '2': 0.8e-3, '3': 0.8e-3, '4':0.8e-3},
                        92: {'1': 0.5e-3, '2': 1e-3, '3': 1e-3, '4': 3.5e-3, '5':0.5e-3, '7': 1.5e-3},
                        91: {'1': 1e-3, '1_1': 1e-3, '2': 1e-3},
                        90: {'1': 0.5e-3, '2': 1e-3, '3': 1e-3},
                        89: {'1': 2e-3, '2': 0.5e-3, '3': 1.5e-3}}

#all_spike_thresholds = {93: {'1': 1e-3},
 #                       92: {'1': 0.5e-3, '2': 2e-3, '3': 0.5e-3, '4': 0.5e-3},
  #                      91: {'1': 0.3e-3},
   #                     90: {'1': 0.5e-3},
    #                    89: {'1': 0.5e-3}}


#good_cells_joana = {93: ['1'],
#                    92: ['1', '2', '3', '4'],
#                    91: ['1'],
#                    90: ['1'],
#                    89: ['1']}

good_cells_joana = {94:['1'],
                    93: ['1', '1_1', '2', '3', '4'],
                    92: ['1', '2', '3', '4', '5','7'],
                    91: ['1', '1_1', '2'],
                    90: ['1', '2', '3'],
                    89: ['1', '2', '3']}


#good_cells_cluster= {93: ['3','5'],
#                    92: ['1'],
#                    91: [],
#                    90: [],
#                   89: []}

good_cells_cluster= {94:['1'],
                    93: ['3'],
                    92: ['1'],
                    91: [],
                    90: [],
                    89: []}

#----------------------------------------------------------------------------------------------------------------------
#
spike_templates= np.load(r'F:\DataKilosort\32chprobe\amplifier2014-05-20T20_45_05\spike_clusters.npy')
spike_templates= np.reshape(spike_templates, ((len(spike_templates)),1))
spike_times= np.load(r'F:\DataKilosort\32chprobe\amplifier2014-05-20T20_45_05\spike_times.npy')

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

#----------------------------------------------------------------------------------------------------------------------

rat = 94
good_cells = good_cells_cluster[rat]
clusterSelected = cluster_indices_good[rat]


date = dates[rat]
data_folder = os.path.join(base_folder + '\\' + date, 'Data')
analysis_folder = os.path.join(base_folder + '\\' + date, 'Analysis')

cell_capture_times = all_cell_capture_times[rat]
spike_thresholds = all_spike_thresholds[rat]



num_of_points_in_spike_trig_ivm = 128
num_of_points_for_padding = 2 * num_of_points_in_spike_trig_ivm

#adc_channel_used = 1
adc_channel_used = 0
num_adc_channels_used = 1
adc_dtype = np.uint16
#adc_dtype=np.int32
inter_spike_time_distance = 30
#amp_gain = 1000
amp_gain = 100
num_ivm_channels = 32
#amp_dtype = np.uint16
amp_dtype = np.int16 #files after concat
#amp_dtype = np.float32 #older files

sampling_freq = 30000
high_pass_freq = 500
filtered_data_type = np.float64

types_of_data_to_load = {'t': 'ivm_data_filtered_cell{}.dat',
                         'f': 'ivm_data_filtered_cell{}',
                         'c': 'ivm_data_filtered_continous_cell{}.dat',
                         'k': 'ivm_data_filtered_klusta_spikes_cell{}.dat',
                         'p': 'patch_data_cell{}.dat',
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

for i in np.arange(0, len(good_cells)):
    raw_data_file_ivm = os.path.join(data_folder, 'amplifier'+date+'T'+cell_capture_times[good_cells[i]]+'.bin')
    raw_data_ivm = ephys.load_raw_data(raw_data_file_ivm, numchannels=num_ivm_channels, dtype=amp_dtype)

    for clus in np.arange(0, len(clusterSelected[good_cells[i]])):
        num_of_spikes = len(kilosort_units[clusterSelected[good_cells[i]][clus]])

        shape_of_filt_spike_trig_ivm = ((num_ivm_channels,
                                     num_of_points_in_spike_trig_ivm,
                                     num_of_spikes))

        ivm_data_filtered = np.memmap(os.path.join(analysis_folder, types_of_sorting[cluster_to_load].format(clusterSelected[good_cells[i]][clus]) + types_of_data_to_load[data_to_load].format(good_cells[i])),
                                                   dtype=filtered_data_type,
                                                   mode='w+',
                                                   shape=shape_of_filt_spike_trig_ivm)

        for spike in np.arange(0, num_of_spikes):
            trigger_point = spike_times[kilosort_units[clusterSelected[good_cells[i]][clus]][spike]]
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

pick_recording = '1' #we need to select the recording


all_cells_ivm_filtered_data = {}
data_to_load = 't'
cluster_to_load = 's'

for clus in np.arange(0, len(clusterSelected[pick_recording])):
    num_of_spikes = len(kilosort_units[clusterSelected[pick_recording][clus]])
    if data_to_load == 't':
        shape_of_filt_spike_trig_ivm = (num_ivm_channels,
                                    num_of_points_in_spike_trig_ivm,
                                    num_of_spikes)
        time_axis = np.arange(-num_of_points_in_spike_trig_ivm/(2*sampling_freq),
                  num_of_points_in_spike_trig_ivm/(2*sampling_freq),
                  1/sampling_freq)

    ivm_data_filtered = np.memmap(os.path.join(analysis_folder, types_of_sorting[cluster_to_load].format(clusterSelected[pick_recording][clus]) + types_of_data_to_load[data_to_load].format(pick_recording)),
                                dtype=filtered_data_type,
                                mode='r',
                                shape=shape_of_filt_spike_trig_ivm)

    all_cells_ivm_filtered_data[clusterSelected[pick_recording][clus]] = ivm_data_filtered


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

# Plot 32channel averages overlaid

voltage_step_size = 0.195e-6
scale_uV = 1000000
scale_ms = 1000
#voltage_step_size = 1
#scale_uV = 1

def plot_average_extra(all_cells_ivm_filtered_data, yoffset=0):

    origin_cmap= plt.get_cmap('hsv')
    shrunk_cmap = shiftedColorMap(origin_cmap, start=0.6, midpoint=0.7, stop=1, name='shrunk')
    cm=shrunk_cmap
    cNorm=colors.Normalize(vmin=0,vmax=32)
    scalarMap= cmx.ScalarMappable(norm=cNorm,cmap=cm)
    sites_order_geometry= [0,31,24,7,1,21,10,30,25,6,15,20,11,16,26,5,14,19,12,17,27,4,8,18,13,23,28,3,9,29,2,22]
    for clus in np.arange(0, len(clusterSelected[pick_recording])):
        num_of_spikes = len(kilosort_units[clusterSelected[pick_recording][clus]])
        extra_average_V = np.average(all_cells_ivm_filtered_data[clusterSelected[pick_recording][clus]][:,:,:],axis=2)
        num_samples=extra_average_V.shape[1]
        sample_axis= np.arange(-(num_samples/2),(num_samples/2))
        time_axis= sample_axis/30000
        extra_average_microVolts = extra_average_V * scale_uV * voltage_step_size
        plt.figure()
        for m in np.arange(np.shape(extra_average_microVolts)[0]):
            colorVal=scalarMap.to_rgba(np.shape(extra_average_microVolts)[0]-m)
            plt.plot(time_axis*scale_ms,extra_average_microVolts[sites_order_geometry[m],:].T, color=colorVal)
            plt.xlim(-2, 2) #window 4ms
            plt.ylim(np.min(extra_average_microVolts)-yoffset,np.max(extra_average_microVolts)+yoffset )
            plt.ylabel('Voltage (\u00B5V)', fontsize=20)
            plt.xlabel('Time (ms)',fontsize=20)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            triggerline(0)

# Plot 32channels averages in space

voltage_step_size = 0.195e-6
scale_uV = 1000000
scale_ms = 1000
#voltage_step_size = 1
#scale_uV = 1

def plot_average_extra_geometry(all_cells_ivm_filtered_data, yoffset=50):


    sites_order_geometry= [0,31,24,7,1,21,10,30,25,6,15,20,11,16,26,5,14,19,12,17,27,4,8,18,13,23,28,3,9,29,2,22]
    subplot_order = [2,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63,65]
    origin_cmap= plt.get_cmap('hsv')
    shrunk_cmap = shiftedColorMap(origin_cmap, start=0.6, midpoint=0.7, stop=1, name='shrunk')
    cm=shrunk_cmap
    cNorm=colors.Normalize(vmin=-4,vmax=28)
    scalarMap= cmx.ScalarMappable(norm=cNorm,cmap=cm)
    for clus in np.arange(0, len(clusterSelected[pick_recording])):
        num_of_spikes = len(kilosort_units[clusterSelected[pick_recording][clus]])
        plt.figure(clus+1)
        extra_average_microVolts= np.average(all_cells_ivm_filtered_data[clusterSelected[pick_recording][clus]][:,:,:],axis=2) * voltage_step_size* scale_uV
        num_samples=extra_average_microVolts.shape[1]
        sample_axis= np.arange(-(num_samples/2),(num_samples/2))
        time_axis= sample_axis/30000
        for i in np.arange(32):
            plt.subplot(22,3,subplot_order [i])
            colorVal=scalarMap.to_rgba(31-i)
            plt.plot(time_axis*scale_ms, extra_average_microVolts[sites_order_geometry[i],:],color=colorVal)
            plt.ylim(np.min(np.min(extra_average_microVolts, axis=1)) - yoffset, np.max(np.max(extra_average_microVolts, axis=1)) +  yoffset)
            plt.xlim(-2, 2)
            plt.axis("OFF")



#--------------------------------------------------------------------------------------------------------------------
# P2P, MIN and MAX 128channels and 32channels

voltage_step_size = 0.195e-6
scale_uV = 1000000
scale_ms = 1000
#voltage_step_size = 1
#scale_uV = 1


def peaktopeak(all_cells_ivm_filtered_data, windowSize=60):

    for clus in np.arange(0, len(clusterSelected[pick_recording])):
        extra_average_V= np.average(all_cells_ivm_filtered_data[clusterSelected[pick_recording][clus]][:,:,:],axis=2) * voltage_step_size
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

        stdv = stats.sem(all_cells_ivm_filtered_data[clusterSelected[pick_recording][clus]][:,:,:],axis=2)
        stdv = stdv * voltage_step_size * scale_uV

        for b in range(NumSites):
           stdv_minimos[b]= stdv[b, argminima[b]]
           stdv_maximos[b]= stdv[b, argmaxima[b]]

        error =  np.sqrt((stdv_minimos * stdv_minimos)+ (stdv_maximos*stdv_maximos))


        np.save(os.path.join(analysis_folder,'stdv_minimos_EXTRA_Cluster'+ str(clusterSelected[pick_recording][clus]) + '.npy'), stdv_minimos)
        np.save(os.path.join(analysis_folder,'stdv_maximos_EXTRA_Cluster'+ str(clusterSelected[pick_recording][clus])  + '.npy'), stdv_maximos)
        np.save(os.path.join(analysis_folder,'stdv_average_EXTRA_Cluster'+ str(clusterSelected[pick_recording][clus])+ '.npy'), stdv)
        np.save(os.path.join(analysis_folder,'error_EXTRA_Cluster'+ str(clusterSelected[pick_recording][clus])+ '.npy'), error)
        np.save(os.path.join(analysis_folder,'p2p_EXTRA_Cluster'+ str(clusterSelected[pick_recording][clus]) + '.npy'), p2p)
        np.save(os.path.join(analysis_folder,'minima_EXTRA_Cluster'+ str(clusterSelected[pick_recording][clus]) + '.npy'), minima)
        np.save(os.path.join(analysis_folder,'maxima_EXTRA_Cluster'+ str(clusterSelected[pick_recording][clus])+ '.npy'), maxima)
        np.save(os.path.join(analysis_folder,'argmaxima_EXTRA_Cluster'+ str(clusterSelected[pick_recording][clus]) + '.npy'), argmaxima)
        np.save(os.path.join(analysis_folder,'argminima_EXTRA_Cluster'+ str(clusterSelected[pick_recording][clus]) + '.npy'), argminima)

    return argmaxima, argminima, maxima, minima, p2p


#plot ALL PEDOT versus Pristine and their average

filename = 'E:\\Data_32ch\\2014-05-20\\Analysis'
pick_recording ='1'

pedot_all= [22,29,3,23,18,4,17,19,5,16,20,6,30,21,7,31]
pristine_all= np.array([2,9,28,13,8,27,12,14,26,11,15,25,10,1,24,0])
yoffset = 100
number_clusters = len(clusterSelected[pick_recording])

fig = plt.figure()
all_axes = fig.get_axes()
outer_grid = gridspec.GridSpec(np.ceil(number_clusters/2).astype(int),2)
for pick_cluster in clusterSelected[pick_recording]:
    filename_p2p = os.path.join(filename + '\p2p_EXTRA_Cluster' + str(pick_cluster) + '.npy')
    filename_error= os.path.join(filename +  '\error_EXTRA_Cluster' + str(pick_cluster) +'.npy')
    amplitudes = np.load(filename_p2p)
    error = np.load(filename_error)

    min_ampl= amplitudes.min()
    max_ampl= amplitudes.max()
    channel = amplitudes.argmax()
    print(min_ampl)
    print(max_ampl)
    print(channel)
    select_indices_biggestAmp = np.where(amplitudes >= (amplitudes.max()/2))
    print(select_indices_biggestAmp)
    Amp_select_indices = amplitudes[select_indices_biggestAmp]
    print(Amp_select_indices)
    thebiggestchannel_pedot = np.intersect1d(channel,pedot_all)
    print(thebiggestchannel_pedot)
    thebiggestchannel_pristine = np.intersect1d(channel,pristine_all)
    print(thebiggestchannel_pristine)
    select_pedot = np.intersect1d (select_indices_biggestAmp, pedot_all)
    print(select_pedot)
    select_pristine = np.intersect1d (select_indices_biggestAmp, pristine_all)
    print(select_pristine)

    extra_average_PEDOT= np.average(all_cells_ivm_filtered_data[pick_cluster][select_pedot,:,:],axis=0) * voltage_step_size* scale_uV
    num_total_spikes_PEDOT= np.shape(extra_average_PEDOT)[1]
    extra_average_PEDOT= np.average(extra_average_PEDOT[:,:],axis=1)
    num_samples=extra_average_PEDOT.shape[0]
    sample_axis= np.arange(-(num_samples/2),(num_samples/2))
    time_axis= sample_axis/30000

    cell = outer_grid[clusterSelected[pick_recording].index(pick_cluster)]
    inner_grid = gridspec.GridSpecFromSubplotSpec(1,3, cell)
    ax0 = plt.subplot(inner_grid[0,0])
    ax1 = plt.subplot(inner_grid[0,1], sharey=ax0, sharex= ax0)
    ax2 = plt.subplot(inner_grid[0,2], sharey=ax1, sharex=ax1)
    #plt.setp(ax1.get_yticklabels(), visible=False)
    #plt.setp(ax2.get_yticklabels(), visible=False)
    plt.setp(ax0.get_xaxis(), visible= False)
    plt.setp(ax1.get_xaxis(), visible= False)
    plt.setp(ax2.get_xaxis(), visible= False)
    plt.setp(ax0.get_yaxis(), visible= False)
    plt.setp(ax1.get_yaxis(), visible= False)
    plt.setp(ax2.get_yaxis(), visible= False)
    #plt.subplot(1,3,2)
    for i in np.arange(len(select_pedot)):
        num_of_spikes = len(kilosort_units[pick_cluster])
        subset = np.random.choice(np.arange(num_of_spikes), math.ceil(num_of_spikes * 0.68))
        ax1.plot(time_axis*scale_ms,all_cells_ivm_filtered_data[pick_cluster][select_pedot[i],:,subset].T * voltage_step_size* scale_uV, color='#f3f3f3', label='%i'%num_total_spikes_PEDOT)
    #plt.plot(time_axis*scale_ms,extra_average_PEDOT,color='#da70d6', linewidth=4)
    ax1.plot(time_axis*scale_ms,extra_average_PEDOT,color='#ff4136', linewidth=3)
    #plt.ylim(np.min(np.min(all_cells_ivm_filtered_data[pick_cluster][:,:,:], axis = 1 ))*voltage_step_size* scale_uV - yoffset, np.max(np.max(all_cells_ivm_filtered_data[2][:,:,:], axis = 1 ))*voltage_step_size* scale_uV + yoffset)
    plt.ylim(-1000, 500)
    #plt.xlim(-2, 2)
    plt.axis("OFF")
    ax1.axis("OFF")
    ax1.text(2, 400.0, 'cluster'+str(pick_cluster)+ ' , ' + str(num_total_spikes_PEDOT) + 'spikes',horizontalalignment='right', verticalalignment='top', fontsize=15)

    extra_average_Pristine= np.average(all_cells_ivm_filtered_data[pick_cluster][select_pristine,:,:],axis=0) * voltage_step_size* scale_uV
    extra_average_Pristine= np.average(extra_average_Pristine[:,:],axis=1)
    num_samples=extra_average_Pristine.shape[0]
    sample_axis= np.arange(-(num_samples/2),(num_samples/2))
    time_axis= sample_axis/30000
    #plt.subplot(1,3,1)
    for i in np.arange(len(select_pristine)):
        num_of_spikes = len(kilosort_units[pick_cluster])
        subset = np.random.choice(np.arange(num_of_spikes), math.ceil(num_of_spikes * 0.68))
        ax0.plot(time_axis*scale_ms,all_cells_ivm_filtered_data[pick_cluster][select_pristine[i],:,subset].T* voltage_step_size* scale_uV, color='#f3f3f3')
    ax0.plot(time_axis*scale_ms,extra_average_Pristine,color='#5da4d6', linewidth=3)
    #plt.ylim(np.min(np.min(np.min(extra_average_microVolts, axis = 1 ))*voltage_step_size* scale_uV - yoffset, np.max(np.max(all_cells_ivm_filtered_data[2][:,:,:], axis = 1 ))*voltage_step_size* scale_uV + yoffset)
    plt.ylim(-1000, 500)
    #plt.xlim(-2, 2)
    #plt.subplot(1,3,3)
    ax2.plot(time_axis*scale_ms,extra_average_Pristine,color='#5da4d6', label ='Pristine', linewidth=1)
    #plt.plot(time_axis*scale_ms,extra_average_PEDOT,color='#da70d6', label ='PEDOT', linewidth=3)
    ax2.plot(time_axis*scale_ms,extra_average_PEDOT,color='#ff4136', label ='PEDOT', linewidth=1)
    plt.ylim(-1000, 500)
    #plt.xlim(-2, 2)
    ax2.axis("OFF")
    ax0.axis("OFF")
    plt.axis("OFF")
plt.vlines(2, -1000,500, color='k', linewidth=5)
plt.hlines(-1000, -2, 2, color='k', linewidth=5)
ax2.text(-1, -1000,'4ms',verticalalignment='bottom', horizontalalignment='right',color='black', fontsize=15)
ax2.text(3.4,400,'1500uV',verticalalignment='top', horizontalalignment='right', color='black', fontsize=15)
#plt.legend()
plt.show()

#nice colors codes hexa
#2ca065 (green)
#ff4136(red)
#cf72ff(purple)
#5da4d6 (blue)
#ff900e (orange)
#____________________________________________________________________________________________________


#GOLD
#plot ALL GOLD versus Pristine and their average
filename = 'E:\\Data_32ch\\2014-05-20\\Analysis'
pick_recording ='1'

gold_all= [22,25,26,27,28,11,12,13]
pristine_all= np.array([2,9,8,14,15,10,1,24,0,29,3,23,18,4,17,19,5,16,20,6,30,21,7,31])
yoffset = 100
number_clusters = len(clusterSelected[pick_recording])

fig = plt.figure()
all_axes = fig.get_axes()
outer_grid = gridspec.GridSpec(np.ceil(number_clusters/2).astype(int),2)
for pick_cluster in clusterSelected[pick_recording]:
    filename_p2p = os.path.join(filename + '\p2p_EXTRA_Cluster' + str(pick_cluster) + '.npy')
    filename_error= os.path.join(filename +  '\error_EXTRA_Cluster' + str(pick_cluster) +'.npy')
    amplitudes = np.load(filename_p2p)
    error = np.load(filename_error)

    min_ampl= amplitudes.min()
    max_ampl= amplitudes.max()
    channel = amplitudes.argmax()
    print(min_ampl)
    print(max_ampl)
    print(channel)
    select_indices_biggestAmp = np.where(amplitudes >= (amplitudes.max()/2))
    print(select_indices_biggestAmp)
    Amp_select_indices = amplitudes[select_indices_biggestAmp]
    print(Amp_select_indices)
    thebiggestchannel_gold = np.intersect1d(channel,gold_all)
    print(thebiggestchannel_gold)
    thebiggestchannel_pristine = np.intersect1d(channel,pristine_all)
    print(thebiggestchannel_pristine)
    select_gold = np.intersect1d (select_indices_biggestAmp, gold_all)
    print(select_pedot)
    select_pristine = np.intersect1d (select_indices_biggestAmp, pristine_all)
    print(select_pristine)

    extra_average_gold= np.average(all_cells_ivm_filtered_data[pick_cluster][select_gold,:,:],axis=0) * voltage_step_size* scale_uV
    num_total_spikes_gold= np.shape(extra_average_gold)[1]
    extra_average_gold= np.average(extra_average_gold[:,:],axis=1)
    num_samples=extra_average_gold.shape[0]
    sample_axis= np.arange(-(num_samples/2),(num_samples/2))
    time_axis= sample_axis/30000

    cell = outer_grid[clusterSelected[pick_recording].index(pick_cluster)]
    inner_grid = gridspec.GridSpecFromSubplotSpec(1,3, cell)
    ax0 = plt.subplot(inner_grid[0,0])
    ax1 = plt.subplot(inner_grid[0,1], sharey=ax0, sharex= ax0)
    ax2 = plt.subplot(inner_grid[0,2], sharey=ax1, sharex=ax1)
    #plt.setp(ax1.get_yticklabels(), visible=False)
    #plt.setp(ax2.get_yticklabels(), visible=False)
    plt.setp(ax0.get_xaxis(), visible= False)
    plt.setp(ax1.get_xaxis(), visible= False)
    plt.setp(ax2.get_xaxis(), visible= False)
    plt.setp(ax0.get_yaxis(), visible= False)
    plt.setp(ax1.get_yaxis(), visible= False)
    plt.setp(ax2.get_yaxis(), visible= False)
    #plt.subplot(1,3,2)
    for i in np.arange(len(select_gold)):
        num_of_spikes = len(kilosort_units[pick_cluster])
        subset = np.random.choice(np.arange(num_of_spikes), math.ceil(num_of_spikes * 0.68))
        ax1.plot(time_axis*scale_ms,all_cells_ivm_filtered_data[pick_cluster][select_gold[i],:,subset].T * voltage_step_size* scale_uV, color='#f3f3f3', label='%i'%num_total_spikes_PEDOT)
    #plt.plot(time_axis*scale_ms,extra_average_PEDOT,color='#da70d6', linewidth=4)
    ax1.plot(time_axis*scale_ms,extra_average_gold,color='#ff4136', linewidth=3)
    #plt.ylim(np.min(np.min(all_cells_ivm_filtered_data[pick_cluster][:,:,:], axis = 1 ))*voltage_step_size* scale_uV - yoffset, np.max(np.max(all_cells_ivm_filtered_data[2][:,:,:], axis = 1 ))*voltage_step_size* scale_uV + yoffset)
    plt.ylim(-300, 500)
    #plt.xlim(-2, 2)
    plt.axis("OFF")
    ax1.axis("OFF")
    ax1.text(2, 300.0, 'cluster'+str(pick_cluster)+ ' , ' + str(num_total_spikes_PEDOT) + 'spikes',horizontalalignment='right', verticalalignment='top', fontsize=15)

    extra_average_Pristine= np.average(all_cells_ivm_filtered_data[pick_cluster][select_pristine,:,:],axis=0) * voltage_step_size* scale_uV
    extra_average_Pristine= np.average(extra_average_Pristine[:,:],axis=1)
    num_samples=extra_average_Pristine.shape[0]
    sample_axis= np.arange(-(num_samples/2),(num_samples/2))
    time_axis= sample_axis/30000
    #plt.subplot(1,3,1)
    for i in np.arange(len(select_pristine)):
        num_of_spikes = len(kilosort_units[pick_cluster])
        subset = np.random.choice(np.arange(num_of_spikes), math.ceil(num_of_spikes * 0.68))
        ax0.plot(time_axis*scale_ms,all_cells_ivm_filtered_data[pick_cluster][select_pristine[i],:,subset].T* voltage_step_size* scale_uV, color='#f3f3f3')
    ax0.plot(time_axis*scale_ms,extra_average_Pristine,color='#5da4d6', linewidth=3)
    #plt.ylim(np.min(np.min(np.min(extra_average_microVolts, axis = 1 ))*voltage_step_size* scale_uV - yoffset, np.max(np.max(all_cells_ivm_filtered_data[2][:,:,:], axis = 1 ))*voltage_step_size* scale_uV + yoffset)
    plt.ylim(-300, 500)
    #plt.xlim(-2, 2)
    #plt.subplot(1,3,3)
    ax2.plot(time_axis*scale_ms,extra_average_Pristine,color='#5da4d6', label ='Pristine', linewidth=1)
    #plt.plot(time_axis*scale_ms,extra_average_PEDOT,color='#da70d6', label ='PEDOT', linewidth=3)
    ax2.plot(time_axis*scale_ms,extra_average_gold,color='#ff4136', label ='PEDOT', linewidth=1)
    plt.ylim(-300, 500)
    #plt.xlim(-2, 2)
    ax2.axis("OFF")
    ax0.axis("OFF")
    plt.axis("OFF")
plt.vlines(2, -300,500, color='k', linewidth=5)
plt.hlines(-300, -2, 2, color='k', linewidth=5)
ax2.text(-1, -300,'4ms',verticalalignment='bottom', horizontalalignment='right',color='black', fontsize=15)
ax2.text(3.4,200,'800uV',verticalalignment='top', horizontalalignment='right', color='black', fontsize=15)
#plt.legend()
plt.show()

#nice colors codes hexa
#2ca065 (green)
#ff4136(red)
#cf72ff(purple)
#5da4d6 (blue)
#ff900e (orange)
#____________________________________________________________________________________________________

#plot  averages where(amplitudes >= (amplitudes.max()/2))
pedot_all= [22,29,3,23,18,4,17,19,5,16,20,6,30,21,7,31]
pristine_all= np.array([2,9,28,13,8,27,12,14,26,11,15,25,10,1,24,0])

sites_order_geometry= [0,31,24,7,1,21,10,30,25,6,15,20,11,16,26,5,14,19,12,17,27,4,8,18,13,23,28,3,9,29,2,22]
subplot_order = [2,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63,65]
yoffset = 100

for pick_cluster in clusterSelected[pick_recording]:
    filename_p2p = os.path.join(filename + '\p2p_EXTRA_Cluster' + str(pick_cluster) + '.npy')
    filename_error= os.path.join(filename +  '\error_EXTRA_Cluster' + str(pick_cluster) +'.npy')
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
    print('atleasthalfofmaxp2p_channels ' + str(select_indices_biggestAmp))
    Amp_select_indices = amplitudes[select_indices_biggestAmp]
    print('atleasthalfofmaxp2p ' + str(Amp_select_indices))
    thebiggestchannel_pedot = np.intersect1d(channel,pedot_all)
    print( 'isthemaxp2p_channel_PEDOT?' + str(thebiggestchannel_pedot))
    thebiggestchannel_pristine = np.intersect1d(channel,pristine_all)
    print('isthemaxp2p_channel_Pristine?' + str(thebiggestchannel_pristine))
    select_pedot = np.intersect1d (select_indices_biggestAmp, pedot_all)
    print('atleasthalfofmaxp2p_channels_PEDOT ' + str(select_pedot))
    select_pristine = np.intersect1d (select_indices_biggestAmp, pristine_all)
    print('atleasthalfofmaxp2p_channels_Pristine ' + str(select_pristine))
    print('#------------------------------------------------------')
    plt.figure()
    extra_average_microVolts= np.average(all_cells_ivm_filtered_data[pick_cluster][:,:,:],axis=2) * voltage_step_size* scale_uV
    num_samples=extra_average_microVolts.shape[1]
    sample_axis= np.arange(-(num_samples/2),(num_samples/2))
    time_axis= sample_axis/30000
    for i in np.arange(32):
        plt.subplot(22,3,subplot_order [i])
        if sites_order_geometry[i] in select_pedot:
            colorVal = 'k'
            alphanumber = 1
        elif sites_order_geometry[i] in select_pristine:
            colorVal = 'grey'
            alphanumber = 1
        else:
            colorVal = 'grey'
            alphanumber = 0.1
        plt.plot(time_axis*scale_ms, extra_average_microVolts[sites_order_geometry[i],:],color=colorVal, linewidth = 3, alpha = alphanumber)
        plt.ylim(np.min(np.min(extra_average_microVolts, axis=1))- yoffset, np.max(np.max(extra_average_microVolts, axis=1))+ yoffset)
        plt.xlim(-2, 2)
        plt.axis("OFF")
    plt.text(6,-300.0,'cluster'+str(pick_cluster),horizontalalignment='right', verticalalignment='top')



#----------------------------------------------------------------------------------------------------------------------
#Autocorrelagram for each cluster

# Spike train autocorelogram
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
for pick_cluster in clusterSelected[pick_recording]:
    times_cluster= np.reshape(spike_times[kilosort_units[pick_cluster]], (spike_times[kilosort_units[pick_cluster]].shape[0]))
    diffs, norm = crosscorrelate_spike_trains((times_cluster/30).astype(np.int64),(times_cluster/30).astype(np.int64),lag=50)
    #hist, edges = np.histogram(diffs)
    plt.figure()
    plt.hist(diffs, bins=101, normed=True, range=(-50,50), align ='mid') #if normed = True, the probability density function at the bin, normalized such that the integral over the range is 1.
    plt.xlim(-lag,lag)
#--------------------------------------------------------------------

# Calculate ELECTRONIC noise

import plotly.plotly as py
import plotly.graph_objs as go



raw_data_file_ivm = os.path.join(data_folder, 'amplifier'+date+'T'+cell_capture_times[pick_recording]+'.bin')
raw_data_ivm = ephys.load_raw_data(raw_data_file_ivm, numchannels=num_ivm_channels, dtype=amp_dtype)
filename = 'E:\\Data_32ch\\2014-11-25\\Analysis'
pick_recording ='3'

#Filter for extracellular recording
def highpass(data,BUTTER_ORDER=3, F_HIGH=14250,sampleFreq=30000.0,passFreq=500.0):
    b, a = signal.butter(BUTTER_ORDER,(passFreq/(sampleFreq/2), F_HIGH/(sampleFreq/2)),'pass')
    return signal.filtfilt(b,a,data)

sampling_freq = 30000
high_pass_freq = 500
temp_unfiltered = raw_data_ivm.dataMatrix
temp_unfiltered = temp_unfiltered.astype(filtered_data_type)
temp_filtered = highpass(temp_unfiltered)
number_clusters = len(clusterSelected[pick_recording])


stdvs_PEDOT = np.median(np.abs(temp_filtered[pedot_all])/0.6745, axis=1)
stdvs_pristine = np.median(np.abs(temp_filtered[pristine_all])/0.6745, axis=1)


voltage_step_size = 0.195e-6
scale_uV = 1000000
#stdvs_uV= stdvs*voltage_step_size*scale_uV
stdvs_PEDOT_uV= stdvs_PEDOT*voltage_step_size*scale_uV
stdvs_pristine_uV= stdvs_pristine*voltage_step_size*scale_uV



#stdv_average = np.average(stdvs_uV)
#stdv_average = np.average(stdvs_uV[np.where(stdvs_uV < 4)])
stdv_PEDOT_average = np.average(stdvs_PEDOT_uV)
stdv_pristine_average = np.average(stdvs_pristine_uV)

#stdv_stdv = stats.sem(stdvs_uV)
#stdv_stdv = stats.sem(stdvs_uV[np.where(stdvs_uV < 4)])
stdv_PEDOT_stdv = stats.sem(stdvs_PEDOT_uV)
stdv_pristine_stdv = stats.sem(stdvs_pristine_uV)

print(stdv_PEDOT_average)
print(stdv_PEDOT_stdv)
print(stdv_pristine_average)
print(stdv_pristine_stdv)



#plot the noise window around each cluster----------------------------------------------------------------------

#raw_data_file_ivm = r'F:\DataKilosort\32chprobe\amplifier2014-11-13T19_01_55\amplifier2014-11-13T19_01_55.bin'

raw_data_file_ivm =r'F:\DataKilosort\32chprobe\amplifier2014-11-25T23_00_08\amplifier2014-11-25T23_00_08.bin'
raw_data_ivm = ephys.load_raw_data(raw_data_file_ivm, numchannels=num_ivm_channels, dtype=amp_dtype)
temp_unfiltered = raw_data_ivm.dataMatrix
temp_filtered = temp_unfiltered.astype(filtered_data_type)


def highpass(data,BUTTER_ORDER=3, F_HIGH=14250,sampleFreq=30000.0,passFreq=250.0):
    b, a = signal.butter(BUTTER_ORDER,(passFreq/(sampleFreq/2), F_HIGH/(sampleFreq/2)),'pass')
    return signal.filtfilt(b,a,data)


temp_filtered = highpass(temp_filtered)



window_start = 0
window_size = 120
fig = plt.figure()
all_axes = fig.get_axes()

outer_grid = gridspec.GridSpec(np.ceil(number_clusters/2).astype(int),2)

for pick_cluster in clusterSelected[pick_recording]:
    filename_p2p = os.path.join(filename + '\p2p_EXTRA_Cluster' + str(pick_cluster) + '.npy')
    filename_error= os.path.join(filename +  '\error_EXTRA_Cluster' + str(pick_cluster) +'.npy')
    amplitudes = np.load(filename_p2p)
    error = np.load(filename_error)

    min_ampl= amplitudes.min()
    max_ampl= amplitudes.max()
    channel = amplitudes.argmax()
    select_indices_biggestAmp = np.where(amplitudes >= (amplitudes.max()/4))
    Amp_select_indices = amplitudes[select_indices_biggestAmp]
    thebiggestchannel_pedot = np.intersect1d(channel,pedot_all)
    thebiggestchannel_pristine = np.intersect1d(channel,pristine_all)
    select_pedot = np.intersect1d (select_indices_biggestAmp, pedot_all)
    select_pristine = np.intersect1d (select_indices_biggestAmp, pristine_all)


    stdvs_PEDOT = np.median(np.abs(temp_filtered[pedot_all])/0.6745, axis=1)
    stdvs_PEDOT_uV= stdvs_PEDOT*voltage_step_size*scale_uV
    stdv_PEDOT_average = np.average(stdvs_PEDOT_uV)
    stdv_PEDOT_stdv = stats.sem(stdvs_PEDOT_uV)

    #num_samples=temp_filtered.shape[1]
    num_samples = window_size
    sample_axis= np.arange(0,num_samples)
    time_axis= sample_axis/30000

    cell = outer_grid[clusterSelected[pick_recording].index(pick_cluster)]
    inner_grid = gridspec.GridSpecFromSubplotSpec(1,3, cell)
    ax0 = plt.subplot(inner_grid[0,0])
    ax1 = plt.subplot(inner_grid[0,1], sharey=ax0, sharex= ax0)
    ax2 = plt.subplot(inner_grid[0,2], sharey=ax1, sharex=ax1)

    #plt.setp(ax1.get_yticklabels(), visible=False)
    #plt.setp(ax2.get_yticklabels(), visible=False)
    #plt.setp(ax0.get_xaxis(), visible= False)
    #plt.setp(ax1.get_xaxis(), visible= False)
    #plt.setp(ax2.get_xaxis(), visible= False)
    #plt.setp(ax0.get_yaxis(), visible= False)
    #plt.setp(ax1.get_yaxis(), visible= False)
    #plt.setp(ax2.get_yaxis(), visible= False)
    for i in np.arange(len(select_pedot)):
        ax1.plot(time_axis*scale_ms,temp_filtered[select_pedot[i],window_start:time_axis.shape[0]]*voltage_step_size*scale_uV, color='#ff4136')
    stdvs_selected_PEDOT = np.median(np.abs(temp_filtered[select_pedot])/0.6745, axis=1)
    stdvs_selected_PEDOT_uV= stdvs_selected_PEDOT*voltage_step_size*scale_uV
    stdv_selected_PEDOT_average = np.average(stdvs_selected_PEDOT_uV)
    stdv_selected_PEDOT_stdv = stats.sem(stdvs_selected_PEDOT_uV)
    plt.axis("OFF")
    ax1.axis("OFF")
    ax1.text(3, 40.0, 'STDV_PEDOT'+ str(format(stdv_selected_PEDOT_average,'.2f')), horizontalalignment='right', verticalalignment='top', fontsize=10)


    stdvs_Pristine = np.median(np.abs(temp_filtered[pristine_all])/0.6745, axis=1)
    stdvs_Pristine_uV= stdvs_Pristine*voltage_step_size*scale_uV
    stdv_Pristine_average = np.average(stdvs_Pristine_uV)
    stdv_Pristine_stdv = stats.sem(stdvs_Pristine_uV)

    for i in np.arange(len(select_pristine)):
        ax0.plot(time_axis*scale_ms,temp_filtered[select_pristine[i],window_start:time_axis.shape[0]]* voltage_step_size* scale_uV, color='#5da4d6')
    stdvs_selected_Pristine = np.median(np.abs(temp_filtered[select_pristine])/0.6745, axis=1)
    stdvs_selected_Pristine_uV= stdvs_selected_Pristine*voltage_step_size*scale_uV
    stdv_selected_Pristine_average = np.average(stdvs_selected_Pristine_uV)
    stdv_selected_Pristine_stdv = stats.sem(stdvs_selected_Pristine_uV)
    ax0.text(3, 40.0,'STDV_Pristine'+ str(format(stdv_selected_Pristine_average, '.2f')), horizontalalignment='right', verticalalignment='top', fontsize=10)


    ax2.plot(time_axis*scale_ms,temp_filtered[pristine_all,window_start:time_axis.shape[0]].T* voltage_step_size* scale_uV, color='#5da4d6')
    #plt.plot(time_axis*scale_ms,extra_average_PEDOT,color='#da70d6', label ='PEDOT', linewidth=3)
    ax2.plot(time_axis*scale_ms,temp_filtered[pedot_all,window_start:time_axis.shape[0]].T*voltage_step_size*scale_uV, color='#ff4136')
    plt.ylim(-40, 40)
    plt.xlim(0, 4)
    ax2.axis("OFF")
    ax0.axis("OFF")
    plt.axis("OFF")

plt.vlines(4, -40,40, color='k', linewidth=5)
plt.hlines(-40, 0, 4, color='k', linewidth=5)
ax2.text(1, -40,'4ms',verticalalignment='bottom', horizontalalignment='right',color='black', fontsize=15)
ax2.text(6,30,'80uV',verticalalignment='top', horizontalalignment='right', color='black', fontsize=15)
#plt.legend()
plt.show()



plt.vlines(2, -1000,500, color='k', linewidth=5)
plt.hlines(-1000, -2, 2, color='k', linewidth=5)
ax2.text(-1, -1000,'4ms',verticalalignment='bottom', horizontalalignment='right',color='black', fontsize=15)
ax2.text(3.4,400,'1500uV',verticalalignment='top', horizontalalignment='right', color='black', fontsize=15)
#plt.legend()
plt.show()

#------------------------------------------------------------------------


#window_start = 0
window_size = 120
fig = plt.figure()
all_axes = fig.get_axes()
outer_grid = gridspec.GridSpec(np.ceil(number_clusters/2).astype(int),2)

for pick_cluster in clusterSelected[pick_recording]:
    #num_samples=temp_filtered.shape[1]
    num_samples = window_size
    sample_axis= np.arange(0,num_samples)
    time_axis= sample_axis/30000

    cell = outer_grid[clusterSelected[pick_recording].index(pick_cluster)]
    inner_grid = gridspec.GridSpecFromSubplotSpec(1,3, cell)
    ax0 = plt.subplot(inner_grid[0,0])
    ax1 = plt.subplot(inner_grid[0,1], sharey=ax0, sharex= ax0)
    ax2 = plt.subplot(inner_grid[0,2], sharey=ax1, sharex=ax1)

    #plt.setp(ax1.get_yticklabels(), visible=False)
    #plt.setp(ax2.get_yticklabels(), visible=False)
    #plt.setp(ax0.get_xaxis(), visible= False)
    #plt.setp(ax1.get_xaxis(), visible= False)
    #plt.setp(ax2.get_xaxis(), visible= False)
    #plt.setp(ax0.get_yaxis(), visible= False)
    #plt.setp(ax1.get_yaxis(), visible= False)
    #plt.setp(ax2.get_yaxis(), visible= False)
    for i in np.arange(len(pedot_all)):
        ax1.plot(time_axis*scale_ms,temp_filtered[pedot_all[i],window_start:time_axis.shape[0]]*voltage_step_size*scale_uV, color='#ff4136')
    plt.axis("OFF")
    ax1.axis("OFF")

    for i in np.arange(len(pristine_all)):
        ax0.plot(time_axis*scale_ms,temp_filtered[pristine_all[i],window_start:time_axis.shape[0]]* voltage_step_size* scale_uV, color='#5da4d6')

    ax2.plot(time_axis*scale_ms,temp_filtered[pristine_all,window_start:time_axis.shape[0]].T* voltage_step_size* scale_uV, color='#5da4d6')
    ax2.plot(time_axis*scale_ms,temp_filtered[pedot_all,window_start:time_axis.shape[0]].T*voltage_step_size*scale_uV, color='#ff4136')
    plt.ylim(-40, 40)
    plt.xlim(0, 4)
    ax2.axis("OFF")
    ax0.axis("OFF")
    plt.axis("OFF")

plt.vlines(4, -40,40, color='k', linewidth=5)
plt.hlines(-40, 0, 4, color='k', linewidth=5)
ax2.text(1, -40,'4ms',verticalalignment='bottom', horizontalalignment='right',color='black', fontsize=15)
ax2.text(6,30,'80uV',verticalalignment='top', horizontalalignment='right', color='black', fontsize=15)
#plt.legend()
plt.show()
























#-----------------------------------------------------------------------------------------------------------------
#SPD spectrum



def load_raw_data(filename, numchannels=32, dtype=np.uint16):
    fdata = np.memmap(filename, dtype)
    numsamples = int(len(fdata) / numchannels)
    dataMatrix = np.reshape(fdata, (numsamples, numchannels))
    dataMatrix = dataMatrix.T
    dimensionOrder = dm.DataDimensions.fromIndividualDimensions(ct.Dimension.CHANNELS, ct.Dimension.TIME)
    data = dm.Data(dataMatrix, dimensionOrder)
    return data



#amp_dtype = np.int16 #files after concat
#amp_dtype = np.float32 #older files
sampling_freq = 30000
filtered_data_type = np.float64
pedot_all= [22,29,3,23,18,4,17,19,5,16,20,6,30,21,7,31]
pristine_all= np.array([2,9,28,13,8,27,12,14,26,11,15,25,10,1,24,0])







#noise saline solution all freq

raw_data_file_ivm = r'F:\Materials paper w Pedro Baiao\Saline noise_11_05_2015_PAPER\RMS saline_probe\amplifier2015-05-11T11_59_54.bin' #noise saline
raw_data_file_ivm = r'F:\Materials paper w Pedro Baiao\Electronic noise_05_05_2015_ PAPER\SC-headstage\amplifier2015-05-05T17_20_09.bin' #noise SC
raw_data_file_ivm = r'F:\Materials paper w Pedro Baiao\Electronic noise_05_05_2015_ PAPER\infinite_not connected_ Ohm\amplifier2015-05-11T12_33_41.bin'
raw_data_file_ivm = r'F:\Materials paper w Pedro Baiao\Electronic noise_05_05_2015_ PAPER\9.9MOhm\amplifier2015-05-05T18_36_49.bin'
raw_data_file_ivm = r'F:\Materials paper w Pedro Baiao\Electronic noise_05_05_2015_ PAPER\1kOhm\amplifier2015-05-11T12_38_44.bin'
raw_data_file_ivm = r'F:\Materials paper w Pedro Baiao\Saline noise_11_05_2015_PAPER\RMS saline_probe and juxta\amplifier2015-05-11T12_16_44.bin' #noise saline


voltage_step_size = 0.195e-6
scale_uV = 1000000
num_ivm_channels= 32
amp_dtype = np.uint16

raw_data_ivm = ephys.load_raw_data(raw_data_file_ivm, numchannels=num_ivm_channels, dtype=amp_dtype)
temp_unfiltered = raw_data_ivm.dataMatrix

num_samples=temp_unfiltered.shape[1]
sample_axis= np.arange(0,num_samples)
time_axis= sample_axis/30000

samplesToplot= 30000

plt.figure();
plt.plot(time_axis[0:samplesToplot], temp_unfiltered[2, 0:samplesToplot].T  )
plt.plot(time_axis[0:samplesToplot], temp_unfiltered[22, 0:samplesToplot].T )



for i in np.arange(num_ivm_channels):
    plt.plot(time_axis[0:samplesToplot], temp_unfiltered[i, 0:samplesToplot].T )
    plt.ylabel('Voltage (\u00B5V)', fontsize=20)
    plt.xlabel('Time (s)',fontsize=20)










#PSD plot
fs=30000
f,Pxx_dens= signal.welch(np.float64(temp_unfiltered[pedot_all]), fs, nperseg=1024)

plt.figure()
for i in np.arange(len(pedot_all)):
    plt. semilogy(f, Pxx_dens[i,:].T,color='r',linewidth=1)
    #plt.ylim([0.5e-3,100])
plt.xlim([100,6000])
plt.xlabel('frequency(Hz)')
plt.ylabel('PSD (V^2/Hz)')

f,Pxx_dens= signal.welch(np.float64(temp_unfiltered[pristine_all]), fs, nperseg=1024)
for i in np.arange(len(pristine_all)):
    plt. semilogy(f, Pxx_dens[i,:].T,color='b',linewidth=1)
    #plt.ylim([0.5e-3,100])
    plt.xlim([100,6000])
    plt.xlabel('frequency(Hz)')
    plt.ylabel('PSD (V^2/Hz)')




#Recording
fs=30000

base_folder = r'E:\Data_32ch'
rat = 93
good_cells = good_cells_cluster[rat]
clusterSelected = cluster_indices_good[rat]
date = dates[rat]
data_folder = os.path.join(base_folder + '\\' + date, 'Data')
analysis_folder = os.path.join(base_folder + '\\' + date, 'Analysis')
cell_capture_times = all_cell_capture_times[rat]
spike_thresholds = all_spike_thresholds[rat]

num_of_points_in_spike_trig_ivm = 128
num_of_points_for_padding = 2 * num_of_points_in_spike_trig_ivm

pick_recording='3'
#filename = 'E:\\Data_32ch\\2014-11-25\\Analysis'

#Filter for extracellular recording
def highpass(data,BUTTER_ORDER=3, F_HIGH=14250,sampleFreq=30000.0,passFreq=1):
    b, a = signal.butter(BUTTER_ORDER,(passFreq/(sampleFreq/2), F_HIGH/(sampleFreq/2)),'pass')
    return signal.filtfilt(b,a,data)

#lowpass freq 1 to 250Hz

#sampling_freq = 30000
low_pass_freq = 250
iir_params = {'order': 3, 'ftype': 'butter', 'padlen': 0}

raw_data_file_ivm = os.path.join(data_folder, 'amplifier'+date+'T'+cell_capture_times[pick_recording]+'.bin')
raw_data_ivm = ephys.load_raw_data(raw_data_file_ivm, numchannels=num_ivm_channels, dtype=amp_dtype)


temp_unfiltered = raw_data_ivm.dataMatrix
temp_unfiltered = temp_unfiltered.astype(filtered_data_type)
temp_filtered = highpass(temp_unfiltered)
temp_filtered= filters.low_pass_filter(temp_filtered, sampling_freq, low_pass_freq, method='iir', iir_params=iir_params)
number_clusters = len(clusterSelected[pick_recording])





fs=30000
f,Pxx_dens= signal.welch(np.float64(temp_filtered[pedot_all]), fs, nperseg=1024)

plt.figure()
for i in np.arange(len(pedot_all)):
    plt. semilogy(f, Pxx_dens[i,:].T,color='r',linewidth=1)
    #plt.ylim([0.5e-3,100])
plt.xlim([400,6000])
plt.xlabel('frequency(Hz)')
plt.ylabel('PSD (V^2/Hz)')

f,Pxx_dens= signal.welch(np.float64(temp_unfiltered[pristine_all]), fs, nperseg=1024)
for i in np.arange(len(pristine_all)):
    plt. semilogy(f, Pxx_dens[i,:].T,color='b',linewidth=1)
    #plt.ylim([0.5e-3,100])
    plt.xlim([400,6000])
    plt.xlabel('frequency(Hz)')
    plt.ylabel('PSD (V^2/Hz)')







for pick_cluster in clusterSelected[pick_recording]:
    filename_p2p = os.path.join(filename + '\p2p_EXTRA_Cluster' + str(pick_cluster) + '.npy')
    filename_error= os.path.join(filename +  '\error_EXTRA_Cluster' + str(pick_cluster) +'.npy')
    amplitudes = np.load(filename_p2p)
    error = np.load(filename_error)

    min_ampl= amplitudes.min()
    max_ampl= amplitudes.max()
    channel = amplitudes.argmax()
    print(min_ampl)
    print(max_ampl)
    print(channel)
    select_indices_biggestAmp = np.where(amplitudes >= (amplitudes.max()/2))
    print(select_indices_biggestAmp)
    Amp_select_indices = amplitudes[select_indices_biggestAmp]
    print(Amp_select_indices)
    thebiggestchannel_pedot = np.intersect1d(channel,pedot_all)
    print(thebiggestchannel_pedot)
    thebiggestchannel_pristine = np.intersect1d(channel,pristine_all)
    print(thebiggestchannel_pristine)
    select_pedot = np.intersect1d (select_indices_biggestAmp, pedot_all)
    print(select_pedot)
    select_pristine = np.intersect1d (select_indices_biggestAmp, pristine_all)
    print(select_pristine)




    fs=30000
    f,Pxx_dens= signal.welch(np.float64(temp_unfiltered[select_pedot]), fs, nperseg=1024)

    plt.figure()
    for i in np.arange(len(select_pedot)):
        plt. semilogy(f, Pxx_dens[i,:].T,color='r',linewidth=1)
        #plt.ylim([0.5e-3,100])
    plt.xlim([0.1,7500])
    plt.xlabel('frequency(Hz)')
    plt.ylabel('PSD (V^2/Hz)')

    f,Pxx_dens= signal.welch(np.float64(temp_unfiltered[select_pristine]), fs, nperseg=1024)
    pristine_average = Pxx_dens
    for i in np.arange(len(select_pristine)):
        plt. semilogy(f, Pxx_dens[i,:].T,color='b',linewidth=1)
        #plt.ylim([0.5e-3,100])
        plt.xlim([1,])
        plt.xlabel('frequency(Hz)')
        plt.ylabel('PSD (V^2/Hz)')




plt.figure()
f,Pxx_dens= signal.welch(np.float64(temp_unfiltered[select_pristine]), fs, nperseg=1024)
plt. semilogy(f, np.average(Pxx_dens[:,:], axis=0),color='b',linewidth=1)
f,Pxx_dens= signal.welch(np.float64(temp_unfiltered[select_pedot]), fs, nperseg=1024)
plt. semilogy(f, np.average(Pxx_dens[:,:], axis=0),color='r',linewidth=1)
#plt.xlim([400,6000])
plt.xlabel('frequency(Hz)')
plt.ylabel('PSD (V^2/Hz)')





num_samples=temp_unfiltered.shape[1]
sample_axis= np.arange(0,num_samples)
time_axis= sample_axis/30000
samplesToplot= num_samples

plt.figure(); plt.plot(time_axis[0:samplesToplot], temp_unfiltered[pristine_all, 0:samplesToplot].T *voltage_step_size*scale_uV, color='g', label= 'Pristine')
plt.plot(time_axis[0:samplesToplot], temp_unfiltered[pedot_all, 0:samplesToplot].T *voltage_step_size*scale_uV, color='k', label = 'PEDOT')
plt.ylabel('Voltage (\u00B5V)', fontsize=20)
plt.xlabel('Time (s)',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend()
plt.ylim(-200,200)


# Value of amplitudes, ERROR (NO ORDER)

filename = r'E:\Data_32ch\2014-11-25\Analysis'
amplitudes = np.load(os.path.join(filename + '\p2p_EXTRA_Cluster215.npy'))
error = np.load(os.path.join(filename +  '\error_EXTRA_Cluster215.npy'))

min_ampl= amplitudes.min()
max_ampl= amplitudes.max()
channel = amplitudes.argmax()
print(min_ampl)
print(max_ampl)
print(channel)

select_indices_biggestAmp = np.where(amplitudes >= (amplitudes.max()/2))
print(select_indices_biggestAmp)

Amp_select_indices = amplitudes[select_indices_biggestAmp]
print(Amp_select_indices)




pedot_all= [22,29,3,23,18,4,17,19,5,16,20,6,30,21,7,31]
pristine_all= np.array([2,9,28,13,8,27,12,14,26,11,15,25,10,1,24,0])


thebiggestchannel_pedot = np.intersect1d(channel,pedot_all)
print(thebiggestchannel_pedot)

thebiggestchannel_pristine = np.intersect1d(channel,pristine_all)
print(thebiggestchannel_pristine)



select_pedot = np.intersect1d (select_indices_biggestAmp, pedot_all)
print(select_pedot)
select_pristine = np.intersect1d (select_indices_biggestAmp, pristine_all)
print(select_pristine)




pick_cluster = 7
for pick_cluster in clusterSelected[pick_recording]:
    extra_average_PEDOT= np.average(all_cells_ivm_filtered_data[pick_cluster][select_pedot,:,:],axis=0) * voltage_step_size* scale_uV
    num_total_spikes_PEDOT= np.shape(extra_average_PEDOT)[1]
    extra_average_PEDOT= np.average(extra_average_PEDOT[:,:],axis=1)
    num_samples=extra_average_PEDOT.shape[0]
    sample_axis= np.arange(-(num_samples/2),(num_samples/2))
    time_axis= sample_axis/30000

    plt.figure()
    plt.subplot(1,2,2)
    for i in np.arange(len(select_pedot)):
        plt.plot(time_axis*scale_ms,all_cells_ivm_filtered_data[pick_cluster][select_pedot[i],:,:] * voltage_step_size* scale_uV, color='k', label='%i'%num_total_spikes_PEDOT, alpha=0.08)

    plt.plot(time_axis*scale_ms,extra_average_PEDOT,color='grey', linewidth=4)
    plt.ylim(-500, 500)
    plt.xlim(-2, 2)
    plt.axis("OFF")
    plt.text(1.5,400.0,'cluster'+str(pick_cluster)+ '=' + str(num_total_spikes_PEDOT) + 'spikes',horizontalalignment='right', verticalalignment='top')

    #plot all Pristine channels and their average
    extra_average_Pristine= np.average(all_cells_ivm_filtered_data[pick_cluster][select_pristine,:,:],axis=0) * voltage_step_size* scale_uV
    extra_average_Pristine= np.average(extra_average_Pristine[:,:],axis=1)
    num_samples=extra_average_Pristine.shape[0]
    sample_axis= np.arange(-(num_samples/2),(num_samples/2))
    time_axis= sample_axis/30000

    plt.subplot(1,2,1)
    for i in np.arange(len(select_pristine)):
        plt.plot(time_axis*scale_ms,all_cells_ivm_filtered_data[pick_cluster][select_pristine[i],:,:]* voltage_step_size* scale_uV, color='k',label='%i'%num_total_spikes_Pristine, alpha=0.08)

    plt.plot(time_axis*scale_ms,extra_average_Pristine,color='grey', linewidth=4)
    plt.ylim(-500, 500)
    plt.xlim(-2, 2)
    plt.axis("OFF")










plt.figure()
extra_average_PEDOT= np.average(all_cells_ivm_filtered_data[clusterSelected[pick_recording][pick_cluster]][select_pedot,:,:],axis=2) * voltage_step_size* scale_uV
#extra_average_PEDOT= np.average(extra_average_PEDOT[:,:],axis=1)
for i in np.arange(len(select_pedot)):
    plt.plot(time_axis*scale_ms,extra_average_PEDOT[i,:])





for i in np.arange(len(select_pristine)):
    plt.plot(all_cells_ivm_filtered_data[pick_cluster][select_pristine[i],:,:], color='g', linewidth=2)

    num_of_spikes = len(kilosort_units[clusterSelected[pick_recording][clus]])






plt.plot(all_cells_ivm_filtered_data[7][select_pedot,:,:].T, color='b', linewidth=2)

extra_average_microVolts= np.average(all_cells_ivm_filtered_data[clusterSelected[pick_recording][clus]][:,:,:],axis=2) * voltage_step_size* scale_uV
        num_samples=extra_average_microVolts.shape[1]
        sample_axis= np.arange(-(num_samples/2),(num_samples/2))
        time_axis= sample_axis/30000



figure()
number_samples=np.shape(data)[1]
time_msec=number_samples/30.0
x_adc=np.linspace(-(time_msec/2), (time_msec/2), num=number_samples)
for i in pedot_select:
    plot(x_adc, data[i,:].T, color='k', linewidth=2)
for i in pristine_select:
    plot(x_adc, data[i,:].T, color='b', linewidth=2)


#--------------------------------------------------------------------------------------------------------------
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





#Plot the stdv and average PEDOT channles versus pristine
filename = 'E:\\Data_32ch\\2014-11-25\\Analysis'
pick_recording ='5'

pedot_all= [22,29,3,23,18,4,17,19,5,16,20,6,30,21,7,31]
pristine_all= np.array([2,9,28,13,8,27,12,14,26,11,15,25,10,1,24,0])
yoffset = 100
for pick_cluster in clusterSelected[pick_recording]:
    filename_p2p = os.path.join(filename + '\p2p_EXTRA_Cluster' + str(pick_cluster) + '.npy')
    filename_error= os.path.join(filename +  '\error_EXTRA_Cluster' + str(pick_cluster) +'.npy')
    amplitudes = np.load(filename_p2p)
    error = np.load(filename_error)

    min_ampl= amplitudes.min()
    max_ampl= amplitudes.max()
    channel = amplitudes.argmax()
    print(min_ampl)
    print(max_ampl)
    print(channel)
    select_indices_biggestAmp = np.where(amplitudes >= (amplitudes.max()/2))
    print(select_indices_biggestAmp)
    Amp_select_indices = amplitudes[select_indices_biggestAmp]
    print(Amp_select_indices)
    thebiggestchannel_pedot = np.intersect1d(channel,pedot_all)
    print(thebiggestchannel_pedot)
    thebiggestchannel_pristine = np.intersect1d(channel,pristine_all)
    print(thebiggestchannel_pristine)
    select_pedot = np.intersect1d (select_indices_biggestAmp, pedot_all)
    print(select_pedot)
    select_pristine = np.intersect1d (select_indices_biggestAmp, pristine_all)
    print(select_pristine)

    extra_average_PEDOT = np.average(all_cells_ivm_filtered_data[pick_cluster][select_pedot,:,:],axis=0) * voltage_step_size* scale_uV
    extra_stdv_PEDOT = stats.sem(all_cells_ivm_filtered_data[pick_cluster][select_pedot,:,:],axis=0) * voltage_step_size* scale_uV
    num_total_spikes_PEDOT= np.shape(extra_average_PEDOT)[1]
    extra_average_PEDOT = np.average(extra_average_PEDOT[:,:],axis=1)
    extra_stdv_PEDOT = np.average(extra_stdv_PEDOT[:,:],axis=1)
    num_samples=extra_average_PEDOT.shape[0]
    sample_axis= np.arange(-(num_samples/2),(num_samples/2))
    time_axis= sample_axis/30000
    plt.figure()
    plt.subplot(1,2,2)
    plt.plot(time_axis*scale_ms,extra_average_PEDOT,color='grey', label='%i'%num_total_spikes_PEDOT, linewidth=4)
    plt.fill_between(time_axis*scale_ms,extra_average_PEDOT-extra_stdv_PEDOT, extra_average_PEDOT + extra_stdv_PEDOT, alpha=1,facecolor='#FF9848')
    #plt.ylim(np.min(np.min(all_cells_ivm_filtered_data[pick_cluster][:,:,:], axis = 1 ))*voltage_step_size* scale_uV - yoffset, np.max(np.max(all_cells_ivm_filtered_data[2][:,:,:], axis = 1 ))*voltage_step_size* scale_uV + yoffset)
    plt.ylim(-800, 800)
    plt.xlim(-2, 2)
    plt.axis("OFF")
    plt.text(1.5,400.0,'cluster'+str(pick_cluster)+ '=' + str(num_total_spikes_PEDOT) + 'spikes',horizontalalignment='right', verticalalignment='top')




    extra_average_Pristine= np.average(all_cells_ivm_filtered_data[pick_cluster][select_pristine,:,:],axis=0) * voltage_step_size* scale_uV
    extra_average_Pristine= np.average(extra_average_Pristine[:,:],axis=1)
    num_samples=extra_average_Pristine.shape[0]
    sample_axis= np.arange(-(num_samples/2),(num_samples/2))
    time_axis= sample_axis/30000
    plt.subplot(1,2,1)
    for i in np.arange(len(select_pristine)):
        plt.plot(time_axis*scale_ms,all_cells_ivm_filtered_data[pick_cluster][select_pristine[i],:,:]* voltage_step_size* scale_uV, color='k', alpha=0.08)
    plt.plot(time_axis*scale_ms,extra_average_Pristine,color='grey', linewidth=4)
    #plt.ylim(np.min(np.min(np.min(extra_average_microVolts, axis = 1 ))*voltage_step_size* scale_uV - yoffset, np.max(np.max(all_cells_ivm_filtered_data[2][:,:,:], axis = 1 ))*voltage_step_size* scale_uV + yoffset)
    plt.ylim(-800, 800)
    plt.xlim(-2, 2)
    #plt.axis("OFF")




outer_grid = gridspec.GridSpec(np.ceil(number_clusters/2).astype(int),2)
cell = outer_grid[1]
inner_grid = gridspec.GridSpecFromSubplotSpec(1,3, cell)

# From here we can plot usinginner_grid's SubplotSpecs
ax1 = plt.subplot(inner_grid[0,0])
ax2 = plt.subplot(inner_grid[0,1])
ax3 = plt.subplot(inner_grid[0,2])


#---------------------

diffs, norm = crosscorrelate_spike_trains((teste2/30).astype(np.int64),(teste2/30).astype(np.int64), lag=1500)


            hist, edges = np.histogram(diffs, bins=autocor_bin_number)
            hist_plot.data_source.data["top"] = hist
            hist_plot.data_source.data["left"] = edges[:-1] / sampling_freq
            hist_plot.data_source.data["right"] = edges[1:] / sampling_freq

plt.hist(diffs, bins=101, range=(-50,50), align='mid')
plt.hist(hist)
plot(hist)
autocor_bin_number =101

diffs, norm = crosscorrelate_spike_trains((teste2/30).astype(np.int64),(teste2/30).astype(np.int64), lag=1500)

        plt.hist(cross[0]/30,bins=201, range=(-50,50), align='mid')



    for clus in np.arange(0, len(clusterSelected[pick_recording])):
        plt.figure()
        plt.hist((cross[clus][0])/30,bins=101, range=(-50,50), align='mid')
        #plt.ylim(0, ysup)
        #plt.xlim(-x,x)


            # update autocorelogram
            diffs, norm = crosscorrelate_spike_trains(all_extra_spike_times[currently_selected_spike_indices].astype(np.int64),
                                                      all_extra_spike_times[currently_selected_spike_indices].astype(np.int64), lag=1500)




            hist, edges = np.histogram((diffs/30).astype(np.int64), bins=autocor_bin_number)


            # update heatmap








def crosscorrelate(sua1, sua2, lag=None, n_pred=1, predictor=None,
                   display=False, kwargs={}):

    assert predictor is 'shuffle' or predictor is None, "predictor must be \
    either None or 'shuffle'. Other predictors are not yet implemented."
    #Check whether sua1 and sua2 are SpikeTrains or arrays
    sua = []
    for x in (sua1, sua2):
        #if isinstance(x, SpikeTrain):
        if hasattr(x, 'spike_times'):
            sua.append(x.spike_times)
        elif x.ndim == 1:
            sua.append(x)
        elif x.ndim == 2 and (x.shape[0] == 1 or x.shape[1] == 1):
            sua.append(x.ravel())
        else:
            raise TypeError("sua1 and sua2 must be either instances of the" \
                            "SpikeTrain class or column/row vectors")
    sua1 = sua[0]
    sua2 = sua[1]
    if sua1.size < sua2.size:
        if lag is None:
            lag = np.ceil(10*np.mean(np.diff(sua1)))
        reverse = False
    else:
        if lag is None:
            lag = np.ceil(20*np.mean(np.diff(sua2)))
        sua1, sua2 = sua2, sua1
        reverse = True
    #construct predictor
    if predictor is 'shuffle':
        isi = np.diff(sua2)
        sua2_ = np.array([])
        for ni in xrange(1,n_pred+1):
            idx = np.random.permutation(isi.size-1)
            sua2_ = np.append(sua2_, np.add(np.insert(
                (np.cumsum(isi[idx])), 0, 0), sua2.min() + (
                np.random.exponential(isi.mean()))))
    #calculate cross differences in spike times
    differences = np.array([])
    pred = np.array([])
    for k in np.arange(0, sua1.size): #changed xrange() for np.arange()
        differences = np.append(differences, sua1[k] - sua2[np.nonzero(
            (sua2 > sua1[k] - lag) & (sua2 < sua1[k] + lag)  & (sua2 != sua1[k] ))])
    if predictor == 'shuffle':
        for k in np.arange(0, sua1.size): #changed xrange() for np.arange()
            pred = np.append(pred, sua1[k] - sua2_[np.nonzero(
                (sua2_ > sua1[k] - lag) & (sua2_ < sua1[k] + lag))])
    if reverse is True:
        differences = -differences
        pred = -pred
    norm = np.sqrt(sua1.size * sua2.size)
    return differences, pred, norm




teste = crosscorrelate_spike_trains(teste2, teste2,lag=1500)
plt.plot(hist)
plt.hist(teste[0], bins=101, range=(-50,50), align='mid')


teste = crosscorrelate_spike_trains(spike_times[kilosort_units[pick_cluster]], spike_times[kilosort_units[pick_cluster]])
spike_times[kilosort_units[pick_cluster]].T.shape
teste2= np.reshape(spike_times[kilosort_units[pick_cluster]], (358))
teste = crosscorrelate_spike_trains(teste2, teste2)
plt.hist(teste[0])


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
    # calculate cross differences in spike times
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




