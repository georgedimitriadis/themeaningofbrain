import os
import random
import time

import h5py as h5
import matplotlib.pyplot as plt
import mne.filter as filters
import numpy as np
import numpy.core.defchararray as np_char
from matplotlib import colors
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE as tsne

import BrainDataAnalysis.timelocked_analysis_functions as tf
import IO.ephys as ephys
import IO.klustakwik as klusta
import Layouts.Probes.probes_imec as pr_imec
from ExperimentSpecificCode._2016_TSne_Paper.t_sne_bhcuda import bhtsne_cuda

#


base_folder = r'H:\Data'

dates = {99: '2015-09-09', 98: '2015-09-04', 97: '2015-09-03', 96: '2015-08-28', 94: '2015-08-25', 93: '2015-08-20'}


all_cell_capture_times = {99: {'1': '14_48_07', '2': '15_17_07', '3': '16_03_31', '4': '16_30_44',
                               '5': '17_33_01', '6': '17_46_43', '7': '18_21_44', '7_1': '18_38_50',
                               '8': '18_58_02', '9': '19_14_01', '10': '19_40_00', '11': '20_26_29',
                               '12': '20_37_00', '13': '20_43_45'},
                          98: {'1': '15_42_02', '2': '15_48_15', '3': '16_14_37', '4': '16_52_15',
                               '4_1': '17_10_23', '5': '18_16_25', '6': '19_08_39', '6_1': '19_24_34',
                               '7': '20_12_32'},
                          97: {'1': '15_32_01', '2': '16_21_00', '2_1': '16_37_50', '3': '17_18_47',
                               '3_1': '17_40_01', '4': '18_13_52', '5': '18_47_20', '6': '19_01_02',
                               '6_1': '19_40_01', '7': '20_11_01', '8': '20_50_17', '9': '21_18_47'},
                          96: {'1': '18_50_42', '2': '19_03_02', '2_1': '19_48_15', '2_2': '20_15_45',
                               '3': '21_04_38', '4': '21_35_13', '5': '22_09_44', '6': '22_43_37',
                               '7': '22_58_30', '8': '23_09_26', '9': '23_28_20'},
                          94: {'1': '20_31_40', '2': '23_21_12', '3': '00_37_00', '4': '00_56_29'},
                          93: {'1': '22_53_29', '2': '23_23_22', '3': '23_57_25', '3_1': '00_23_38'}}

all_spike_thresholds = {99: {'1': 1e-3, '2': 1e-3, '3': 1e-3, '4': 1e-3,
                             '5': 3e-4, '6': 1e-3, '7': 1e-3, '7_1': 1e-3,
                             '8': 1e-3, '9': 1e-3, '10': 1e-3, '11': 1e-3,
                             '12': 1e-3, '13': 1e-3},
                        98: {'1': 6e-4, '2': 1e-3, '3': 1e-3, '4': 1e-3,
                             '4_1': 3e-4, '5': -1e-3, '6': 1e-3, '6_1': 1e-3,
                             '7': 1e-3},
                        97: {'1': 1e-3, '2': 1e-3, '2_1': 1e-3, '3': 1e-3,
                             '3_1': 1e-3, '4': 2e-3, '5': 1e-3, '6': 1e-3,
                             '6_1': 2e-3, '7': 5e-4, '8': 1e-3, '9': -2e-4},
                        96: {'1': 1e-3, '2': 1e-3, '2_1': 1e-3, '2_2': 1e-3,
                             '3': 1e-3, '4': 1e-3, '5': 1e-3, '6': 1e-3,
                             '7': 1e-3, '8': 1e-3, '9': 1e-3},
                        94: {'1': 1e-3, '2': 1e-3, '3': 1e-3, '4': -5e-4},
                        93: {'1': 1e-3, '2': 5e-4, '3': 4e-4, '3_1': 4e-4}}

all_throwing_away_spikes_thresholds = {99: {'4': -300, '6': -300, '7': -300, '7_1': -300},
                                       98: {'5': -800},
                                       97: {'6': -1000, '6_1': -1000, '9': -2000},
                                       96: {'2_1': -400, '2_2': -400, '7': -400, '9': -500},
                                       94: {'1': -300, '2': -300, '3': -300, '4': -500},
                                       93: {'1': -300, '2': -300, '3': -300, '3_1': -400}}

all_average_over_n_spikes = {99: {'4': 60, '6': 60, '7': 60, '7_1': 60},
                             98: {'5': 30},
                             97: {'6': 200, '6_1': 200, '9': 10},
                             96: {'2_1': 100, '2_2': 100, '7': 100, '9': 300},
                             94: {'1': 100, '2': 100, '3': 100, '4': 100},
                             93: {'1': 100, '2': 100, '3': 100, '3_1': 50}}

good_cells_joana = {99: ['4', '6', '7', '7_1'],
                    98: ['3', '4_1', '5', '6_1'],
                    97: ['4', '6', '6_1', '7', '9'],
                    96: ['2_1', '2_2', '7', '9'],
                    94: ['4'],
                    93: ['1', '2', '3']}

good_cells_george = {99: ['4', '6', '7', '7_1'],
                     98: ['5'],
                     97: ['6', '6_1', '9'],
                     96: ['2_1', '2_2', '9'],
                     94: ['4'],
                     93: ['3_1']}

good_cells_small = {99: ['4', '6', '7', '7_1'],
                    98: ['5'],
                    97: ['6', '6_1'],
                    96: ['2_1', '2_2', '9'],
                    94: ['4'],
                    93: ['3_1']}

good_cells_large = {97: ['9']}

rat = 96

good_cells = good_cells_george[rat]



date = dates[rat]
data_folder = os.path.join(base_folder + '\\' + date, 'Data')
analysis_folder = os.path.join(base_folder + '\\' + date, 'Analysis')

cell_capture_times = all_cell_capture_times[rat]
spike_thresholds = all_spike_thresholds[rat]
throw_away_spikes_thresholds = all_throwing_away_spikes_thresholds[rat]
average_over_n_spikes = all_average_over_n_spikes[rat]


num_of_points_in_spike_trig_ivm = 64
num_of_points_for_padding = 2 * num_of_points_in_spike_trig_ivm

adc_channel_used = 0
num_adc_channels_used = 1
adc_dtype = np.uint16
inter_spike_time_distance = 30
amp_gain = 100
num_ivm_channels = 128
amp_dtype = np.uint16

sampling_freq = 30000
high_pass_freq = 500
filtered_data_type = np.float64

types_of_data_to_load = {'t': 'ivm_data_filtered_cell{}.dat',
                         'c': 'ivm_data_filtered_continous_cell{}.dat',
                         'k': 'ivm_data_filtered_klusta_spikes_cell{}.dat',
                         'p': 'patch_data_cell{}.dat'}




# Generate the spike time triggers (and the adc traces in Volts) for all cells
all_cells_spike_triggers = {}
all_cells_spike_data_in_V = {}
all_cells_spike_triggers_sec = {}
for i in np.arange(0, len(good_cells)):
    raw_data_file_pipette = os.path.join(data_folder, 'adc'+date+'T'+cell_capture_times[good_cells[i]]+'.bin')

    raw_data_patch = ephys.load_raw_event_trace(raw_data_file_pipette, number_of_channels=8,
                                                  channel_used=adc_channel_used, dtype=adc_dtype)
    spike_triggers, spike_data_in_V = tf.find_peaks(raw_data_patch.dataMatrix,
                                                                       threshold=spike_thresholds[good_cells[i]],
                                                                       inter_spike_time_distance=inter_spike_time_distance,
                                                                       amp_gain=amp_gain)
    all_cells_spike_triggers[good_cells[i]] = spike_triggers
    all_cells_spike_data_in_V[good_cells[i]] = spike_data_in_V
    all_cells_spike_triggers_sec[good_cells[i]] = spike_triggers / sampling_freq
    np.save(os.path.join(analysis_folder,'triggers_Cell'+ good_cells[i] + '.npy'), all_cells_spike_triggers[good_cells[i]])
    print(len(spike_triggers))



# Generate the (channels x time_points x spikes) high passed extracellular recordings datasets for all cells
all_cells_ivm_filtered_data = {}
data_to_load = 't'
for i in np.arange(0, len(good_cells)):
    raw_data_file_ivm = os.path.join(data_folder, 'amplifier'+date+'T'+cell_capture_times[good_cells[i]]+'.bin')

    raw_data_ivm = ephys.load_raw_data(raw_data_file_ivm, numchannels=num_ivm_channels, dtype=amp_dtype)

    num_of_spikes = len(all_cells_spike_triggers[good_cells[i]])

    shape_of_filt_spike_trig_ivm = ((num_ivm_channels,
                                     num_of_points_in_spike_trig_ivm,
                                     num_of_spikes))
    ivm_data_filtered = np.memmap(os.path.join(analysis_folder, types_of_data_to_load[data_to_load].format(good_cells[i])),
                                    dtype=filtered_data_type,
                                    mode='w+',
                                    shape=shape_of_filt_spike_trig_ivm)

    for spike in np.arange(0, num_of_spikes):
        trigger_point = all_cells_spike_triggers[good_cells[i]][spike]
        start_point = int(trigger_point - (num_of_points_in_spike_trig_ivm + num_of_points_for_padding)/2)
        if start_point < 0:
            break
        end_point = int(trigger_point + (num_of_points_in_spike_trig_ivm + num_of_points_for_padding)/2)
        if end_point > raw_data_ivm.shape()[1]:
            break
        temp_unfiltered = raw_data_ivm.dataMatrix[:, start_point:end_point]
        temp_unfiltered = temp_unfiltered.astype(filtered_data_type)
        iir_params = {'order': 4, 'ftype': 'butter', 'padlen': 0}
        temp_filtered = filters.high_pass_filter(temp_unfiltered, sampling_freq, high_pass_freq, method='iir',
                                                 iir_params=iir_params)  # 4th order Butter with no padding
        temp_filtered = temp_filtered[:, int(num_of_points_for_padding / 2):-int(num_of_points_for_padding / 2)]
        ivm_data_filtered[:, :, spike] = temp_filtered
    del ivm_data_filtered


# Generate the (channels x time_points x time_slots) high passed extracellular recordings datasets for all cells
# but where the 3rd dimension is not juxta spikes but just consecutive time windows
all_cells_ivm_filtered_data = {}
time_window = 0.2 #seconds
data_to_load = 'c'
memmap_filename = 'ivm_data_filtered_continous_cell{}.dat'
for i in np.arange(0, len(good_cells)):
    raw_data_file_ivm = os.path.join(data_folder, 'amplifier'+date+'T'+cell_capture_times[good_cells[i]]+'.bin')

    raw_data_ivm = ephys.load_raw_data(raw_data_file_ivm, numchannels=num_ivm_channels, dtype=amp_dtype)

    time_points_in_window = time_window * sampling_freq
    num_of_windows = np.shape(raw_data_ivm.dataMatrix)[1] / time_points_in_window
    print('Total number of windows: {}'.format(num_of_windows))
    shape_of_filt_spike_trig_ivm = (num_ivm_channels,
                                    time_points_in_window,
                                    num_of_windows)
    ivm_data_filtered = np.memmap(os.path.join(analysis_folder, types_of_data_to_load[data_to_load].format(good_cells[i])),
                                    dtype=filtered_data_type,
                                    mode='w+',
                                    shape=shape_of_filt_spike_trig_ivm)
    for window in np.arange(num_of_windows):
        start_point = int((time_points_in_window + num_of_points_for_padding) * window)
        if start_point < 0:
            break
        end_point = int(start_point + (time_points_in_window + num_of_points_for_padding))
        if end_point > raw_data_ivm.shape()[1]:
            break
        temp_unfiltered = raw_data_ivm.dataMatrix[:, start_point:end_point]
        temp_unfiltered = temp_unfiltered.astype(filtered_data_type)
        iir_params = {'order': 4, 'ftype': 'butter', 'padlen': 0}
        temp_filtered = filters.high_pass_filter(temp_unfiltered, sampling_freq, high_pass_freq, method='iir',
                                                 iir_params=iir_params)  # 4th order Butter with no padding
        temp_filtered = temp_filtered[:, int(num_of_points_for_padding / 2):-int(num_of_points_for_padding / 2)]
        ivm_data_filtered[:, :, window] = temp_filtered
        if window % 100 == 0:
            print(window)
    del ivm_data_filtered

# Generate the (channels x time_points x spikes) juxtacellular recordings datasets for all cells
all_cells_patch_data = {}
data_to_load = 'p'
amp_gain=100
amp_y_digitization=65536
amp_y_range=10

scaling_factor = amp_y_range / (amp_y_digitization * amp_gain)
for i in np.arange(0, len(good_cells)):
    raw_data_file_patch = os.path.join(data_folder, 'adc'+date+'T'+cell_capture_times[good_cells[i]]+'.bin')

    raw_data_patch = ephys.load_raw_event_trace(raw_data_file_patch, number_of_channels=8,
                                                  channel_used=adc_channel_used, dtype=adc_dtype)

    raw_data_patch = (raw_data_patch.dataMatrix - np.mean(raw_data_patch.dataMatrix)) * scaling_factor

    num_of_spikes = len(all_cells_spike_triggers[good_cells[i]])

    shape_of_filt_spike_trig_patch = ((num_of_points_in_spike_trig_ivm,
                                       num_of_spikes))
    patch_data = np.memmap(os.path.join(analysis_folder, types_of_data_to_load[data_to_load].format(good_cells[i])),
                                    dtype=filtered_data_type,
                                    mode='w+',
                                    shape=shape_of_filt_spike_trig_patch)
    for spike in np.arange(0, num_of_spikes):
        trigger_point = all_cells_spike_triggers[good_cells[i]][spike]
        start_point = int(trigger_point - num_of_points_in_spike_trig_ivm / 2)
        if start_point < 0:
            break
        end_point = int(trigger_point + num_of_points_in_spike_trig_ivm / 2)
        if end_point > raw_data_patch.shape[0]:
            break
        patch_data[:, spike] = raw_data_patch[start_point:end_point]
    del patch_data



# Load the extracellular recording cut data from the .dat files on hard disk onto memmaped arrays
all_cells_ivm_filtered_data = {}
data_to_load = 't'
for i in np.arange(0, len(good_cells)):
    all_cells_spike_triggers_cell= np.load(os.path.join(analysis_folder,'triggers_Cell'+ good_cells[i] + '.npy'))
    num_of_spikes = len(all_cells_spike_triggers_cell)
    if data_to_load == 't':
        shape_of_filt_spike_trig_ivm = (num_ivm_channels,
                                        num_of_points_in_spike_trig_ivm,
                                        num_of_spikes)
        time_axis = np.arange(-num_of_points_in_spike_trig_ivm/(2*sampling_freq),
                      num_of_points_in_spike_trig_ivm/(2*sampling_freq),
                      1/sampling_freq)
    if data_to_load == 'c':
        shape_of_filt_spike_trig_ivm = (num_ivm_channels,
                                        time_points_in_window,
                                        num_of_windows)
        time_axis = np.arange(-time_points_in_window/2,
                      time_points_in_window/2,
                      1/time_points_in_window)
    ivm_data_filtered = np.memmap(os.path.join(analysis_folder, types_of_data_to_load[data_to_load].format(good_cells[i])),
                                    dtype=filtered_data_type,
                                    mode='r',
                                    shape=shape_of_filt_spike_trig_ivm)
    all_cells_ivm_filtered_data[good_cells[i]] = ivm_data_filtered




# Load the juxtacellular recording cut data from the .dat files on hard disk onto memmaped arrays
all_cells_patch_data = {}
data_to_load = 'p'
for i in np.arange(0, len(good_cells)):
    num_of_spikes = len(all_cells_spike_triggers[good_cells[i]])
    if data_to_load == 'p':
        shape_of_filt_spike_trig_patch = ((num_of_points_in_spike_trig_ivm,
                                           num_of_spikes))
        time_axis = np.arange(-num_of_points_in_spike_trig_ivm/(2*sampling_freq),
                      num_of_points_in_spike_trig_ivm/(2*sampling_freq),
                      1/sampling_freq)
        patch_data = np.memmap(os.path.join(analysis_folder, types_of_data_to_load[data_to_load].format(good_cells[i])),
                                        dtype=filtered_data_type,
                                        mode='r',
                                        shape=shape_of_filt_spike_trig_patch)
        all_cells_patch_data[good_cells[i]] = patch_data



# Remove large extracellular spikes from data to see if the average over spikes is due to the random extracellular spiking
cell = good_cells[0]
t = np.zeros(shape=all_cells_ivm_filtered_data[cell].shape[:2])
added = 0
for i in np.arange(0, all_cells_ivm_filtered_data[cell].shape[2]):
    if np.min(all_cells_ivm_filtered_data[cell][:, :, i]) > -2000:
        t = t + all_cells_ivm_filtered_data[cell][:, :, i]
        added += 1
t = t / added
print(added)
plt.plot(t.T)


# Generate averaged extracellular spikes but using as few a number of spikes as possible (after throwing away traces
# with too large amplitudes (extracellular spiking of other cells)
num_of_points_in_spike_trig_ivm = 64
all_cells_ivm_filtered_averaged_data = {}
for cell in good_cells:
    t = []
    for i in np.arange(0, all_cells_ivm_filtered_data[cell].shape[2]):
        if np.min(all_cells_ivm_filtered_data[cell][:, :, i]) > throw_away_spikes_thresholds[cell]:
            t.append(all_cells_ivm_filtered_data[cell][:, :, i])
    t = np.array(t)
    t = np.transpose(t, axes=[1, 2, 0])

    avg_number = average_over_n_spikes[cell]
    t_avg = [np.average(t[:, :, avg_number*i:(avg_number*i+avg_number)], axis=2)
             for i in np.arange(0, int(t.shape[2] / avg_number))]
    t_avg = np.array(t_avg)
    t_avg = np.transpose(t_avg, axes=[1, 2, 0])
    all_cells_ivm_filtered_averaged_data[cell] = t_avg


# Delete some time points from the edges of the timeseries
for cell in good_cells:
    all_cells_ivm_filtered_averaged_data[cell] = all_cells_ivm_filtered_averaged_data[cell][:, 16:48, :]
num_of_points_in_spike_trig_ivm = 32






# Cluster with TSNE averaged out spikes of many cells
# Put the data in a 2D (samples x features -> spikes x (times*channels in a spike)) matrix ready for PCA or T_SNE
X = []
for i in np.arange(0, len(good_cells)):
    num_of_spikes = all_cells_ivm_filtered_averaged_data[good_cells[i]].shape[2]
    newshape = (num_ivm_channels*num_of_points_in_spike_trig_ivm, num_of_spikes)
    t = all_cells_ivm_filtered_averaged_data[good_cells[i]][:, :, :num_of_spikes]
    t = np.reshape(t, newshape=newshape)
    X.extend(t.T)


all_rats_all_cells_ivm_filtered_averaged_data = {}
all_rats_all_cells_ivm_filtered_averaged_data[rat] =  all_cells_ivm_filtered_averaged_data


import pickle
file = open(r'E:\Data\George\Projects\SpikeSorting\Joana_Paired_128ch\AnalysisGlobal' + \
            r'\all_rats_all_cells_ivm_filtered_averaged_data.pkl', 'rb')
all_rats_all_cells_ivm_filtered_averaged_data = pickle.load(file)
file.close()


data_for_tsne = []
cell_info = {'num_of_spikes': [], 'label': [], 'color': []}
colors_used = []
rats = list(all_rats_all_cells_ivm_filtered_averaged_data.keys())
time_points = 32
for rat in rats:
    cells = list(all_rats_all_cells_ivm_filtered_averaged_data[rat].keys())
    for cell in cells:
        num_of_spikes = all_rats_all_cells_ivm_filtered_averaged_data[rat][cell].shape[2]
        start_time_point = int((num_of_points_in_spike_trig_ivm - time_points)/2)
        end_time_point = num_of_points_in_spike_trig_ivm - start_time_point
        newshape = (num_ivm_channels*(end_time_point - start_time_point), num_of_spikes)
        t = all_rats_all_cells_ivm_filtered_averaged_data[rat][cell][:, start_time_point:end_time_point, :]
        t = np.reshape(t, newshape=newshape)
        data_for_tsne.extend(t.T)

        cell_info['num_of_spikes'].append(num_of_spikes)
        cell_info['label'].append('rat: {} cell: {}'.format(rat, cell))
        color = random.choice(list(colors.cnames))
        while color in colors_used:
            color = random.choice(colors.cnames)
        cell_info['color'].append(color)


# Generate the start and end points of each cell in the spikes axis
start = [0]
end = []
total_spikes = 0
for i in np.arange(0, len(cell_info['num_of_spikes'])):
    if i < len(cell_info['num_of_spikes']) - 1:
        start.append(cell_info['num_of_spikes'][i] + total_spikes)
    end.append(cell_info['num_of_spikes'][i] + total_spikes)
    total_spikes += cell_info['num_of_spikes'][i]


# T-SNE
perplexity = 5
early_exaggeration = 1.0
learning_rate = 100
model = tsne(n_components=2, perplexity=perplexity, early_exaggeration=early_exaggeration,
             learning_rate=learning_rate, metric='euclidean',
             random_state=None, init='random', verbose=10)
t_tsne = model.fit_transform(data_for_tsne)
t_tsne = t_tsne.T

#  2D plot
fig = plt.figure()
ax = fig.add_subplot(111)
#c = ['r', 'g', 'b', 'y', 'sienna', 'palegreen', 'm', 'pink', 'olivedrab', 'silver', 'forestgreen', 'palegoldenrod']
s = 20
for i in np.arange(0, len(cell_info['num_of_spikes'])):
    ax.scatter(t_tsne[0][start[i]:end[i]], t_tsne[1][start[i]:end[i]], color=cell_info['color'][i], s=s,
               label=cell_info['label'][i])
# Now add the legend with some customizations.
legend = ax.legend(loc='upper center', shadow=True)

# The frame is matplotlib.patches.Rectangle instance surrounding the legend.
frame = legend.get_frame()
frame.set_facecolor('0.90')

# Set the fontsize
for label in legend.get_texts():
    label.set_fontsize('large')

for label in legend.get_lines():
    label.set_linewidth(1.5)  # the legend line width
fig.suptitle('T-SNE with Per={}, exag={} and l_rate={}'.format(perplexity, early_exaggeration, learning_rate))
plt.show()

#  3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
c = ['r', 'g', 'b', 'y', 'sienna', 'palegreen', 'm', 'pink', 'olivedrab', 'silver', 'forestgreen', 'palegoldenrod']
s = 20
for i in np.arange(0, len(cell_info['num_of_spikes'])):
    ax.scatter(t_tsne[0][start[i]:end[i]], t_tsne[1][start[i]:end[i]], t_tsne[2][start[i]:end[i]],
               color=c[i])


# Testing of the idea of "cleaning" the data before t-sneeing. Does not make a difference
#  Clean the data set to be fed into T-sne
# High pass over 2000Hz
num_of_spikes = ivm_data_filtered.shape[2]
ivm_data_double_filtered = np.zeros(shape=ivm_data_filtered.shape)
for spike in np.arange(0, num_of_spikes):
    ivm_data_double_filtered[:, :, spike] = filters.high_pass_filter(ivm_data_filtered[:, :, spike], Fs=sampling_freq,
                                                                     Fp=2000)
# Set to zero the channels that have to spike features
num_of_spikes = ivm_data_filtered.shape[2]
ivm_data_double_filtered_zeroed = np.zeros(shape=ivm_data_filtered.shape)
for spike in np.arange(0, num_of_spikes):
    for channel in np.arange(0, ivm_data_filtered.shape[0]):
        if np.std(ivm_data_filtered[channel, :, spike]) > 150:
            ivm_data_double_filtered_zeroed[channel, :, spike] = ivm_data_filtered[channel, :, spike]


spikes_to_include = 1500  # (about 3.8 minutes)
fin_time_point = all_cells_spike_triggers['9'][spikes_to_include] + 500

raw_data_file_ivm = os.path.join(data_folder, 'amplifier'+date+'T'+cell_capture_times['9']+'.bin')
raw_data_ivm = ephys.load_raw_data(raw_data_file_ivm, numchannels=num_ivm_channels, dtype=amp_dtype)
raw_data_ivm = raw_data_ivm.dataMatrix[:, :fin_time_point]
shape_of_filt_spike_trig_ivm = np.shape(raw_data_ivm)
temp = np.memmap(os.path.join(analysis_folder, 'ivm_data_filtered_full_trace_cell{}.dat'.format(good_cells[i])),
                                dtype=np.float64,
                                mode='w+',
                                shape=shape_of_filt_spike_trig_ivm)
raw_data_ivm = raw_data_ivm.astype(np.float64)
raw_data_ivm_filtered = filters.high_pass_filter(raw_data_ivm, Fs=sampling_freq, Fp=400)
temp = raw_data_ivm_filtered



###################################################################################
# jpak97_cell9



base_folder = r'D:\Data\George\Projects\SpikeSorting\Joana_Paired_128ch'
rat = 97
good_cells = '9'
date = '2015-09-03'
data_folder = os.path.join(base_folder + '\\' + date, 'Data')
analysis_folder = os.path.join(base_folder + '\\' + date, 'Analysis')

cell_capture_times = '21_18_47'
spike_thresholds = -2e-4

adc_channel_used = 0
adc_dtype = np.uint16
inter_spike_time_distance = 0.002
amp_gain = 100
num_ivm_channels = 128
amp_dtype = np.uint16

sampling_freq = 30000
high_pass_freq = 500
filtered_data_type = np.float64

types_of_data_to_load = {'t': 'ivm_data_filtered_cell{}.dat',
                         'k': 'ivm_data_filtered_klusta_first_{}_spikes_cell_{}.dat',
                         'p': 'patch_data_cell{}.dat'}

num_of_points_in_spike_trig_ivm = 20
num_of_points_for_padding = 2 * num_of_points_in_spike_trig_ivm



# Generate the spike time triggers (and the adc traces in Volts)
raw_data_file_patch = os.path.join(data_folder, 'adc'+date+'T'+cell_capture_times+'.bin')

raw_data_patch = ephys.load_raw_event_trace(raw_data_file_patch, number_of_channels=8,
                                              channel_used=adc_channel_used, dtype=adc_dtype)
spike_triggers, spike_peaks, spike_data_in_V = tf.create_spike_triggered_events(raw_data_patch.dataMatrix,
                                                                   threshold=spike_thresholds,
                                                                   inter_spike_time_distance=inter_spike_time_distance,
                                                                   amp_gain=amp_gain)
num_of_spikes = len(spike_triggers)
print(num_of_spikes)



# Seperate the juxta spikes into a number of groups according to their size
num_of_spike_groups = 4
spike_thresholds_groups = np.arange(np.min(spike_peaks), np.max(spike_peaks),
                                    (np.max(spike_peaks) - np.min(spike_peaks)) / num_of_spike_groups)
spike_thresholds_groups = np.append(spike_thresholds_groups, np.max(spike_peaks))
spike_triggers_grouped = {}
spike_peaks_grouped = {}
spike_triggers_grouped_withnans = {}
spike_peaks_grouped_withnans = {}
for t in range(1, len(spike_thresholds_groups)):
    spike_triggers_grouped[t] = []
    spike_peaks_grouped[t] = []
    spike_peaks_grouped_withnans[t] = np.empty(len(spike_peaks))
    spike_peaks_grouped_withnans[t][:] = np.NAN
    spike_triggers_grouped_withnans[t] = np.empty(len(spike_peaks))
    spike_triggers_grouped_withnans[t][:] = np.NAN
for s in range(len(spike_peaks)):
    for t in range(1, len(spike_thresholds_groups)):
        if spike_peaks[s] < spike_thresholds_groups[t]:
            spike_triggers_grouped[t].append(spike_triggers[s])
            spike_peaks_grouped[t].append(spike_peaks[s])
            break
for t in range(1, len(spike_thresholds_groups)):
    spike_peaks_grouped_withnans[t][np.in1d(spike_triggers, spike_triggers_grouped[t])] = \
        spike_peaks[np.in1d(spike_triggers, spike_triggers_grouped[t])]
    spike_triggers_grouped_withnans[t][np.in1d(spike_triggers, spike_triggers_grouped[t])] = \
        spike_triggers[np.in1d(spike_triggers, spike_triggers_grouped[t])]






# Cut the adc trace into a time x spikes matrix
data_to_load = 'p'
shape_of_filt_spike_trig_patch = ((num_of_points_in_spike_trig_ivm,
                                   num_of_spikes))
patch_data = np.memmap(os.path.join(analysis_folder, types_of_data_to_load[data_to_load].format(good_cells)),
                                dtype=filtered_data_type,
                                mode='w+',
                                shape=shape_of_filt_spike_trig_patch)
for spike in np.arange(0, num_of_spikes):
    trigger_point = spike_triggers[spike]
    start_point = int(trigger_point - num_of_points_in_spike_trig_ivm / 2)
    if start_point < 0:
        break
    end_point = int(trigger_point + num_of_points_in_spike_trig_ivm / 2)
    if end_point > raw_data_patch.dataMatrix.shape[0]:
        break
    patch_data[:, spike] = raw_data_patch.dataMatrix[start_point:end_point]



# Cut the ivm traces into a channels x time x spikes cube and high pass them
data_to_load = 't'
raw_data_file_ivm = os.path.join(data_folder, 'amplifier'+date+'T'+cell_capture_times+'.bin')

raw_data_ivm = ephys.load_raw_data(raw_data_file_ivm, numchannels=num_ivm_channels, dtype=amp_dtype)

shape_of_filt_spike_trig_ivm = ((num_ivm_channels,
                                 num_of_points_in_spike_trig_ivm,
                                 num_of_spikes))
ivm_data_filtered = np.memmap(os.path.join(analysis_folder, types_of_data_to_load[data_to_load].format(good_cells)),
                                dtype=filtered_data_type,
                                mode='w+',
                                shape=shape_of_filt_spike_trig_ivm)
for spike in np.arange(0, num_of_spikes):
    trigger_point = spike_triggers[spike]
    start_point = int(trigger_point - (num_of_points_in_spike_trig_ivm + num_of_points_for_padding)/2)
    if start_point < 0:
        break
    end_point = int(trigger_point + (num_of_points_in_spike_trig_ivm + num_of_points_for_padding)/2)
    if end_point > raw_data_ivm.shape()[1]:
        break
    temp_unfiltered = raw_data_ivm.dataMatrix[:, start_point:end_point]
    temp_unfiltered = temp_unfiltered.astype(filtered_data_type)
    iir_params = {'order': 4, 'ftype': 'butter', 'padlen': 0}
    temp_filtered = filters.high_pass_filter(temp_unfiltered, sampling_freq, high_pass_freq, method='iir',
                                             iir_params=iir_params)  # 4th order Butter with no padding
    temp_filtered = temp_filtered[:, int(num_of_points_for_padding / 2):-int(num_of_points_for_padding / 2)]
    ivm_data_filtered[:, :, spike] = temp_filtered


# Load the already saved ivm_data_filtered
shape_of_filt_spike_trig_ivm = (num_ivm_channels,
                                num_of_points_in_spike_trig_ivm,
                                num_of_spikes)
time_axis = np.arange(-num_of_points_in_spike_trig_ivm/(2*sampling_freq),
              num_of_points_in_spike_trig_ivm/(2*sampling_freq),
              1/sampling_freq)
ivm_data_filtered = np.memmap(os.path.join(analysis_folder, types_of_data_to_load[data_to_load].format(good_cells)),
                                dtype=filtered_data_type,
                                mode='r',
                                shape=shape_of_filt_spike_trig_ivm)









# Single Cell comparison between Klusta and tsne
# 0) Create the required .dat file and the probe .prb file to throw into klustakwik (i.e. the phy module)
spikes_to_include = num_of_spikes
fin_time_point = spike_triggers[spikes_to_include] + 500
start_time_point = 0
time_limits = [start_time_point, fin_time_point]

raw_data_file_ivm = os.path.join(data_folder, 'amplifier'+date+'T'+cell_capture_times+'.bin')
raw_data_ivm = ephys.load_raw_data(raw_data_file_ivm, numchannels=num_ivm_channels, dtype=amp_dtype)
file_dat = os.path.join(analysis_folder, r'klustakwik\raw_data_ivm_klusta.dat')

klusta.make_dat_file(raw_data_ivm.dataMatrix, filename=file_dat, num_channels=num_ivm_channels, time_limits=time_limits)

file_prb = os.path.join(analysis_folder, r'klustakwik\128ch_passive_imec.prb')
electrode_structure = pr_imec.create_128channels_imec_prb(file_prb)



# 0.5) Grab the mask and the PCA components for all spikes from the .kwx file
filename = r'D:\Data\George\Projects\SpikeSorting\Joana_Paired_128ch\2015-09-03\Analysis\klustakwik\threshold_6_5std\threshold_6_5std.kwx'
h5file = h5.File(filename, mode='r')
pca_and_masks = np.array(list(h5file['channel_groups/0/features_masks']))
masks = np.array(pca_and_masks[:, :, 1])
pca_features = np.array(pca_and_masks[:, :, 0])
masked_pca_features = pca_features * masks



# 1) Grab the spike times from the .kwik file
filename = r'D:\Data\George\Projects\SpikeSorting\Joana_Paired_128ch\2015-09-03\Analysis\klustakwik\threshold_6_5std\threshold_6_5std.kwik'
h5file = h5.File(filename, mode='r')
all_extra_spike_times = np.array(list(h5file['channel_groups/0/spikes/time_samples']))
#klusta_clusters = np.array(list(h5file['channel_groups/0/spikes/clusters/main']))
h5file.close()
print('All extra spikes = {}'.format(len(all_extra_spike_times)))


# 2) Function to find the common spikes between the klusta spikedetct results and the juxtacellular spikes
def find_juxta_spikes_in_extra_detected(juxta_spikes_to_include, juxta_spike_triggers, all_extra_spike_times, d_time_points):
    common_spikes = []
    indices_of_common_spikes_in_klusta = []
    clusters_of_common_spikes = []
    prev_spikes_added = 0
    curr_spikes_added = 0
    ks = 0
    juxta_spikes_not_found = []
    index_of_klusta_spike_found = 0
    for juxta_spike in juxta_spike_triggers[index_of_klusta_spike_found:juxta_spikes_to_include]:
        for possible_extra_spike in np.arange(juxta_spike - d_time_points, juxta_spike + d_time_points):
            possible_possitions = np.where(all_extra_spike_times == possible_extra_spike)[0]
            if len(possible_possitions) != 0:
                index_of_klusta_spike_found = possible_possitions[0]
                common_spikes.append(all_extra_spike_times[index_of_klusta_spike_found])
                indices_of_common_spikes_in_klusta.append(index_of_klusta_spike_found)
                #clusters_of_common_spikes.append(klusta_clusters[index_of_klusta_spike_found])
                curr_spikes_added += 1
                break
        if curr_spikes_added > prev_spikes_added:
            prev_spikes_added = curr_spikes_added
        else:
            juxta_spikes_not_found.append(juxta_spike)
    print(np.shape(common_spikes))
    print(str(100 * (np.shape(common_spikes)[0] / len(juxta_spike_triggers[:juxta_spikes_to_include])))+'% found')
    return common_spikes, indices_of_common_spikes_in_klusta, juxta_spikes_not_found


spikes_to_include = num_of_spikes
# 3) Find the common spikes between the klusta spikedetct results and
# the juxtacellular spikes for all the juxta spikes and for all the sub groups of juxta spikes
common_spikes, indices_of_common_spikes_in_klusta, juxta_spikes_not_found = \
    find_juxta_spikes_in_extra_detected(juxta_spikes_to_include=spikes_to_include,
                                        juxta_spike_triggers = spike_triggers,
                                        all_extra_spike_times=all_extra_spike_times,
                                        d_time_points=7)
common_spikes_grouped = {}
juxta_spikes_not_found_grouped = {}
indices_of_common_spikes_in_klusta_grouped = {}
for g in range(1, num_of_spike_groups+1):
    common_spikes_grouped[g], indices_of_common_spikes_in_klusta_grouped[g], juxta_spikes_not_found_grouped[g] = \
        find_juxta_spikes_in_extra_detected(juxta_spikes_to_include=spikes_to_include,
                                            juxta_spike_triggers=spike_triggers_grouped[g],
                                            all_extra_spike_times=all_extra_spike_times,
                                            d_time_points=7)

# 4) Plot the change in percentage of found juxta spikes with time window size
delta_time = range(15)
num_of_common_spikes = []
for i in delta_time:
    common_spikes, indices_of_common_spikes_in_klusta, juxta_spikes_not_found = \
        find_juxta_spikes_in_extra_detected(juxta_spikes_to_include=spikes_to_include,
                                            juxta_spike_triggers=spike_triggers,
                                            all_extra_spike_times=all_extra_spike_times,
                                            d_time_points=i)
    num_of_common_spikes.append(np.shape(common_spikes))
plt.plot(2000*np.array(delta_time)/sampling_freq, np.array(num_of_common_spikes)/spikes_to_include)


up_to_extra_spike = len(all_extra_spike_times)
subset_of_largest_channels = num_ivm_channels
# Generate the data to go into t-sne for all spikes detected by klusta's spikedetect
# 1) Cut the correct data, filter them (hp) and put them in a channels x time x spikes ivm data cube
data_to_load = 'k'
raw_data_file_ivm = os.path.join(data_folder, 'amplifier'+date+'T'+cell_capture_times+'.bin')

raw_data_ivm = ephys.load_raw_data(raw_data_file_ivm, numchannels=num_ivm_channels, dtype=amp_dtype)

shape_of_filt_spike_trig_ivm = (subset_of_largest_channels,
                                num_of_points_in_spike_trig_ivm,
                                up_to_extra_spike)
filename_for_kluster = os.path.join(analysis_folder,
                                     types_of_data_to_load[data_to_load].format(up_to_extra_spike, good_cells))
ivm_data_filtered = np.memmap(filename_for_kluster,
                              dtype=filtered_data_type,
                              mode='w+',
                              shape=shape_of_filt_spike_trig_ivm)
for spike in np.arange(0, up_to_extra_spike):
    trigger_point = all_extra_spike_times[spike]
    start_point = int(trigger_point - (num_of_points_in_spike_trig_ivm + num_of_points_for_padding)/2)
    if start_point < 0:
        break
    end_point = int(trigger_point + (num_of_points_in_spike_trig_ivm + num_of_points_for_padding)/2)
    if end_point > raw_data_ivm.shape()[1]:
        break
    temp_unfiltered = raw_data_ivm.dataMatrix[:, start_point:end_point]
    temp_unfiltered = temp_unfiltered.astype(filtered_data_type)
    if subset_of_largest_channels != num_ivm_channels:
        mins = np.min(temp_unfiltered, axis=1)
        indices = np.argsort(mins)
        temp_unfiltered = temp_unfiltered[indices[:subset_of_largest_channels]]
    iir_params = {'order': 4, 'ftype': 'butter', 'padlen': 0}
    temp_filtered = filters.high_pass_filter(temp_unfiltered, sampling_freq, high_pass_freq, method='iir',
                                             iir_params=iir_params)  # 4th order Butter with no padding
    temp_filtered = temp_filtered[:, int(num_of_points_for_padding / 2):-int(num_of_points_for_padding / 2)]
    ivm_data_filtered[:, :, spike] = temp_filtered
del raw_data_ivm
del ivm_data_filtered
time_axis = np.arange(-num_of_points_in_spike_trig_ivm/(2*sampling_freq),
                      num_of_points_in_spike_trig_ivm / (2 * sampling_freq),
                      1 / sampling_freq)

# 2) Load the ivm data cube
data_to_load = 'k'
num_of_spikes = len(all_extra_spike_times)
shape_of_filt_spike_trig_ivm = (subset_of_largest_channels,
                                num_of_points_in_spike_trig_ivm,
                                up_to_extra_spike)
filename_for_kluster = os.path.join(analysis_folder,
                                    types_of_data_to_load[data_to_load].format(up_to_extra_spike, good_cells))
ivm_data_filtered = np.memmap(filename_for_kluster,
                              dtype=filtered_data_type,
                              mode='r',
                              shape=shape_of_filt_spike_trig_ivm)

# 2.5) Generate a mask for the required time points (spikes x (channels * time_points)) from the pca mask
# (spikes x (channels * 3 pca features))
index_every_3 = np.arange(0, num_ivm_channels * 3, 3)
masks_single_numbers = masks[:up_to_extra_spike, index_every_3]
mask_for_time_points = np.repeat(masks_single_numbers, repeats=num_of_points_in_spike_trig_ivm, axis=1)


# 3) Put the ivm cube in a 2D matrix ready for tsne
X = []
use_features = True
mask_data = False

if not use_features:
    data_to_use = ivm_data_filtered
    #data_to_use = np.mean(ivm_data_filtered, axis=0)
    #data_to_use = data_to_use.reshape((1, data_to_use.shape[0], data_to_use.shape[1]))
    num_of_spikes = data_to_use.shape[2]
    number_of_time_points = data_to_use.shape[1]
    number_of_channels = data_to_use.shape[0]
    start = int((num_of_points_in_spike_trig_ivm - number_of_time_points) / 2)
    end = start + number_of_time_points
else:
    data_to_use = pca_features[:up_to_extra_spike, :].T
    data_to_use = data_to_use.reshape((1, data_to_use.shape[0], data_to_use.shape[1]))
    num_of_spikes = data_to_use.shape[2]
    number_of_time_points = data_to_use.shape[1]
    number_of_channels = data_to_use.shape[0]
    start = 0
    end = number_of_time_points

newshape = (number_of_channels * number_of_time_points, num_of_spikes)
t = data_to_use[:, start:end, :num_of_spikes]
t = np.reshape(t, newshape=newshape)
if mask_data:
    t = t * np.transpose(mask_for_time_points)
X.extend(t.T)
X_np = np.array(X)
del t, X


#indices_of_data_for_tsne = [(i, k)[0] for i, k in enumerate(X) if random.random() > 0] #For random choice
indices_of_data_for_tsne = range(up_to_extra_spike) #For the first n spikes
data_for_tsne = X_np[indices_of_data_for_tsne]
juxta_cluster_indices_grouped = {}
for g in range(1, num_of_spike_groups+1):
    juxta_cluster_indices_temp = np.intersect1d(indices_of_data_for_tsne, indices_of_common_spikes_in_klusta_grouped[g])
    juxta_cluster_indices_grouped[g] = [i for i in np.arange(0, len(indices_of_data_for_tsne)) if
                             len(np.where(juxta_cluster_indices_temp == indices_of_data_for_tsne[i])[0])]
    print(len(juxta_cluster_indices_grouped[g]))



# T-SNE
# Python scikit-learn t-sne
t0 = time.time()
perplexity = 500.0
early_exaggeration = 100.0
learning_rate = 3000.0
theta = 0.0
model = tsne(n_components=2, perplexity=perplexity, early_exaggeration=early_exaggeration,
             learning_rate=learning_rate, n_iter=1000, n_iter_without_progress=500,
             min_grad_norm=1e-7, metric="euclidean", init="random", verbose=3,
             random_state=None, method='barnes_hut', angle=theta)
t_tsne = model.fit_transform(data_for_tsne)
t_tsne = t_tsne.T
t1 = time.time()
print("Scikit t-sne took {} seconds, ({} minutes), for {} spikes".format(t1-t0, (t1-t0)/60, up_to_extra_spike))

# save the python scikit generated t-sne results
threshold = 5.5
file_name = r'D:\Data\George\Projects\SpikeSorting\Joana_Paired_128ch\2015-09-03\Analysis\t_tsne_ivm_data_{}sp_{}per_{}ee_{}lr_{}tp_{}thres.pkl'\
    .format(len(indices_of_data_for_tsne), perplexity, early_exaggeration, learning_rate, number_of_time_points, threshold)
file = open(file_name, 'bw')
pickle.dump((ivm_data_filtered, t_tsne, juxta_cluster_indices_grouped, perplexity, early_exaggeration, learning_rate), file)
file.close()


bhtsne_cuda.save_data_for_tsne(data_for_tsne, r'E:\George\SourceCode\Repos\t_sne_gpu\t_sne_gpu\windows', 'data.dat',
                               theta=0.6, perplexity=50, eta=200, no_dims=2, iterations=1000, gpu_mem=0.8, randseed=-1)
t_tsne = np.transpose(
    bhtsne_cuda.load_tsne_result(r'E:\George\SourceCode\Repos\t_sne_gpu\t_sne_gpu\windows', 'result.dat'))

# C++ wrapper t-sne using CPU
t0 = time.time()
perplexity = 50.0
theta = 0.2
learning_rate = 200.0
iterations = 5000
gpu_mem = 0
t_tsne = bhtsne_cuda.bh_tsne(data_for_tsne,
                             tmp_dir_path=r'D:\Data\George\Projects\SpikeSorting\Joana_Paired_128ch\2015-09-03\Analysis\tsne_results',
                             no_dims=2, perplexity=perplexity, eta=learning_rate, theta=theta,
                             iterations=iterations, gpu_mem=gpu_mem, randseed=-1, verbose=3)
t_tsne = np.transpose(t_tsne)
t1 = time.time()
print("C++ t-sne took {} seconds, ({} minutes), for {} spikes".format(t1-t0, (t1-t0)/60, up_to_extra_spike))


# C++ wrapper t-sne using GPU
t0 = time.time()
perplexity = 50.0
theta = 0.2
learning_rate = 200.0
iterations = 5000
gpu_mem = 0.8
t_tsne = bhtsne_cuda.bh_tsne(data_for_tsne,
                             tmp_dir_path=r'D:\Data\George\Projects\SpikeSorting\Joana_Paired_128ch\2015-09-03\Analysis\tsne_results',
                             no_dims=2, perplexity=perplexity, eta=learning_rate, theta=theta,
                             iterations=iterations, gpu_mem=gpu_mem, randseed=-1, verbose=3)
t_tsne = np.transpose(t_tsne)
t1 = time.time()
print("CUDA t-sne took {} seconds, ({} minutes), for {} spikes".format(t1-t0, (t1-t0)/60, up_to_extra_spike))


#  2D plot
fig = plt.figure()
ax = fig.add_subplot(111)
juxta_scatters = []
s = 10
ax.scatter(t_tsne[0], t_tsne[1], s=3)
c = ['r', 'g', 'c', 'm', 'y', 'k', 'w', 'b']
for g in range(1, num_of_spike_groups+1):
    juxta_scatters.append(ax.scatter(t_tsne[0][juxta_cluster_indices_grouped[g]],
                                     t_tsne[1][juxta_cluster_indices_grouped[g]],
                                     s=s, color=c[g-1]))
fig.suptitle('T-SNE on Thres={} with Per={}, l_rate={}, extra spikes={}'.format(5.5, perplexity, learning_rate,
                                                                       len(indices_of_data_for_tsne)))
threshold_legend = np_char.add(np.char.mod('%i', spike_thresholds_groups * 1e6), ' uV')
plt.legend(juxta_scatters, threshold_legend)
plt.tight_layout(rect=[0,0,1,1])


#  3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
s = 10
c = ['r', 'g', 'c', 'm', 'y', 'k', 'w', 'b']
ax.scatter(t_tsne[0], t_tsne[1], t_tsne[2], s=3)
for g in np.arange(1, num_of_spike_groups+1):
    ax.scatter(t_tsne[0][juxta_cluster_indices_grouped[g]], t_tsne[1][juxta_cluster_indices_grouped[g]],
               t_tsne[2][juxta_cluster_indices_grouped[g]], s=s, color=c[g-1])

# load saved t-sne results
spike_number = 53497
perplexity = 800.0
early_exaggeration = 100.0
learning_rate = 3000.0
number_of_time_points = 20
threshold = 11
file_name = r'D:\Data\George\Projects\SpikeSorting\Joana_Paired_128ch\2015-09-03\Analysis\t_tsne_ivm_data_{}sp_{}per_{}ee_{}lr_{}tp_{}thres.pkl'\
    .format(spike_number, perplexity, early_exaggeration, learning_rate, number_of_time_points, threshold)
file = open(file_name, 'br')
ivm_data_filtered, t_tsne, juxta_cluster_indices_grouped, perplexity, early_exaggeration, learning_rate = pickle.load(file)
file.close()

# Load the c++ bhtsne results
tmp_dir_path = r'D:\Data\George\Projects\SpikeSorting\Joana_Paired_128ch\2015-09-03\Analysis\tsne_results'
results_filename = 'result_0to50kexsp_40timpoints_per500_ee100_lr3k_theta07.dat'
t_tsne = bhtsne_cuda.load_tsne_result(tmp_dir_path, results_filename)
t_tsne = np.transpose(t_tsne)


# Clustering
def fit_dbscan(data, eps, min_samples, show=True, juxta_cluster_indices_grouped=None, threshold_legend=None):
    X = np.transpose(data)

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    score = metrics.silhouette_score(X, labels, sample_size=5000)
    print('For eps={}, min_samples={}, estimated number of clusters={}'.format(eps, min_samples, n_clusters_))
    print("Silhouette Coefficient: {}".format(score))

    if show:
        show_clustered_tsne(db, X, juxta_cluster_indices_grouped, threshold_legend)

    return db, n_clusters_, labels, core_samples_mask, score


from sklearn.cluster import KMeans
kmeans_est_35 = KMeans(n_clusters=35)
kmeans_est_35.fit(X)


def show_clustered_tsne(dbscan_result, X, juxta_cluster_indices_grouped=None, threshold_legend=None):
    core_samples_mask = np.zeros_like(dbscan_result.labels_, dtype=bool)
    core_samples_mask[dbscan_result.core_sample_indices_] = True
    labels = dbscan_result.labels_

    fig = plt.figure()
    ax = fig.add_subplot(111)
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        ms = 4
        if k == -1:
            # Black used for noise.
            col = 'k'
            ms = 2

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                markeredgecolor='k', markersize=6)

        xy = X[class_member_mask & ~core_samples_mask]
        ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                markeredgecolor='k', markersize=ms)

    if juxta_cluster_indices_grouped is not None:
        c = ['r', 'g', 'c', 'm', 'y', 'k', 'w', 'b']
        juxta_scatters = []
        for g in range(1, len(juxta_cluster_indices_grouped)+1):
            line, = ax.plot(X[juxta_cluster_indices_grouped[g], 0], X[juxta_cluster_indices_grouped[g], 1], '*',
                            markersize=4.5, markerfacecolor=c[g-1], markeredgecolor=c[g-1])
            juxta_scatters.append(line)
        if threshold_legend is not None:
            ax.legend(juxta_scatters, threshold_legend)

    plt.tight_layout(rect=[0,0,1,1])

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    plt.title('DBSCAN clustering of T-sne with {} estimated number of clusters'.format(n_clusters_))
    plt.show()



# Loop over values of eps and min_samples to find best fit for DBSCAN
epses = np.arange(0.1, 0.5, 0.05)
min_sampleses = np.arange(5, 100, 5)
clustering_scores = []
params = []
for eps in epses:
    for min_samples in min_sampleses:
        db, n_clusters_, labels, core_samples_mask, score = fit_dbscan(t_tsne, eps, min_samples, show=False)
        clustering_scores.append(score)
        params.append((eps, min_samples))
params = np.array(params)
clustering_scores = np.array(clustering_scores)


# Define TP / FP / TN and FN
X = np.transpose(t_tsne)
juxta_cluster_indices = []
for g in range(1, num_of_spike_groups+1):
    juxta_cluster_indices.extend(juxta_cluster_indices_grouped[g])
means_of_juxta = np.array([np.median(X[juxta_cluster_indices, 0]), np.median(X[juxta_cluster_indices, 1])])

means_of_labels = np.zeros((n_clusters_, 2))
dmeans = np.zeros(n_clusters_)
for l in range(n_clusters_):
    class_member_mask = (labels == l)
    xy = X[class_member_mask & core_samples_mask]
    means_of_labels[l, 0] = np.median(xy[:, 0])
    means_of_labels[l, 1] = np.median(xy[:, 1])
    dmeans[l] = np.linalg.norm((means_of_labels[l,:]-means_of_juxta))
juxta_cluster_index = np.argmin(dmeans)

# have a look where the averages are
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(means_of_juxta[0], means_of_juxta[1])
ax.scatter(means_of_labels[:, 0], means_of_labels[:, 1], color='r')

# calculate prec, rec and f factor
class_member_mask = (labels == juxta_cluster_index)
extra_cluster_indices = [i for i, x in enumerate(class_member_mask & core_samples_mask) if x]
tp_indices = np.intersect1d(juxta_cluster_indices, extra_cluster_indices)
tp = len(tp_indices)
all_pos = len(extra_cluster_indices)
all_true = len(juxta_cluster_indices)
precision = tp / all_pos
recall = tp / all_true
f_factor = 2*(precision*recall)/(precision+recall)
print('Precision = {}, Recall = {}, F1 factor = {}'.format(precision, recall, f_factor))

#Distances
import numpy as np
import matplotlib.pyplot as plt
import itertools

#ALL_ELECTRODES
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

#####ORDEREDSITES
electrode_coordinate_grid = list(itertools.product(np.arange(1, 33), np.arange(1, 5)))
orderedSites = electrode_coordinate_grid


#####SITESPOSITION
scaleX=25
scaleY=25
scale = np.array([scaleX,scaleY])
probeConfiguration=np.zeros((32,4,2))
for i in range(32):
    for j in range(4):
        probeConfiguration[i,j]= np.array([i,j])

probeConfiguration = np.flipud(probeConfiguration) # the (0,0) for Y,Z is the elect (31,0) left bottom
probeConfiguration = probeConfiguration * scale
sitesPosition = probeConfiguration



###REFERENCE POINT EXTRACELLULAR

referencecoordinates = (0, 37.5) #ref point bottom center (z,y)
referencecoordinates = (-310, 37.5) #ref point Tip (z,y)





def eval_dist(referencecoordinates, IVM, juxta):
    '''
    referenceSite = site to which the IVM coordinates refer to
    IVM = coordinates of the reference site on the probe
    juxta = juxta probe coordinates
    sitesPositionsFN = path to sites positions file.
    '''

    scaleX=25
    scaleY=25
    scale = np.array([scaleX,scaleY])
    probeConfiguration=np.zeros((32,4,2))
    for i in range(32):
        for j in range(4):
            probeConfiguration[i,j]= np.array([i,j])

    probeConfiguration = np.flipud(probeConfiguration) # the (0,0) for Y,Z is the elect (31,0) left bottom
    probeConfiguration = probeConfiguration * scale
    sitesPosition = probeConfiguration

    electrode_coordinate_grid = list(itertools.product(np.arange(1, 33), np.arange(1, 5)))
    orderedSites = electrode_coordinate_grid

    referencePosition = np.zeros(2)
    referencePosition[0] = referencecoordinates[0]
    referencePosition[1] = referencecoordinates[1]

    #orderedSites = id_rever
    NumSites =  len(orderedSites)


    for i in range(32):
         for j in range(4):
            sitesPosition[i][j][0]= sitesPosition[i][j][0] - referencePosition[0]
            sitesPosition[i][j][1]= sitesPosition[i][j][1] - referencePosition[1]



    newPositions = np.zeros((128, 3))

    #newPositions[:, 1] = np.copy(sitesPositions[:, 0])
    #newPositions[:, 2] = np.copy(sitesPositions[:, 1])
    newPositions = np.zeros((NumSites, 3))
    newPositions[:, 2] = np.copy(sitesPosition[:,:,0].reshape(1,128))
    newPositions[:, 1] = np.copy(sitesPosition[:,:,1].reshape(1,128))

    for i in range(NumSites):
        newPositions[i, 0] = newPositions[i, 2] * np.cos(0.8412486985)  # 48.2º

    for i in range(NumSites):
        newPositions[i, 2] = newPositions[i, 2] * np.sin(0.8412486985)  # 48.2º

    spikesPositions = np.zeros((NumSites, 3))
    spikesDistances = np.zeros(NumSites)

    spikesPositions = np.copy(newPositions)
    for i in range(NumSites):
        spikesPositions[i] = spikesPositions[i] - IVM + juxta

    for j in range(NumSites):
        spikesDistances[j] = np.sqrt(spikesPositions[j][0]**2 +
                                     spikesPositions[j][1]**2 +
                                     spikesPositions[j][2]**2)

    return spikesPositions, spikesDistances

######################
cellname = 'cell4_0'



#cell7''(196.5;0;-4865)IVM(408.7;0;-5282.4)

#cell7(204;0;-4850)IVM(408.7;0;-5282.4)

#cell5(331.5 0 -4594.8)IVM(494.5399 0 -4920.042)

# extraPos and juxtaPos have to be numpy arrays
# I had to add a decimal place in all axes
# The Z axis has to be flipped
#cell4(-185;y;-4094)IVM(313.3;y;-4640)
#IVM 468.9159 33.3 -4695.594 Patch 437.4 33.4 -4462.1

#IVM 468.9159 33.3 -4695.594 Patch 437.4 33.4 -4462.1


#494.5399 0 -4920.042	161.4 47.2 -4415.2

extraPos = np.array([494.5399, 0, -4920.042 ])*np.array([1, 1, -1])
juxtaPos = np.array([161.4, 47.2, -4415.2 ])*np.array([1, 1, -1])
#referencecoordinates = (-310, 37.5) #ref point Tip (z,y)
referencecoordinates = (0, 37.5) #ref point bottom center (z,y)

extraPos = np.array([487.2, 0.0, -5333.3])*np.array([1, 1, -1])
juxtaPos = np.array([266.2, 20.0, -5075.2])*np.array([1, 1, -1])
#referencecoordinates = (-310, 37.5) #ref point Tip (z,y)
referencecoordinates = (0, 37.5) #ref point bottom center (z,y)

referencecoordinates = (0, 37.5)
refSite = referencecoordinates

pos, dist = eval_dist(refSite, extraPos, juxtaPos)


min_dist = dist.min()
channel_min_dist = electrode_coordinate_grid[dist.argmin()]
#channel_intan = orderedSites[0, dist.argmin()]
print(min_dist)
print(channel_min_dist)
#print(channel_intan)

i=3

np.save(os.path.join(analysis_folder,'distances_Cell'+ good_cells[i] + '.npy'), dist)
np.save(os.path.join(analysis_folder,'positions_Cell'+ good_cells[i] + '.npy'), pos)


# Schematic of the relative positions of the juxtacellular probe and the
# electrodes of the silicon probe ()
fig, ax = plt.subplots()
ax.scatter(pos[:, 0], pos[:, 2], color='b')
ax.scatter(0, 0, color='r')
ax.set_title('XoZ plane\n'+cellname, fontsize=20)
ax.set_aspect('equal')
plt.draw()
plt.show()

####scheme of heatmap

A = np.copy(dist)

orderedSites = all_electrodes.reshape(1,128)
Amod = np.reshape(A,(32,4))
fig, ax = plt.subplots()
#ax.set_title('Distances Heatmap',fontsize=40, position=(0.8,1.02))
plt.axis('off')
im = ax.imshow(Amod, cmap=plt.get_cmap('jet'),vmin = np.min(A),vmax= np.max(A))
cb = fig.colorbar(im, ticks = [np.min(A), 0,np.max(A)])
cb.ax.tick_params(labelsize = 20)
plt.show()

####3D

spikesPositions =pos
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(spikesPositions[:,0],spikesPositions[:,1],spikesPositions[:,2])
ax.scatter(0,0,0,c='g')


###average
###first load data 3D and average

def heatmapp_amplituide(all_cells_ivm_filtered_data, good_cells_number = 0):

    voltage_step_size = 0.195e-6
    extra_average_V = np.average(all_cells_ivm_filtered_data[good_cells[good_cells_number]][:,:,:],axis=2) * voltage_step_size
    extra_average_microVolts = extra_average_V * 1000000
    orderedSites = all_electrodes.reshape(1,128)
    amplitude = np.zeros(128)
    for j in np.arange(128):
        amplitude[j] = abs(np.min(extra_average_microVolts[orderedSites[0][j],:])) + abs(np.max(extra_average_microVolts[orderedSites[0][j],:]))
    return amplitude

i = 0
amplitude = heatmapp_amplituide(all_cells_ivm_filtered_data, good_cells_number = i)

np.save(os.path.join(analysis_folder,'amplitude_EXTRA_Cell'+ good_cells[i] + '.npy'), amplitude)

####scheme of heatmap

B = np.copy(amplitude)

orderedSites = all_electrodes.reshape(1,128)
Bmod = np.reshape(B,(32,4))
fig, ax = plt.subplots()
#ax.set_title('Distances Heatmap',fontsize=10, position=(0.8,1.02))
plt.axis('off')
im = ax.imshow(Bmod, cmap=plt.get_cmap('jet'),vmin = np.min(B),vmax= np.max(B))
cb = fig.colorbar(im,ticks = [np.min(B), 0,np.max(B)])
cb.ax.tick_params(labelsize = 20)
plt.show()

