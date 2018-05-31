__author__ = 'George Dimitriadis'

import os
import numpy as np
import BrainDataAnalysis.timelocked_analysis_functions as tf
import IO.ephys as ephys
import scipy.signal as signal


base_folder = r'E:\Data_32ch'

#dates = {99: '2015-09-09', 98: '2015-09-04', 97: '2015-09-03', 96: '2015-08-28', 94: '2015-08-26', 93: '2015-08-21'}

dates = {89: '2014-03-20', 90: '2014-03-26', 91: '2014-10-17', 92: '2014-11-13', 93: '2014-11-25'}

#dates = {89: '2014-02-19', 90: '2013-12-20', 91: '2014-02-14', 92: '2014-10-10', 93: '2015-04-24'}


all_cell_capture_times = {93: {'1': '21_27_13', '1_1': '22_09_28', '2': '22_44_57', '3': '23_00_08', '4': '22_31_54'},
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

good_cells_joana = {93: ['1', '1_1', '2', '3', '4'],
                    92: ['1', '2', '3', '4', '5','7'],
                    91: ['1', '1_1', '2'],
                    90: ['1', '2', '3'],
                    89: ['1', '2', '3']}


rat = 93
good_cells = good_cells_joana[rat]

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
amp_dtype = np.uint16
#amp_dtype = np.float32

sampling_freq = 30000
high_pass_freq = 100
filtered_data_type = np.float64

types_of_data_to_load = {'t': 'ivm_data_filtered_cell{}.dat',
                         'c': 'ivm_data_filtered_continous_cell{}.dat',
                         'k': 'ivm_data_filtered_klusta_spikes_cell{}.dat',
                         'p': 'patch_data_cell{}.dat',
                         'm': 'ivm_data_raw_cell{}.dat',
                         }


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

#Filter for extracellular recording
def highpass(data,BUTTER_ORDER=3, F_HIGH=14250,sampleFreq=30000.0,passFreq=100.0):
    b, a = signal.butter(BUTTER_ORDER,(passFreq/(sampleFreq/2), F_HIGH/(sampleFreq/2)),'pass')
    return signal.filtfilt(b,a,data)


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
        temp_filtered = highpass(temp_unfiltered)
        temp_filtered = temp_filtered[:, int(num_of_points_for_padding / 2):-int(num_of_points_for_padding / 2)]
        ivm_data_filtered[:, :, spike] = temp_filtered
    del ivm_data_filtered


# Generate the (channels x time_points ) high passed extracellular recordings datasets for all cells

all_cells_ivm_filtered_data = {}
time_window = 0.1 #seconds
data_to_load = 'c'
for i in np.arange(0, len(good_cells)):
    raw_data_file_ivm = os.path.join(data_folder, 'amplifier'+date+'T'+cell_capture_times[good_cells[i]]+'.bin')

    raw_data_ivm = ephys.load_raw_data(raw_data_file_ivm, numchannels=num_ivm_channels, dtype=amp_dtype)

    time_points_in_window = time_window * sampling_freq
    shape_of_filt_spike_trig_ivm = (num_ivm_channels,
                                    time_points_in_window)
    ivm_data_filtered = np.memmap(os.path.join(analysis_folder, types_of_data_to_load[data_to_load].format(good_cells[i])),
                                    dtype=filtered_data_type,
                                    mode='w+',
                                    shape=shape_of_filt_spike_trig_ivm)
    start_point = int((time_points_in_window + num_of_points_for_padding))
    if start_point < 0:
        break
    end_point = int(start_point + (time_points_in_window + num_of_points_for_padding))
    if end_point > raw_data_ivm.shape()[1]:
        break
    temp_unfiltered = raw_data_ivm.dataMatrix[:, start_point:end_point]
    temp_unfiltered = temp_unfiltered.astype(filtered_data_type)
    temp_filtered = highpass(temp_unfiltered)
    temp_filtered = temp_filtered[:, int(num_of_points_for_padding / 2):-int(num_of_points_for_padding / 2)]
    ivm_data_filtered[:, :] = temp_filtered
    del ivm_data_filtered

# Generate the (channels x time_points x windows) high passed extracellular recordings datasets for all cells

all_cells_ivm_filtered_data = {}
time_window = 0.1 #seconds
data_to_load = 'c'
num_of_windows = 10
for i in np.arange(0, len(good_cells)):
    raw_data_file_ivm = os.path.join(data_folder, 'amplifier'+date+'T'+cell_capture_times[good_cells[i]]+'.bin')

    raw_data_ivm = ephys.load_raw_data(raw_data_file_ivm, numchannels=num_ivm_channels, dtype=amp_dtype)

    time_points_in_window = time_window * sampling_freq
    shape_of_filt_spike_trig_ivm = (num_ivm_channels,
                                    time_points_in_window, num_of_windows)
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
        temp_filtered = highpass(temp_unfiltered)
        temp_filtered = temp_filtered[:, int(num_of_points_for_padding / 2):-int(num_of_points_for_padding / 2)]
        ivm_data_filtered[:, :, window] = temp_filtered
    del ivm_data_filtered


# Generate the (channels x time_points x spikes) juxtacellular recordings datasets for all cells
all_cells_patch_data = {}
data_to_load = 'p'
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



# Generate the (channels x time_points x spikes) raw extracellular recordings datasets for all cells
all_cells_ivm_raw_data = {}
data_to_load = 'm'
high_pass_freq = 0.1

for i in np.arange(0, len(good_cells)):
    raw_data_file_ivm = os.path.join(data_folder, 'amplifier'+date+'T'+cell_capture_times[good_cells[i]]+'.bin')

    raw_data_ivm = ephys.load_raw_data(raw_data_file_ivm, numchannels=num_ivm_channels, dtype=amp_dtype)

    num_of_spikes = len(all_cells_spike_triggers[good_cells[i]])

    shape_of_raw_spike_trig_ivm = ((num_ivm_channels,
                                     num_of_points_in_spike_trig_ivm,
                                     num_of_spikes))
    ivm_data_raw = np.memmap(os.path.join(analysis_folder, types_of_data_to_load[data_to_load].format(good_cells[i])),
                                    dtype=filtered_data_type,
                                    mode='w+',
                                    shape=shape_of_raw_spike_trig_ivm)

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
        temp_unfiltered = highpass(temp_unfiltered)
        temp_unfiltered = temp_unfiltered[:, int(num_of_points_for_padding / 2):-int(num_of_points_for_padding / 2)]
        ivm_data_raw[:, :, spike] = temp_unfiltered
    del ivm_data_raw


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
        time_points_in_window = time_window * sampling_freq
        shape_of_filt_spike_trig_ivm = (num_ivm_channels,
                                        time_points_in_window)
        time_axis = np.arange(-time_points_in_window/2,
                      time_points_in_window/2,
                      1/time_points_in_window)

    ivm_data_filtered = np.memmap(os.path.join(analysis_folder, types_of_data_to_load[data_to_load].format(good_cells[i])),
                                    dtype=filtered_data_type,
                                    mode='r',
                                    shape=shape_of_filt_spike_trig_ivm)
    all_cells_ivm_filtered_data[good_cells[i]] = ivm_data_filtered

# Load the extracellular filtered (channels x time_points x windows)
all_cells_ivm_filtered_data = {}
data_to_load = 'c'
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
        time_points_in_window = time_window * sampling_freq
        shape_of_filt_spike_trig_ivm = (num_ivm_channels,
                                        time_points_in_window,num_of_windows)
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


# Load the raw extracellular recording cut data from the .dat files on hard disk onto memmaped arrays
all_cells_ivm_raw_data = {}
data_to_load = 'm'
num_of_points_in_spike_trig_ivm = 60000
for i in np.arange(0, len(good_cells)):
    all_cells_spike_triggers_cell= np.load(os.path.join(analysis_folder,'triggers_Cell'+ good_cells[i] + '.npy'))
    num_of_spikes = len(all_cells_spike_triggers_cell)
    if data_to_load == 'm':
        shape_of_raw_spike_trig_ivm= (num_ivm_channels,
                                        num_of_points_in_spike_trig_ivm,
                                        num_of_spikes)
        time_axis = np.arange(-num_of_points_in_spike_trig_ivm/(2*sampling_freq),
                      num_of_points_in_spike_trig_ivm/(2*sampling_freq),
                      1/sampling_freq)
    if data_to_load == 'c':
        shape_of_raw_spike_trig_ivm = (num_ivm_channels,
                                        time_points_in_window,
                                        num_of_windows)
        time_axis = np.arange(-time_points_in_window/2,
                      time_points_in_window/2,
                      1/time_points_in_window)
    ivm_data_raw = np.memmap(os.path.join(analysis_folder, types_of_data_to_load[data_to_load].format(good_cells[i])),
                                    dtype=filtered_data_type,
                                    mode='r',
                                    shape=shape_of_raw_spike_trig_ivm)
    all_cells_ivm_raw_data[good_cells[i]] = ivm_data_raw


