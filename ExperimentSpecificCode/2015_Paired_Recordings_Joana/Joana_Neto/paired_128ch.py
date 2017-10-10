__author__ = 'George Dimitriadis'



import os
import numpy as np
import BrainDataAnalysis.timelocked_analysis_functions as tf
import IO.ephys as ephys
import scipy.signal as signal

base_folder = r'D:\Protocols\PairedRecordings\Neuroseeker128\Data'

dates = {99: '2015-09-09', 98: '2015-09-04', 97: '2015-09-03', 96: '2015-08-28', 94: '2015-08-26', 93: '2015-08-21', 92: '2015-08-27'}


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
                          93: {'1': '22_53_29', '2': '23_23_22', '3': '00_02_48', '3_1': '00_23_38'},
                          92: {'1': '19_59_51', '2': '20_07_41', '3': '20_17_03', '4': '20_42_42',
                               '5': '20_54_45'}}

all_spike_thresholds = {99: {'1': 1e-3, '2': 1e-3, '3': 1e-3, '4': 1e-3,
                             '5': 3e-4, '6': 1e-3, '7': 1e-3, '7_1': 1e-3,
                             '8': 1e-3, '9': 1e-3, '10': 1e-3, '11': 0.2e-3,
                             '12': 1e-3, '13': 1e-3},
                        98: {'1': 6e-4, '2': 1e-3, '3': 0.5e-3, '4': 1e-3,
                             '4_1': 3e-4, '5': -1e-3, '6': 0.25e-3, '6_1': 0.25e-3,
                             '7': 0.25e-3},
                        97: {'1': 0.5e-3, '2': 1e-3, '2_1': 1e-3, '3': 1e-3,
                             '3_1': 1e-3, '4': 2e-3, '5': 1e-3, '6': 1e-3,
                             '6_1': 2e-3, '7': 5e-4, '8': 1e-3, '9': -2e-4},
                        96: {'1': 0.3e-3, '2': 0.5e-3, '2_1': 1e-3, '2_2': 1e-3,
                             '3': 1e-3, '4': 0.5e-3, '5': 1e-3, '6': 1e-3,
                             '7': 1e-3, '8': 1e-3, '9': 1e-3},
                        94: {'1': 0.4e-3, '2': -0.3e-3, '3': 1e-3, '4': -5e-4},
                        93: {'1': 0.5e-3, '2': 5e-4, '3': 4e-4, '3_1': 4e-4},
                        92: {'1': -0.8e-3, '2': 1e-3, '3': 1e-3, '4': 0.45e-3,
                             '5': 4e-4}}

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
                    98: ['5', '6', '6_1'],
                    97: ['6', '6_1','9'],
                    96: ['2', '2_1', '2_2', '9'],
                    94: ['1', '2', '4'],
                    93: ['3','3_1']}

all_cells = {99: ['1', '2', '3', '4', '5', '6', '7', '7_1', '8', '9', '10', '11', '12', '13'],
             98: ['1', '2','3','4', '4_1', '5', '6', '6_1', '7'],
             97: ['1', '2', '2_1', '3', '3_1', '4', '5', '6', '6_1', '7', '8', '9'],
             96: ['1', '2', '2_1', '2_2', '3', '4', '5', '6', '7', '8', '9'],
             94: ['1', '2', '3', '4'],
             93: ['1', '2', '3', '3_1'],
             92: ['1', '2', '3', '4', '5']}


good_cells_large = {97: ['9']}

good_cells_george = {97: ['6']}

rat = 97

good_cells = good_cells_george [rat]

date = dates[rat]
data_folder = os.path.join(base_folder + '\\' + date, 'Data')
analysis_folder = os.path.join(base_folder + '\\' + date, 'Analysis')

cell_capture_times = all_cell_capture_times[rat]
spike_thresholds = all_spike_thresholds[rat]
#throw_away_spikes_thresholds = all_throwing_away_spikes_thresholds[rat]
#average_over_n_spikes = all_average_over_n_spikes[rat]


num_of_points_in_spike_trig_ivm = 128
num_of_points_for_padding = 2 * num_of_points_in_spike_trig_ivm

adc_channel_used = 0
num_adc_channels_used = 1
adc_dtype = np.uint16
inter_spike_time_distance = 30
amp_gain = 50
num_ivm_channels = 128
amp_dtype = np.uint16

sampling_freq = 30000
high_pass_freq = 100
low_pass_freq = 5000
filtered_data_type = np.float64

types_of_data_to_load = {'t': 'ivm_data_filtered_cell{}.dat',
                         'c': 'ivm_data_filtered_continous_cell{}.dat',
                         'k': 'ivm_data_filtered_klusta_spikes_cell{}.dat',
                         'p': 'patch_data_cell{}.dat',
                         'm': 'ivm_data_raw_cell{}.dat',
                         'l': 'ivm_data_filteredlow_cell{}.dat'
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
def highpass(data,BUTTER_ORDER=3, F_HIGH=5000, sampleFreq=30000.0, passFreq=100.0):
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



# Load the raw extracellular recording cut data from the .dat files on hard disk onto memmaped arrays
all_cells_ivm_raw_data = {}
data_to_load = 'm'
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
