
import numpy as np
from os.path import join
from joblib import Parallel, delayed


def load_extracellular_data_cube(data_cube_filename,
                                 cube_type,
                                 shape_of_spike_trig_avg):
    cut_extracellular_data = np.memmap(data_cube_filename,
                                       dtype=cube_type,
                                       mode='r',
                                       shape=shape_of_spike_trig_avg)
    return cut_extracellular_data


def generate_average_over_selected_spikes_multiprocess(base_folder,
                                                       binary_data_filename,
                                                       number_of_channels_in_binary_file,
                                                       spike_times,
                                                       cube_type,
                                                       cut_time_points_around_spike=100,
                                                       num_of_points_for_baseline=None):
    channel_map = np.load(join(base_folder, 'channel_map.npy'))
    used_electrodes = np.squeeze(channel_map, axis=1)

    num_of_channels = used_electrodes.size
    num_of_points_in_spike_trig = cut_time_points_around_spike * 2

    data_raw = np.memmap(binary_data_filename, dtype=np.int16, mode='r')
    number_of_timepoints_in_raw = int(data_raw.shape[0] / number_of_channels_in_binary_file)
    raw_extracellular_data = np.reshape(data_raw, (number_of_channels_in_binary_file, number_of_timepoints_in_raw),
                                        order='F')

    unordered_data = Parallel(n_jobs=8)(delayed(_avg_of_eigth_of_selected_spikes)(i,
                                                                                  spike_times,
                                                                                  used_electrodes,
                                                                                  num_of_points_in_spike_trig,
                                                                                  cube_type,
                                                                                  raw_extracellular_data,
                                                                                  num_of_points_for_baseline)
                                        for i in np.arange(8))

    data = np.zeros((num_of_channels, cut_time_points_around_spike * 2))
    for partial_data, process_idx in unordered_data:
        data += partial_data
    data /= 8

    return data


def _avg_of_eigth_of_selected_spikes(process,
                                     spike_times,
                                     used_electrodes,
                                     num_of_points_in_spike_trig,
                                     cube_type,
                                     raw_extracellular_data,
                                     num_of_points_for_baseline):

    num_of_spikes = len(spike_times)
    num_of_channels = len(used_electrodes)

    partial_data = np.zeros((num_of_channels, num_of_points_in_spike_trig))

    num_of_spikes_in_this_process = np.ceil(num_of_spikes / 8)
    if process == 7:
        num_of_spikes_in_this_process = num_of_spikes - num_of_spikes_in_this_process * 7
    start_spike_index = int(num_of_spikes_in_this_process * process)

    num_spikes_done = 0
    for spike in np.arange(start_spike_index, num_of_spikes_in_this_process + start_spike_index):
        spike = int(spike)
        trigger_point = spike_times[spike]
        start_point = int(trigger_point - num_of_points_in_spike_trig / 2)
        if start_point < 0:
            break
        end_point = int(trigger_point + num_of_points_in_spike_trig / 2)
        if end_point > raw_extracellular_data.shape[1]:
            break
        temp = raw_extracellular_data[used_electrodes, start_point:end_point]
        if num_of_points_for_baseline is not None:
            baseline = np.mean(temp[:, [0, num_of_points_for_baseline]], 1)
            temp = (temp.T - baseline.T).T

        partial_data += temp.astype(cube_type)
        num_spikes_done += 1
        del temp

    partial_data /= num_spikes_done

    del baseline

    return partial_data, process


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


types = {0: 'Noise', 1: 'SS', 2: 'SS_Contaminated', 3: 'SS_Putative', 4: 'MUA', 5: 'Unspesified_1', 6: 'Unspecified_2',
         7: 'Unspecified_3'}


def key_from_type(type_to_find):
    for key, type in types.items():
        if type == type_to_find:
            return key

    return None


def symbol_from_type(type_to_find):
    if type_to_find == types[1]:
        symbol = '+'
    elif type_to_find == types[2]:
        symbol = 'star'
    elif type_to_find == types[3]:
        symbol = 'o'
    elif type_to_find == types[4]:
        symbol = 's'
    elif type_to_find == types[5]:
        symbol = 't'
    elif type_to_find == types[6]:
        symbol = 't2'
    else:
        symbol = 't3'

    return symbol


def get_templates_with_spike_indices_from_spike_info(spike_info):
    templates = np.unique(spike_info['template_after_cleaning'])
    templates_with_spike_indices = dict()
    for template in templates:
        spike_indices = np.argwhere(np.in1d(spike_info['template_after_cleaning'], template))
        templates_with_spike_indices[template] = spike_indices

    return templates_with_spike_indices


def get_templates_with_number_of_spikes_from_spike_info(spike_info):
    templates = np.unique(spike_info['template_after_cleaning'])
    templates_with_number_of_spikes = np.empty((len(templates), 2)).astype(np.int)
    for t in np.arange(len(templates)):
        spike_indices = np.argwhere(np.in1d(spike_info['template_after_cleaning'], templates[t]))
        templates_with_number_of_spikes[t] = [templates[t], len(spike_indices)]

    return templates_with_number_of_spikes


def find_templates_of_spikes(spike_info, spikes_indices):
    return np.unique(spike_info['template_after_cleaning'][spikes_indices])
