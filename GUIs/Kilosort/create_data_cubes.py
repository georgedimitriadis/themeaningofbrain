

import numpy as np
from os.path import join
from joblib import Parallel, delayed
from BrainDataAnalysis import neuroseeker_specific_functions as nf


def generate_average_over_spikes_per_template(base_folder,
                                              binary_data_filename,
                                              number_of_channels_in_binary_file,
                                              cut_time_points_around_spike=100):
    channel_map = np.load(join(base_folder, 'channel_map.npy'))
    active_channel_map = np.squeeze(channel_map, axis=1)

    spike_templates = np.load(join(base_folder, r'spike_templates.npy'))
    template_feature_ind = np.load(join(base_folder, 'template_feature_ind.npy'))
    number_of_templates = template_feature_ind.shape[0]

    spike_times = np.squeeze(np.load(join(base_folder, 'spike_times.npy')).astype(np.int))

    num_of_channels = active_channel_map.size

    data_raw = np.memmap(binary_data_filename, dtype=np.int16, mode='r')
    number_of_timepoints_in_raw = int(data_raw.shape[0] / number_of_channels_in_binary_file)
    data_raw_matrix = np.reshape(data_raw, (number_of_channels_in_binary_file, number_of_timepoints_in_raw),
                                 order='F')

    number_of_timepoints_in_raw = data_raw_matrix.shape[1]
    data = np.zeros((number_of_templates, num_of_channels, cut_time_points_around_spike * 2))

    for template in np.arange(number_of_templates):
        spike_indices_in_template = np.argwhere(np.in1d(spike_templates, template))
        spike_times_in_template = np.squeeze(spike_times[spike_indices_in_template])
        num_of_spikes_in_template = spike_indices_in_template.shape[0]
        y = np.zeros((num_of_channels, cut_time_points_around_spike * 2))
        if num_of_spikes_in_template != 0:
            # remove any spikes that don't have enough time points
            too_early_spikes = np.squeeze(np.argwhere(spike_times_in_template < cut_time_points_around_spike), axis=1)
            too_late_spikes = np.squeeze(np.argwhere(spike_times_in_template > number_of_timepoints_in_raw - cut_time_points_around_spike), axis=1)
            out_of_time_spikes = np.concatenate((too_early_spikes, too_late_spikes))
            spike_indices_in_template = np.delete(spike_indices_in_template, out_of_time_spikes)
            num_of_spikes_in_template = spike_indices_in_template.shape[0]

            for spike_in_template in spike_indices_in_template:
                y = y + data_raw_matrix[active_channel_map,
                                        spike_times[spike_in_template] - cut_time_points_around_spike:
                                        spike_times[spike_in_template] + cut_time_points_around_spike]

            y = y / num_of_spikes_in_template
        data[template, :, :] = y
        del y
        print('Added template ' + str(template) + ' with ' + str(num_of_spikes_in_template) + ' spikes')

    np.save(join(base_folder, 'avg_spike_template2.npy'), data)


def generate_average_over_spikes_per_template_multiprocess(base_folder,
                                                           binary_data_filename,
                                                           number_of_channels_in_binary_file,
                                                           cut_time_points_around_spike=100):
    channel_map = np.load(join(base_folder, 'channel_map.npy'))
    if len(channel_map.shape) > 1:
        active_channel_map = np.squeeze(channel_map, axis=1)
    else:
        active_channel_map = channel_map

    spike_templates = np.load(join(base_folder, r'spike_templates.npy'))
    #template_feature_ind = np.load(join(base_folder, 'template_feature_ind.npy'))
    #number_of_templates = template_feature_ind.shape[0]
    templates = np.load(join(base_folder, 'templates.npy'))
    number_of_templates = templates.shape[0]
    spike_times = np.squeeze(np.load(join(base_folder, 'spike_times.npy')).astype(np.int))

    num_of_channels = active_channel_map.size

    data_raw = np.memmap(binary_data_filename, dtype=np.int16, mode='r')
    number_of_timepoints_in_raw = int(data_raw.shape[0] / number_of_channels_in_binary_file)
    data_raw_matrix = np.reshape(data_raw, (number_of_channels_in_binary_file, number_of_timepoints_in_raw),
                                 order='F')

    number_of_timepoints_in_raw = data_raw_matrix.shape[1]

    unordered_data = Parallel(n_jobs=8)(delayed(_avg_of_single_template)(i,
                                                                         spike_times,
                                                                         spike_templates,
                                                                         num_of_channels,
                                                                         cut_time_points_around_spike,
                                                                         number_of_timepoints_in_raw,
                                                                         data_raw_matrix,
                                                                         active_channel_map)
                                        for i in np.arange(number_of_templates))
    data = np.zeros((number_of_templates, num_of_channels, cut_time_points_around_spike * 2))
    for idx, info in unordered_data:
        data[idx, ...] = info

    np.save(join(base_folder, 'avg_spike_template.npy'), data)


def _avg_of_single_template(template,
                            spike_times,
                            spike_templates,
                            num_of_channels,
                            cut_time_points_around_spike,
                            number_of_timepoints_in_raw,
                            data_raw_matrix,
                            active_channel_map):

    spike_indices_in_template = np.argwhere(np.in1d(spike_templates, template))
    spike_times_in_template = np.squeeze(spike_times[spike_indices_in_template])
    num_of_spikes_in_template = spike_indices_in_template.shape[0]
    y = np.zeros((num_of_channels, cut_time_points_around_spike * 2))
    if num_of_spikes_in_template != 0:
        # remove any spikes that don't have enough time points
        too_early_spikes = np.squeeze(np.argwhere(spike_times_in_template < cut_time_points_around_spike), axis=1)
        too_late_spikes = np.squeeze(np.argwhere(spike_times_in_template > number_of_timepoints_in_raw - cut_time_points_around_spike), axis=1)
        out_of_time_spikes = np.concatenate((too_early_spikes, too_late_spikes))
        spike_indices_in_template = np.delete(spike_indices_in_template, out_of_time_spikes)
        num_of_spikes_in_template = spike_indices_in_template.shape[0]

        for spike_in_template in spike_indices_in_template:
            y = y + data_raw_matrix[active_channel_map,
                                    spike_times[spike_in_template] - cut_time_points_around_spike:
                                    spike_times[spike_in_template] + cut_time_points_around_spike]

        y = y / num_of_spikes_in_template
        print('Added template ' + str(template) + ' with ' + str(num_of_spikes_in_template) + ' spikes')
    return template, y





