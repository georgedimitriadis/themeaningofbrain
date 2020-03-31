
from os.path import join
import numpy as np
import BrainDataAnalysis.neuroseeker_specific_functions as ns_funcs
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2019_06_25_AK_47p2 import constants as const
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight.Common_functions \
    import events_sync_funcs as sync_funcs
from BrainDataAnalysis.Statistics import binning
from BrainDataAnalysis.LFP import emd
from mne.time_frequency import multitaper as mt
import BrainDataAnalysis.neuroseeker_specific_functions as ns_funcs

import pandas as pd
import matplotlib.pyplot as plt

import common_data_transforms as cdt
import sequence_viewer as sv
import slider as sl

from scipy.signal import hilbert

import pickle


# -------------------------------------------------
# <editor-fold desc="LOAD FOLDERS AND DATA">
date_folder = 8

data_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Data')

kilosort_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Analysis',
                       'Kilosort')

events_folder = join(data_folder, "events")

analysis_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Analysis')
results_folder = join(analysis_folder, 'Results')
spike_lfp_folder = join(results_folder, 'SpikeLfpCorrelations', 'RandomSpikesAlongRecording')
spike_lfp_images_folder = join(spike_lfp_folder, 'SingleNeuronImages')

template_info = pd.read_pickle(join(kilosort_folder, 'template_info.df'))

spike_info = pd.read_pickle(join(kilosort_folder, 'spike_info_after_cleaning.df'))

lfps_file = join(data_folder, 'Amplifier_LFPs_Downsampled_x4.bin')
lfps = ns_funcs.load_binary_amplifier_data((lfps_file), number_of_channels=const.NUMBER_OF_LFP_CHANNELS_IN_BINARY_FILE)

imfs_file = join(analysis_folder, 'Lfp', 'EMD', 'imfs.bin')
imfs = emd.load_memmaped_imfs(imfs_file, const.NUMBER_OF_IMFS, const.NUMBER_OF_LFP_CHANNELS_IN_BINARY_FILE)

num_of_spikes = 4000
half_time_window = 2000

# </editor-fold>


# -------------------------------------------------
# <editor-fold desc="CREATE TIMES OF RANDOM SPIKES FOR ALL HIGH FIRING NEURONS">

neurons_with_high_frs = template_info[template_info['number of spikes'] >= num_of_spikes].index
spikes = np.array([np.random.choice(n, num_of_spikes, replace=False)
                   for n in template_info.iloc[neurons_with_high_frs]['spikes in template'].values])

spike_times = []
for i in np.arange(len(spikes)):
    spike_times.append(spike_info[np.isin(spike_info['original_index'], spikes[i])]['times'].values)
    print(i, len(spike_times[i]))

spike_times = np.array(spike_times)

np.save(join(spike_lfp_folder, 'random_spike_indices.npy'), spikes)
np.save(join(spike_lfp_folder, 'random_spike_times.npy'), spike_times)

# </editor-fold>


# -------------------------------------------------
# <editor-fold desc="LOAD TIMES OF RANDOM SPIKES FOR ALL HIGH FIRING NEURONS">

neurons_with_high_frs = template_info[template_info['number of spikes'] >= num_of_spikes].index
spikes = np.load(join(spike_lfp_folder, 'random_spike_indices.npy'))
spike_times = np.load(join(spike_lfp_folder, 'random_spike_times.npy'))

# </editor-fold>


# -------------------------------------------------
# <editor-fold desc="GENERATE SPIKE TRIGGERED LFPS">

spike_times_subsampled = (spike_times / const.LFP_DOWNSAMPLE_FACTOR).astype(np.int)

spike_triggered_lfp_avg_shape = (len(spike_times), const.NUMBER_OF_LFP_CHANNELS_IN_BINARY_FILE, 2 * half_time_window)
spike_triggered_lfps_avg = np.memmap(join(spike_lfp_folder, 'spike_triggered_lfps_avg.bin'), np.float32, 'w+',
                                     shape=spike_triggered_lfp_avg_shape)
spike_triggered_lfps_std = np.memmap(join(spike_lfp_folder, 'spike_triggered_lfps_std.bin'), np.float32, 'w+',
                                     shape=spike_triggered_lfp_avg_shape)


for neuron in np.arange(len(spike_times)):
    neuron_spike_times = spike_times_subsampled[neuron]
    for channel in np.arange(const.NUMBER_OF_LFP_CHANNELS_IN_BINARY_FILE):
        channel_lfp = lfps[channel, :]

        spike_triggered_channel_lfp = np.array(
            [channel_lfp[x-half_time_window:x+half_time_window] for x in neuron_spike_times if x-half_time_window >= 0
             and x + half_time_window <= lfps.shape[1]])

        spike_triggered_lfps_avg[neuron, channel, :] = spike_triggered_channel_lfp.mean(axis=0)
        spike_triggered_lfps_std[neuron, channel, :] = spike_triggered_channel_lfp.std(axis=0)

    print('Done neuron {}'.format(neuron))


# </editor-fold>


# -------------------------------------------------
# <editor-fold desc="VISUALISE SPIKE TRIGGERED LFPS">

spike_times_subsampled = (spike_times / const.LFP_DOWNSAMPLE_FACTOR).astype(np.int)

spike_triggered_lfp_avg_shape = (len(spike_times), const.NUMBER_OF_LFP_CHANNELS_IN_BINARY_FILE, 2 * half_time_window)
spike_triggered_lfps_avg = np.memmap(join(spike_lfp_folder, 'spike_triggered_lfps_avg.bin'), np.float32, 'r',
                                     shape=spike_triggered_lfp_avg_shape)
spike_triggered_lfps_std = np.memmap(join(spike_lfp_folder, 'spike_triggered_lfps_std.bin'), np.float32, 'r',
                                     shape=spike_triggered_lfp_avg_shape)


positions_on_probe = template_info['position Y'].iloc[neurons_with_high_frs].values
neuron_indices_pos_sorted = np.argsort(positions_on_probe)[::-1]

# Save all the pics
for n in neuron_indices_pos_sorted:
    plt.figure(figsize=(18, 12), dpi=80)
    spike_triggered_lfps_avg_spaced = cdt.space_data(spike_triggered_lfps_avg[n, :, :], 20)
    plt.plot(spike_triggered_lfps_avg_spaced.T)
    neuron_template_index = neurons_with_high_frs[n]
    position = int(template_info['position Y'].iloc[neuron_template_index]) * const.POSITION_MULT
    template = template_info['template number'].iloc[neuron_template_index]
    plt.title('Neuron : {}, Index_on_all_neurons = {}, Index_on_fast_neurons = {}, Y Position = {}'.format(template,
                                                                                                           neuron_template_index,
                                                                                                           n,
                                                                                                           position))
    plt.savefig(join(spike_lfp_images_folder, 'position_{}_index_{}_template_{}.png'.format(position,
                                                                                            n,
                                                                                            template)))
    plt.close()
# ------------------


# Show live all the pics
def draw_lfps(i):
    plt.clf()
    n = neuron_indices_pos_sorted[i]
    spike_triggered_lfps_avg_spaced = cdt.space_data(spike_triggered_lfps_avg[n, :, :], 20)
    plt.plot(spike_triggered_lfps_avg_spaced.T)
    neuron_template_index = neurons_with_high_frs[n]
    position = int(template_info['position Y'].iloc[neuron_template_index]) * const.POSITION_MULT
    template = template_info['template number'].iloc[neuron_template_index]
    plt.title('Neuron : {}, Index_on_all_neurons = {}, Index_on_fast_neurons = {}, Y Position = {}'
              .format(template, neuron_template_index, n, position))

input = 0
output = None
sl.connect_repl_var(globals(), 'input', 'output', 'draw_lfps', slider_limits=[0, len(neurons_with_high_frs)])
# ---------------------

# </editor-fold>
