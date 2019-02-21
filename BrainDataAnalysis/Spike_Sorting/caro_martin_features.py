

import numpy as np
from os.path import join


# -------------------------------
# Load test data for development
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2018_04_AK_33p1 import constants as const
import BrainDataAnalysis.neuroseeker_specific_functions as ns_funcs
from spikesorting_tsne import preprocessing_kilosort_results as preproc_kilo, constants as ct, visualization as vis


date = 8
kilosort_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date],
                       'Analysis', 'Kilosort')
binary_data_filename = join(const.base_save_folder, const.rat_folder, const.date_folders[date],
                            'Data', 'Amplifier_APs.bin')
tsne_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date],
                   'Analysis', 'Tsne', 'Using_PCs_from_12_Kilosort_electrodes')

number_of_spikes_in_large_mua_templates = 10000
large_mua_templates = preproc_kilo.find_large_mua_templates(kilosort_folder, number_of_spikes_in_large_mua_templates)


mua_template_to_look = 2
mua_template = large_mua_templates[mua_template_to_look]

template_folder = join(tsne_folder, 'template_{}'.format(mua_template))
data_for_tsne = np.load(join(template_folder, 'data_to_tsne_(50491, 36).npy'))

spike_times = np.squeeze(np.load(join(kilosort_folder, 'spike_times.npy')))
spikes = np.squeeze(np.load(join(template_folder, 'indices_of_spikes_used.npy')))
spike_times = spike_times[spikes]

pc_features = np.load(join(kilosort_folder, 'pc_features.npy'))
pc_feature_ind = np.load(join(kilosort_folder, 'pc_feature_ind.npy'))

avg_spike_template = np.load(join(kilosort_folder, 'avg_spike_template.npy'))
large_channels = pc_feature_ind[mua_template, :]
mua_template_data = avg_spike_template[mua_template, :, :]

raw_data = ns_funcs.load_binary_amplifier_data(binary_data_filename,
                                               number_of_channels=const.NUMBER_OF_AP_CHANNELS_IN_BINARY_FILE)


large_channels_full = np.arange(int(large_channels[0] - 30), int(large_channels[0] + 30))


spike_data = []
i = 0
for spike_time in spike_times:
    spike_data.append(raw_data[large_channels_full, int(spike_time-20):int(spike_time+20)])

spike_data = np.array(spike_data)

# -------------------------------


class Point:
    def __init__(self, time, value):
        self.time = time
        self.value = value


class SpikePointNumbers:
    def __init__(self, single_channel_single_spike_data):
        self.data = single_channel_single_spike_data





class CaroMartinFeatureSet:
    def __init__(self, spike_data, std_threshold=6):
        self.data = spike_data

        self.std_threshold = std_threshold
        self.selected_channels = []
        self.find_relevant_channels(self)

        self.P1 = None
        self.P2 = None

    def find_relevant_channels(self):
        mean_spike_data = np.mean(self.spike_data, axis=0)
        for channel in mean_spike_data:
            if np.abs(np.min(mean_spike_data[channel, :]) / np.std(mean_spike_data[channel, :10])) > self.std_threshold:
                self.selected_channels.append(channel)

