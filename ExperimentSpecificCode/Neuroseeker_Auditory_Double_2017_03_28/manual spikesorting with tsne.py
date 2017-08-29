

import numpy as np
from os.path import join
from BrainDataAnalysis import neuroseeker_specific_functions as nf

base_folder_v = r'F:\Data\George\Projects\SpikeSorting\Neuroseeker\\' + \
                r'Neuroseeker_2017_03_28_Anesthesia_Auditory_DoubleProbes\Vertical\Analysis\\' + \
                r'Experiment_2_T18_48_25_And_Experiment_3_T19_41_07\Kilosort'

base_folder_a = r'F:\Data\George\Projects\SpikeSorting\Neuroseeker\\' + \
                r'Neuroseeker_2017_03_28_Anesthesia_Auditory_DoubleProbes\Angled\Analysis\\' + \
                r'Experiment_2_T18_48_25_And_Experiment_3_T19_41_07\Kilosort'  # desktop

#base_folder_a = r'D:\Data\Brain\Neuroseeker_2017_03_28_Anesthesia_Auditory_DoubleProbes\Angled\Analysis\\' + \
#                 r'Experiment_2_T18_48_25_And_Experiment_3_T19_41_07\Kilosort'  # laptop

binary_v = r'Exp2and3_2017_03_28T18_48_25_Amp_S16_LP3p5KHz_mV.bin'
binary_a = r'Exp2and3_2017_03_28T18_48_25_Amp_S16_LP3p5KHz_mV.bin'
number_of_channels_in_binary_file = 1440


'''
template_markings_v = np.load(join(base_folder_v, 'template_marking.npy'))
spike_templates_v = np.load(join(base_folder_v, 'spike_templates.npy'))
spike_times_v = np.load(join(base_folder_v, 'spike_times.npy'))
templates_clean_index_v = np.argwhere(template_markings_v)
spikes_clean_index_v = np.squeeze(np.argwhere(np.in1d(spike_templates_v, templates_clean_index_v)))
spike_templates_clean_v = spike_templates_v[spikes_clean_index_v]
spike_times_clean_v = spike_times_v[spikes_clean_index_v]
'''

channel_map = np.squeeze(np.load(join(base_folder_a, 'channel_map.npy')))
template_markings_a = np.load(join(base_folder_a, 'template_marking.npy'))
spike_templates_a = np.load(join(base_folder_a, 'spike_templates.npy'))
spike_times_a = np.load(join(base_folder_a, 'spike_times.npy'))
templates_clean_index_a = np.argwhere(template_markings_a)
spikes_clean_index_a = np.squeeze(np.argwhere(np.in1d(spike_templates_a, templates_clean_index_a)))
spike_templates_clean_a = spike_templates_a[spikes_clean_index_a]
spike_times_clean_a = spike_times_a[spikes_clean_index_a]


binary_file_a = join(base_folder_a, binary_a)
raw_extracellular_data_a = nf.load_binary_amplifier_data(binary_file_a)




