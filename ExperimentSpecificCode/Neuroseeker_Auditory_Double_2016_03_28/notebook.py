
from os.path import join
import numpy as np
from GUIs.Kilosort import cleaning_kilosorting_results as clean
from tsne_for_spikesort import spikes

base_folder_v = r'F:\Data\George\Projects\SpikeSorting\Neuroseeker\\' + \
                r'Neuroseeker_2017_03_28_Anesthesia_Auditory_DoubleProbes\Vertical\\' + \
                r'Experiment_2_T18_48_25_And_Experiment_3_T19_41_07\Kilosort'

base_folder_a = r'F:\Data\George\Projects\SpikeSorting\Neuroseeker\\' + \
                r'Neuroseeker_2017_03_28_Anesthesia_Auditory_DoubleProbes\Angled\\' + \
                r'Experiment_2_T18_48_25_And_Experiment_3_T19_41_07\Kilosort'

binary_v = r'Exp2and3_2017_03_28T18_48_25_Amp_S16_LP3p5KHz_mV.bin'
binary_a = r'Exp2and3_2017_03_28T18_48_25_Amp_S16_LP3p5KHz_mV.bin'
number_of_channels_in_binary_file = 1440

clean.cleanup_kilosorted_data(base_folder_v, number_of_channels_in_binary_file=number_of_channels_in_binary_file,
                              binary_data_filename=binary_v, generate_all=False,
                              overwrite_avg_spike_template_file=False, overwrite_template_marking_file=False, freq=20000,
                              time_points=100)


clean.cleanup_kilosorted_data(base_folder_a, number_of_channels_in_binary_file=number_of_channels_in_binary_file,
                              binary_data_filename=binary_a, generate_all=False,
                              overwrite_avg_spike_template_file=False, overwrite_template_marking_file=False, freq=20000,
                              time_points=100)


template_markings_v = np.load(join(base_folder_v, 'template_marking.npy'))
spike_templates_v = np.load(join(base_folder_v, 'spike_templates.npy'))
templates_clean_index_v = np.argwhere(template_markings_v)
spikes_clean_index_v = np.squeeze(np.argwhere(np.in1d(spike_templates_v, templates_clean_index_v)))
spike_templates_clean_v = spike_templates_v[spikes_clean_index_v]

template_markings_a = np.load(join(base_folder_a, 'template_marking.npy'))
spike_templates_a = np.load(join(base_folder_a, 'spike_templates.npy'))
templates_clean_index_a = np.argwhere(template_markings_a)
spikes_clean_index_a = np.squeeze(np.argwhere(np.in1d(spike_templates_a, templates_clean_index_a)))
spike_templates_clean_a = spike_templates_a[spikes_clean_index_a]


weighted_average_postions_v, spike_distance_on_probe_v = \
    spikes.generate_probe_positions_of_spikes(base_folder_v, binary_v, number_of_channels_in_binary_file,
                                              spikes_clean_index_v)


weighted_average_postions_a, spike_distance_on_probe_a = \
    spikes.generate_probe_positions_of_spikes(base_folder_a, binary_a, number_of_channels_in_binary_file,
                                              spikes_clean_index_a)


position_mult = 2.25
probe_dimensions = [100, 8100]

spike_positions_v_cor = weighted_average_postions_v * position_mult
spikes.view_spike_positions(spike_positions_v_cor,brain_regions=None, probe_dimensions=probe_dimensions)

spike_positions_a_cor = weighted_average_postions_a * position_mult
spikes.view_spike_positions(spike_positions_a_cor,brain_regions=None, probe_dimensions=probe_dimensions)