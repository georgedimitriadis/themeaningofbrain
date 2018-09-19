
from os.path import join
import numpy as np
from spikesorting_tsne import spike_positioning_on_probe as pos

base_folder_v = r'F:\Neuroseeker\\' + \
                r'Neuroseeker_2017_03_28_Anesthesia_Auditory_DoubleProbes\Vertical\Analysis\\' + \
                r'Experiment_2_T18_48_25_And_Experiment_3_T19_41_07\Kilosort'

base_folder_a = r'F:\Neuroseeker\\' + \
                r'Neuroseeker_2017_03_28_Anesthesia_Auditory_DoubleProbes\Angled\Analysis\\' + \
                r'Experiment_2_T18_48_25_And_Experiment_3_T19_41_07\Kilosort'

binary_v = join(base_folder_v, r'Exp2and3_2017_03_28T18_48_25_Amp_S16_LP3p5KHz_mV.bin')
binary_a = join(base_folder_a, r'Exp2and3_2017_03_28T18_48_25_Amp_S16_LP3p5KHz_mV.bin')
number_of_channels_in_binary_file = 1440


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
    pos.generate_probe_positions_of_spikes(base_folder_v, binary_v, number_of_channels_in_binary_file,
                                              spikes_clean_index_v)

weighted_template_positions_v = pos.generate_probe_positions_of_templates(base_folder_v, threshold=0.1)

weighted_average_postions_a, spike_distance_on_probe_a = \
    pos.generate_probe_positions_of_spikes(base_folder_a, binary_a, number_of_channels_in_binary_file,
                                              spikes_clean_index_a)

weighted_template_positions_a = pos.generate_probe_positions_of_templates(base_folder_a, threshold=0.1)

position_mult = 2.25
probe_dimensions = [100, 8100]

brain_regions_v = {'AuD': 5208, 'Au1': 4748, 'AuV': 2890, 'TeA': 2248, 'Ectorhinal': 1933,
                   'Perirhinal': 1418, 'Entorhinal': 808}

spike_positions_v_cor = weighted_average_postions_v * position_mult
pos.view_spike_positions(spike_positions_v_cor, brain_regions=brain_regions_v, probe_dimensions=probe_dimensions)

template_positions_v_cor = weighted_template_positions_v * position_mult
pos.view_spike_positions(template_positions_v_cor, brain_regions=brain_regions_v, probe_dimensions=probe_dimensions,
                            labels_offset=250, font_size=12)

brain_regions_a = {'Au1 L1-3': 6396, 'Au1 L4': 5983, 'Au1 L5': 5783, 'Au1 L6': 5443, 'CA1': 5038,
                   'Dentate \nGyrus': 4353, 'MGN - PP': 2733, 'Substantia \nNigra': 1440}

spike_positions_a_cor = weighted_average_postions_a * position_mult
pos.view_spike_positions(spike_positions_a_cor,brain_regions=None, probe_dimensions=probe_dimensions)

template_positions_a_cor = weighted_template_positions_a * position_mult
pos.view_spike_positions(template_positions_a_cor,brain_regions=brain_regions_a, probe_dimensions=probe_dimensions,
                            labels_offset=160, font_size=12)





# ADDENTUM
# Plotting positions on probe acolor coded acording to the type of the unit (red: su, green: suc, blue: sup,
# black: mu, grey: unclassified

import matplotlib.pyplot as plt
template_markings_a_clean = template_markings_a[np.argwhere(template_markings_a > 0)]
fig = plt.figure()
ax = fig.add_axes([0.08, 0.05, 0.9, 0.9])
colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 0, 1], [0.4, 0.4, 0.4, 1]]
for i in np.arange(1, 6):
    indices = np.squeeze(np.argwhere(template_markings_a_clean == i))
    ax.scatter(template_positions_a_cor[indices, 0], template_positions_a_cor[indices, 1], s=5, c=colors[i-1])
ax.set_xlim(0, probe_dimensions[0])
ax.set_ylim(0, probe_dimensions[1])
ax.yaxis.set_ticks(np.arange(0, probe_dimensions[1], 100))
ax.tick_params(axis='y', direction='in', length=5, width=1, colors='b')
for region in brain_regions_a:
    ax.text(2, brain_regions_a[region] - 80, region, fontsize=20)
    ax.plot([0, probe_dimensions[0]], [brain_regions_a[region], brain_regions_a[region]], 'k--', linewidth=2)


template_markings_v_clean = template_markings_v[np.argwhere(template_markings_v > 0)]
fig = plt.figure()
ax = fig.add_axes([0.08, 0.05, 0.9, 0.9])
colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 0, 1], [0.4, 0.4, 0.4, 1]]
for i in np.arange(1, 6):
    indices = np.squeeze(np.argwhere(template_markings_v_clean == i))
    ax.scatter(template_positions_v_cor[indices, 0], template_positions_v_cor[indices, 1], s=5, c=colors[i-1])
ax.set_xlim(0, probe_dimensions[0])
ax.set_ylim(0, probe_dimensions[1])
ax.yaxis.set_ticks(np.arange(0, probe_dimensions[1], 100))
ax.tick_params(axis='y', direction='in', length=5, width=1, colors='b')
for region in brain_regions_v:
    ax.text(2, brain_regions_v[region] - 80, region, fontsize=20)
    ax.plot([0, probe_dimensions[0]], [brain_regions_v[region], brain_regions_v[region]], 'k--', linewidth=2)

