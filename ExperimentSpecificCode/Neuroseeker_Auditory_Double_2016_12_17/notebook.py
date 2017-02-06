

import os
import numpy as np
from TSne_Numba import spikes, gpu
from numba import cuda

base_folder_a = r'D:\Data\George\Projects\SpikeSorting\Neuroseeker\Neuroseeker_2016_12_17_Anesthesia_Auditory_DoubleProbes\AngledProbe\KilosortResults'
binary_data_filename_a = r'AngledProbe_BinaryAmplifier_12Regions_Penetration1_2016-12-17T19_02_12.bin'

base_folder_v = r'D:\Data\George\Projects\SpikeSorting\Neuroseeker\Neuroseeker_2016_12_17_Anesthesia_Auditory_DoubleProbes\VerticalProbe\KilosortResults\Penetration1_2016-12-17T19_02_15'
binary_data_filename_v = r'VerticalProbe_BinaryAmplifier_12Regions_Penetration1_2016-12-17T19_02_15.bin'

number_of_channels_in_binary_file = 1440

# ------- LOAD STUFF ---------------------------------------------------------------------------------------------------
template_marking_a = np.load(os.path.join(base_folder_a, 'template_marking.npy'))
spike_templates_a = np.load(os.path.join(base_folder_a, 'spike_templates.npy'))
template_features_a = np.load(os.path.join(base_folder_a, 'template_features.npy'))
template_features_ind_a = np.load(os.path.join(base_folder_a, 'template_feature_ind.npy'))

template_marking_v = np.load(os.path.join(base_folder_v, 'template_marking.npy'))
spike_templates_v = np.load(os.path.join(base_folder_v, 'spike_templates.npy'))
template_features_v = np.load(os.path.join(base_folder_v, 'template_features.npy'))
template_features_ind_v = np.load(os.path.join(base_folder_v, 'template_feature_ind.npy'))
# ----------------------------------------------------------------------------------------------------------------------

# ------- TEMPLATES AND SPIKES REMAINING AFTER MANUALL CLEANING --------------------------------------------------------
templates_clean_index_a = np.argwhere(template_marking_a)
spikes_clean_index_a = np.squeeze(np.argwhere(np.in1d(spike_templates_a, templates_clean_index_a)))

templates_clean_index_v = np.argwhere(template_marking_v)
spikes_clean_index_v = np.squeeze(np.argwhere(np.in1d(spike_templates_v, templates_clean_index_v)))
# ----------------------------------------------------------------------------------------------------------------------

# ------- GENERATING A SPARSE SPIKE FEATURES MATRIX FOR T-SNE USING ONLY THE GOOD SPIKES AND TEMPLATES -----------------
template_features_sparse_clean_a = np.zeros((spikes_clean_index_a.size, templates_clean_index_a.size))
for spike in spikes_clean_index_a:
    cluster = spike_templates_a[spike][0]
    indices = template_features_ind_a[cluster, :]
    for i in np.arange(len(indices)):
        template_features_sparse_clean_a[np.argwhere(spikes_clean_index_a == spike),
                                         np.argwhere(templates_clean_index_a == indices[i])] = template_features_a[spike, i]
np.save(os.path.join(base_folder_a, 'data_to_tsne_'+str(template_features_sparse_clean_a.shape)+'.npy'),
        template_features_sparse_clean_a)


template_features_sparse_clean_v = np.zeros((spikes_clean_index_v.size, templates_clean_index_v.size))
for spike in spikes_clean_index_v:
    cluster = spike_templates_v[spike][0]
    indices = template_features_ind_v[cluster, :]
    for i in np.arange(len(indices)):
        template_features_sparse_clean_v[np.argwhere(spikes_clean_index_v == spike),
                                         np.argwhere(templates_clean_index_v == indices[i])] = template_features_v[spike, i]
np.save(os.path.join(base_folder_v, 'data_to_tsne_'+str(template_features_sparse_clean_v.shape)+'.npy'),
        template_features_sparse_clean_v)

template_features_sparse_clean_a = np.load(os.path.join(base_folder_a, 'data_to_tsne_(110728, 292).npy'))
# ----------------------------------------------------------------------------------------------------------------------

# --------- CREATE THE AVERAGE PROBE POSITION OF EACH SPIKE AND ITS DISTANCE TO THE PROBE'S 0,0 ------------------------
std = 2
position_mult = 2.25
probe_dimensions = [100, 8100]

spike_positions_a, spike_distances_on_probe_a = \
    spikes.generate_probe_positions_of_spikes(base_folder=base_folder_a,
                                              binary_data_filename=binary_data_filename_a,
                                              number_of_channels_in_binary_file=
                                              number_of_channels_in_binary_file,
                                              used_spikes_indices=spikes_clean_index_a, threshold=2)

spike_positions_a_cor = spike_positions_a * position_mult
np.save(os.path.join(base_folder_a, 'spike_positions_on_probe_std'+str(std)+'.npy'), spike_positions_a_cor)

spike_positions_a_cor = np.load(os.path.join(base_folder_a, 'spike_positions_on_probe_std'+str(std)+'.npy'))
spike_distances_on_probe_a = np.sqrt(np.power(spike_positions_a_cor[:, 0], 2) + np.power(spike_positions_a_cor[:, 1], 2))


brain_regions_a = {'Layer 1/2/3': 3330*position_mult, 'Layer 4': 3090*position_mult, 'Layer 5': 3030*position_mult,
                   'Layer 6': 2800*position_mult,
                   'CA1': 2580*position_mult, ' ': 2220*position_mult, 'CA3': 2000*position_mult,
                   'Thalamus': 1760*position_mult, 'Medial Geniculate Nucleus': 1470*position_mult,
                   'Rest of Thalamus': 840*position_mult}
spikes.view_spike_positions(spike_positions_a_cor, brain_regions_a, probe_dimensions)


spike_positions_v, spike_distances_on_probe_v = \
    spikes.generate_probe_positions_of_spikes(base_folder=base_folder_v,
                                              binary_data_filename=binary_data_filename_v,
                                              number_of_channels_in_binary_file=
                                              number_of_channels_in_binary_file,
                                              used_spikes_indices=spikes_clean_index_v, threshold=2)

spike_positions_v_cor = spike_positions_v * position_mult
np.save(os.path.join(base_folder_v, 'spike_positions_on_probe_std'+str(std)+'.npy'), spike_positions_v_cor)
brain_regions_v = {'AuD L1/2/3': 2960*position_mult, 'Au1 L4': 2630*position_mult, 'Au1 L5': 2380*position_mult,
                   'AuV L5': 1640*position_mult,
                   'Temporal Association L5': 1340*position_mult, 'Ectorhinal': 1100*position_mult,
                   'Perirhinal': 800*position_mult, 'Dorsolateral Entorhinal': 500*position_mult}
spikes.view_spike_positions(spike_positions_v_cor, brain_regions_v, probe_dimensions)
# ----------------------------------------------------------------------------------------------------------------------

# ---------- CREATE ARRAY OF LISTS KEEPING THE SPIKES THAT ARE CLOSE TO EACH SPIKE ON PROBE ---------------------------
spike_indices_sorted_by_probe_distance_a = np.array([b[0] for b in sorted(enumerate(spike_distances_on_probe_a),
                                                                          key=lambda dist: dist[1])])
spike_distances_on_probe_sorted_a = np.array([b[1] for b in sorted(enumerate(spike_distances_on_probe_a),
                                                                   key=lambda dist: dist[1])])

spike_indices_sorted_by_probe_distance_v = np.array([b[0] for b in sorted(enumerate(spike_distances_on_probe_v),
                                                                          key=lambda dist: dist[1])])
spike_distances_on_probe_sorted_v = np.array([b[1] for b in sorted(enumerate(spike_distances_on_probe_v),
                                                                   key=lambda dist: dist[1])])


distance_threshold = 100
max_elements_in_matrix = 2e9

indices_of_first_arrays, indices_of_second_arrays = \
    spikes.define_all_spike_spike_matrices_for_distance_calc(spike_distances_on_probe_sorted_a,
                                                             max_elements_in_matrix=max_elements_in_matrix,
                                                             probe_distance_threshold=distance_threshold)
np.save(os.path.join(base_folder_a, r'first_second_matrices_indices_for_distance_cal.npy'), (indices_of_first_arrays,
                                                                                             indices_of_second_arrays))

(indices_of_first_arrays, indices_of_second_arrays) = np.load(os.path.join(base_folder_a,
                                                              r'first_second_matrices_indices_for_distance_cal.npy'))



# ----------- GET THE DISTANCES ON THE KILOSORT TEMPLATE FEATURE SPACE -------------------------------------------------
template_features_sparce_clean_probesorted = template_features_sparse_clean_a[spike_indices_sorted_by_probe_distance_a,
                                                                              :]

fas_indices_path = os.path.join(base_folder_a, r'first_second_matrices_indices_for_distance_cal.npy')
selected_sorted_indices, selected_sorted_distances = \
    gpu.calculate_knn_distances_close_on_probe(template_features_sorted=template_features_sparce_clean_probesorted,
                                               indices_of_first_and_second_matrices=np.load(fas_indices_path))
