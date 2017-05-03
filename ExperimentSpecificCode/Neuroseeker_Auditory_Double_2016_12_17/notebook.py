

import os
import numpy as np
from tsne_for_spikesort import spikes, t_sne, gpu

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

# ---------- CREATE ARRAY OF LISTS KEEPING THE SPIKES THAT ARE CLOSE TO EACH SPIKE ON PROBE ----------------------------
# ---------- DOES NOT WORK. NOT ENOUGH MEMORY --------------------------------------------------------------------------
spike_indices_sorted_by_probe_distance_a = np.array([b[0] for b in sorted(enumerate(spike_distances_on_probe_a),
                                                                          key=lambda dist: dist[1])])
spike_distances_on_probe_sorted_a = np.array([b[1] for b in sorted(enumerate(spike_distances_on_probe_a),
                                                                   key=lambda dist: dist[1])])

spike_indices_sorted_by_probe_distance_v = np.array([b[0] for b in sorted(enumerate(spike_distances_on_probe_v),
                                                                          key=lambda dist: dist[1])])
spike_distances_on_probe_sorted_v = np.array([b[1] for b in sorted(enumerate(spike_distances_on_probe_v),
                                                                   key=lambda dist: dist[1])])

# ---------- CALCULATE THE INDICES OF THE 1st AND 2nd MATRICES USING THE PROBE DISTANCE SORTED INDICES -----------------
distance_threshold = 100
max_elements_in_matrix = 2e9

indices_of_first_arrays, indices_of_second_arrays = \
    spikes.define_all_spike_spike_matrices_for_distance_calc(spike_distances_on_probe_sorted_a,
                                                             max_elements_in_matrix=max_elements_in_matrix,
                                                             probe_distance_threshold=distance_threshold)
indices_of_first_arrays = np.array(indices_of_first_arrays)
indices_of_second_arrays = np.array(indices_of_second_arrays)
np.save(os.path.join(base_folder_a, r'first_second_matrices_indices_for_distance_cal.npy'), (indices_of_first_arrays,
                                                                                             indices_of_second_arrays))

# --------- CALCULATE THE CLOSEST DISTANCE PAIRS AND THEIR DISTANCES ON THE GPU-----------------------------------------
template_features_sparce_clean_probesorted = template_features_sparse_clean_a[spike_indices_sorted_by_probe_distance_a,
                                                                              :]
selected_sorted_indices, selected_sorted_distances = \
    gpu.calculate_knn_distances_close_on_probe(template_features_sorted=template_features_sparce_clean_probesorted,
                                               indices_of_first_and_second_matrices=(indices_of_first_arrays,
                                                                                     indices_of_second_arrays))
np.save(os.path.join(base_folder_a, r'selected_hdspace_sorted_distances.npy'), selected_sorted_distances)
np.save(os.path.join(base_folder_a, r'selected_hdspace_sorted_indices.npy'), selected_sorted_indices)

# --------- CALCULATE THE GAUSSIAN PERPLEXITY OF THE HIGH DIMENSIONAL SPACE --------------------------------------------
perplexity = 100
indices_p, values_p = t_sne._compute_gaussian_perplexity(selected_sorted_indices, selected_sorted_distances,
                                                         perplexity=perplexity)
np.save(os.path.join(base_folder_a, r'values_p.npy'), values_p)
np.save(os.path.join(base_folder_a, r'indices_p.npy'), indices_p)

# ----------------------------------------------------------------------------------------------------------------------
# -----------------------ONE OF PIPELINE--------------------------------------------------------------------------------

# Probe Angled
import os
import numpy as np

perplexity = 100

# define the path for all files and filename of raw data----------------------------------------------------------------
base_folder = r'D:\Data\George\Projects\SpikeSorting\Neuroseeker\\' + \
                  r'Neuroseeker_2016_12_17_Anesthesia_Auditory_DoubleProbes\AngledProbe\KilosortResults'


# load the spikes x features matrix that is cleaned (no bad templates)--------------------------------------------------
template_features_sparse_clean = np.load(os.path.join(base_folder, 'data_to_tsne_(110728, 292).npy'))

# get the distance of each spike on the probe---------------------------------------------------------------------------
std = 2
spike_positions_cor = np.load(os.path.join(base_folder, 'spike_positions_on_probe_std'+str(std)+'.npy'))
spike_distances_on_probe = np.sqrt(np.power(spike_positions_cor[:, 0], 2) + np.power(spike_positions_cor[:, 1], 2))

# sort the spike distances and their original indices according to probe distance---------------------------------------
spike_indices_sorted_by_probe_distance = np.array([b[0] for b in sorted(enumerate(spike_distances_on_probe),
                                                                        key=lambda dist: dist[1])])
spike_distances_on_probe_sorted = np.array([b[1] for b in sorted(enumerate(spike_distances_on_probe),
                                                                 key=lambda dist: dist[1])])

# use the sorted indices to sort the spike x features matrix according to the probe distance of each spike--------------
template_features_sparce_clean_probesorted = template_features_sparse_clean[spike_indices_sorted_by_probe_distance, :]

# load the saved gpu calculated and sorted distances and indices -------------------------------------------------------
(indices_of_first_arrays, indices_of_second_arrays) = np.load(os.path.join(base_folder,
                                                              r'first_second_matrices_indices_for_distance_cal.npy'))

selected_sorted_indices = np.load(os.path.join(base_folder, 'selected_hdspace_sorted_indices.npy'))
selected_sorted_distances = np.load(os.path.join(base_folder, 'selected_hdspace_sorted_distances.npy'))

# Get the gaussian perplexities ----------------------------------------------------------------------------------------

indices_p = np.load(os.path.join(base_folder, r'indices_p.npy'))
values_p = np.load(os.path.join(base_folder, r'values_p.npy'))


# -------- creating fewer spikes matrices for debugging c++ and numba --------------
import numpy as np
import os
from tsne_for_spikesort import spikes, t_sne
from t_sne_bhcuda import bhtsne_cuda as tsne_exe
import  matplotlib.pyplot as plt

def get_some_data(number_of_spikes):
    number_of_channels_in_binary_file = 1440
    base_folder = r'D:\Data\George\Projects\SpikeSorting\Neuroseeker\Neuroseeker_2016_12_17_Anesthesia_Auditory_DoubleProbes\AngledProbe\KilosortResults'
    binary_data_filename = r'AngledProbe_BinaryAmplifier_12Regions_Penetration1_2016-12-17T19_02_12.bin'

    template_marking = np.load(os.path.join(base_folder, 'template_marking.npy'))
    spike_templates = np.load(os.path.join(base_folder, 'spike_templates.npy'))
    template_features = np.load(os.path.join(base_folder, 'template_features.npy'))
    template_features_ind = np.load(os.path.join(base_folder, 'template_feature_ind.npy'))

    templates_clean_index = np.argwhere(template_marking)
    spikes_clean_index = np.squeeze(np.argwhere(np.in1d(spike_templates, templates_clean_index)))


    spikes_clean_index = spikes_clean_index[:number_of_spikes]
    templates_clean_index = np.unique(spike_templates[spikes_clean_index])

    template_features_sparse_clean = np.zeros((spikes_clean_index.size, templates_clean_index.size))
    for spike in spikes_clean_index:
        cluster = spike_templates[spike][0]
        indices = template_features_ind[cluster, :]
        for i in np.arange(len(indices)):
            template_features_sparse_clean[np.argwhere(spikes_clean_index == spike),
                                             np.argwhere(templates_clean_index == indices[i])] = template_features[spike, i]



    std = 2
    position_mult = 2.25
    probe_dimensions = [100, 8100]

    spike_positions, spike_distances_on_probe = \
        spikes.generate_probe_positions_of_spikes(base_folder=base_folder,
                                                  binary_data_filename=binary_data_filename,
                                                  number_of_channels_in_binary_file=
                                                  number_of_channels_in_binary_file,
                                                  used_spikes_indices=spikes_clean_index, threshold=2)

    spike_positions_cor = spike_positions * position_mult
    spike_distances_on_probe = np.sqrt(np.power(spike_positions_cor[:, 0], 2) + np.power(spike_positions_cor[:, 1], 2))




    spike_indices_sorted_by_probe_distance = np.array([b[0] for b in sorted(enumerate(spike_distances_on_probe),
                                                                              key=lambda dist: dist[1])])
    spike_distances_on_probe_sorted = np.array([b[1] for b in sorted(enumerate(spike_distances_on_probe),
                                                                       key=lambda dist: dist[1])])



    distance_threshold = 100
    max_elements_in_matrix = 2e9

    indices_of_first_arrays, indices_of_second_arrays = \
        spikes.define_all_spike_spike_matrices_for_distance_calc(spike_distances_on_probe_sorted,
                                                                 max_elements_in_matrix=max_elements_in_matrix,
                                                                 probe_distance_threshold=distance_threshold)
    indices_of_first_arrays = np.array(indices_of_first_arrays)
    indices_of_second_arrays = np.array(indices_of_second_arrays)

    template_features_sparce_clean_probesorted = template_features_sparse_clean[spike_indices_sorted_by_probe_distance, :]

    return indices_of_first_arrays, indices_of_second_arrays, template_features_sparce_clean_probesorted


number_of_spikes = 50
indices_of_first_arrays, indices_of_second_arrays, template_features_sparce_clean_probesorted = \
    get_some_data(number_of_spikes=number_of_spikes)


# Save and Load----------------
np.save(os.path.join(base_folder, 'template_features_sparce_clean_probesorted_'+str(number_of_spikes)+'.npy'),
        template_features_sparce_clean_probesorted)
np.save(os.path.join(base_folder, r'first_second_matrices_indices_for_distance_cal_'+str(number_of_spikes)+'.npy'),
        (indices_of_first_arrays, indices_of_second_arrays))

template_features_sparce_clean_probesorted = np.load(os.path.join(base_folder,
                                                                  'template_features_sparce_clean_probesorted_' +
                                                                  str(number_of_spikes)+'.npy'))
(indices_of_first_arrays, indices_of_second_arrays) = np.load(os.path.join(base_folder,
                                                                           r'first_second_matrices_indices_for_distance_cal_'+
                                                                           str(number_of_spikes)+'.npy'))
#-------------------------------


perplexity = 20
y = t_sne.run(data=template_features_sparce_clean_probesorted,
              indices_of_first_and_second_matrices=(indices_of_first_arrays, indices_of_second_arrays),
              iters=100, perplexity=perplexity, eta=200, num_dims=2, verbose=3)

plt.scatter(y[:, 0], y[:, 1])

tsne_exe.save_data_for_tsne(template_features_sparce_clean_probesorted,
                            r'E:\George\SourceCode\Repos\t_sne_bhcuda\build\vs2013\t_sne_bhcuda', 'data.dat',
                            theta=0.05, perplexity=perplexity, eta=200, no_dims=2, iterations=20,
                            seed=0, gpu_mem=0.8, verbose=2, randseed=1)









# -------- testing stuff --------------
import numpy as np
import os
from tsne_for_spikesort import t_sne, gpu
import matplotlib.pyplot as plt
import tsne_for_spikesort.io_with_cpp as io
import matplotlib.pylab as pylab

number_of_spikes = 100
# Generate data
indices_of_first_arrays, indices_of_second_arrays, template_features_sparce_clean_probesorted = \
    get_some_data(number_of_spikes=number_of_spikes)

# OR Load Stuff
base_folder = r'D:\Data\George\Projects\SpikeSorting\Neuroseeker\Neuroseeker_2016_12_17_Anesthesia_Auditory_DoubleProbes\AngledProbe\KilosortResults'
template_features_sparce_clean_probesorted = np.load(os.path.join(base_folder,
                                                                  'template_features_sparce_clean_probesorted_' +
                                                                  str(number_of_spikes)+'.npy'))
(indices_of_first_arrays, indices_of_second_arrays) = np.load(os.path.join(base_folder,
                                                                           r'first_second_matrices_indices_for_distance_cal_'+
                                                                           str(number_of_spikes)+'.npy'))





indices_of_first_and_second_matrices = (indices_of_first_arrays, indices_of_second_arrays)
perplexity = 20
theta = 0.1
eta = 200.0
num_dims = 2
iterations = 2000
verbose = 2

data = pylab.demean(template_features_sparce_clean_probesorted, axis=0)
data /= data.max()
num_samples = data.shape[0]
closest_indices_in_hd, closest_distances_in_hd = \
    gpu.calculate_knn_distances_close_on_probe(template_features_sorted=data,
                                               indices_of_first_and_second_matrices=
                                               indices_of_first_and_second_matrices,
                                               perplexity=perplexity,
                                               verbose=verbose)

# Test that the gpou distances are correct
from scipy.spatial.distance import cdist

for matrix_index in np.arange(indices_of_first_arrays.shape[0]):
    first_matrix = np.array(data[indices_of_first_arrays[matrix_index][0]:
    indices_of_first_arrays[matrix_index][1], :],
                            dtype=np.float32)
    second_matrix = np.array(data[indices_of_second_arrays[matrix_index][0]:
    indices_of_second_arrays[matrix_index][1], :],
                             dtype=np.float32)

cpu_closest_indices_in_hd = cdist(first_matrix, second_matrix, 'euclidean')

indices_p, values_p = t_sne._compute_gaussian_perplexity(closest_indices_in_hd, closest_distances_in_hd,
                                                   perplexity=perplexity)
sum_p = np.sum(values_p)
values_p /= sum_p

indices_p = indices_p.astype(np.uint)
values_p = values_p.astype(np.float64)
num_knns = indices_p.shape[1]
y = np.random.random((num_samples, num_dims)) * 0.0001
y = np.array(y, dtype=np.float64)


filename = 'data.dat'
folder_debug = r'E:\George\SourceCode\Repos\BarnesHutCpp\Barnes_Hut'
folder_release = r'E:\George\SourceCode\Repos\BarnesHutCpp\Barnes_Hut\x64\Release'

io.save_data_for_tsne(files_dir=folder_debug, y=y, col_p=indices_p, val_p=values_p, theta=theta,
                      perplexity=perplexity, eta=eta, iterations=iterations, verbose=verbose)


y = np.array(io.load_tsne_result(folder_debug, filename='result.dat'))


y = t_sne.run(data=template_features_sparce_clean_probesorted,
              indices_of_first_and_second_matrices=(indices_of_first_arrays, indices_of_second_arrays),
              intermediate_file_dir=r'E:\George\Temporary',
              iters=iterations, perplexity=perplexity, eta=eta, num_dims=num_dims, verbose=2,
              exe_dir=r'E:\Software\Develop\Languages\Pythons\Miniconda35\Scripts\Barnes_Hut.exe')


plt.scatter(y[:, 0], y[:, 1])




from struct import pack, unpack, calcsize
def _read_unpack(fmt, fh):
    return unpack(fmt, fh.read(calcsize(fmt)))

with open(os.path.join(folder, filename), 'rb') as output_file:
    theta, eta, n, no_dims, k, iterations, verbose, perplexity = _read_unpack('ddiiiiii', output_file)
    print(theta)
    print(perplexity)
    # Collect the results, but they may be out of order
    results = [_read_unpack('{}d'.format(no_dims), output_file) for _ in range(n)]

results = np.array(results)