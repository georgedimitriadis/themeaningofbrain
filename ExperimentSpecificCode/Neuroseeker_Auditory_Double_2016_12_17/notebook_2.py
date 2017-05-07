
import numpy as np
import os
from tsne_for_spikesort import spikes
import matplotlib.pylab as pylab
from tsne_for_spikesort import t_sne, gpu


def get_some_data(base_folder, binary_data_filename, number_of_spikes):
    number_of_channels_in_binary_file = 1440

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

    position_mult = 2.25

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


# GET SOME TEMPLATE FEATURES DATA
base_folder = r'E:\Data\Brain\Neuroseeker_2016_12_17_Anesthesia_Auditory_DoubleProbes\AngledProbe\Kilosort results'
binary_data_filename = r'AngledProbe_BinaryAmplifier_12Regions_Penetration1_2016-12-17T19_02_12.bin'
number_of_spikes = 20000
indices_of_first_arrays, indices_of_second_arrays, template_features_sparce_clean_probesorted = \
    get_some_data(base_folder=base_folder, binary_data_filename=binary_data_filename, number_of_spikes=number_of_spikes)

data = pylab.demean(template_features_sparce_clean_probesorted, axis=0)
data /= data.max()


# CALCULATE DISTANCES ON THE GPU
indices_of_first_and_second_matrices = (indices_of_first_arrays, indices_of_second_arrays)
perplexity = 100
theta = 0.1
eta = 200.0
num_dims = 2
iterations = 2000
verbose = 2

num_samples = data.shape[0]
closest_indices_in_hd, closest_distances_in_hd = \
    gpu.calculate_knn_distances_close_on_probe(template_features_sorted=data,
                                               indices_of_first_and_second_matrices=
                                               indices_of_first_and_second_matrices,
                                               perplexity=perplexity,
                                               verbose=verbose)


# PASS THE DATA TO C++ TO GENERATE SORTED DISTANCES
from tsne_for_spikesort import io_with_cpp as io
from os.path import join
from struct import calcsize, unpack

debug_folder = r'E:\Projects\Analysis\Brain\BarnesHutCpp\Barnes_Hut'
exe_folder = r'E:\Projects\Analysis\Brain\BarnesHutCpp\Barnes_Hut\x64\Release'
io.save_data_for_barneshut(exe_folder, closest_distances_in_hd, closest_indices_in_hd, eta=eta, iterations=iterations,
                           num_of_dims=num_dims, perplexity=perplexity, theta=theta, verbose=verbose)

# Run now the C++ exe
# Then load the results
tsne = io.load_tsne_result(exe_folder)
import matplotlib.pyplot as plt
plt.scatter(tsne[:, 0], tsne[:, 1])


# LOAD SAVED DATA TO CHECK WHAT HAS BEEN SAVED
def _read_unpack(fmt, fh):
    return unpack(fmt, fh.read(calcsize(fmt)))

filename = 'data.dat'
with open(join(debug_folder, filename), 'rb') as output_file:
    theta_, eta_, num_of_spikes_, num_of_dims_, num_of_nns_, iterations_, verbose_, perplexity_ =\
        _read_unpack('ddiiiiii', output_file)

    sorted_distances_ = [_read_unpack('{}d'.format(num_of_nns_), output_file) for _ in range(num_of_spikes_)]

    sorted_indices_ = [_read_unpack('{}i'.format(num_of_nns_), output_file) for _ in range(num_of_spikes_)]



# CALCULATE DISTANCES ON THE CPU FOR TESTING
from scipy.spatial.distance import cdist
from numba import cuda
import numpy as np

for matrix_index in np.arange(indices_of_first_arrays.shape[0]):
    first_matrix = np.array(data[indices_of_first_arrays[matrix_index][0]:
    indices_of_first_arrays[matrix_index][1], :],
                            dtype=np.float32)
    second_matrix = np.array(data[indices_of_second_arrays[matrix_index][0]:
    indices_of_second_arrays[matrix_index][1], :],
                             dtype=np.float32)


m = first_matrix.shape[0]
n = second_matrix.shape[0]
temp = np.array(np.zeros((m, n), dtype=np.float32))
distances_on_gpu = cuda.to_device(np.asfortranarray(temp))
gpu._calculate_distances_on_gpu(first_matrix, second_matrix, distances_on_gpu, verbose=True)
gpu_distances = distances_on_gpu.copy_to_host()
gpu_distances[np.argwhere(np.isnan(gpu_distances))] = 0
gpu_distances = np.sqrt(np.abs(gpu_distances))

cpu_distances = cdist(first_matrix, second_matrix, 'euclidean')
np.sqrt(((gpu_distances - cpu_distances) ** 2).sum())


