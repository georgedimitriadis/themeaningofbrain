from tsne_for_spikesort_old import spikes
import matplotlib.pylab as pylab
from tsne_for_spikesort_old import gpu
from struct import calcsize, unpack
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from numba import cuda
from ExperimentSpecificCode._2016_TSne_Paper.t_sne_bhcuda import bhtsne_cuda as tsne_exe, tsne_cluster as tsne_cl, \
    tsne_cluster
from BrainDataAnalysis import ploting_functions as pl


def get_some_data(base_folder, binary_data_filename, number_of_spikes):
    number_of_channels_in_binary_file = 1440

    template_marking = np.load(join(base_folder, 'template_marking.npy'))
    spike_templates = np.load(join(base_folder, 'spike_templates.npy'))
    template_features = np.load(join(base_folder, 'template_features.npy'))
    template_features_ind = np.load(join(base_folder, 'template_feature_ind.npy'))

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

    spike_positions, spike_distances_on_probe, \
        spike_indices_sorted_by_probe_distance, spike_distances_on_probe_sorted = \
        spikes.generate_probe_positions_of_spikes(base_folder=base_folder,
                                                  binary_data_filename=binary_data_filename,
                                                  number_of_channels_in_binary_file=
                                                  number_of_channels_in_binary_file,
                                                  used_spikes_indices=spikes_clean_index,
                                                  position_mult=position_mult, threshold=2)



    distance_threshold = 1000
    max_elements_in_matrix = 0.5e9

    indices_of_first_arrays, indices_of_second_arrays = \
        spikes.define_all_spike_spike_matrices_for_distance_calc(spike_distances_on_probe_sorted,
                                                                 max_elements_in_matrix=max_elements_in_matrix,
                                                                 probe_distance_threshold=distance_threshold)


    indices_of_first_arrays = np.array(indices_of_first_arrays)
    indices_of_second_arrays = np.array(indices_of_second_arrays)

    template_features_sparce_clean_probesorted = template_features_sparse_clean[spike_indices_sorted_by_probe_distance, :]

    return indices_of_first_arrays, indices_of_second_arrays, \
           spike_indices_sorted_by_probe_distance, template_features_sparce_clean_probesorted


# GET SOME TEMPLATE FEATURES DATA
base_folder = r'E:\Data\Brain\Neuroseeker_2016_12_17_Anesthesia_Auditory_DoubleProbes\AngledProbe\Kilosort results'
binary_data_filename = r'AngledProbe_BinaryAmplifier_12Regions_Penetration1_2016-12-17T19_02_12.bin'
number_of_spikes = 100000
indices_of_first_arrays, indices_of_second_arrays, \
spike_indices_sorted_by_probe_distance, template_features_sparce_clean_probesorted = \
    get_some_data(base_folder=base_folder, binary_data_filename=binary_data_filename, number_of_spikes=number_of_spikes)

data = pylab.demean(template_features_sparce_clean_probesorted, axis=0)
data /= data.max()


# CALCULATE DISTANCES ON THE GPU
indices_of_first_and_second_matrices = (indices_of_first_arrays, indices_of_second_arrays)

perplexity = 100
theta = 0.4
eta = 200.0
num_dims = 2
iterations = 1000
verbose = 2

num_samples = data.shape[0]
closest_indices_in_hd, closest_distances_in_hd = \
    gpu.calculate_knn_distances_close_on_probe(template_features_sorted=data,
                                               indices_of_first_and_second_matrices=
                                               indices_of_first_and_second_matrices,
                                               perplexity=perplexity,
                                               verbose=verbose)



import numpy as np
import matplotlib.pylab as pylab
from tsne_for_spikesort_old import gpu
from os.path import join
from tsne_for_spikesort_old import io_with_cpp as io
base_folder = r'E:\Data\Brain\Neuroseeker_2016_12_17_Anesthesia_Auditory_DoubleProbes\AngledProbe\Kilosort results'

perplexity = 100
template_features_sparse_clean = np.load(join(base_folder, 'template_features_sparce_clean_100k.npy'))
data = pylab.demean(template_features_sparse_clean, axis=0)
data /= data.max()
closest_indices_in_hd, closest_distances_in_hd = \
    gpu.calculate_knn_distances_all_to_all(template_features_sparse_clean=data, perplexity=perplexity, mem_usage=0.9,
                                           verbose=True)


# PASS THE SORTED DISTANCES TO BARNES HUT C++ TO GENERATE T-SNE RESULTS
theta = 0.4
eta = 200.0
num_dims = 2
iterations = 1000
verbose = 2
debug_folder = r'E:\Projects\Analysis\Brain\spikesorting_tsne_bhpart\Barnes_Hut'
exe_folder = r'E:\Projects\Analysis\Brain\spikesorting_tsne_bhpart\Barnes_Hut\x64\Release'
io.save_data_for_barneshut(exe_folder, closest_distances_in_hd, closest_indices_in_hd, eta=eta, iterations=iterations,
                           num_of_dims=num_dims, perplexity=perplexity, theta=theta, verbose=verbose)

# pass the cleaned probe sorted sparce features to the full bhcuda_tsne.exe
tsne_exe.save_data_for_tsne(template_features_sparce_clean_probesorted,
                            r'E:\Projects\Analysis\Brain\t_sne_bhcuda\build\vs2013', 'data.dat',
                            theta=0.05, perplexity=perplexity, eta=eta, no_dims=num_dims, iterations=2000,
                            seed=0, gpu_mem=0.8, verbose=2, randseed=1)


# Run now manually the Barnes Hut C++ exe
# Then load the results
tsne = io.load_tsne_result(exe_folder)


# and then plot the t-sne color coded with the kilosort templates
spike_templates = np.load(join(base_folder, 'spike_templates.npy'))
template_marking = np.load(join(base_folder, 'template_marking.npy'))
# clean out the spike_templates array (remove the spikes in the noise templates)
templates_clean_index = np.argwhere(template_marking)
spikes_clean_index = np.squeeze(np.argwhere(np.in1d(spike_templates, templates_clean_index)))
spike_templates_clean = spike_templates[spikes_clean_index][:number_of_spikes]

spike_templates_clean_sorted = spike_templates_clean[spike_indices_sorted_by_probe_distance]
cluster_info = tsne_cl.create_cluster_info_from_kilosort_spike_templates(join(base_folder, 'cluster_info.pkl'),
                                                                         spike_templates_clean_sorted)
labels_dict = pl.generate_labels_dict_from_cluster_info_dataframe(cluster_info=cluster_info)
pl.plot_tsne(tsne.T, cm=plt.cm.prism, labels_dict=labels_dict)


# LOAD SAVED DATA TO CHECK WHAT HAS BEEN SAVED
def _read_unpack(fmt, fh):
    return unpack(fmt, fh.read(calcsize(fmt)))

filename = 'data.dat'
with open(join(debug_folder, filename), 'rb') as output_file:
    theta_, eta_, num_of_spikes_, num_of_dims_, num_of_nns_, iterations_, verbose_, perplexity_ =\
        _read_unpack('ddiiiiii', output_file)

    sorted_distances_ = [_read_unpack('{}d'.format(num_of_nns_), output_file) for _ in range(num_of_spikes_)]

    sorted_indices_ = [_read_unpack('{}i'.format(num_of_nns_), output_file) for _ in range(num_of_spikes_)]



# CALCULATE DISTANCES ON THE CPU AND THE GPU FOR TESTING
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
cpu_distances_sorted = np.sort(cpu_distances, axis=-1)






# ----------------------------------------------------
# ----------------------------------------------------
# DO THE FULL ANGLED PROBE DATA SET WITH A ALL TO ALL SPIKES T-SNE


import numpy as np
import matplotlib.pylab as pylab
from tsne_for_spikesort_old import gpu
from tsne_for_spikesort_old import io_with_cpp as io
from t_sne_bhcuda import tsne_cluster as tsne_cl
from os.path import join
from BrainDataAnalysis import ploting_functions as pl
import matplotlib.pyplot as plt

base_folder = r'D:\Data\Brain\Neuroseeker_2016_12_17_Anesthesia_Auditory_DoubleProbes\AngledProbe\Kilosort results'
binary_data_filename = r'AngledProbe_BinaryAmplifier_12Regions_Penetration1_2016-12-17T19_02_12.bin'

template_marking = np.load(join(base_folder, 'template_marking.npy'))
spike_templates = np.load(join(base_folder, 'spike_templates.npy'))
template_features = np.load(join(base_folder, 'template_features.npy'))
template_features_ind = np.load(join(base_folder, 'template_feature_ind.npy'))

clean_templates = np.argwhere(template_marking)
spikes_clean_index = np.squeeze(np.argwhere(np.in1d(spike_templates, clean_templates)))

'''
spike_num_to_use = 20000
if spike_num_to_use is not None:
    spikes_clean_index = random.sample(list(spikes_clean_index), spike_num_to_use) # Or None
    templates_clean_index = np.unique(spike_templates[spikes_clean_index])
'''

template_features_sparse_clean = np.zeros((spikes_clean_index.size, templates_clean_index.size))
for spike in spikes_clean_index:
    cluster = spike_templates[spike][0]
    indices = template_features_ind[cluster, :]
    for i in np.arange(len(indices)):
        template_features_sparse_clean[np.argwhere(spikes_clean_index == spike),
                                       np.argwhere(templates_clean_index == indices[i])] = template_features[
            spike, i]


np.save(join(base_folder, 'data_to_tsne_' + str(template_features_sparse_clean.shape) + '.npy'),
        template_features_sparse_clean)


# FIND DISTANCES AND PICK THE PER*3 SMALLEST
template_features_sparse_clean = np.load(join(base_folder, r'data_to_tsne_(20000, 172).npy'))

data = pylab.demean(template_features_sparse_clean, axis=0)
data /= data.max()
perplexity = 100
closest_indices_in_hd, closest_distances_in_hd = \
    gpu.calculate_knn_distances_all_to_all(template_features_sparse_clean=data, perplexity=perplexity, mem_usage=0.9,
                                           verbose=True)

# PASS THE SORTED DISTANCES TO BARNES HUT C++ TO GENERATE T-SNE RESULTS
theta = 0.4
eta = 200.0
num_dims = 2
iterations = 1000
verbose = 2
exe_folder = r'E:\Projects\Analysis\Brain\spikesorting_tsne_bhpart\Barnes_Hut\x64\Release'
io.save_data_for_barneshut(exe_folder, closest_distances_in_hd, closest_indices_in_hd, eta=eta, iterations=iterations,
                           num_of_dims=num_dims, perplexity=perplexity, theta=theta, verbose=verbose)

# CALL THE C++ EXECUTABLE HERE!!
# Then load the results
tsne = io.load_tsne_result(exe_folder)
tsne = tsne.T

#  NOW PLOT
spike_templates_clean = spike_templates[spikes_clean_index]
cluster_info = tsne_cl.create_cluster_info_from_kilosort_spike_templates(join(base_folder, 'cluster_info.pkl'),
                                                                         spike_templates_clean)
labels_dict = pl.generate_labels_dict_from_cluster_info_dataframe(cluster_info=cluster_info)
markers = ['.', '*', 'o', '>', '<', '_', ',']
labeled_sizes = range(20, 100, 20)
pl.plot_tsne(tsne, cm=plt.cm.prism, labels_dict=labels_dict, markers=markers, labeled_sizes=labeled_sizes)

spike_times = np.load(join(base_folder, 'spike_times.npy'))[spikes_clean_index]
channel_positions = np.load(join(base_folder, 'channel_positions.npy'))
num_of_electrodes = 1440
raw_extracellular_data = np.memmap(join(base_folder, binary_data_filename), dtype=np.int16)
num_of_time_points = int(raw_extracellular_data.shape[0]/num_of_electrodes)
raw_extracellular_data = np.reshape(raw_extracellular_data, (num_of_electrodes, num_of_time_points), order='F')
data_cube_filename = join(base_folder, 'raw_data_around_spikes_cube.npy')
channel_map = np.squeeze(np.load(join(base_folder, 'channel_map.npy')))

# Make cube
data_cube = tsne_cluster.create_data_cube_from_raw_extra_data(raw_extracellular_data, data_cube_filename,
                                                              num_of_points_in_spike_trig=80, cube_type=np.int16, extra_spike_times=spike_times,
                                                              num_of_electrodes=None, used_electrodes=channel_map,
                                                              num_of_points_for_baseline=15)
# Or load it if already made
data_cube = np.memmap(data_cube_filename, dtype=np.int16, shape=(channel_map.shape[0], 80, len(spike_times)))


tsne_cluster.gui_manual_cluster_tsne_spikes(tsne_array_or_filename=tsne, spike_times_list_or_filename=spike_times,
                                            raw_extracellular_data=None,
                                            num_of_points_for_baseline=10, cut_extracellular_data_or_filename=data_cube,
                                            shape_of_cut_extracellular_data=data_cube.shape, cube_type=data_cube.dtype,
                                            sampling_freq=20000, autocor_bin_number=2000,
                                            cluster_info_file=join(base_folder, 'cluster_info.pkl'),
                                            use_existing_cluster=True, time_samples_h5_dir=None,
                                            spike_indices_to_use=None, prb_file=None, k4=False,
                                            verbose=True)