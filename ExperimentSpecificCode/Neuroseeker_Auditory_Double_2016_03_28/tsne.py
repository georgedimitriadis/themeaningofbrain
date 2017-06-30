


import numpy as np
import matplotlib.pylab as pylab
from tsne_for_spikesort import gpu
from tsne_for_spikesort import io_with_cpp as io
from os.path import join
from BrainDataAnalysis import ploting_functions as pf
import matplotlib.pyplot as plt
from t_sne_bhcuda import tsne_cluster as tsne_cl


'''
#base_folder_v = r'D:\Data\George\Projects\SpikeSorting\\Neuroseeker\\' + \
#                r'Neuroseeker_2017_03_28_Anesthesia_Auditory_DoubleProbes\Vertical\\' + \
#                r'Experiment_2_T18_48_25_And_Experiment_3_T19_41_07\Kilosort'
#binary_v = r'Exp2and3_2017_03_28T18_48_25_Amp_S16_LP3p5KHz_mV.bin'
'''


base_folder = r'D:\Data\Brain\Neuroseeker_2017_03_28_Anesthesia_Auditory_DoubleProbes\Angled\Analysis\Experiment_2_T18_48_25_And_Experiment_3_T19_41_07\Kilosort'

template_marking = np.load(join(base_folder, 'template_marking.npy'))
spike_templates = np.load(join(base_folder, 'spike_templates.npy'))
template_features = np.load(join(base_folder, 'template_features.npy'))
template_features_ind = np.load(join(base_folder, 'template_feature_ind.npy'))

clean_templates = np.argwhere(template_marking)
spikes_clean_index = np.squeeze(np.argwhere(np.in1d(spike_templates, clean_templates)))

number_of_spikes = 200000  # Or all = spikes_clean_index.size

spikes_clean_index = spikes_clean_index[:number_of_spikes]
clean_templates = np.unique(spike_templates[spikes_clean_index])

# -------
template_features_sparse_clean = np.zeros((spikes_clean_index.size, clean_templates.size))
for spike in spikes_clean_index:
    cluster = spike_templates[spike][0]
    indices = template_features_ind[cluster, :]
    for i in np.arange(len(indices)):
        template_features_sparse_clean[np.argwhere(spikes_clean_index == spike),
                                       np.argwhere(clean_templates == indices[i])] = template_features[
            spike, i]

np.save(join(base_folder, 'data_to_tsne_' + str(template_features_sparse_clean.shape) + '.npy'),
        template_features_sparse_clean)
# --------

template_features_sparse_clean = np.load(join(base_folder, r'data_to_tsne_(20000, 172).npy'))
# --------


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



# Run now manually the Barnes Hut C++ exe
# Then load the results
tsne = io.load_tsne_result(exe_folder)


# and then plot the t-sne color coded with the kilosort templates
spike_templates = np.load(join(base_folder, 'spike_templates.npy'))
spike_templates_clean = spike_templates[spikes_clean_index]

cluster_info = tsne_cl.create_cluster_info_from_kilosort_spike_templates(join(base_folder, 'cluster_info.pkl'),
                                                                         spike_templates_clean)
labels_dict = pf.generate_labels_dict_from_cluster_info_dataframe(cluster_info=cluster_info)

markers = ['.', '*', 'o', '>', '<', '_', ',']
labeled_sizes = range(20, 100, 20)

pf.plot_tsne(tsne.T, cm=plt.cm.prism, labels_dict=labels_dict, legent_on=False, markers=markers, labeled_sizes=labeled_sizes)