


import numpy as np
import matplotlib.pylab as pylab
from tsne_for_spikesort_old import gpu
from tsne_for_spikesort_old import io_with_cpp as io
from os.path import join


base_folder = r'F:\Data\George\Projects\SpikeSorting\Neuroseeker\\' + \
              r'Neuroseeker_2017_03_28_Anesthesia_Auditory_DoubleProbes\Angled\Analysis\\' + \
              r'Experiment_2_T18_48_25_And_Experiment_3_T19_41_07\Kilosort' # Desktop
#base_folder = r'D:\Data\Brain\Neuroseeker_2017_03_28_Anesthesia_Auditory_DoubleProbes\Angled\Analysis\Experiment_2_T18_48_25_And_Experiment_3_T19_41_07\Kilosort' # Laptop


# ------------- MANUAL PIPELINE ----------------------------------------------------------------------------------------

template_marking = np.load(join(base_folder, 'template_marking.npy'))
spike_templates = np.load(join(base_folder, 'spike_templates.npy'))
template_features = np.load(join(base_folder, 'template_features.npy'))
template_features_ind = np.load(join(base_folder, 'template_feature_ind.npy'))

clean_templates = np.argwhere(template_marking)
spikes_clean_index = np.squeeze(np.argwhere(np.in1d(spike_templates, clean_templates)))

number_of_spikes = spikes_clean_index.size  # Or all = spikes_clean_index.size


# -------
template_features_sparse_clean = np.zeros((spikes_clean_index.size, clean_templates.size))
s = 0
for spike in spikes_clean_index:
    cluster = spike_templates[spike][0]
    indices = template_features_ind[cluster, :]
    if s % 5000 == 0:
        print(s)
    s+=1
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
iterations = 3000
verbose = 3
exe_folder = r'E:\Software\Develop\Source\Repos\spikesorting_tsne_bhpart\Barnes_Hut\x64\Release'  # Desktop
#exe_folder = r'E:\Projects\Analysis\Brain\spikesorting_tsne_bhpart\Barnes_Hut\x64\Release' # Laptop
io.save_data_for_barneshut(exe_folder, closest_distances_in_hd, closest_indices_in_hd, eta=eta, iterations=iterations,
                           num_of_dims=num_dims, perplexity=perplexity, theta=theta, verbose=verbose)



# Run now manually the Barnes Hut C++ exe
# Then load the results
tsne = io.load_tsne_result(exe_folder)


# ----------------------------------------------------------------------------------------------------------------------
# -------------PACKAGE PILELINE-----------------------------------------------------------------------------------------
from spikesorting_tsne import preprocessing_kilosort_results as preproc
from spikesorting_tsne import tsne as TSNE
from os.path import join
from BrainDataAnalysis import ploting_functions as pf
import numpy as np
from ExperimentSpecificCode._2016_TSne_Paper.t_sne_bhcuda import tsne_cluster as tsne_cl
import matplotlib.pyplot as plt
from tsne_for_spikesort_old import io_with_cpp as io


base_folder = r'F:\Neuroseeker\\' + \
              r'Neuroseeker_2017_03_28_Anesthesia_Auditory_DoubleProbes\Angled\Analysis\\' + \
              r'Experiment_2_T18_48_25_And_Experiment_3_T19_41_07\Kilosort'  # Desktop

files_dir = join(base_folder, 'Tsne_Results', '2017_08_20_dis700K_579templates')



total_spikes_required = 700000
full_inclusion_threshold = 15000  # number of spikes or fewer for a template to have to be fully included

spikes_used, small_clean_templates_with_spike_indices, large_clean_templates_with_spike_indices\
    = preproc.find_spike_indices_for_representative_tsne(base_folder=base_folder,
                                                         save_to_folder=files_dir,
                                                         threshold=full_inclusion_threshold,
                                                         total_spikes_required=total_spikes_required)

spikes_used = None  # Use this to load all spikes for tsne
template_features_sparse_clean = \
    preproc.calculate_template_features_matrix_for_tsne(base_folder, save_to_folder=files_dir,
                                                        spikes_used_with_original_indexing=spikes_used)

# OR Load it
template_features_sparse_clean = np.load(join(files_dir, 'data_to_tsne_(1091229, 579).npy'))


exe_dir = r'E:\Software\Develop\Source\Repos\spikesorting_tsne_bhpart\Barnes_Hut\x64\Release'
theta = 0.4
eta = 200.0
exageration = 20
num_dims = 2
perplexity = 100
iterations = 4000
random_seed = 1
verbose = 3
tsne = TSNE.t_sne(samples=template_features_sparse_clean, files_dir=files_dir, exe_dir=exe_dir, num_dims=num_dims,
                  perplexity=perplexity, theta=theta, eta=eta, exageration=exageration, iterations=iterations,
                  random_seed=random_seed, verbose=verbose)

# OR
tsne = io.load_tsne_result(join(base_folder, files_dir))

# OR
tsne = TSNE.t_sne_from_existing_distances(files_dir=files_dir, exe_dir=exe_dir, num_dims=num_dims,
                                          perplexity=perplexity, theta=theta, eta=eta, exageration=exageration,
                                          iterations=iterations, random_seed=random_seed, verbose=verbose)

spikes_used = np.load(join(files_dir, 'indices_of_spikes_used.npy'))

spike_templates = np.load(join(base_folder, 'spike_templates.npy'))
spike_templates_clean = spike_templates[spikes_used]

cluster_info = tsne_cl.create_cluster_info_from_kilosort_spike_templates(join(base_folder, 'cluster_info.pkl'),
                                                                         spike_templates_clean)
labels_dict = pf.generate_labels_dict_from_cluster_info_dataframe(cluster_info=cluster_info)

markers = ['.', '*', 'o', '>', '<', '_', ',']
labeled_sizes = range(20, 100, 20)

pf.plot_tsne(tsne.T, cm=plt.cm.prism, labels_dict=labels_dict, legent_on=False, markers=None, labeled_sizes=None)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(tsne[:, 0], tsne[:, 1], tsne[:, 2], zdir='z', s=20, c='b', depthshade=True)


pf.make_video_of_tsne_iterations(iterations=3000, video_dir=files_dir, data_file_name='interim_{:0>6}.dat',
                                 video_file_name='tsne_video.mp4', figsize=(15, 15), dpi=200, fps=30,
                                 labels_dict=labels_dict, cm=plt.cm.prism,
                                 label_name='Label', legent_on=False, labeled_sizes=None, markers=None,
                                 max_screen=True)




spike_info = preproc.generate_spike_info(base_folder, files_dir)






