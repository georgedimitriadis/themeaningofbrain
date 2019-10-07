from spikesorting_tsne import tsne as TSNE
from spikesorting_tsne import preprocessing_kilosort_results as preproc


import pandas as pd
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from spikesorting_tsne import io_with_cpp as io
from BrainDataAnalysis.Graphics import ploting_functions as pf

base_folder = r'Z:\n\Neuroseeker Probe Recordings\Neuroseeker_2017_08_08\Analysis\kilosort\18_26_30_afterREFeachGroup'

files_dir = join(base_folder, 'Tsne_Results')


# Get features of all clean spikes
template_features_sparse_clean = \
    preproc.calculate_template_features_matrix_for_tsne(base_folder, save_to_folder=files_dir)

# OR Load it
template_features_sparse_clean = np.load(join(files_dir, 'data_to_tsne_(513722, 588).npy'))


exe_dir = r'E:\Software\Develop\Source\Repos\spikesorting_tsne_bhpart\Barnes_Hut\x64\Release'
theta = 0.4
eta = 200.0
exageration = 12
num_dims = 2
perplexity = 100
iterations = 4000
random_seed = 1
verbose = 3
tsne = TSNE.t_sne(samples=template_features_sparse_clean, files_dir=files_dir, exe_dir=exe_dir, num_dims=num_dims,
                  perplexity=perplexity, theta=theta, eta=eta, exageration=exageration, iterations=iterations,
                  random_seed=random_seed, verbose=verbose)

# OR LOAD
tsne = io.load_tsne_result(files_dir=files_dir)

# -------------------------------------------------------------------
# PLOTTING
spike_templates = np.load(join(base_folder, 'spike_templates.npy'))
template_markings = np.load(join(base_folder, 'template_marking.npy'))
clean_templates_index = np.argwhere(template_markings > 0)
clean_spikes_index = np.squeeze(np.argwhere(np.in1d(spike_templates, clean_templates_index)))

np.save(join(files_dir, 'indices_of_spikes_used.npy'), clean_spikes_index)
spikes_used = np.load(join(files_dir, 'indices_of_spikes_used.npy'))

spike_templates_clean = spike_templates[spikes_used]


def create_cluster_info_from_kilosort_spike_templates(cluster_info_filename, spike_templates):
    kilosort_units = {}
    for i in np.arange(len(spike_templates)):
        cluster = spike_templates[i][0]
        if cluster in kilosort_units:
            kilosort_units[cluster] = np.append(kilosort_units[cluster], i)
        else:
            kilosort_units[cluster] = i

    cluster_info = pd.DataFrame(columns=['Cluster', 'Num_of_Spikes', 'Spike_Indices'])
    cluster_info = cluster_info.set_index('Cluster')
    cluster_info['Spike_Indices'] = cluster_info['Spike_Indices'].astype(list)

    cluster_info.set_value('UNLABELED', 'Num_of_Spikes', 0)
    cluster_info.set_value('UNLABELED', 'Spike_Indices', [])
    cluster_info.set_value('NOISE', 'Num_of_Spikes', 0)
    cluster_info.set_value('NOISE', 'Spike_Indices', [])
    cluster_info.set_value('MUA', 'Num_of_Spikes', 0)
    cluster_info.set_value('MUA', 'Spike_Indices', [])
    for g in kilosort_units.keys():
        if np.size(kilosort_units[g]) == 1:
            kilosort_units[g] = [kilosort_units[g]]
        cluster_name = str(g)
        cluster_info.set_value(cluster_name, 'Num_of_Spikes', len(kilosort_units[g]))
        cluster_info.set_value(cluster_name, 'Spike_Indices', kilosort_units[g])

    cluster_info.to_pickle(cluster_info_filename)
    return cluster_info

cluster_info = create_cluster_info_from_kilosort_spike_templates(join(base_folder, 'cluster_info.pkl'),
                                                                         spike_templates_clean)
labels_dict = pf.generate_labels_dict_from_cluster_info_dataframe(cluster_info=cluster_info)

markers = ['.', '*', 'o', '>', '<', '_', ',']
labeled_sizes = range(20, 100, 20)

pf.plot_tsne(tsne.T, cm=plt.cm.prism, labels_dict=labels_dict, legent_on=False, markers=None, labeled_sizes=None)


spike_info = preproc.generate_spike_info_from_full_tsne(base_folder, files_dir)