

from BrainDataAnalysis import tsne_analysis_functions as tsne_funcs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os.path import join
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2018_04_AK_33p1 import constants as const
from spikesorting_tsne import preprocessing_kilosort_results as preproc_kilo, constants as ct, visualization as vis
import BrainDataAnalysis.neuroseeker_specific_functions as ns_funcs
from spikesorting_tsne_guis import helper_functions as hf
import sequence_viewer as seq_v
import transform as tr
import one_shot_viewer as one_s_v
from sklearn.decomposition import PCA
from spikesorting_tsne import tsne
import os

# FOLDERS NAMES --------------------------------------------------
date = 8
kilosort_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date],
                       'Analysis', 'Kilosort')
binary_data_filename = join(const.base_save_folder, const.rat_folder, const.date_folders[date],
                            'Data', 'Amplifier_APs.bin')
tsne_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date],
                   'Analysis', 'Tsne')
barnes_hut_exe_dir=r'E:\Software\Develop\Source\Repos\spikesorting_tsne_bhpart\Barnes_Hut\win\x64\Release'
# ----------------------------------------------------------------

'''
# ----------------------------------------------------------------
# CREATE INDIVIDUAL T-SNE FOR EACH MUA TEMPLATE WITH MORE THAN number_of_spikes_in_large_mua_templates SPIKES

preproc_kilo.t_sne_each_one_of_the_large_mua_templates_by_itself(kilosort_folder, tsne_folder, barnes_hut_exe_dir,
                                                               number_of_spikes_in_large_mua_templates=10000,
                                                               num_dims=2, perplexity=100, theta=0.2, iterations=2000,
                                                               random_seed=1, verbose=2)
# ----------------------------------------------------------------
'''

# ----------------------------------------------------------------
# HAVE A LOOK
# Find the mua templates with large spike count
number_of_spikes_in_large_mua_templates = 10000
large_mua_templates = preproc_kilo.find_large_mua_templates(kilosort_folder, number_of_spikes_in_large_mua_templates)


mua_template_to_look = 2
mua_template = large_mua_templates[mua_template_to_look]

template_folder = join(tsne_folder, 'template_{}'.format(mua_template))
data_for_tsne = np.load(join(template_folder, 'data_to_tsne_(50491, 36).npy'))

spike_times = np.squeeze(np.load(join(kilosort_folder, 'spike_times.npy')))
spikes = np.squeeze(np.load(join(template_folder, 'indices_of_spikes_used.npy')))
spike_times = spike_times[spikes]

pc_features = np.load(join(kilosort_folder, 'pc_features.npy'))
pc_feature_ind = np.load(join(kilosort_folder, 'pc_feature_ind.npy'))

avg_spike_template = np.load(join(kilosort_folder, 'avg_spike_template.npy'))
large_channels = pc_feature_ind[mua_template, :]
mua_template_data = avg_spike_template[mua_template, :, :]

#plt.plot(mua_template_data[large_channels, :].T)


raw_data = ns_funcs.load_binary_amplifier_data(binary_data_filename,
                                               number_of_channels=const.NUMBER_OF_AP_CHANNELS_IN_BINARY_FILE)

#large_channels_full = np.arange(np.min(large_channels), np.max(large_channels))

large_channels_full = np.arange(int(large_channels[0] - 30), int(large_channels[0] + 30))

#plt.plot(mua_template_data[large_channels_full, :].T)


spike_data = []
i = 0
for spike_time in spike_times:
    spike_data.append(raw_data[large_channels_full, int(spike_time-20):int(spike_time+20)])
    i += 1
    if i%1000 == 0:
        print(i)

spike_data = np.array(spike_data)


# TSNE The 10 PCs of all the channels that are within the group of largest channels according to Kilosort
# -------------------------------------------------------------
n_components = 10
principal_components = np.empty((spike_data.shape[0], spike_data.shape[1] * n_components))
pca = PCA(n_components=n_components)

i = 0
for index in range(len(spike_data)):
    pca.fit(spike_data[index, :, :].transpose())
    components = pca.components_.flatten()
    principal_components[index, :] = components
    i += 1
    if i % 1000 == 0:
        print(pca.explained_variance_)


template_folder = join(tsne_folder, 'template_{}_morePCs'.format(mua_template))
num_dims = 2
perplexity = 100
theta = 0.2
iterations = 3000
random_seed = 1
verbose = 2
template_features_matrix = principal_components

tsne.t_sne(template_features_matrix, files_dir=template_folder,
           exe_dir=barnes_hut_exe_dir,
           num_dims=num_dims, perplexity=perplexity,
           theta=theta, iterations=iterations, random_seed=random_seed, verbose=verbose)
preproc_kilo.generate_spike_info(kilosort_folder=kilosort_folder, tsne_folder=template_folder)

spike_info = np.load(join(template_folder, 'spike_info.df'))
vis.plot_tsne_of_spikes(spike_info)

# DBSCAN
data = np.array([spike_info['tsne_x'], spike_info['tsne_y']])
db, n_clusters, labels, core_samples_mask, score = tsne_funcs.fit_dbscan(data, eps=0.025, min_samples=43, normalize=True,
                                                                         show=True)
spike_info_for_labels = spike_info.copy()
spike_info_for_labels[ct.TEMPLATE_AFTER_CLEANING] = labels
spike_info_for_labels[ct.TEMPLATE_AFTER_SORTING] = labels
#spike_info_for_labels.to_pickle(join(template_folder, 'spike_info_for_labels.df'))
spike_info_for_labels.to_pickle(join(template_folder, 'spike_info.df'))
vis.plot_tsne_of_spikes(spike_info_for_labels)
# -------------------------------------------------------------


# TSNE ALL LARGE MUA
number_of_spikes_in_large_mua_templates = 10000
large_mua_templates = preproc_kilo.find_large_mua_templates(kilosort_folder, number_of_spikes_in_large_mua_templates)

templates_of_spikes = np.load(join(kilosort_folder, 'spike_templates.npy'))
spike_times = np.squeeze(np.load(join(kilosort_folder, 'spike_times.npy')))

pc_feature_ind = np.load(join(kilosort_folder, 'pc_feature_ind.npy'))

raw_data = ns_funcs.load_binary_amplifier_data(binary_data_filename,
                                               number_of_channels=const.NUMBER_OF_AP_CHANNELS_IN_BINARY_FILE)

large_mua_templates_minus = np.delete(large_mua_templates, [2])

for mua_template in large_mua_templates_minus:
    print('DOING TEMPLATE NUMBER {}, {} OUT OF {}'.format(mua_template,
                                                          np.argwhere(large_mua_templates_minus == mua_template),
                                                          len(large_mua_templates_minus)))

    large_channels = pc_feature_ind[mua_template, :]
    tsne_folder_single_template = join(tsne_folder, 'template_{}'.format(mua_template))
    os.mkdir(tsne_folder_single_template)

    mua_spikes = np.argwhere(np.in1d(templates_of_spikes, mua_template) > 0)
    np.save(join(tsne_folder_single_template, ct.INDICES_OF_SPIKES_USED_FILENAME), mua_spikes)

    mua_spike_times = spike_times[mua_spikes]

    large_channels_full = np.arange(int(large_channels[0] - 30), int(large_channels[0] + 30))
    spike_data = []

    for spike_time in mua_spike_times:
        spike_data.append(raw_data[large_channels_full, int(spike_time - 20):int(spike_time + 20)])

    spike_data = np.array(spike_data)

    n_components = 10
    principal_components = np.empty((spike_data.shape[0], spike_data.shape[1] * n_components))

    i = 0
    for index in range(len(spike_data)):
        pca = PCA(n_components=n_components)
        pca.fit(spike_data[index, :, :].transpose())
        components = pca.components_.flatten()
        principal_components[index, :] = components
        i += 1
        if i % 5000 == 0:
            print(pca.explained_variance_)

    template_features_matrix = principal_components
    num_dims = 2
    perplexity = 100
    theta = 0.2
    iterations = 4000
    random_seed = 1
    verbose = 2

    tsne.t_sne(template_features_matrix, files_dir=tsne_folder_single_template,
               exe_dir=barnes_hut_exe_dir,
               num_dims=num_dims, perplexity=perplexity,
               theta=theta, iterations=iterations, random_seed=random_seed, verbose=verbose)
    preproc_kilo.generate_spike_info(kilosort_folder=kilosort_folder, tsne_folder=tsne_folder_single_template)
# -------------------------------------------------------------


s = 0
seq_v.graph_pane(globals(), 's', 'spike_data')

spike_data_fd = np.diff(spike_data, axis=2)
spike_data_fd = np.concatenate((spike_data_fd, np.zeros((spike_data.shape[0], spike_data.shape[1], 1))), axis=2)
seq_v.graph_pane(globals(), 's', 'spike_data_fd')

spike_data_sd = np.diff(spike_data_fd, axis=2)
spike_data_sd = np.concatenate((spike_data_sd, np.zeros((spike_data.shape[0], spike_data.shape[1], 1))), axis=2)

largest_channel_index = np.squeeze(np.argwhere(large_channels_full == large_channels[0]))

def get_phase_x_of_channel_for_spike(s):
    return spike_data_sd[s, largest_channel_index, :]

def get_phase_y_of_channel_for_spike(s):
    return spike_data_fd[s, largest_channel_index, :]

phase_x = None
phase_y = None

tr.connect_repl_var(globals(), 's', 'get_phase_x_of_channel_for_spike', 'phase_x')
tr.connect_repl_var(globals(), 's', 'get_phase_y_of_channel_for_spike', 'phase_y')

s = 1
one_s_v.graph(globals(), 'phase_y', 'phase_x')