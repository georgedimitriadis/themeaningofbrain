

import numpy as np
from os.path import join
from GUIs.Kilosort import clean_kilosort_templates as ckt
from GUIs.Kilosort import create_data_cubes as cdc
from spikesorting_tsne import preprocessing_kilosort_results as preproc
from spikesorting_tsne import tsne as TSNE
from spikesorting_tsne import io_with_cpp as io
from spikesorting_tsne import positions_on_probe as pos
from BrainDataAnalysis import ploting_functions as pf

kilosort_folder = r'F:\JoanaPaired\128ch\2015-09-03\Analysis\kilosort\thres4_10_10_Fe16_Pc12\kilosort output'
tsne_folder = r'F:\JoanaPaired\128ch\2015-09-03\Analysis\kilosort\thres4_10_10_Fe16_Pc12\tsne\tsne_cleaned'
data_folder = r'F:\JoanaPaired\128ch\2015-09-03\Data'
adc_file = 'adc2015-09-03T21_18_47.bin'
amplifier_file = r'amplifier2015-09-03T21_18_47.bin'
prb_file = r'F:\JoanaPaired\128ch\2015-09-03\Analysis\klustakwik\128ch_passive_imec.prb'


number_of_channels_in_binary_file = 128
type_of_binary = np.uint16

# CLEANING KILOSORT RESULTS -------------------------
cdc.generate_average_over_spikes_per_template_multiprocess(base_folder=kilosort_folder,
                                                           binary_data_filename=join(data_folder, amplifier_file),
                                                           number_of_channels_in_binary_file=number_of_channels_in_binary_file,
                                                           cut_time_points_around_spike=80)

ckt.cleanup_kilosorted_data(kilosort_folder, number_of_channels_in_binary_file=number_of_channels_in_binary_file,
                            binary_data_filename=join(data_folder, amplifier_file), prb_file=prb_file,
                            type_of_binary=type_of_binary, order_of_binary='F', sampling_frequency=30000)

# ---------------------------------------------------

# Generate positions on probe
weighted_template_positions = pos.generate_probe_positions_of_templates(kilosort_folder, threshold=0.1)
np.save(join(kilosort_folder, 'weighted_template_positions.npy'), weighted_template_positions)
# ---------------------------

# REDOING TSNE ON CLEAN KILOSORT RESULTS ------------
spike_clusters = np.load(join(kilosort_folder, 'spike_clusters.npy'))
template_marking = np.load(join(kilosort_folder, 'template_marking.npy'))
spikes_used = np.array([spike_index for spike_index in np.arange(len(spike_clusters)) if template_marking[spike_clusters[spike_index]] > 0])
np.save(join(tsne_folder, 'indices_of_spikes_used.npy'), spikes_used)

template_features_sparse_clean = \
    preproc.calculate_template_features_matrix_for_tsne(kilosort_folder, save_to_folder=tsne_folder,
                                                        spikes_used_with_original_indexing=spikes_used)

template_features_sparse_clean = np.load(join(tsne_folder, 'data_to_tsne_(272886, 140).npy'))

exe_dir = r'E:\Software\Develop\Source\Repos\spikesorting_tsne_bhpart\Barnes_Hut\x64\Release'
theta = 0.2
eta = 200.0
num_dims = 2
perplexity = 100
iterations = 4000
random_seed = 1
verbose = 3
tsne = TSNE.t_sne(samples=template_features_sparse_clean, files_dir=tsne_folder, exe_dir=exe_dir, num_dims=num_dims,
                  perplexity=perplexity, theta=theta, eta=eta, iterations=iterations, random_seed=random_seed,
                  verbose=verbose)

tsne = io.load_tsne_result(tsne_folder)


# and plotting results ------------------------------
spike_clusters_clean = spike_clusters[spikes_used]
labels_dict = {}
clean_templates = np.argwhere(template_marking > 0)
total_spikes = 0
for i in np.arange(len(clean_templates)):
    spikes = np.squeeze(np.argwhere(np.in1d(spike_clusters_clean, clean_templates[i])))
    labels_dict[i] = spikes


pf.plot_tsne(tsne.T, labels_dict=labels_dict, legent_on=False)
# ---------------------------------------------------



tsne_folder = r'F:\JoanaPaired\128ch\2015-09-03\Analysis\kilosort\thres4_10_10_Fe16_Pc12\tsne\tsne_uncleaned_466Kspikes'
spike_info = preproc.generate_spike_info_from_full_tsne(kilosort_folder, tsne_folder)


# CREATE SPIKE_INFO FROM KLUSTAKWIK RESULTS AND RUN THIS IN THE TSNE SPIKESORT GUI
import numpy as np
from os.path import join
import pandas as pd
import h5py as h5
import pickle

klusta_tsne_folder = r'F:\JoanaPaired\128ch\2015-09-03\Analysis\klustakwik\threshold_6_5std'
tsne = np.load(join(klusta_tsne_folder, 't_sne_results_128820s_100per_200lr_2000it.npy'))


kwik_filename = join(klusta_tsne_folder, r'threshold_6_5std.kwik')
h5file = h5.File(kwik_filename, mode='r')
spike_times = np.array(list(h5file['channel_groups/0/spikes/time_samples']))
h5file.close()
print('All extra spikes = {}'.format(len(spike_times)))


cluster_info_file = join(klusta_tsne_folder, r'cluster_info.pkl')
pkl_file = open(cluster_info_file, 'rb')
labels = pickle.load(pkl_file)
pkl_file.close()
spike_templates = np.zeros(len(spike_times)) - 1
for template_name in labels.index:
    for spike_index in labels.loc[template_name]['Spike_Indices']:
        spike_templates[spike_index] = np.argwhere(labels.index == template_name)[0][0]

types = {0: 'Noise', 1: 'SS', 2: 'SS_Contaminated', 3: 'SS_Putative', 4: 'MUA', 5: 'Unspesified_1',
         6: 'Unspecified_2',
         7: 'Unspecified_3'}

columns = ['original_index', 'times', 'template_after_cleaning', 'type_after_cleaning', 'template_after_sorting',
           'type_after_sorting', 'template_with_all_spikes_present', 'tsne_x', 'tsne_y', 'probe_position_x',
           'probe_position_z']

spike_info = pd.DataFrame(index=np.arange(len(spike_times)), columns=columns)

spike_info['original_index'] = np.arange(len(tsne[0]))
spike_info['times'] = spike_times
spike_info['template_after_cleaning'] = spike_templates
spike_info['type_after_cleaning'] = [types[5] for i in spike_templates]
spike_info['template_after_sorting'] = spike_info['template_after_cleaning']
spike_info['type_after_sorting'] = spike_info['type_after_cleaning']
spike_info['template_with_all_spikes_present'] = [True for i in spike_templates]
spike_info['tsne_x'] = tsne[0, :]
spike_info['tsne_y'] = tsne[1, :]

spike_info.to_pickle(join(klusta_tsne_folder, 'spike_info.df'))


# COMPARE MASKED PCS BETWEEN DIFFERENT EMBEDDINGS
import numpy as np
from os.path import join
from sklearn import manifold, datasets
import matplotlib.pyplot as plt
from time import time
from matplotlib.ticker import NullFormatter
from spikesorting_tsne import io_with_cpp as io

klusta_tsne_folder = r'F:\JoanaPaired\128ch\2015-09-03\Analysis\klustakwik\threshold_6_5std'
masked_pcas_for_128k_spikes = np.load(join(klusta_tsne_folder, 'masked_pcas_for_128k_spikes.npy'))
masked_pcas_for_128k_spikes_juxta_indices = np.load(join(klusta_tsne_folder, 'masked_pcas_for_128k_spikes_juxta_indices.npy'))

n_points = 20000
masked_pcas_for_128k_spikes = masked_pcas_for_128k_spikes[:n_points, :]
masked_pcas_for_128k_spikes_juxta_indices = masked_pcas_for_128k_spikes_juxta_indices[np.argwhere(
    masked_pcas_for_128k_spikes_juxta_indices < n_points)]

X = masked_pcas_for_128k_spikes
color = np.ones(n_points) * 3
color[masked_pcas_for_128k_spikes_juxta_indices] = - 3
n_neighbors = 300
n_components = 2

spikes_folder = '20K_spikes'

methods = ['standard', 'ltsa', 'hessian', 'modified']
labels = ['LLE', 'LTSA', 'Hessian LLE', 'Modified LLE']


t0 = time()
Y = manifold.LocallyLinearEmbedding(n_neighbors, n_components,
                                    eigen_solver='auto',
                                    method=methods[0]).fit_transform(X)
t1 = time()
print("%s: %.2g sec" % (methods[0], t1 - t0))
np.save(join(klusta_tsne_folder, 'other_embeddings', spikes_folder, methods[0]+'.npy'), Y)

# Did not run--------------
t0 = time()
Y = manifold.LocallyLinearEmbedding(n_neighbors, n_components,
                                    eigen_solver='auto',
                                    method=methods[1]).fit_transform(X)
t1 = time()
print("%s: %.2g sec" % (methods[i], t1 - t0))
np.save(join(klusta_tsne_folder, 'other_embeddings', methods[1]+'.npy'), Y)

t0 = time()
Y = manifold.LocallyLinearEmbedding(n_neighbors, n_components,
                                    eigen_solver='auto',
                                    method=methods[2]).fit_transform(X)
t1 = time()
print("%s: %.2g sec" % (methods[i], t1 - t0))
np.save(join(klusta_tsne_folder, 'other_embeddings', methods[2]+'.npy'), Y)
# ------------------------

t0 = time()
Y = manifold.LocallyLinearEmbedding(n_neighbors, n_components,
                                    eigen_solver='auto',
                                    method=methods[3]).fit_transform(X)
t1 = time()
print("%s: %.2g sec" % (methods[3], t1 - t0))
np.save(join(klusta_tsne_folder, 'other_embeddings', spikes_folder, methods[3]+'.npy'), Y)





t0 = time()
Y = manifold.Isomap(n_neighbors, n_components).fit_transform(X)
t1 = time()
print("Isomap: %.2g sec" % (t1 - t0))
np.save(join(klusta_tsne_folder, 'other_embeddings', spikes_folder, 'Isomap.npy'), Y)


# Did not run
t0 = time()
mds = manifold.MDS(n_components, max_iter=100, n_init=1)
Y = mds.fit_transform(X)
t1 = time()
print("MDS: %.2g sec" % (t1 - t0))
# --------

t0 = time()
se = manifold.SpectralEmbedding(n_components=n_components,
                                n_neighbors=n_neighbors)
Y = se.fit_transform(X)
t1 = time()
print("SpectralEmbedding: %.2g sec" % (t1 - t0))
np.save(join(klusta_tsne_folder, 'other_embeddings', spikes_folder, 'SpectralEmbedding.npy'), Y)



tsne_folder = r'F:\JoanaPaired\128ch\2015-09-03\Analysis\klustakwik\threshold_6_5std'
tsne_file = r'result_128820s_100per_200lr_2000it.dat'
tsne = io.load_tsne_result(tsne_folder, tsne_file)


embeddings = ['standard', 'modified', 'T-sne', 'Isomap', 'SpectralEmbedding']
labels = ['LLE', 'Modified LLE', 'T-sne', 'Isomap', 'SpectralEmbedding']
times = [720, 1100, 2300, 460]

fig = plt.figure(figsize=(15, 8))

for i in np.arange(len(embeddings)):
    if i != 2:
        Y = np.load(join(klusta_tsne_folder, 'other_embeddings', spikes_folder, embeddings[i] + '.npy'))
    if i == 2:
        Y = tsne[:n_points, :]

    ax = fig.add_subplot(231 + i)
    plt.scatter(Y[:, 0], Y[:, 1], c=color, s=10, cmap=plt.cm.Spectral)
    plt.title("%s" % (labels[i]))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')

Y = tsne[:n_points, :]
ax = fig.add_subplot(235)
plt.scatter(Y[:, 0], Y[:, 1], c=color, s=10, cmap=plt.cm.Spectral)
plt.title("%s" % ('T-sne'))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')


pf.plot_tsne(tsne.T)


# RECREATING FIGURE 5 (KILOSORT TSNE)
from os.path import join
from BrainDataAnalysis import ploting_functions as pf
from spikesorting_tsne import io_with_cpp as io
import pandas as pd

tsne_folder = r'Z:\g\George\DataAndResults\Experiments\Anesthesia\Joana_Paired_128ch\2015-09-03\Analysis\kilosort\thres4_10_10_Fe16_Pc12\tsne'
tsne_file = r'tsne_template_features_466313sp_100per_2000its_02theta.dat'
cluster_info_file = join(tsne_folder, 'cluster_info_full_original.pkl')


tsne = io.load_tsne_result(tsne_folder, tsne_file)

cluster_info = pd.read_pickle(cluster_info_file)
labels_dict = pf.generate_labels_dict_from_cluster_info_dataframe(cluster_info)

pf.plot_tsne(tsne.T, labels_dict=labels_dict, legent_on=False)