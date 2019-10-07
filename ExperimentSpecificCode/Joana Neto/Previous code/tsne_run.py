from spikesorting_tsne import preprocessing_kilosort_results as preproc
from BrainDataAnalysis import ploting_functions as pf
from t_sne_bhcuda import tsne_cluster as tsne_cl
from spikesorting_tsne import io_with_cpp as io
from __future__ import print_function, absolute_import, division
import numpy as np
from os.path import join
import matplotlib
matplotlib.use('Qt5Agg')
import pandas as pd

base_folder = r'Z:\n\Neuroseeker Probe Recordings\Neuroseeker_2017_08_08\Analysis\kilosort\18_26_30_afterREFeachGroup'

files_dir = join(base_folder, 'Tsne_Results')



spike_info = preproc.generate_spike_info_from_full_tsne(base_folder, files_dir)



spike_templates = np.load(join(base_folder, 'spike_templates.npy'))
template_markings = np.load(join(base_folder, 'template_marking.npy'))
clean_templates_index = np.argwhere(template_markings > 0)
clean_spikes_index = np.squeeze(np.argwhere(np.in1d(spike_templates, clean_templates_index)))

np.save(join(files_dir, 'indices_of_spikes_used.npy'), clean_spikes_index)
spikes_used = np.load(join(files_dir, 'indices_of_spikes_used.npy'))

base_folder = r'Z:\n\Neuroseeker Probe Recordings\Neuroseeker_2017_08_08\Analysis\kilosort\18_26_30_afterREFeachGroup'

files_dir = join(base_folder, 'Tsne_Results')

labels_dict = pf.generate_labels_dict_from_cluster_info_dataframe(cluster_info=cluster_info)

cluster_info = create_cluster_info_from_kilosort_spike_templates(join(base_folder, 'cluster_info.pkl'),
                                                                 spike_templates_clean)


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



spike_templates = np.load(join(base_folder, 'spike_templates.npy'))
template_markings = np.load(join(base_folder, 'template_marking.npy'))
clean_templates_index = np.argwhere(template_markings > 0)
clean_spikes_index = np.squeeze(np.argwhere(np.in1d(spike_templates, clean_templates_index)))

np.save(join(files_dir, 'indices_of_spikes_used.npy'), clean_spikes_index)
spikes_used = np.load(join(files_dir, 'indices_of_spikes_used.npy'))

spike_templates_clean = spike_templates[spikes_used]

cluster_info = tsne_cl.create_cluster_info_from_kilosort_spike_templates(join(base_folder, 'cluster_info.pkl'),
                                                                         spike_templates_clean)
labels_dict = pf.generate_labels_dict_from_cluster_info_dataframe(cluster_info=cluster_info)


tsne = io.load_tsne_result(files_dir=files_dir)


base_folder = r'Z:\n\Neuroseeker Probe Recordings\Neuroseeker_2017_08_08\Analysis\kilosort\18_26_30_afterREFeachGroup'

files_dir = join(base_folder, 'Tsne_Results')

from os.path import join
import numpy as np

base_folder = r'Z:\n\Neuroseeker Probe Recordings\Neuroseeker_2017_08_08\Analysis\kilosort\18_26_30_afterREFeachGroup'

files_dir = join(base_folder, 'Tsne_Results')

np.sum(q)
indices.shape
r = np.in1d(spiketemplates,indices)
indices.shape
indices.max()
indices.min()
np.sum(q)
q.max()
t.shape
spiketemplates.shape
clean_spike_indices.shape
clean_spike_indices = np.argwhere(np.in1d(spiketemplates,indices))
indices = np.argwhere(q)
q = t>0
spiketemplates = np.load(r"F:\kilosort_ch4\kilosort\18_26_30_afterREFeachGroup\spike_templates.npy")
t= np.load(r"F:\kilosort_ch4\kilosort\18_26_30_afterREFeachGroup\template_marking.npy")



cl.cleanup_kilosorted_data(r'F:\kilosort_ch4\kilosort\18_26_30_afterREFeachGroup', 1440, r'F:\kilosort_ch4\18_26_30_afterREFeachGroup.bin', r'F:\kilosort_ch4\prb.txt',num_of_shanks_for_vis=3)
