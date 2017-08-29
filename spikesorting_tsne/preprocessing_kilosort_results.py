

import numpy as np
from os.path import join, isfile
import pickle
import pandas as pd
from . import io_with_cpp as io


def spike_indices_of_template(spike_templates, clean_templates, clean_template_index):
    original_template_index = clean_templates[clean_template_index]
    spikes_indices = np.squeeze(np.argwhere(np.in1d(spike_templates, original_template_index)))
    return spikes_indices


def find_templates_with_number_of_spikes_under_threshold(spike_templates, clean_templates, threshold):

    small_clean_templates_with_indices = dict()
    large_clean_templates_with_indices = dict()
    num_of_spikes_in_small_templates = 0
    for i in np.arange(len(clean_templates)):
        spike_indices = spike_indices_of_template(spike_templates, clean_templates, i)
        num_of_spikes = spike_indices.size
        if num_of_spikes <= threshold:
            small_clean_templates_with_indices[i] = spike_indices
            num_of_spikes_in_small_templates += num_of_spikes
        else:
            large_clean_templates_with_indices[i] = spike_indices

    return small_clean_templates_with_indices, num_of_spikes_in_small_templates, large_clean_templates_with_indices


def get_template_marking(base_folder):
    # If there is no template_marking.npy then all templates are considered good
    if isfile(join(base_folder, 'template_marking.npy')):
        template_marking = np.load(join(base_folder, 'template_marking.npy'))
    else:
        templates = np.load(join(base_folder, 'templates.npy'))
        num_of_templates = templates.shape[2]
        template_marking = np.ones(num_of_templates)
        del templates
        print('No template_marking.npy found. Using all templates.')

    return template_marking


def find_spike_indices_for_representative_tsne(base_folder, save_to_folder, threshold, total_spikes_required):

    spike_templates = np.load(join(base_folder, 'spike_templates.npy'))
    template_marking = get_template_marking(base_folder)

    clean_templates = np.argwhere(template_marking)

    small_clean_templates_with_spike_indices, num_of_spikes_in_small_templates, large_clean_templates_with_spike_indices = \
        find_templates_with_number_of_spikes_under_threshold(spike_templates, clean_templates, threshold)

    extra_spikes_required = total_spikes_required - num_of_spikes_in_small_templates

    spikes_clean_index = np.squeeze(np.argwhere(np.in1d(spike_templates, clean_templates)))
    percentage_of_kept_spikes_in_large_templates = extra_spikes_required / (spikes_clean_index.size -
                                                                            num_of_spikes_in_small_templates)

    spikes_chosen = []
    num_of_spikes_in_large_templates = 0
    for large_template_index in large_clean_templates_with_spike_indices.keys():
        spike_indices = large_clean_templates_with_spike_indices[large_template_index]
        num_of_spikes = spike_indices.size
        num_of_spikes_in_large_templates += num_of_spikes
        chosen_num_of_spikes = int(num_of_spikes * percentage_of_kept_spikes_in_large_templates)
        chosen_spike_indices = np.random.choice(spike_indices, chosen_num_of_spikes, replace=False)
        spikes_chosen.append(chosen_spike_indices)

    num_of_spikes_in_small_templates = 0
    for small_template_index in small_clean_templates_with_spike_indices.keys():
        spike_indices = small_clean_templates_with_spike_indices[small_template_index]
        spikes_chosen.append(spike_indices)
        num_of_spikes_in_small_templates += spike_indices.size

    print('{} templates with more than {} spikes (total spikes in those = {}). \n{} templates with less '
          'than {} spikes (total spikes in those = {})'.format(len(large_clean_templates_with_spike_indices),
                                                                         threshold, num_of_spikes_in_large_templates,
                                                                         len(small_clean_templates_with_spike_indices),
                                                                         threshold, num_of_spikes_in_small_templates))

    spikes_chosen_flat = []
    for sublist in spikes_chosen:
        if np.size(sublist) > 1:
            for item in sublist:
                spikes_chosen_flat.append(item)
        else:
            spikes_chosen_flat.append(sublist)

    spikes_chosen_flat = np.array(spikes_chosen_flat)
    small_clean_templates_indices = np.fromiter(small_clean_templates_with_spike_indices.keys(), int,
                                                len(small_clean_templates_with_spike_indices))
    large_clean_tempalates_indices = np.fromiter(large_clean_templates_with_spike_indices.keys(), int,
                                                 len(large_clean_templates_with_spike_indices))
    pickle.dump(small_clean_templates_with_spike_indices,
                open(join(save_to_folder, "small_clean_templates_with_spike_indices.pkl"), "wb"))
    pickle.dump(large_clean_templates_with_spike_indices,
                open(join(save_to_folder, "large_clean_templates_with_spike_indices.pkl"), "wb"))
    np.save(join(save_to_folder, 'indices_of_spikes_used'), spikes_chosen_flat)
    np.save(join(save_to_folder, 'indices_of_small_templates'), small_clean_templates_indices)
    np.save(join(save_to_folder, 'indices_of_large_templates'), large_clean_tempalates_indices)

    return spikes_chosen_flat, small_clean_templates_with_spike_indices, large_clean_templates_with_spike_indices


def calculate_template_features_matrix_for_tsne(base_folder, save_to_folder, spikes_used_with_original_indexing=None,
                                                spikes_used_wth_clean_indexing=None):

    spike_templates = np.load(join(base_folder, 'spike_templates.npy'))
    template_features = np.load(join(base_folder, 'template_features.npy'))
    template_features_ind = np.load(join(base_folder, 'template_feature_ind.npy'))

    template_marking = get_template_marking(base_folder)

    clean_templates = np.argwhere(template_marking)
    spikes_clean_index = np.squeeze(np.argwhere(np.in1d(spike_templates, clean_templates)))

    if spikes_used_with_original_indexing is not None and spikes_used_wth_clean_indexing is not None:
        print('Use one of the spikes_used_... variable')
        return None
    elif spikes_used_with_original_indexing is None and spikes_used_wth_clean_indexing is None:
        pass
    elif spikes_used_with_original_indexing is None and spikes_used_wth_clean_indexing is not None:
        spikes_clean_index = spikes_clean_index[spikes_used_wth_clean_indexing]
    elif spikes_used_with_original_indexing is not None and spikes_used_wth_clean_indexing is None:
        spikes_clean_index = spikes_used_with_original_indexing

    clean_templates = np.unique(spike_templates[spikes_clean_index])

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
                                           np.argwhere(clean_templates == indices[i])] = template_features[spike, i]

    np.save(join(save_to_folder, 'data_to_tsne_' + str(template_features_sparse_clean.shape) + '.npy'),
            template_features_sparse_clean)

    return template_features_sparse_clean


def load_template_features_matrix_for_tsne(save_to_folder, shape):

    template_features_sparse_clean = np.load(join(save_to_folder, r'data_to_tsne_' + str(shape) + '.npy'))
    return template_features_sparse_clean


def generate_spike_info(base_folder, files_dir):
    spikes_used = np.load(join(files_dir, 'indices_of_spikes_used.npy'))
    template_marking = np.load(join(base_folder, 'template_marking.npy'))
    spike_templates = np.load(join(base_folder, 'spike_templates.npy'))[spikes_used]
    spike_times = np.load(join(base_folder, 'spike_times.npy'))[spikes_used]
    indices_of_small_templates = np.load(join(files_dir, 'indices_of_small_templates.npy'))
    tsne = io.load_tsne_result(join(base_folder, files_dir))

    types = {0: 'Noise', 1: 'SS', 2: 'SS_Contaminated', 3: 'SS_Putative', 4: 'MUA', 5: 'Unspesified_1',
             6: 'Unspecified_2',
             7: 'Unspecified_3'}

    columns = ['original_index', 'times', 'template_after_cleaning', 'type_after_cleaning', 'template_after_sorting',
               'type_after_sorting', 'template_with_all_spikes_present', 'tsne_x', 'tsne_y', 'probe_position_x',
               'probe_position_z']

    spike_info = pd.DataFrame(index=np.arange(spikes_used.size), columns=columns)

    spike_info['original_index'] = spikes_used
    spike_info['times'] = spike_times
    spike_info['template_after_cleaning'] = spike_templates
    spike_info['type_after_cleaning'] = [types[int(template_marking[i])] for i in spike_templates]
    spike_info['template_after_sorting'] = spike_info['template_after_cleaning']
    spike_info['type_after_sorting'] = spike_info['type_after_cleaning']
    spike_info['template_with_all_spikes_present'] = [bool(np.in1d(spike_template, indices_of_small_templates))
                                                      for spike_template in spike_templates]
    spike_info['tsne_x'] = tsne[:, 0]
    spike_info['tsne_y'] = tsne[:, 1]

    spike_info.to_pickle(join(files_dir, 'spike_info.df'))

    return spike_info
