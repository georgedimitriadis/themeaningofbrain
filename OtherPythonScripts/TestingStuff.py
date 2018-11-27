


import numpy as np
import pandas as pd
import os.path as ospath
import matplotlib.pyplot as plt

main_path = r'D:\Data\George\Projects\SpikeSorting\Joana_Paired_128ch\2015-09-03\Analysis\klustakwik\threshold_6_5std'
cluster_filename = ospath.join(main_path, 'cluster_info.pkl')
cluster_info = pd.read_pickle(cluster_filename)


num_of_clusters = cluster_info.shape[0]
indices_list_of_lists = cluster_info['Spike_Indices'].tolist()
indices = [item for sublist in indices_list_of_lists for item in sublist]
cluster_indices = np.arange(num_of_clusters)
colors = np.empty((len(cluster_indices), 4))
for c in cluster_indices:
    r = np.random.random(size=1).astype(np.float32)[0]
    g = np.random.random(size=1).astype(np.float32)[0]
    for i in np.arange(len(indices_list_of_lists[c])):
        colors[c] = np.array([r, g, 0.2, 1]).astype(np.float32)


tsne = np.load(ospath.join(main_path, r't_sne_results_3d_theta02.npy'))
t_sne = tsne.T

in_cluster = np.empty((len(t_sne)))
for t in range(len(t_sne)):
    in_cluster[t] = 0
    for i in cluster_indices:
        if np.in1d(t, indices_list_of_lists[i]).any():
            in_cluster[t] = i
            break


colored_3d_tsne = np.empty((len(t_sne), 7)).astype(np.float32)
for t in range(t_sne.shape[0]):
    black = np.array([0.1, 0.1, 0.1, 1.0]).astype(np.float32)
    colored_3d_tsne[t, :3] = t_sne[t]
    colored_3d_tsne[t, 3:] = black
    if in_cluster[t]:
        colored_3d_tsne[t, 3:] = colors[in_cluster[t]]
colored_3d_tsne_T = colored_3d_tsne.T


#binary array with timestamp as the 8th column
import h5py as h5
from os.path import join
kwik_filename = join(main_path, 'threshold_6_5std.kwik')
h5file = h5.File(kwik_filename, mode='r')
time_samples_h5_dir = r'channel_groups/0/spikes/time_samples'
all_extra_spike_times = np.array(list(h5file[time_samples_h5_dir]))
colored_3d_tsne = np.empty((len(t_sne), 8)).astype(np.float32)
for t in range(t_sne.shape[0]):
    black = np.array([0.1, 0.1, 0.1, 1.0]).astype(np.float32)
    colored_3d_tsne[t, :3] = t_sne[t]
    colored_3d_tsne[t, 3:7] = black
    if in_cluster[t]:
        colored_3d_tsne[t, 3:7] = colors[in_cluster[t]]
    colored_3d_tsne[t, 7] = all_extra_spike_times[t]
colored_3d_tsne_T = colored_3d_tsne.T



file_of_colored_3d_tsne = ospath.join(main_path, r'colored_3d_tsne_withtimes.dat')
colored_3d_tsne.tofile(file_of_colored_3d_tsne)




fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(tsne[0], tsne[1], tsne[2], zdir='z', s=20, c='b', depthshade=True)


# -----------------------------------------------------------------------------
# make 3d binary file from cell positions npy array
file = r'C:\Users\KAMPFF-LAB_ANALYSIS2\Downloads\positions.npy'
array = np.load(file)
array_baselined = array
for i in np.arange(1,4):
    array_baselined[:, i] = array[:, i] - np.mean(array[:, i])
types = array[:, 0]
cluster_indices = np.arange(np.min(types)-1, np.max(types))
colors = np.empty((len(cluster_indices), 4))
for c in cluster_indices:
    r = np.random.random(size=1).astype(np.float32)[0]
    g = np.random.random(size=1).astype(np.float32)[0]
    b = np.random.random(size=1).astype(np.float32)[0]
    colors[c] = np.array([r, g, b, 1]).astype(np.float32)



colored_array = np.empty((len(array), 7)).astype(np.float32)
for t in range(len(array)):
    colored_array[t, :3] = array[t, 1:4]/10
    colored_array[t, 3:] = colors[types[t]-1]
colored_array.tofile(r'E:\George\Temporary\cell_positions.bin')

import glob


# Plot TSNE pciture(s)
def plot_interim_tsne_results(folder_path):
    os.chdir(folder_path)
    #os.mkdir(folder_path + '\interim_frames')
    plt.figure('Summary',  figsize=(12, 8), dpi=120, facecolor='w', edgecolor='k')
    count = 1
    for file in glob.glob("interim*.dat"):
        print(file)
        tsne_data = TSNE.load_tsne_result(folder_path, file)
        tsne_data = np.transpose(tsne_data)
        plt.cla()
        plt.plot(tsne_data[0, :], tsne_data[1, :], '.', MarkerSize = 2, Color = [0.0, 0.0, 0.0, 0.25])
        minX = np.min(tsne_data[0, :])
        maxX = np.max(tsne_data[0, :])
        minY = np.min(tsne_data[1, :])
        maxY = np.max(tsne_data[1, :])
        rangeX = np.max(np.abs([minX, maxX]))
        rangeY = np.max(np.abs([minY, maxY]))
        #print(rangeX)
        plt.ylim([-rangeY, rangeY])
        plt.xlim([-rangeX, rangeX])
        plt.savefig(folder_path + '/interim_frames'+'/'+ file[:-4] +'.png')
        count = count + 1

# Plot TSNE pciture(s) with labels
def plot_interim_tsne_results_with_labels(folder_path):
    # Import labels
    pkl_file = open(label_path, 'rb')
    labels = pickle.load(pkl_file)
    pkl_file.close()
    juxta = labels.loc['Juxta']
    juxta_indices = juxta['Spike_Indices']
    juxta_indices = juxta_indices[juxta_indices < len(spike_indices)]
    num_labels = len(labels)
    label_indices = labels['Spike_Indices']
    color_indices = plt.Normalize(0, num_labels)
    cm = plt.cm.gist_ncar

    os.chdir(folder_path)
    #os.mkdir(folder_path + '\interim_frames')
    plt.figure('Summary',  figsize=(12, 8), dpi=120, facecolor='w', edgecolor='k')
    interim_dat_files = glob.glob("interim*.dat")
    for c in range(1,len(interim_dat_files), 1):
        file = interim_dat_files[c]
        print(file)
        tsne_data = TSNE.load_tsne_result(folder_path, file)
        tsne_data = np.transpose(tsne_data)
        plt.cla()
        plt.plot(tsne_data[0, :], tsne_data[1, :], '.', MarkerSize = 2, Color = [0.0, 0.0, 0.0, 0.15])
        #plt.plot(tsne_data[0, juxta_indices], tsne_data[1, juxta_indices], '.', MarkerSize = 2, Color = [1.0, 0.0, 0.0, 0.3])
        for g in range(0, num_labels):
            l_idx = label_indices[g]
            l_idx = l_idx[l_idx < len(spike_indices)]
            plt.plot(tsne_data[0][l_idx], tsne_data[1][l_idx], '.', MarkerSize=2, Color=cm(color_indices(g)), Alpha=0.25)

        minX = np.min(tsne_data[0, :])
        maxX = np.max(tsne_data[0, :])
        minY = np.min(tsne_data[1, :])
        maxY = np.max(tsne_data[1, :])
        rangeX = np.max(np.abs([minX, maxX]))
        rangeY = np.max(np.abs([minY, maxY]))
        #print(rangeX)
        plt.ylim([-rangeY, rangeY])
        plt.xlim([-rangeX, rangeX])
        plt.savefig(folder_path + '/interim_frames'+'/'+ file[:-4] +'.png')







# ======================================================================================================================
# Running t-sne manual clustering

import numpy as np
import h5py as h5
import os
from ExperimentSpecificCode._2016_TSne_Paper.t_sne_bhcuda import bhtsne_cuda as TSNE, tsne_cluster

filename = r'D:\Data\George\Projects\SpikeSorting\Joana_Paired_128ch\2015-09-03\Analysis\klustakwik\threshold_6_5std\threshold_6_5std.kwik'
h5file = h5.File(filename, mode='r')
spike_times_phy = np.array(list(h5file['channel_groups/0/spikes/time_samples']))
h5file.close()


kilosort_experiment_folder = r'thres4_10_10_Fe16_Pc12'  # thres4_10_10_Fe16_Pc12 OR thres4_10_10_Fe256_Pc128 OR thres6_12_12_Fe256_Pc128
kilosort_path = os.path.join(r'D:\Data\George\Projects\SpikeSorting\Joana_Paired_128ch\2015-09-03\Analysis\kilosort',
                             kilosort_experiment_folder)
spike_clusters_kilosort = np.load(os.path.join(kilosort_path, 'kilosort output\\spike_templates.npy'))
spike_times_kilosort = np.load(os.path.join(kilosort_path, 'kilosort output\spike_times.npy'))
template_features = np.load(os.path.join(kilosort_path, 'kilosort output\\template_features.npy'))
template_features_ind = np.load(os.path.join(kilosort_path, 'kilosort output\\template_feature_ind.npy'))

template_features_tsne = TSNE.load_tsne_result(os.path.join(kilosort_path, 'tsne'),
                                               'tsne_template_features_466313sp_100per_2000its_02theta.dat')
pc_features_tsne = TSNE.load_tsne_result(os.path.join(kilosort_path, r'tsne\pc_features_results'),
                                         'tsne_pc_features_466ksp_per100_theta02_it2k_rs1.dat')


spikes_used = np.arange(len(template_features_tsne))#np.arange(150000) #np.random.choice(np.arange(len(template_features_tsne)), 150000) #np.arange(len(template_features_tsne))
template_features_tsne = np.transpose(np.array(template_features_tsne)[spikes_used])

pc_features_tsne = np.transpose(np.array(pc_features_tsne)[spikes_used])

spike_clusters_kilosort = spike_clusters_kilosort[spikes_used]
spike_times_kilosort = spike_times_kilosort[spikes_used]

cluster_info_filename = os.path.join(kilosort_path, 'tsne\\cluster_info_full_final.pkl')
#cluster_info = tsne_cluster.create_cluster_info_from_kilosort_spike_templates(cluster_info_filename,
#                                                                              spike_clusters_kilosort)


import IO.ephys as ephys
import numpy as np
import os

# Parameters for testing (128 channels)
base_dir = r'D:\Data\George\Projects\SpikeSorting\Joana_Paired_128ch\2015-09-03'
tsne_dir = r'Analysis\klustakwik\threshold_6_5std'
data_dir = r'Data'
data_cube_dir = r'Analysis\TempCube'
kilosort_experiment_folder = r'thres4_10_10_Fe16_Pc12'  # thres4_10_10_Fe16_Pc12 OR thres4_10_10_Fe256_Pc128 OR thres6_12_12_Fe256_Pc128
kilosort_path = os.path.join(r'D:\Data\George\Projects\SpikeSorting\Joana_Paired_128ch\2015-09-03\Analysis\kilosort',
                             kilosort_experiment_folder)


num_ivm_channels = 128
amp_dtype = np.uint16
cube_type = np.int32
sampling_freq = 30000
num_of_points_in_spike_trig = 64
num_of_points_for_baseline = 10

data_cube_filename = os.path.join(kilosort_path, 'tsne\\raw_data_cube_for_pcs.npy')

autocor_bin_number = 100
prb_file = r'D:\Data\George\Projects\SpikeSorting\Joana_Paired_128ch\2015-09-03\Analysis\klustakwik\128ch_passive_imec.prb'

spike_indices_to_use = None
num_of_spikes = np.shape(template_features_tsne)[1]

shape_of_cut_extracellular_data = (num_ivm_channels, num_of_points_in_spike_trig, num_of_spikes)

raw_data_filename = os.path.join(base_dir, data_dir, 'amplifier2015-09-03T21_18_47.bin')
raw_data_ivm = ephys.load_raw_data(raw_data_filename, numchannels=num_ivm_channels, dtype=amp_dtype).dataMatrix

# manual clustering the kilosort data
spike_indices_to_use = None
num_of_spikes = len(spikes_used)

shape_of_cut_extracellular_data = (num_ivm_channels, num_of_points_in_spike_trig, num_of_spikes)

raw_data_filename = os.path.join(base_dir, data_dir, 'amplifier2015-09-03T21_18_47.bin')
raw_data_ivm = ephys.load_raw_data(raw_data_filename, numchannels=num_ivm_channels, dtype=amp_dtype).dataMatrix

tsne_cluster.gui_manual_cluster_tsne_spikes(tsne_array_or_filename=template_features_tsne,
                                            spike_times_list_or_filename=np.reshape(spike_times_kilosort,
                                                                                    (len(spike_times_kilosort))),
                                            raw_extracellular_data=None,
                                            num_of_points_for_baseline=num_of_points_for_baseline,
                                            cut_extracellular_data_or_filename=data_cube_filename,
                                            shape_of_cut_extracellular_data=shape_of_cut_extracellular_data,
                                            cube_type=cube_type,
                                            sampling_freq=sampling_freq,
                                            autocor_bin_number=autocor_bin_number,
                                            cluster_info_file=cluster_info_filename,
                                            use_existing_cluster=True,
                                            spike_indices_to_use=spike_indices_to_use,
                                            prb_file=prb_file,
                                            k4=True,
                                            verbose=True)




# ======================================================================================================================
# Test code to sort out the SHIFT bug in Bokeh
from bokeh.client import push_session
from bokeh.layouts import column
from bokeh.models import BoxSelectTool, LassoSelectTool, ColumnDataSource, Circle
from bokeh.plotting import curdoc, figure
from bokeh.models.widgets import Button
import numpy as np

global previously_selected_spike_indices
global previous_tsne_source_selected

# Setup data
tsne = [np.random.random(50), np.random.random(50)]
tsne_figure_size = [500, 500]

# Scatter plot
tsne_fig_tools = "pan,wheel_zoom,box_zoom,box_select,lasso_select,tap,resize,reset,save"
tsne_figure = figure(tools=tsne_fig_tools, plot_width=tsne_figure_size[0], plot_height=tsne_figure_size[1],
                     title='T-sne', min_border=10, min_border_left=50, webgl=True)

tsne_source = ColumnDataSource({'tsne-x': tsne[0], 'tsne-y': tsne[1]})

tsne_selected_points_glyph = Circle(x='tsne-x', y='tsne-y', size=7, line_alpha=0, fill_alpha=1, fill_color='red')
tsne_nonselected_points_glyph = Circle(x='tsne-x', y='tsne-y', size=7,
                                       line_alpha=0, fill_alpha=1, fill_color='blue')

tsne_glyph_renderer = tsne_figure.add_glyph(tsne_source, tsne_nonselected_points_glyph,
                                            selection_glyph=tsne_selected_points_glyph,
                                            nonselection_glyph=tsne_nonselected_points_glyph,
                                            name='tsne_nonselected_glyph_renderer')

tsne_figure.select(BoxSelectTool).select_every_mousemove = False
tsne_figure.select(LassoSelectTool).select_every_mousemove = False


def on_tsne_data_update(attr, old, new):
    global previously_selected_spike_indices

    previously_selected_spike_indices = np.array(old['1d']['indices'])


tsne_source.on_change('selected', on_tsne_data_update)



# Undo button
undo_selected_points_button = Button(label='Undo last selection')

def on_button_undo_selection():
    global previously_selected_spike_indices
    tsne_source.data = {'tsne-x': tsne[0], 'tsne-y': tsne[1]}
    tsne_source.selected['1d']['indices'] = previously_selected_spike_indices
    old = new = tsne_source.selected
    tsne_source.trigger('selected', old, new)

undo_selected_points_button.on_click(on_button_undo_selection)


# Layout
lay = column(tsne_figure, undo_selected_points_button)

session = push_session(curdoc())
session.show(lay)  # open the document in a browser
session.loop_until_closed()

# ======================================================================================================================

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
import os
from t_sne_bhcuda import tsne_cluster
from t_sne_bhcuda import bhtsne_cuda as TSNE
import pickle
import pandas as pd
import BrainDataAnalysis.ploting_functions as pf


kilosort_experiment_folder = r'thres4_10_10_Fe16_Pc12'  # thres4_10_10_Fe16_Pc12 OR thres4_10_10_Fe256_Pc128 OR thres6_12_12_Fe256_Pc128
kilosort_path = os.path.join(r'D:\Data\George\Projects\SpikeSorting\Joana_Paired_128ch\2015-09-03\Analysis\kilosort',
                             kilosort_experiment_folder)
spike_times_kilosort = np.load(os.path.join(kilosort_path, 'kilosort output\spike_times.npy'))
cluster_info_filename = os.path.join(kilosort_path, 'tsne\\cluster_info_150k_workedOnIt.pkl')
cluster_info = pd.read_pickle(cluster_info_filename)

data = pd.concat([cluster_info.loc['10':'10'],
                  cluster_info.loc['10a':'10a'],
                  cluster_info.loc['10b':'10b'],
                  cluster_info.loc['10c':'10c'],
                  cluster_info.loc['10d':'10d'],
                  cluster_info.loc['10e':'10e']])

indices_array_of_arrays = np.array(data.Spike_Indices.tolist())
spike_times_array_of_arrays = [spike_times_kilosort[i] for i in indices_array_of_arrays]

indices_all = np.concatenate(data.Spike_Indices.tolist()).ravel()
spike_times_of_indices_all = spike_times_kilosort[indices_all]


tsne = TSNE.load_tsne_result(os.path.join(kilosort_path, 'tsne'),
                                          'tsne_template_features_466313sp_100per_2000its_02theta.dat')

tsne_indices_all = np.array(tsne)[indices_all]

# max + [:, 1] = top, min + [:, 1] = bottom, max + [:, 0] = right, min + [:, 0] = left
starting_point_index = np.argmin(tsne_indices_all[:, 1])
distances_to_starting_point = [distance.euclidean(tsne_indices_all[starting_point_index], tsne_indices_all[i]) \
                               for i in np.arange(len(indices_all))]
local_indices_sorted_to_distance = [i[0] for i in sorted(enumerate(distances_to_starting_point), key=lambda x:x[1])]


middle_bin_size = []
for index in np.arange(len(local_indices_sorted_to_distance)):
    spike_times_used = spike_times_of_indices_all[local_indices_sorted_to_distance[index:]]
    diffs, norm = tsne_cluster.crosscorrelate_spike_trains(spike_times_used.astype(np.int64),
                                                           spike_times_used.astype(np.int64), lag=1500)
    hist, edges = np.histogram(diffs, bins=autocor_bin_number)
    middle_bin_size.append(hist[49])
    if hist[49] == 0:
        break

spike_times_used = spike_times_of_indices_all[local_indices_sorted_to_distance[len(middle_bin_size):]]
diffs, norm = tsne_cluster.crosscorrelate_spike_trains(spike_times_used.astype(np.int64),
                                                       spike_times_used.astype(np.int64),
                                                       lag=1500)

autocor_bin_number = 100
hist, edges = np.histogram(diffs, bins=autocor_bin_number)



# T-sne again just the selected blob
template_features_sparse = np.zeros((template_features.shape[0], template_features_ind.shape[0]))
for spike in np.arange(template_features.shape[0]):
    cluster = spike_clusters_kilosort[spike][0]
    indices = template_features_ind[cluster, :]
    for i in np.arange(len(indices)):
        template_features_sparse[spike, indices[i]] = template_features[spike, i]

template_features_tsne_selected_indices = TSNE.t_sne(template_features_sparse[indices_all], perplexity=100, theta=0.2,
                                        files_dir=os.path.join(kilosort_path, 'tsne'),
                                        iterations=2000, randseed=1)


pc_features = np.load(os.path.join(kilosort_path, 'kilosort output\\pc_features.npy'))
pc_features_ind = np.load(os.path.join(kilosort_path, 'kilosort output\\pc_feature_ind.npy'))
pc_features_sparse = np.zeros((pc_features.shape[0], pc_features.shape[1], pc_features_ind.shape[0]))
for spike in np.arange(template_features.shape[0]):
    cluster = spike_clusters_kilosort[spike][0]
    indices = pc_features_ind[cluster, :]
    for i in np.arange(len(indices)):
        for pc in np.arange(3):
            pc_features_sparse[spike, pc, indices[i]] = pc_features[spike, pc, i]
pc_features_sparse_flatten = np.reshape(pc_features_sparse, (pc_features.shape[0], pc_features.shape[1] * pc_features_ind.shape[0]))

pc_features_tsne_selected_indices = TSNE.t_sne(pc_features_sparse_flatten[indices_all[:-10]],
                              perplexity=20, theta=0.01,
                              files_dir=os.path.join(kilosort_path, 'tsne'),
                              results_filename='tsne_pc_features_blob.npy',
                              gpu_mem=0.8, iterations=2000, randseed=1)


pc_and_template_features_sparse = np.concatenate((pc_features_sparse_flatten, template_features_sparse), axis=1)

pc_and_template_features_tsne_selected_indices = TSNE.t_sne(pc_and_template_features_sparse[indices_all],
                              perplexity=100, theta=0.2,
                              files_dir=os.path.join(kilosort_path, 'tsne'),
                              results_filename='tsne_pc_template_features_blob.npy',
                              gpu_mem=0.8, iterations=2000, randseed=1)

pf.plot_tsne(np.transpose(pc_and_template_features_tsne_selected_indices), legend_on=False,
             subtitle='T-sne', cm=plt.cm.coolwarm, sizes=[2, 8])

plt.bar(edges[:-1], hist, width=30)

result = plt.hist(diffs, bins=autocor_bin_number)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(tsne_indices_all[:, 0],
            tsne_indices_all[:, 1],
           color='blue')
ax.scatter(tsne_indices_all[local_indices_sorted_to_distance[len(middle_bin_size):], 0],
            tsne_indices_all[local_indices_sorted_to_distance[len(middle_bin_size):], 1],
           color='red')









# Create a data set with only the selected spikes
import IO.ephys as ephys
import BrainDataAnalysis.filters as filters

base_dir = r'D:\Data\George\Projects\SpikeSorting\Joana_Paired_128ch\2015-09-03'
data_dir = r'Data'
raw_data_filename = os.path.join(base_dir, data_dir, 'amplifier2015-09-03T21_18_47.bin')

raw_data_ivm = ephys.load_raw_data(raw_data_filename, numchannels=num_ivm_channels, dtype=amp_dtype).dataMatrix
data_cube_filename = os.path.join(kilosort_path, 'tsne\\raw_data_cube_of_selected_spikes.npy')
num_ivm_channels = 128
amp_dtype = np.uint16
cube_type = np.int32
sampling_freq = 30000
num_of_points_in_spike_trig = 64
num_of_points_for_baseline = 10

selected_spikes_data_cube = tsne_cluster.create_data_cube_from_raw_extra_data(raw_data_ivm, data_cube_filename,
                                                                              num_ivm_channels,
                                                                              num_of_points_in_spike_trig,
                                                                              cube_type, spike_times_of_indices_all,
                                                                              num_of_points_in_spike_trig - 1)

selected_spikes_concatenated = np.reshape(selected_spikes_data_cube, ((num_ivm_channels *
                                                                      num_of_points_in_spike_trig),
                                                                      len(spike_times_of_indices_all))).transpose()

selected_spikes_data_cube_filtered = np.zeros((num_ivm_channels, num_of_points_in_spike_trig,
                                                 len(spike_times_of_indices_all)))
for spike in np.arange(len(spike_times_of_indices_all)):
    selected_spikes_data_cube_filtered[:, :, spike] = filters.high_pass_filter(selected_spikes_data_cube[:, :, spike],
                                                                                  30000, 30000 / 128) # removes the slant in each trial

selected_spikes_concatenated_filtered = np.reshape(selected_spikes_data_cube_filtered, ((num_ivm_channels *
                                                                      num_of_points_in_spike_trig),
                                                                      len(spike_times_of_indices_all))).transpose()

from sklearn.neural_network import BernoulliRBM as RBM

X = np.zeros((selected_spikes_concatenated_filtered.shape[0], selected_spikes_concatenated_filtered.shape[1]))
for i in np.arange(selected_spikes_concatenated_filtered.shape[0]):
    X[i, :] = (selected_spikes_concatenated_filtered[i, :] - np.min(selected_spikes_concatenated_filtered[i, :], 0)) / (np.max(selected_spikes_concatenated_filtered[i, :], 0) - np.min(selected_spikes_concatenated_filtered[i, :], 0))  # 0-1 scaling

rbm = RBM(n_components=200, learning_rate=0.05, batch_size=100, n_iter=500)
rbm.fit(X.transpose())

rbm_features_tsne_selected_indices = TSNE.t_sne(rbm.components_.transpose(),
                                              perplexity=100, theta=0.1,
                                              files_dir=os.path.join(kilosort_path, 'tsne'),
                                              gpu_mem=0.8, iterations=2000, randseed=1)
pf.plot_tsne(np.transpose(rbm_features_tsne_selected_indices), legend_on=False,
             subtitle='T-sne', cm=plt.cm.coolwarm, sizes=[2, 8])




# MAKING THE T-SNE PAPER'S LAST FIGURE
indices_list_of_lists = []
# Group 1 = All the easy to cluster SUs
clusters_su_easy = np.char.mod('%d', [0,2,5,6,11,12,16,17,24,25,28,29,31,37,43,48,55,56,65,66,68,69,75,79,81,83,84,87,
                                      89, 92, 96, 97,110,117,118,119,120,123,124,131,133,138,139,141,142,144,148,153,
                                      154, 155, 157, 162, 163, 166, 173,174,175,178,179, 180,181,188,197,198,202,208,
                                      211,212,216,228,230,233,243,254])

cluster_info_su_easy = cluster_info[cluster_info.index.isin(clusters_su_easy)]
indices_list_of_lists.append(np.concatenate(cluster_info_su_easy.Spike_Indices.real).ravel())

# Group 2 = All the impossible to cluster MUs
clusters_mu = np.char.mod('%d', [7,10,13,14,18,21,30,36,38,40,41,45,46,53,57,59,60,67,74,95,101,104,107,108,112,113,
                                 134,135,156,158,169,205,207,209,210,215,218,221,244])

cluster_info_mu = cluster_info[cluster_info.index.isin(clusters_mu)]
indices_list_of_lists.append(np.concatenate(cluster_info_mu.Spike_Indices.real).ravel())

# Group 3 = All the Noise
clusters_noise = 'NOISE'

cluster_info_noise = cluster_info[cluster_info.index == clusters_noise]
indices_list_of_lists.append(np.concatenate(cluster_info_noise.Spike_Indices.real).ravel())

# Group 4 = All the unassigned
clusters_noise = 'UNLABELED'

cluster_info_noise = cluster_info[cluster_info.index == clusters_noise]
indices_list_of_lists.append(np.concatenate(cluster_info_noise.Spike_Indices.real).ravel())

# Group 5 = All the merges and splits
clusters_su_sam = ['3', '8', '177', '19', '44', '32', '32a', '50', '50a', '50b', '56a', '68a', '70', '70b', '70c',
                   '70d', '73', '73a', '15a', '15b', '99', '99a', '105', '174a']

cluster_info_su_sam = cluster_info[cluster_info.index.isin(clusters_su_sam)]
indices_list_of_lists.append(np.concatenate(cluster_info_su_sam.Spike_Indices.real).ravel())

labels_dict = {}
for c in np.arange(len(indices_list_of_lists)):
    labels_dict[c] = indices_list_of_lists[c]

pf.plot_tsne(np.transpose(template_features_tsne), labels_dict, sizes=[1, 3], markers = ['.', 'o'], legend_on=False,
             cm=plt.cm.gnuplot, cm_remapping=[0, 1, 2, 3, 4],
             label_array=np.array(['Simple SUs', 'MUs', 'NOISE', 'UNLABELED', 'SUs FROM MERGES AND SPLITS', 'NONE']).astype(str))













import numpy as np
import psutil
import time

debug_matrix_file = r'E:\Data\debug_matrix.dat'
debug_matrix_dimensions = (10**4, 10**4)

debug_matrix_mm = np.memmap(debug_matrix_file, np.float32, mode='w+', shape=debug_matrix_dimensions)

for i in range(debug_matrix_mm.shape[0]):
    debug_matrix_mm[i, :] = np.random.random(debug_matrix_mm.shape[1])
print('Finished making matrix')

t = time.clock()
np_cov = np.cov(debug_matrix_mm)
print('Full numpy covariance took ' + str(time.clock() - t) + ' secs')


def calculate_number_of_elements_fitting_in_ram():
    remaining_memory = psutil.virtual_memory()[1]
    mem_to_use = 0.7 * remaining_memory

    total_elements_to_load_to_ram = int(mem_to_use / np.dtype(np.float32).itemsize)

    # Use this to simulate smaller ram than what is in the computer for debugging
    total_elements_to_load_to_ram = 1000

    elements_of_matrix_to_ram = int(total_elements_to_load_to_ram / 2)

    return elements_of_matrix_to_ram


def get_number_of_iterations(matrix, num_of_elements):
    total_rows = matrix.shape[0]
    return int(np.ceil(total_rows / num_of_elements))


def load_part_of_memmaped_matrix(matrix, num_of_elements_to_load, iteration_number):
    starting_index = iteration_number * num_of_elements_to_load
    ending_index = (iteration_number + 1) * num_of_elements_to_load
    if ending_index > matrix.shape[0]:
        ending_index = matrix.shape[0]

    return matrix[starting_index:ending_index, :]


def fill_in_covariance_mm(covariance, part_matrix1, part_matrix2, it1, it2, elements1, elements2):
    start_index1 = elements1 * it1
    end_index1 = elements1 * (it1 + 1)
    if end_index1 > covariance.shape[0]:
        end_index1 = covariance.shape[0]
    start_index2 = elements2 * it2
    end_index2 = elements2 * (it2 + 1)
    if end_index2 > covariance.shape[0]:
        end_index2 = covariance.shape[0]

    full_covariance = np.cov(part_matrix1, part_matrix2)
    cov_shape = full_covariance.shape
    covariance[start_index1:end_index1, start_index2:end_index2] = full_covariance[int(cov_shape[0]/2):int(cov_shape[0])
                                                                                   , 0:int(cov_shape[1]/2)]
    covariance[start_index2:end_index2, start_index1:end_index1] = full_covariance[0:int(cov_shape[0]/2),
                                                                                   int(cov_shape[1]/2):int(cov_shape[1])]


def covariance_mm(matrix1, matrix2, result_file):

    covariance = np.memmap(result_file, np.float32, mode='w+', shape=(matrix1.shape[0], matrix2.shape[0]))

    elements_of_matrix_to_ram = calculate_number_of_elements_fitting_in_ram()

    for it1 in range(get_number_of_iterations(matrix1, elements_of_matrix_to_ram)):
        for it2 in range(get_number_of_iterations(matrix2, elements_of_matrix_to_ram)):
            part_matrix1 = load_part_of_memmaped_matrix(matrix1, elements_of_matrix_to_ram, it1)
            part_matrix2 = load_part_of_memmaped_matrix(matrix2, elements_of_matrix_to_ram, it2)
            fill_in_covariance_mm(covariance, part_matrix1, part_matrix2, it1, it2, elements_of_matrix_to_ram,
                                  elements_of_matrix_to_ram)
            print(it1, it2)

    return covariance

t = time.clock()
result = covariance_mm(debug_matrix_mm, debug_matrix_mm, r'D:\Data\George\temp\result_debug.dat')
print('Partial numpy covariance took ' + str(time.clock() - t) + ' secs')


dif = np.mean(np.power(result - np_cov,2))
print(dif)


import numpy as np
debug_matrix_dimensions = (3, 3)

m1 = np.random.random(debug_matrix_dimensions)
m2 = np.random.random(debug_matrix_dimensions)

t = np.cov(m1, m2)
t1 = np.cov(m1, m1)
t2 = np.cov(m2, m2)



#-------------------------------------------------



import numpy as np
import pandas as pd
import os.path as path

base_folder = r'E:\Code\Mine\pystarters_scientific_computing'
events_file = path.join(base_folder, 'Events.csv')
video_file = path.join(base_folder, 'Video.csv')


events_df = pd.read_csv(events_file, sep="\+01:00| |\(|\)|,", engine='python', header=None,
                        names=['Time','Type', 'X', 'Y'], usecols=[0,2,4,6], parse_dates=[0], infer_datetime_format=True)

video_df = pd.read_csv(video_file, sep='\+01:00', engine='python', header=None, names=['Time', 'Frame'],
                       usecols=[0, 1], parse_dates=[0], infer_datetime_format=True)





events_df = events_df.set_index('Time')
video_df = video_df.set_index(('Time'))

partial_video_reindexed = video_df.reindex(events_df.index, method='nearest')

events_with_frame_df = pd.merge(partial_video_reindexed, events_df, left_index =True, right_index=True)

events_df.reset_index(level=0, inplace=True)
video_df.reset_index(level=0, inplace=True)
events_with_frame_df.reset_index(level=0, inplace=True)



import matplotlib.pyplot as plt
plt.scatter(events_df['X'], events_df['Y'])



from os.path import join




import numpy as np
import matplotlib.pyplot as plt

wires = {1952:1, 1983:4, 1990:8}
passive = {1995:16, 2005:32, 2013:64, 2014:204}
active_electrodes = {1986:8, 2011:752, 2016:966, 2017:1356, 2022:8000}
active_channels = {1986:8, 2011:16, 2016:384, 2017:1356, 2022:1024}

plt.plot(wires.keys(), wires.values(), passive.keys(), passive.values(), active_channels.keys(), active_channels.values(), active_electrodes.keys(), active_electrodes.values())





from GUIs.Kilosort import clean_kilosort_templates as clean
from os.path import join


data_folder = r'E:\Data\Neuroseeker_2017_03_28_Anesthesia_Auditory_DoubleProbes\Angled\Data\Experiment_Bach_1_2017-03-28T18_10_41'
kilosort_folder = r'E:\Data\Neuroseeker_2017_03_28_Anesthesia_Auditory_DoubleProbes\Angled\Analysis\Experiment_Bach_1_2017-03-28T18_10_41\Kilosort'
binary_filename = join(data_folder, 'Amplifier_APs.bin')66
number_of_channels = 1368
prb_file = join(kilosort_folder, 'ap_only_prb.txt')

clean.cleanup_kilosorted_data(kilosort_folder,
                              number_of_channels_in_binary_file=number_of_channels,
                              binary_data_filename=binary_filename,
                              prb_file=prb_file,
                              type_of_binary=np.int16,
                              order_of_binary='F',
                              sampling_frequency=20000,
                              num_of_shanks_for_vis=5)