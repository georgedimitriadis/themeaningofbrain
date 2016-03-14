__author__ = 'George Dimitriadis'

import os
import numpy as np
import numpy.core.defchararray as np_char
import BrainDataAnalysis.timelocked_analysis_functions as tf
import BrainDataAnalysis.ploting_functions as pf
import IO.ephys as ephys
import mne.filter as filters
from sklearn.manifold import TSNE as tsne
import t_sne_bhcuda.bhtsne_cuda as tsne_bhcuda
import t_sne_bhcuda.t_sne_spikes as tsne_spikes
import matplotlib.pyplot as plt
import Layouts.Probes.probes_imec as pr_imec
import h5py as h5
import IO.klustakwik as klusta
import pickle
from sklearn.cluster import DBSCAN
from sklearn import metrics
import time


base_folder = r'D:\Data\George\Projects\SpikeSorting\Joana_Paired_128ch'
rat = 97
good_cells = '9'
date = '2015-09-03'
data_folder = os.path.join(base_folder + '\\' + date, 'Data')
analysis_folder = os.path.join(base_folder + '\\' + date, 'Analysis')

cell_capture_times = '21_18_47'
spike_thresholds = -2e-4

adc_channel_used = 0
adc_dtype = np.uint16
inter_spike_time_distance = 0.002
amp_gain = 100
num_ivm_channels = 128
amp_dtype = np.uint16

sampling_freq = 30000
high_pass_freq = 500
filtered_data_type = np.float64

types_of_data_to_load = {'t': 'ivm_data_filtered_cell{}.dat',
                         'k': 'ivm_data_filtered_klusta_first_{}_spikes_cell_{}.dat',
                         'p': 'patch_data_cell{}.dat'}

num_of_points_in_spike_trig_ivm = 20
num_of_points_for_padding = 2 * num_of_points_in_spike_trig_ivm



# Generate the spike time triggers (and the adc traces in Volts)
raw_data_file_patch = os.path.join(data_folder, 'adc'+date+'T'+cell_capture_times+'.bin')

raw_data_patch = ephys.load_raw_event_trace(raw_data_file_patch, number_of_channels=8,
                                              channel_used=adc_channel_used, dtype=adc_dtype)
spike_triggers, spike_peaks, spike_data_in_V = tf.create_spike_triggered_events(raw_data_patch.dataMatrix,
                                                                   threshold=spike_thresholds,
                                                                   inter_spike_time_distance=inter_spike_time_distance,
                                                                   amp_gain=amp_gain)
num_of_spikes = len(spike_triggers)
print(num_of_spikes)



# Seperate the juxta spikes into a number of groups according to their size
num_of_spike_groups = 12
spike_thresholds_groups = np.arange(np.min(spike_peaks), np.max(spike_peaks),
                                    (np.max(spike_peaks) - np.min(spike_peaks)) / num_of_spike_groups)
spike_thresholds_groups = np.append(spike_thresholds_groups, np.max(spike_peaks))
spike_triggers_grouped = {}
spike_peaks_grouped = {}
spike_triggers_grouped_withnans = {}
spike_peaks_grouped_withnans = {}
for t in range(1, len(spike_thresholds_groups)):
    spike_triggers_grouped[t] = []
    spike_peaks_grouped[t] = []
    spike_peaks_grouped_withnans[t] = np.empty(len(spike_peaks))
    spike_peaks_grouped_withnans[t][:] = np.NAN
    spike_triggers_grouped_withnans[t] = np.empty(len(spike_peaks))
    spike_triggers_grouped_withnans[t][:] = np.NAN
for s in range(len(spike_peaks)):
    for t in range(1, len(spike_thresholds_groups)):
        if spike_peaks[s] < spike_thresholds_groups[t]:
            spike_triggers_grouped[t].append(spike_triggers[s])
            spike_peaks_grouped[t].append(spike_peaks[s])
            break
for t in range(1, len(spike_thresholds_groups)):
    spike_peaks_grouped_withnans[t][np.in1d(spike_triggers, spike_triggers_grouped[t])] = \
        spike_peaks[np.in1d(spike_triggers, spike_triggers_grouped[t])]
    spike_triggers_grouped_withnans[t][np.in1d(spike_triggers, spike_triggers_grouped[t])] = \
        spike_triggers[np.in1d(spike_triggers, spike_triggers_grouped[t])]






# Cut the adc trace into a time x spikes matrix
data_to_load = 'p'
shape_of_filt_spike_trig_patch = ((num_of_points_in_spike_trig_ivm,
                                   num_of_spikes))
patch_data = np.memmap(os.path.join(analysis_folder, types_of_data_to_load[data_to_load].format(good_cells)),
                                dtype=filtered_data_type,
                                mode='w+',
                                shape=shape_of_filt_spike_trig_patch)
for spike in np.arange(0, num_of_spikes):
    trigger_point = spike_triggers[spike]
    start_point = int(trigger_point - num_of_points_in_spike_trig_ivm / 2)
    if start_point < 0:
        break
    end_point = int(trigger_point + num_of_points_in_spike_trig_ivm / 2)
    if end_point > raw_data_patch.dataMatrix.shape[0]:
        break
    patch_data[:, spike] = raw_data_patch.dataMatrix[start_point:end_point]



# Cut the ivm traces into a channels x time x spikes cube and high pass them
data_to_load = 't'
raw_data_file_ivm = os.path.join(data_folder, 'amplifier'+date+'T'+cell_capture_times+'.bin')

raw_data_ivm = ephys.load_raw_data(raw_data_file_ivm, numchannels=num_ivm_channels, dtype=amp_dtype)

shape_of_filt_spike_trig_ivm = ((num_ivm_channels,
                                 num_of_points_in_spike_trig_ivm,
                                 num_of_spikes))
ivm_data_filtered = np.memmap(os.path.join(analysis_folder, types_of_data_to_load[data_to_load].format(good_cells)),
                                dtype=filtered_data_type,
                                mode='w+',
                                shape=shape_of_filt_spike_trig_ivm)
for spike in np.arange(0, num_of_spikes):
    trigger_point = spike_triggers[spike]
    start_point = int(trigger_point - (num_of_points_in_spike_trig_ivm + num_of_points_for_padding)/2)
    if start_point < 0:
        break
    end_point = int(trigger_point + (num_of_points_in_spike_trig_ivm + num_of_points_for_padding)/2)
    if end_point > raw_data_ivm.shape()[1]:
        break
    temp_unfiltered = raw_data_ivm.dataMatrix[:, start_point:end_point]
    temp_unfiltered = temp_unfiltered.astype(filtered_data_type)
    iir_params = {'order': 4, 'ftype': 'butter', 'padlen': 0}
    temp_filtered = filters.high_pass_filter(temp_unfiltered, sampling_freq, high_pass_freq, method='iir',
                                             iir_params=iir_params)  # 4th order Butter with no padding
    temp_filtered = temp_filtered[:, int(num_of_points_for_padding / 2):-int(num_of_points_for_padding / 2)]
    ivm_data_filtered[:, :, spike] = temp_filtered


# Load the already saved ivm_data_filtered
shape_of_filt_spike_trig_ivm = (num_ivm_channels,
                                num_of_points_in_spike_trig_ivm,
                                num_of_spikes)
time_axis = np.arange(-num_of_points_in_spike_trig_ivm/(2*sampling_freq),
              num_of_points_in_spike_trig_ivm/(2*sampling_freq),
              1/sampling_freq)
ivm_data_filtered = np.memmap(os.path.join(analysis_folder, types_of_data_to_load[data_to_load].format(good_cells)),
                                dtype=filtered_data_type,
                                mode='r',
                                shape=shape_of_filt_spike_trig_ivm)









# Single Cell comparison between Klusta and tsne
# 0) Create the required .dat file and the probe .prb file to throw into klustakwik (i.e. the phy module)
spikes_to_include = num_of_spikes
fin_time_point = spike_triggers[spikes_to_include] + 500
start_time_point = 0
time_limits = [start_time_point, fin_time_point]

raw_data_file_ivm = os.path.join(data_folder, 'amplifier'+date+'T'+cell_capture_times+'.bin')
raw_data_ivm = ephys.load_raw_data(raw_data_file_ivm, numchannels=num_ivm_channels, dtype=amp_dtype)
file_dat = os.path.join(analysis_folder, r'klustakwik\raw_data_ivm_klusta.dat')

klusta.make_dat_file(raw_data_ivm.dataMatrix, filename=file_dat, num_channels=num_ivm_channels, time_limits=time_limits)

file_prb = os.path.join(analysis_folder, r'klustakwik\128ch_passive_imec.prb')
electrode_structure = pr_imec.create_128channels_imec_prb(file_prb)



# 0.5) Grab the mask and the PCA components for all spikes from the .kwx file
filename = r'D:\Data\George\Projects\SpikeSorting\Joana_Paired_128ch\2015-09-03\Analysis\klustakwik\threshold_6_5std\threshold_6_5std.kwx'
h5file = h5.File(filename, mode='r')
pca_and_masks = np.array(list(h5file['channel_groups/0/features_masks']))
masks = np.array(pca_and_masks[:, :, 1])
pca_features = np.array(pca_and_masks[:, :, 0])
masked_pca_features = pca_features * masks



# 1) Grab the spike times from the .kwik file
filename = r'D:\Data\George\Projects\SpikeSorting\Joana_Paired_128ch\2015-09-03\Analysis\klustakwik\threshold_6_5std\threshold_6_5std.kwik'
h5file = h5.File(filename, mode='r')
all_extra_spike_times = np.array(list(h5file['channel_groups/0/spikes/time_samples']))
#klusta_clusters = np.array(list(h5file['channel_groups/0/spikes/clusters/main']))
h5file.close()
print('All extra spikes = {}'.format(len(all_extra_spike_times)))


# 2) Function to find the common spikes between the klusta spikedetct results and the juxtacellular spikes
def find_juxta_spikes_in_extra_detected(juxta_spikes_to_include, juxta_spike_triggers, all_extra_spike_times, d_time_points):
    common_spikes = []
    indices_of_common_spikes_in_klusta = []
    clusters_of_common_spikes = []
    prev_spikes_added = 0
    curr_spikes_added = 0
    ks = 0
    juxta_spikes_not_found = []
    index_of_klusta_spike_found = 0
    for juxta_spike in juxta_spike_triggers[index_of_klusta_spike_found:juxta_spikes_to_include]:
        for possible_extra_spike in np.arange(juxta_spike - d_time_points, juxta_spike + d_time_points):
            possible_possitions = np.where(all_extra_spike_times == possible_extra_spike)[0]
            if len(possible_possitions) != 0:
                index_of_klusta_spike_found = possible_possitions[0]
                common_spikes.append(all_extra_spike_times[index_of_klusta_spike_found])
                indices_of_common_spikes_in_klusta.append(index_of_klusta_spike_found)
                #clusters_of_common_spikes.append(klusta_clusters[index_of_klusta_spike_found])
                curr_spikes_added += 1
                break
        if curr_spikes_added > prev_spikes_added:
            prev_spikes_added = curr_spikes_added
        else:
            juxta_spikes_not_found.append(juxta_spike)
    print(np.shape(common_spikes))
    print(str(100 * (np.shape(common_spikes)[0] / len(juxta_spike_triggers[:juxta_spikes_to_include])))+'% found')
    return common_spikes, indices_of_common_spikes_in_klusta, juxta_spikes_not_found


spikes_to_include = num_of_spikes
# 3) Find the common spikes between the klusta spikedetct results and
# the juxtacellular spikes for all the juxta spikes and for all the sub groups of juxta spikes
common_spikes, indices_of_common_spikes_in_klusta, juxta_spikes_not_found = \
    find_juxta_spikes_in_extra_detected(juxta_spikes_to_include=spikes_to_include,
                                        juxta_spike_triggers = spike_triggers,
                                        all_extra_spike_times=all_extra_spike_times,
                                        d_time_points=7)
common_spikes_grouped = {}
juxta_spikes_not_found_grouped = {}
indices_of_common_spikes_in_klusta_grouped = {}
for g in range(1, num_of_spike_groups+1):
    common_spikes_grouped[g], indices_of_common_spikes_in_klusta_grouped[g], juxta_spikes_not_found_grouped[g] = \
        find_juxta_spikes_in_extra_detected(juxta_spikes_to_include=spikes_to_include,
                                            juxta_spike_triggers=spike_triggers_grouped[g],
                                            all_extra_spike_times=all_extra_spike_times,
                                            d_time_points=7)

# 4) Plot the change in percentage of found juxta spikes with time window size
delta_time = range(15)
num_of_common_spikes = []
for i in delta_time:
    common_spikes, indices_of_common_spikes_in_klusta, juxta_spikes_not_found = \
        find_juxta_spikes_in_extra_detected(juxta_spikes_to_include=spikes_to_include,
                                            juxta_spike_triggers=spike_triggers,
                                            all_extra_spike_times=all_extra_spike_times,
                                            d_time_points=i)
    num_of_common_spikes.append(np.shape(common_spikes))
plt.plot(2000*np.array(delta_time)/sampling_freq, np.array(num_of_common_spikes)/spikes_to_include)


up_to_extra_spike = len(all_extra_spike_times)
subset_of_largest_channels = num_ivm_channels
# Generate the data to go into t-sne for all spikes detected by klusta's spikedetect
# 1) Cut the correct data, filter them (hp) and put them in a channels x time x spikes ivm data cube
data_to_load = 'k'
raw_data_file_ivm = os.path.join(data_folder, 'amplifier'+date+'T'+cell_capture_times+'.bin')

raw_data_ivm = ephys.load_raw_data(raw_data_file_ivm, numchannels=num_ivm_channels, dtype=amp_dtype)

shape_of_filt_spike_trig_ivm = (subset_of_largest_channels,
                                num_of_points_in_spike_trig_ivm,
                                up_to_extra_spike)
filename_for_kluster = os.path.join(analysis_folder,
                                     types_of_data_to_load[data_to_load].format(up_to_extra_spike, good_cells))
ivm_data_filtered = np.memmap(filename_for_kluster,
                              dtype=filtered_data_type,
                              mode='w+',
                              shape=shape_of_filt_spike_trig_ivm)
for spike in np.arange(0, up_to_extra_spike):
    trigger_point = all_extra_spike_times[spike]
    start_point = int(trigger_point - (num_of_points_in_spike_trig_ivm + num_of_points_for_padding)/2)
    if start_point < 0:
        break
    end_point = int(trigger_point + (num_of_points_in_spike_trig_ivm + num_of_points_for_padding)/2)
    if end_point > raw_data_ivm.shape()[1]:
        break
    temp_unfiltered = raw_data_ivm.dataMatrix[:, start_point:end_point]
    temp_unfiltered = temp_unfiltered.astype(filtered_data_type)
    if subset_of_largest_channels != num_ivm_channels:
        mins = np.min(temp_unfiltered, axis=1)
        indices = np.argsort(mins)
        temp_unfiltered = temp_unfiltered[indices[:subset_of_largest_channels]]
    iir_params = {'order': 4, 'ftype': 'butter', 'padlen': 0}
    temp_filtered = filters.high_pass_filter(temp_unfiltered, sampling_freq, high_pass_freq, method='iir',
                                             iir_params=iir_params)  # 4th order Butter with no padding
    temp_filtered = temp_filtered[:, int(num_of_points_for_padding / 2):-int(num_of_points_for_padding / 2)]
    ivm_data_filtered[:, :, spike] = temp_filtered
del raw_data_ivm
del ivm_data_filtered
time_axis = np.arange(-num_of_points_in_spike_trig_ivm/(2*sampling_freq),
                      num_of_points_in_spike_trig_ivm / (2 * sampling_freq),
                      1 / sampling_freq)

# 2) Load the ivm data cube
data_to_load = 'k'
num_of_spikes = len(all_extra_spike_times)
shape_of_filt_spike_trig_ivm = (subset_of_largest_channels,
                                num_of_points_in_spike_trig_ivm,
                                up_to_extra_spike)
filename_for_kluster = os.path.join(analysis_folder,
                                    types_of_data_to_load[data_to_load].format(up_to_extra_spike, good_cells))
ivm_data_filtered = np.memmap(filename_for_kluster,
                              dtype=filtered_data_type,
                              mode='r',
                              shape=shape_of_filt_spike_trig_ivm)

# 2.5) Generate a mask for the required time points (spikes x (channels * time_points)) from the pca mask
# (spikes x (channels * 3 pca features))
index_every_3 = np.arange(0, num_ivm_channels * 3, 3)
masks_single_numbers = masks[:up_to_extra_spike, index_every_3]
mask_for_time_points = np.repeat(masks_single_numbers, repeats=num_of_points_in_spike_trig_ivm, axis=1)


# 3) Put the ivm cube in a 2D matrix ready for tsne
X = []
use_features = True
mask_data = False

if not use_features:
    data_to_use = ivm_data_filtered
    #data_to_use = np.mean(ivm_data_filtered, axis=0)
    #data_to_use = data_to_use.reshape((1, data_to_use.shape[0], data_to_use.shape[1]))
    num_of_spikes = data_to_use.shape[2]
    number_of_time_points = data_to_use.shape[1]
    number_of_channels = data_to_use.shape[0]
    start = int((num_of_points_in_spike_trig_ivm - number_of_time_points) / 2)
    end = start + number_of_time_points
else:
    data_to_use = masked_pca_features[:up_to_extra_spike, :].T
    data_to_use = data_to_use.reshape((1, data_to_use.shape[0], data_to_use.shape[1]))
    num_of_spikes = data_to_use.shape[2]
    number_of_time_points = data_to_use.shape[1]
    number_of_channels = data_to_use.shape[0]
    start = 0
    end = number_of_time_points

newshape = (number_of_channels * number_of_time_points, num_of_spikes)
t = data_to_use[:, start:end, :num_of_spikes]
t = np.reshape(t, newshape=newshape)
if mask_data:
    t = t * np.transpose(mask_for_time_points)
X.extend(t.T)
X_np = np.array(X)
del t, X


#indices_of_data_for_tsne = [(i, k)[0] for i, k in enumerate(X) if random.random() > 0] #For random choice
indices_of_data_for_tsne = range(up_to_extra_spike) #For the first n spikes
data_for_tsne = X_np[indices_of_data_for_tsne]

s = 128820
indices_of_data_for_tsne = range(s)
juxta_cluster_indices_grouped = {}
for g in range(0, num_of_spike_groups):
    juxta_cluster_indices_temp = np.intersect1d(indices_of_data_for_tsne, indices_of_common_spikes_in_klusta_grouped[g+1])
    juxta_cluster_indices_grouped[g] = [i for i in np.arange(0, len(indices_of_data_for_tsne)) if
                             len(np.where(juxta_cluster_indices_temp == indices_of_data_for_tsne[i])[0])]
    #print(len(juxta_cluster_indices_grouped[g]))

per = 100
lr = 200
it = 2000

t_tsne = np.load(r'D:\Data\George\Projects\SpikeSorting\Joana_Paired_128ch\2015-09-03\Analysis\klustakwik'+\
                 r'\threshold_6_5std\t_sne_results_{}s_{}per_{}lr_{}it.npy'.format(s, per, lr, it))






# T-SNE
# Python scikit-learn t-sne
t0 = time.time()
perplexity = 500.0
early_exaggeration = 100.0
learning_rate = 3000.0
theta = 0.0
model = tsne(n_components=2, perplexity=perplexity, early_exaggeration=early_exaggeration,
             learning_rate=learning_rate, n_iter=1000, n_iter_without_progress=500,
             min_grad_norm=1e-7, metric="euclidean", init="random", verbose=3,
             random_state=None, method='barnes_hut', angle=theta)
t_tsne = model.fit_transform(data_for_tsne)
t_tsne = t_tsne.T
t1 = time.time()
print("Scikit t-sne took {} seconds, ({} minutes), for {} spikes".format(t1-t0, (t1-t0)/60, up_to_extra_spike))

# save the python scikit generated t-sne results
threshold = 5.5
file_name = r'D:\Data\George\Projects\SpikeSorting\Joana_Paired_128ch\2015-09-03\Analysis\t_tsne_ivm_data_{}sp_{}per_{}ee_{}lr_{}tp_{}thres.pkl'\
    .format(len(indices_of_data_for_tsne), perplexity, early_exaggeration, learning_rate, number_of_time_points, threshold)
file = open(file_name, 'bw')
pickle.dump((ivm_data_filtered, t_tsne, juxta_cluster_indices_grouped, perplexity, early_exaggeration, learning_rate), file)
file.close()


tsne_bhcuda.save_data_for_tsne(data_for_tsne, r'E:\George\SourceCode\Repos\t_sne_bhcuda\bin\windows', 'data.dat',
                               theta=0.6, perplexity=50, eta=200, no_dims=2, iterations=1000, seed=0, gpu_mem=0.8,
                               randseed=-1)
t_tsne = np.transpose(tsne_bhcuda.load_tsne_result(
            r'D:\Data\George\Projects\SpikeSorting\Joana_Paired_128ch\2015-09-03\Analysis\klustakwik\threshold_6_5std',
            'result_final_allspikes.dat'))



# T-sne with my conda package
kwx_file_path = r'''D:\Data\George\Projects\SpikeSorting\Joana_Paired_128ch\2015-09-03\Analysis\\klustakwik_128to32_probe\threshold_6_5std\threshold_6_5std.kwx'''
indices_of_data_for_tsne = None#range(128820)
seed = 0
perplexity = 100.0
theta = 0.2
learning_rate = 200.0
iterations = 2000
gpu_mem = 0.8
t_tsne2 = tsne_spikes.t_sne_spikes(kwx_file_path=kwx_file_path, hdf5_dir_to_pca=r'channel_groups/0/features_masks',
                                  mask_data=True, path_to_save_tmp_data=None,
                                  indices_of_spikes_to_tsne=indices_of_data_for_tsne, use_scikit=False,
                                  perplexity=perplexity, theta=theta, eta=learning_rate,
                                  iterations=iterations, seed=seed, verbose=2, gpu_mem=gpu_mem)


# C++ wrapper t-sne using CPU
t0 = time.time()
perplexity = 50.0
theta = 0.2
learning_rate = 200.0
iterations = 5000
gpu_mem = 0
t_tsne = tsne_bhcuda.t_sne(data_for_tsne,
                           files_dir=r'D:\Data\George\Projects\SpikeSorting\Joana_Paired_128ch\2015-09-03\Analysis\tsne_results',
                           no_dims=2, perplexity=perplexity, eta=learning_rate, theta=theta,
                           iterations=iterations, gpu_mem=gpu_mem, randseed=-1, verbose=3)
t_tsne = np.transpose(t_tsne)
t1 = time.time()
print("C++ t-sne took {} seconds, ({} minutes), for {} spikes".format(t1-t0, (t1-t0)/60, up_to_extra_spike))


# C++ wrapper t-sne using GPU
t0 = time.time()
perplexity = 1000.0
theta = 0.2
learning_rate = 200.0
iterations = 2000
gpu_mem = 0.8
t_tsne = tsne_bhcuda.t_sne(data_for_tsne,
                           files_dir=r'D:\Data\George\Projects\SpikeSorting\Joana_Paired_128ch\2015-09-03\Analysis\tsne_results',
                           no_dims=2, perplexity=perplexity, eta=learning_rate, theta=theta,
                           iterations=iterations, gpu_mem=gpu_mem, randseed=-1, verbose=3)
t_tsne = np.transpose(t_tsne)
t1 = time.time()
print("CUDA t-sne took {} seconds, ({} minutes), for {} spikes".format(t1-t0, (t1-t0)/60, up_to_extra_spike))


#  2D plot
pf.plot_tsne(t_tsne, juxta_cluster_indices_grouped, subtitle='T-sne', label_name='Peak size in uV',
             label_array=(spike_thresholds_groups*1e6).astype(int))
pf.plot_tsne(t_tsne, subtitle='T-sne of 129000 spikes from Juxta Paired recordings, not labeled',
             label_name=None,
             label_array=None)



#  3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
s = 10
c = ['r', 'g', 'c', 'm', 'y', 'k', 'w', 'b']
ax.scatter(t_tsne[0], t_tsne[1], t_tsne[2], s=3)
for g in np.arange(1, num_of_spike_groups+1):
    ax.scatter(t_tsne[0][juxta_cluster_indices_grouped[g]], t_tsne[1][juxta_cluster_indices_grouped[g]],
               t_tsne[2][juxta_cluster_indices_grouped[g]], s=s, color=c[g-1])

# load saved t-sne results
spike_number = 53497
perplexity = 800.0
early_exaggeration = 100.0
learning_rate = 200.0
number_of_time_points = 20
threshold = 11
file_name = r'D:\Data\George\Projects\SpikeSorting\Joana_Paired_128ch\2015-09-03\Analysis\t_tsne_ivm_data_{}sp_{}per_{}ee_{}lr_{}tp_{}thres.pkl'\
    .format(spike_number, perplexity, early_exaggeration, learning_rate, number_of_time_points, threshold)
file = open(file_name, 'br')
ivm_data_filtered, t_tsne, juxta_cluster_indices_grouped, perplexity, early_exaggeration, learning_rate = pickle.load(file)
file.close()

# Load the c++ bhtsne results
tmp_dir_path = r'D:\Data\George\Projects\SpikeSorting\Joana_Paired_128ch\2015-09-03\Analysis\tsne_results'
results_filename = 'result_0to50kexsp_40timpoints_per500_ee100_lr3k_theta07.dat'
t_tsne = tsne_bhcuda.load_tsne_result(tmp_dir_path, results_filename)
t_tsne = np.transpose(t_tsne)
#or
t_tsne = np.load(r'D:\Data\George\Projects\SpikeSorting\Joana_Paired_128ch\2015-09-03\Analysis\klustakwik\threshold_6_5std\t_sne_results_final_allspikes.npy')


# Clustering
def fit_dbscan(data, eps, min_samples, show=True, juxta_cluster_indices_grouped=None, threshold_legend=None):
    X = np.transpose(data)

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    score = metrics.silhouette_score(X, labels, sample_size=5000)
    print('For eps={}, min_samples={}, estimated number of clusters={}'.format(eps, min_samples, n_clusters_))
    print("Silhouette Coefficient: {}".format(score))

    if show:
        pf.show_clustered_tsne(db, X, juxta_cluster_indices_grouped, threshold_legend)

    return db, n_clusters_, labels, core_samples_mask, score


from sklearn.cluster import KMeans
kmeans_est_35 = KMeans(n_clusters=35)
kmeans_est_35.fit(X)

db, n_clusters_, labels, core_samples_mask, score = fit_dbscan(t_tsne, 0.6, 20, show=True)
pf.show_clustered_tsne(db, X, juxta_cluster_indices_grouped=None, threshold_legend=None)


# Loop over values of eps and min_samples to find best fit for DBSCAN
epses = np.arange(0.1, 0.5, 0.05)
min_sampleses = np.arange(5, 100, 5)
clustering_scores = []
params = []
for eps in epses:
    for min_samples in min_sampleses:
        db, n_clusters_, labels, core_samples_mask, score = fit_dbscan(t_tsne, eps, min_samples, show=False)
        clustering_scores.append(score)
        params.append((eps, min_samples))
params = np.array(params)
clustering_scores = np.array(clustering_scores)


# Define TP / FP / TN and FN
X = np.transpose(t_tsne)
juxta_cluster_indices = []
for g in range(0, num_of_spike_groups):
    juxta_cluster_indices.extend(juxta_cluster_indices_grouped[g])
means_of_juxta = np.array([np.median(X[juxta_cluster_indices, 0]), np.median(X[juxta_cluster_indices, 1])])

means_of_labels = np.zeros((n_clusters_, 2))
dmeans = np.zeros(n_clusters_)
for l in range(n_clusters_):
    class_member_mask = (labels == l)
    xy = X[class_member_mask & core_samples_mask]
    means_of_labels[l, 0] = np.median(xy[:, 0])
    means_of_labels[l, 1] = np.median(xy[:, 1])
    dmeans[l] = np.linalg.norm((means_of_labels[l,:]-means_of_juxta))
juxta_cluster_index = np.argmin(dmeans)



# calculate prec, rec and f factor
class_member_mask = (labels == juxta_cluster_index)
extra_cluster_indices = [i for i, x in enumerate(class_member_mask & core_samples_mask) if x]
tp_indices = np.intersect1d(juxta_cluster_indices, extra_cluster_indices)
tp = len(tp_indices)
all_pos = len(extra_cluster_indices)
all_true = len(juxta_cluster_indices)
precision = tp / all_pos
recall = tp / all_true
f_factor = 2*(precision*recall)/(precision+recall)
print('Precision = {}, Recall = {}, F1 factor = {}'.format(precision, recall, f_factor))

# have a look where the averages are
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(means_of_juxta[0], means_of_juxta[1])
ax.scatter(means_of_labels[:, 0], means_of_labels[:, 1], color='r')