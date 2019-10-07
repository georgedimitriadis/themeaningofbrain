__author__ = 'George Dimitriadis'

import os
import numpy as np
from IO import ephys as ioep
import BrainDataAnalysis.timelocked_analysis_functions as tf
import BrainDataAnalysis.Graphics.ploting_functions as pf
import IO.ephys as ephys
import mne.filter as filters
from sklearn.manifold import TSNE as tsne
import ExperimentSpecificCode._2016_TSne_Paper.t_sne_bhcuda.bhtsne_cuda as tsne_bhcuda
import ExperimentSpecificCode._2016_TSne_Paper.t_sne_bhcuda.t_sne_spikes as tsne_spikes
import matplotlib.pyplot as plt
import Layouts.Probes.probes_imec as pr_imec
import h5py as h5
import IO.klustakwik as klusta
import pickle
import time


base_folder = r'D:\Data\George\Projects\SpikeSorting\Joana_Paired_128ch'
base_folder = r'Z:\g\George\DataAndResults\Experiments\Anesthesia\Joana_Paired_128ch'
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
    if spike % 100 == 0:
        print('Done ' + str(spike) + 'spikes')


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
filename = os.path.join(analysis_folder, 'klustakwik', 'threshold_6_5std', r'threshold_6_5std.kwx')
h5file = h5.File(filename, mode='r')
pca_and_masks = np.array(list(h5file['channel_groups/0/features_masks']))
masks = np.array(pca_and_masks[:, :, 1])
pca_features = np.array(pca_and_masks[:, :, 0])
masked_pca_features = pca_features * masks



# 1) Grab the spike times from the .kwik file
filename = os.path.join(analysis_folder, 'klustakwik', 'threshold_6_5std', r'threshold_6_5std.kwik')
h5file = h5.File(filename, mode='r')
spike_times_phy = np.array(list(h5file['channel_groups/0/spikes/time_samples']))
#klusta_clusters = np.array(list(h5file['channel_groups/0/spikes/clusters/main']))
h5file.close()
print('All extra spikes = {}'.format(len(spike_times_phy)))


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
                                        all_extra_spike_times=spike_times_phy,
                                        d_time_points=10)
common_spikes_grouped = {}
juxta_spikes_not_found_grouped = {}
indices_of_common_spikes_in_klusta_grouped = {}
for g in range(1, num_of_spike_groups+1):
    common_spikes_grouped[g], indices_of_common_spikes_in_klusta_grouped[g], juxta_spikes_not_found_grouped[g] = \
        find_juxta_spikes_in_extra_detected(juxta_spikes_to_include=spikes_to_include,
                                            juxta_spike_triggers=spike_triggers_grouped[g],
                                            all_extra_spike_times=spike_times_phy,
                                            d_time_points=7)

# 4) Plot the change in percentage of found juxta spikes with time window size
delta_time = range(15)
num_of_common_spikes = []
for i in delta_time:
    common_spikes, indices_of_common_spikes_in_klusta, juxta_spikes_not_found = \
        find_juxta_spikes_in_extra_detected(juxta_spikes_to_include=spikes_to_include,
                                            juxta_spike_triggers=spike_triggers,
                                            all_extra_spike_times=spike_times_phy,
                                            d_time_points=i)
    num_of_common_spikes.append(np.shape(common_spikes))
plt.plot(2000*np.array(delta_time)/sampling_freq, np.array(num_of_common_spikes)/spikes_to_include)


up_to_extra_spike = len(spike_times_phy)
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
    trigger_point = spike_times_phy[spike]
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
num_of_spikes = len(spike_times_phy)
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



# Throw away labeled extra spikes that are not in the cluster of electrodes that show the juxta spikes
def select_spikes_in_certain_channels(spike_thresholds, common_spikes, indices_of_common_spikes_in_extra,
                                      good_channels, num_of_raw_data_channels):
    raw_data = ioep.load_raw_data(filename=os.path.join(data_folder, 'amplifier'+date+'T'+cell_capture_times+'.bin'),
                                  numchannels=num_of_raw_data_channels)
    common_spikes = np.array(common_spikes)
    indices_of_common_spikes_in_extra = np.array(indices_of_common_spikes_in_extra)
    t = raw_data.dataMatrix[:, common_spikes]
    if spike_thresholds > 0:
        spike_channels = np.argmin(t, axis=0)
    if spike_thresholds < 0:
        spike_channels = np.argmax(t, axis=0)
    good_spike_indices = [i for i,x in list(enumerate(spike_channels)) if np.in1d(x, good_channels)]
    common_spikes = common_spikes[good_spike_indices]
    indices_of_common_spikes_in_extra = indices_of_common_spikes_in_extra[good_spike_indices]
    return common_spikes, indices_of_common_spikes_in_extra

r1 = np.array([103, 101,  99,  97,  95,  93,  91,  89,  87, 70, 66, 84, 82, 80, 108, 110, 47, 45, 43, 41,  1, 61, 57, 36, 34, 32, 30, 28, 26, 24, 22, 20])
r2 = np.array([106,	104, 115, 117, 119,	121, 123, 125, 127, 71,	67,	74,	76,	78,	114, 112, 49, 51, 53, 55,  2, 62, 58,  4,  6,  8, 10, 12, 14, 21, 19, 16])
r3 = np.array([102,	100,  98,  96,  94,  92,  90,  88,  86, 72, 68, 65, 85, 83,  81, 111, 46, 44, 42, 40, 38, 63, 59, 39, 37, 35, 33, 31, 29, 27, 25, 23])
r4 = np.array([109,	107, 105, 116, 118,	120, 122, 124, 126, 73,	69,	64,	75,	77,	 79, 113, 48, 50, 52, 54, 56,  0, 60,  3,  5,  7,  9, 11, 13, 15, 18, -1])

spike_channels = np.concatenate((r1[:8], r2[:8], r3[:8]))

common_spikes_grouped[g], indices_of_common_spikes_in_klusta_grouped[g] \
    = select_spikes_in_certain_channels(spike_thresholds, common_spikes_grouped[g],
                                        indices_of_common_spikes_in_klusta_grouped[g],
                                        spike_channels, num_of_raw_data_channels=128)



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

t_tsne = np.load(r'D:\Data\George\Projects\SpikeSorting\Joana_Paired_128ch\2015-09-03\Analysis\klustakwik'+\
                 r'\threshold_6_5std\t_sne_results_final_allspikes.npy')




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
path = r'D:\Data\George\Projects\SpikeSorting\Joana_Paired_128ch\2015-09-03\Analysis\klustakwik\threshold_6_5std'
kwx_file_path = os.path.join(path, r'threshold_6_5std.kwx')
video = os.path.join(path, r'video')
indices_of_data_for_tsne = None #range(40000)
seed = 0
perplexity = 100.0
theta = 0.2
learning_rate = 200.0
iterations = 5000
gpu_mem = 0.2
no_dims = 2
tsne = tsne_spikes.t_sne_spikes(kwx_file_path=kwx_file_path, hdf5_dir_to_pca=r'channel_groups/0/features_masks',
                                  mask_data=True, path_to_save_tmp_data=path,
                                  indices_of_spikes_to_tsne=indices_of_data_for_tsne, use_scikit=False,
                                  perplexity=perplexity, theta=theta, no_dims=no_dims, eta=learning_rate,
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
pf.plot_tsne(t_tsne, juxta_cluster_indices_grouped, subtitle='T-sne', cm=plt.cm.coolwarm, label_name='Peak size in uV',
             label_array=(spike_thresholds_groups*1e6).astype(int), labeled_sizes=[2, 15])
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
def fit_dbscan(data, eps, min_samples, normalize=True,
               show=True, juxta_cluster_indices_grouped=None, threshold_legend=None):
    X = np.transpose(data)

    if normalize:
        from sklearn.preprocessing import minmax_scale
        minmax_scale(X, feature_range=(-1, 1), axis=0, copy=False)

    from sklearn.cluster import DBSCAN
    from sklearn import metrics
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


db, n_clusters_, labels, core_samples_mask, score = fit_dbscan(t_tsne, eps=0.0155, min_samples=43, normalize=True,
                                                               show=True)
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


# ---------------------------------------------------------------------
# Manual Clustering using the GUI

from os.path import join
import numpy as np
import IO.ephys as ephys

# Parameters for testing (128 channels)
base_dir = r'Z:\g\George\DataAndResults\Experiments\Anesthesia\Joana_Paired_128ch\2015-09-03'
tsne_dir = r'Analysis\klustakwik\threshold_6_5std'
data_dir = r'Data'
data_cube_dir = r'Analysis\TempCube'
per = 100
lr = 200
it = 2000
s = 128820

tsne_filename = join(base_dir, tsne_dir, 't_sne_results_{}s_{}per_{}lr_{}it.npy'.format(s, per, lr, it))
kwik_filename = join(base_dir, tsne_dir, 'threshold_6_5std.kwik')
time_samples_h5_dir = r'channel_groups/0/spikes/time_samples'

spike_indices_to_use = None#np.arange(20000)
num_of_spikes = s
if spike_indices_to_use is not None:
    num_of_spikes = len(spike_indices_to_use)

num_ivm_channels = 128
amp_dtype = np.uint16
cube_type = np.int32
sampling_freq = 30000
num_of_points_in_spike_trig = 64
num_of_points_for_baseline = 10

shape_of_cut_extracellular_data = (num_ivm_channels, num_of_points_in_spike_trig, num_of_spikes)

raw_data_filename = join(base_dir, data_dir, 'amplifier2015-09-03T21_18_47.bin')
raw_data_ivm = ephys.load_raw_data(raw_data_filename, numchannels=num_ivm_channels, dtype=amp_dtype).dataMatrix

data_cube_filename = join(base_dir, data_cube_dir, 'baselined_data_cube.npy')

autocor_bin_number = 100
cluster_info_filename = join(base_dir, tsne_dir, 'cluster_info.pkl')
prb_file = join(base_dir, 'Analysis', 'klustakwik', '128ch_passive_imec.prb')

tsne_cluster.gui_manual_cluster_tsne_spikes(tsne_array_or_filename=tsne_filename,
                                            spike_times_list_or_filename=kwik_filename,
                                            time_samples_h5_dir=time_samples_h5_dir,
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
                                            k4=False,
                                            verbose=True)



# Make video from interim data
label_path = r'D:\Data\George\Projects\SpikeSorting\Joana_Paired_128ch\2015-09-03\Analysis\klustakwik\threshold_6_5std\cluster_info.pkl'
pkl_file = open(label_path, 'rb')
labels = pickle.load(pkl_file)
pkl_file.close()
spikes_labeled_dict = {}
i = 0
for name in labels.index:
    spikes_labeled_dict[i] = labels['Spike_Indices'][name]
    i += 1

video_dir = r'D:\Data\George\Projects\SpikeSorting\Joana_Paired_128ch\2015-09-03\Analysis\klustakwik\threshold_6_5std\video'
iterations = 10000
cm = plt.cm.jet
markers = ['.', '^']
sizes = [1, 6]
pf.make_video_of_tsne_iterations(iterations, video_dir, data_file_name='interim_{:0>6}.dat',
                                 video_file_name='tsne_video.mp4', figsize=(15, 15), dpi=200, fps=30,
                                 labels_dict=spikes_labeled_dict, legent_on=False, cm=cm, sizes=sizes, markers=markers)


# ______________ KILOSORT __________________________________________________________
# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
# KILOSORT UNITS ON TSNE PLOT (made from old phy's masked PCs)

from BrainDataAnalysis import Utilities as util
from BrainDataAnalysis.Graphics import ploting_functions as pf
import matplotlib.pyplot as plt
import numpy as np
import h5py as h5
import os
from ExperimentSpecificCode._2016_TSne_Paper.t_sne_bhcuda import bhtsne_cuda as TSNE
import pickle
import pandas as pd

filename = r'D:\Data\George\Projects\SpikeSorting\Joana_Paired_128ch\2015-09-03\Analysis\klustakwik\threshold_6_5std\threshold_6_5std.kwik'
h5file = h5.File(filename, mode='r')
spike_times_phy = np.array(list(h5file['channel_groups/0/spikes/time_samples']))
h5file.close()

t_tsne = np.load(r'D:\Data\George\Projects\SpikeSorting\Joana_Paired_128ch\2015-09-03\Analysis\klustakwik'+\
                 r'\threshold_6_5std\t_sne_results_final_allspikes.npy')

kilosort_experiment_folder = r'thres4_10_10_Fe16_Pc12'  # thres4_10_10_Fe16_Pc12 OR thres4_10_10_Fe256_Pc128 OR thres6_12_12_Fe256_Pc128
kilosort_path = os.path.join(r'D:\Data\George\Projects\SpikeSorting\Joana_Paired_128ch\2015-09-03\Analysis\kilosort', kilosort_experiment_folder)
spike_clusters_kilosort = np.load(os.path.join(kilosort_path, 'kilosort output\\spike_templates.npy'))
spike_times_kilosort = np.load(os.path.join(kilosort_path, 'kilosort output\spike_times.npy'))
template_features = np.load(os.path.join(kilosort_path, 'kilosort output\\template_features.npy'))
template_features_ind = np.load(os.path.join(kilosort_path, 'kilosort output\\template_feature_ind.npy'))
pc_features = np.load(os.path.join(kilosort_path, 'kilosort output\\pc_features.npy'))
pc_features_ind = np.load(os.path.join(kilosort_path, 'kilosort output\\pc_feature_ind.npy'))

common_spikes, indices_of_common_spikes_in_phy, indices_of_common_spikes_in_kilosort, small_train_spikes_not_found = util.find_points_in_array_with_jitter(spike_times_phy, spike_times_kilosort, 6)

np.save(os.path.join(kilosort_path, 'tsne\\small_train_spikes_not_found.npy'), small_train_spikes_not_found)
np.save(os.path.join(kilosort_path, 'tsne\\indices_of_common_spikes.npy'), indices_of_common_spikes_in_phy)
np.save(os.path.join(kilosort_path, 'tsne\common_spikes_in_tsne_train.npy'), common_spikes)
np.save(os.path.join(kilosort_path, 'tsne\indices_of_common_spikes_in_kilosort_train.npy'), indices_of_common_spikes_in_kilosort)
# OR
common_spikes = np.load(os.path.join(kilosort_path, 'tsne\\common_spikes_in_tsne_train.npy'))
indices_of_common_spikes_in_phy = np.load(os.path.join(kilosort_path, 'tsne\\indices_of_common_spikes.npy'))
indices_of_common_spikes_in_kilosort = np.load(os.path.join(kilosort_path, 'tsne\\indices_of_common_spikes_in_kilosort_train.npy'))
small_train_spikes_not_found = np.load(os.path.join(kilosort_path, 'tsne\\small_train_spikes_not_found.npy'))


# plot the phy tsne using the clusters defined by kilosort
kilosort_units = {} # dict has all the phy spikes
for i in np.arange(indices_of_common_spikes_in_kilosort.__len__()):
    index_in_kilosort = indices_of_common_spikes_in_kilosort[i]
    if spike_clusters_kilosort[index_in_kilosort][0] in kilosort_units:
        kilosort_units[spike_clusters_kilosort[index_in_kilosort][0]] = np.append(kilosort_units[spike_clusters_kilosort[index_in_kilosort][0]], indices_of_common_spikes_in_phy[i])
    else:
        kilosort_units[spike_clusters_kilosort[index_in_kilosort][0]] = [indices_of_common_spikes_in_phy[i]]

pf.plot_tsne(t_tsne, kilosort_units, subtitle='T-sne', cm=plt.cm.coolwarm, sizes=[2, 8])


# make sparse template_features matrix and t-sne this
template_features_sparse = np.zeros((template_features.shape[0], template_features_ind.shape[0]))
for spike in np.arange(template_features.shape[0]):
    cluster = spike_clusters_kilosort[spike][0]
    indices = template_features_ind[cluster, :]
    for i in np.arange(len(indices)):
        template_features_sparse[spike, indices[i]] = template_features[spike, i]

template_features_tsne = TSNE.t_sne(template_features_sparse, perplexity=100, theta=0.2,
                                        files_dir=os.path.join(kilosort_path, 'tsne'),
                                        iterations=2000, randseed=1)

template_features_tsne = TSNE.t_sne(template_features_sparse[indices_of_common_spikes_in_kilosort], perplexity=100, theta=0.2,
                                        files_dir=os.path.join(kilosort_path, 'tsne'),
                                        iterations=2000, randseed=1)


template_features_tsne = TSNE.load_tsne_result(os.path.join(kilosort_path, 'tsne'), 'tsne_template_features_466313sp_100per_2000its_02theta.dat')


kilosort_units = {} # dict has all the kilosort spikes
for i in np.arange(len(spike_clusters_kilosort)):
    cluster = spike_clusters_kilosort[i][0]
    if cluster in kilosort_units:
        kilosort_units[cluster] = np.append(kilosort_units[cluster], i)
    else:
        kilosort_units[cluster] = i


kilosort_units = {} # dict has all the kilosort spikes
for i in np.arange(indices_of_common_spikes_in_kilosort.__len__()):
    index_in_kilosort = indices_of_common_spikes_in_kilosort[i]
    if spike_clusters_kilosort[index_in_kilosort][0] in kilosort_units:
        kilosort_units[spike_clusters_kilosort[index_in_kilosort][0]] = np.append(kilosort_units[spike_clusters_kilosort[index_in_kilosort][0]], i)
    else:
        kilosort_units[spike_clusters_kilosort[index_in_kilosort][0]] = i

cluster_info = pd.DataFrame(columns=['Cluster', 'Num_of_Spikes', 'Spike_Indices'])
cluster_info = cluster_info.set_index('Cluster')
cluster_info['Spike_Indices'] = cluster_info['Spike_Indices'].astype(list)
for g in kilosort_units.keys():
    if np.size(kilosort_units[g]) == 1:
        kilosort_units[g] = [kilosort_units[g]]
    cluster_name = str(g)
    cluster_info.set_value(cluster_name, 'Num_of_Spikes', len(kilosort_units[g]))
    cluster_info.set_value(cluster_name, 'Spike_Indices', kilosort_units[g])

cluster_info.to_pickle(os.path.join(kilosort_path, r'tsne\cluster_info_full.pkl'))

pf.plot_tsne(np.transpose(template_features_tsne), labels_dict=kilosort_units, legend_on=False,
             subtitle='T-sne', cm=plt.cm.coolwarm, sizes=[2, 8])



# get the phy manual clusters and swap their indices with the ones in the template_features_tsne
pkl_file = open(r'D:\Data\George\Projects\SpikeSorting\Joana_Paired_128ch\2015-09-03\Analysis\klustakwik\threshold_6_5std\cluster_info.pkl', 'rb')
labels = pickle.load(pkl_file)
pkl_file.close()
phy_units = {}
i = 0
for name in labels.index:
    phy_cluster_indices = labels['Spike_Indices'][name]
    phy_units[i] = np.empty((0)).astype(np.int32)
    for k in np.arange(len(phy_cluster_indices)):
        t = np.where(indices_of_common_spikes_in_phy == phy_cluster_indices[k])[0].astype(np.int32)
        if len(t) > 0:
            phy_units[i] = np.append(phy_units[i], t.astype(int)[0])
    i += 1


ind_in_ks_list = indices_of_common_spikes_in_kilosort.tolist()
pf.plot_tsne(np.transpose(template_features_tsne)[:, ind_in_ks_list], labels_dict=phy_units,
              subtitle='T-sne', cm=plt.cm.coolwarm, sizes=[2, 8])


# make sparse pc_features matrix and t-sne this (or the part that is common with the phy tsne)
pc_features_sparse = np.zeros((pc_features.shape[0], pc_features.shape[1], pc_features_ind.shape[0]))
for spike in np.arange(template_features.shape[0]):
    cluster = spike_clusters_kilosort[spike][0]
    indices = pc_features_ind[cluster, :]
    for i in np.arange(len(indices)):
        for pc in np.arange(3):
            pc_features_sparse[spike, pc, indices[i]] = pc_features[spike, pc, i]
pc_features_sparse_flatten = np.reshape(pc_features_sparse, (pc_features.shape[0], pc_features.shape[1] * pc_features_ind.shape[0]))

indices_of_common_spikes_in_kilosort_updated = indices_of_common_spikes_in_kilosort[:-100]
pc_features_tsne = TSNE.t_sne(pc_features_sparse_flatten[indices_of_common_spikes_in_kilosort_updated, :],
                              perplexity=100, theta=0.2,
                              files_dir=os.path.join(kilosort_path, 'tsne'),
                              results_filename='tsne_pc_features_128ksp_per100_theta02_it2k_rs1.dat',
                              gpu_mem=0.8, iterations=2000, randseed=1)


pc_features_tsne = np.transpose(pc_features_tsne)

kilosort_units_minus100 = {} # dict has all the kilosort spikes
for i in np.arange(indices_of_common_spikes_in_kilosort_updated.__len__()):
    index_in_kilosort = indices_of_common_spikes_in_kilosort_updated[i]
    if spike_clusters_kilosort[index_in_kilosort][0] in kilosort_units_minus100:
        kilosort_units_minus100[spike_clusters_kilosort[index_in_kilosort][0]] = np.append(kilosort_units_minus100[spike_clusters_kilosort[index_in_kilosort][0]], i)
    else:
        kilosort_units_minus100[spike_clusters_kilosort[index_in_kilosort][0]] = i

pf.plot_tsne(pc_features_tsne, labels_dict=kilosort_units_minus100, legend_on=False,
             subtitle='T-sne', cm=plt.cm.coolwarm, sizes=[2, 8])

# make sparse pc_features matrix with all the kilosort spikes and t-sne this
pc_features_tsne = TSNE.t_sne(pc_features_sparse_flatten,
                              perplexity=100, theta=0.2,
                              files_dir=os.path.join(kilosort_path, 'tsne\pc_features_results'),
                              results_filename='tsne_pc_features_466ksp_per100_theta02_it2k_rs1.dat',
                              gpu_mem=0.8, iterations=2000, randseed=1, verbose=3)

pf.plot_tsne(np.transpose(pc_features_tsne), labels_dict=kilosort_units, legend_on=False,
             subtitle='T-sne', cm=plt.cm.coolwarm, sizes=[2, 8])

# manual clustering setup
import IO.ephys as ephys
from t_sne_bhcuda import tsne_cluster
from BrainDataAnalysis import ploting_functions as pf
import numpy as np
import os
from t_sne_bhcuda import bhtsne_cuda as TSNE

# Parameters for testing (128 channels)
base_dir = r'D:\Data\George\Projects\SpikeSorting\Joana_Paired_128ch\2015-09-03'
tsne_dir = r'Analysis\klustakwik\threshold_6_5std'
data_dir = r'Data'
data_cube_dir = r'Analysis\TempCube'
kilosort_experiment_folder = r'thres4_10_10_Fe16_Pc12'  # thres4_10_10_Fe16_Pc12 OR thres4_10_10_Fe256_Pc128 OR thres6_12_12_Fe256_Pc128
kilosort_path = os.path.join(r'D:\Data\George\Projects\SpikeSorting\Joana_Paired_128ch\2015-09-03\Analysis\kilosort', kilosort_experiment_folder)

template_features_tsne = TSNE.load_tsne_result(os.path.join(kilosort_path, 'tsne'), 'tsne_template_features_466313sp_100per_2000its_02theta.dat')
pc_features_tsne = TSNE.load_tsne_result(os.path.join(kilosort_path, r'tsne\pc_features_results'), 'tsne_pc_features_466ksp_per100_theta02_it2k_rs1.dat')
common_spikes = np.load(os.path.join(kilosort_path, 'tsne\\common_spikes_in_tsne_train.npy'))
indices_of_common_spikes_in_kilosort = np.load(os.path.join(kilosort_path, 'tsne\\indices_of_common_spikes_in_kilosort_train.npy'))

num_ivm_channels = 128
amp_dtype = np.uint16
cube_type = np.int32
sampling_freq = 30000
num_of_points_in_spike_trig = 64
num_of_points_for_baseline = 10

data_cube_filename = os.path.join(kilosort_path, 'tsne\\raw_data_cube.npy')

autocor_bin_number = 100
prb_file = r'D:\Data\George\Projects\SpikeSorting\Joana_Paired_128ch\2015-09-03\Analysis\klustakwik\128ch_passive_imec.prb'


# manual clustering the kilosort template_feature_tsne data that are common to the phy tsne
cluster_info_filename = os.path.join(kilosort_path, 'tsne\\cluster_info.pkl')
spike_indices_to_use = None#np.arange(20000)
num_of_spikes = len(common_spikes)
if spike_indices_to_use is not None:
    num_of_spikes = len(spike_indices_to_use)

shape_of_cut_extracellular_data = (num_ivm_channels, num_of_points_in_spike_trig, num_of_spikes)

raw_data_filename = os.path.join(base_dir, data_dir, 'amplifier2015-09-03T21_18_47.bin')
raw_data_ivm = ephys.load_raw_data(raw_data_filename, numchannels=num_ivm_channels, dtype=amp_dtype).dataMatrix

tsne_cluster.gui_manual_cluster_tsne_spikes(tsne_array_or_filename=np.transpose(template_features_tsne)[:, indices_of_common_spikes_in_kilosort.tolist()],
                                            spike_times_list_or_filename=common_spikes,
                                            raw_extracellular_data=raw_data_ivm,
                                            num_of_points_for_baseline=num_of_points_for_baseline,
                                            cut_extracellular_data_or_filename=data_cube_filename,
                                            shape_of_cut_extracellular_data=shape_of_cut_extracellular_data,
                                            cube_type=cube_type,
                                            sampling_freq=sampling_freq,
                                            autocor_bin_number=autocor_bin_number,
                                            cluster_info_file=cluster_info_filename,
                                            use_existing_cluster=False,
                                            spike_indices_to_use=spike_indices_to_use,
                                            prb_file=prb_file,
                                            k4=True,
                                            verbose=True)

# manual clustering all of the kilosort template_feature_tsne data
cluster_info_filename = os.path.join(kilosort_path, 'tsne\\cluster_info_full.pkl')
spike_indices_to_use = None
num_of_spikes = np.shape(template_features_tsne)[0]

shape_of_cut_extracellular_data = (num_ivm_channels, num_of_points_in_spike_trig, num_of_spikes)

raw_data_filename = os.path.join(base_dir, data_dir, 'amplifier2015-09-03T21_18_47.bin')
raw_data_ivm = ephys.load_raw_data(raw_data_filename, numchannels=num_ivm_channels, dtype=amp_dtype).dataMatrix

tsne_cluster.gui_manual_cluster_tsne_spikes(tsne_array_or_filename=np.transpose(template_features_tsne),
                                            spike_times_list_or_filename=np.reshape(spike_times_kilosort, (len(spike_times_kilosort))),
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

# manual clustering the kilosort pc_feature_tsne data
cluster_info_filename = os.path.join(kilosort_path, 'tsne\\cluster_info_full.pkl')
spike_indices_to_use = None
num_of_spikes = np.shape(pc_features_tsne)[0]

shape_of_cut_extracellular_data = (num_ivm_channels, num_of_points_in_spike_trig, num_of_spikes)

raw_data_filename = os.path.join(base_dir, data_dir, 'amplifier2015-09-03T21_18_47.bin')
raw_data_ivm = ephys.load_raw_data(raw_data_filename, numchannels=num_ivm_channels, dtype=amp_dtype).dataMatrix

tsne_cluster.gui_manual_cluster_tsne_spikes(tsne_array_or_filename=pc_features_tsne,
                                            spike_times_list_or_filename=np.reshape(spike_times_kilosort, (len(spike_times_kilosort))),
                                            raw_extracellular_data=raw_data_ivm,
                                            num_of_points_for_baseline=num_of_points_for_baseline,
                                            cut_extracellular_data_or_filename=data_cube_filename,
                                            shape_of_cut_extracellular_data=shape_of_cut_extracellular_data,
                                            cube_type=cube_type,
                                            sampling_freq=sampling_freq,
                                            autocor_bin_number=autocor_bin_number,
                                            cluster_info_file=cluster_info_filename,
                                            use_existing_cluster=False,
                                            spike_indices_to_use=spike_indices_to_use,
                                            prb_file=prb_file,
                                            k4=True,
                                            verbose=True)