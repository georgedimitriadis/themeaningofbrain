

import os
import numpy as np
import BrainDataAnalysis.timelocked_analysis_functions as tf
import IO.ephys as ephys
import mne.filter as filters
from sklearn.manifold import TSNE as tsne
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import BrainDataAnalysis.ploting_functions as pf
import random
from matplotlib import colors
import Layouts.Probes.probes_imec as pr_imec
import h5py as h5
import IO.klustakwik as klusta
import pickle


base_folder = r'D:\Data\George\Projects\SpikeSorting\Joana_Paired_128ch'
rat = 97
good_cells = '9'
date = '2015-09-03'
data_folder = os.path.join(base_folder + '\\' + date, 'Data')
analysis_folder = os.path.join(base_folder + '\\' + date, 'Analysis')

cell_capture_times = '21_18_47'
spike_thresholds = -2e-4

num_of_points_in_spike_trig_ivm = 64
num_of_points_for_padding = 2 * num_of_points_in_spike_trig_ivm

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
                         'c': 'ivm_data_filtered_continous_cell{}.dat',
                         'k': 'ivm_data_filtered_klusta_{}spikes_cell{}.dat'}


# Generate the spike time triggers (and the adc traces in Volts)
raw_data_file_pipette = os.path.join(data_folder, 'adc'+date+'T'+cell_capture_times+'.bin')

raw_data_pipette = ephys.load_raw_event_trace(raw_data_file_pipette, number_of_channels=8,
                                              channel_used=adc_channel_used, dtype=adc_dtype)
spike_triggers, spike_data_in_V = tf.create_spike_triggered_events(raw_data_pipette.dataMatrix,
                                                                   threshold=spike_thresholds,
                                                                   inter_spike_time_distance=inter_spike_time_distance,
                                                                   amp_gain=amp_gain)
print(len(spike_triggers))

















# Single Cell comparison between Klusta and tsne
# Create the required .dat file and the probe .prb file to throw into klustakwik (i.e. the phy module)
spikes_to_include = 3893
fin_time_point = spike_triggers[spikes_to_include] + 500
start_time_point = 0
time_limits = [start_time_point, fin_time_point]

raw_data_file_ivm = os.path.join(data_folder, 'amplifier'+date+'T'+cell_capture_times+'.bin')
raw_data_ivm = ephys.load_raw_data(raw_data_file_ivm, numchannels=num_ivm_channels, dtype=amp_dtype)
file_dat = os.path.join(analysis_folder, r'klustakwik\raw_data_ivm_klusta.dat')

klusta.make_dat_file(raw_data_ivm.dataMatrix, num_channels=num_ivm_channels, time_limits=time_limits, filename=file_dat)


file_prb = os.path.join(analysis_folder, r'klustakwik\128ch_passive_imec.prb')
electrode_structure = pr_imec.create_128channels_imec_prb(file_prb)



# Grab the spike times from the .kwik file
filename = r'E:\Linux Mint\Shared\klustakwik_analysis\Joana_Paired_128ch\rat97_cell9\1500juxta_20455extra\rat97_cell9_1500spikes.kwik'
h5file = h5.File(filename, mode='r')
all_extra_spike_times = np.array(list(h5file['channel_groups/0/spikes/time_samples']))
klusta_clusters = np.array(list(h5file['channel_groups/0/spikes/clusters/main']))
h5file.close()


# Find the common spikes between the klusta spikedetct results and the juxtacellular spikes
common_spikes = []
indices_of_common_spikes_in_klusta = []
clusters_of_common_spikes = []
d_time_points = 5
prev_spikes_added = 0
curr_spikes_added = 0
ks = 0
juxta_spikes_not_found = []
index_of_klusta_spike_found = 0
index_of_klusta_spike_found = 0
for juxta_spike in spike_triggers[index_of_klusta_spike_found:spikes_to_include]:
    for possible_extra_spike in np.arange(juxta_spike - d_time_points, juxta_spike + d_time_points):
        possible_possitions = np.where(all_extra_spike_times == possible_extra_spike)[0]
        if len(possible_possitions) != 0:
            index_of_klusta_spike_found = possible_possitions[0]
            common_spikes.append(all_extra_spike_times[index_of_klusta_spike_found])
            indices_of_common_spikes_in_klusta.append(index_of_klusta_spike_found)
            clusters_of_common_spikes.append(klusta_clusters[index_of_klusta_spike_found])
            curr_spikes_added += 1
            break
    if curr_spikes_added > prev_spikes_added:
        prev_spikes_added = curr_spikes_added
    else:
        juxta_spikes_not_found.append(juxta_spike)



# Generate the data to go into t-sne for all spikes detected by klusta's spikedetect
data_to_load = 'k'
for i in np.arange(0, len(good_cells)):
    raw_data_file_ivm = os.path.join(data_folder, 'amplifier'+date+'T'+cell_capture_times+'.bin')

    raw_data_ivm = ephys.load_raw_data(raw_data_file_ivm, numchannels=num_ivm_channels, dtype=amp_dtype)

    num_of_spikes = len(all_extra_spike_times)

    shape_of_filt_spike_trig_ivm = (num_ivm_channels,
                                    num_of_points_in_spike_trig_ivm,
                                    num_of_spikes)
    ivm_data_filtered = np.memmap(os.path.join(analysis_folder, types_of_data_to_load[data_to_load].format(spikes_to_include, good_cells[i])),
                                    dtype=filtered_data_type,
                                    mode='w+',
                                    shape=shape_of_filt_spike_trig_ivm)
    for spike in np.arange(0, num_of_spikes):
        trigger_point = all_extra_spike_times[spike]
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
    del ivm_data_filtered
    time_axis = np.arange(-num_of_points_in_spike_trig_ivm/(2*sampling_freq),
                      num_of_points_in_spike_trig_ivm/(2*sampling_freq),
                      1/sampling_freq)

spikes_to_include = ''
data_to_load = 'k'
for i in np.arange(0, len(good_cells)):
    num_of_spikes = len(all_extra_spike_times)
    shape_of_filt_spike_trig_ivm = (num_ivm_channels,
                                    num_of_points_in_spike_trig_ivm,
                                    num_of_spikes)
    ivm_data_filtered = np.memmap(os.path.join(analysis_folder, types_of_data_to_load[data_to_load].format(spikes_to_include, good_cells[i])),
                                    dtype=filtered_data_type,
                                    mode='r',
                                    shape=shape_of_filt_spike_trig_ivm)


X = []
data_to_use = ivm_data_filtered
num_of_spikes = data_to_use.shape[2]
number_of_time_points = 20
newshape = (num_ivm_channels*number_of_time_points, num_of_spikes)
start = int((num_of_points_in_spike_trig_ivm - number_of_time_points) / 2)
end = start + number_of_time_points
t = data_to_use[:, start:end, :num_of_spikes]
t = np.reshape(t, newshape=newshape)
X.extend(t.T)
X_np = np.array(X)

indices_of_data_for_tsne = [(i, k)[0] for i, k in enumerate(X) if random.random() > 0]
data_for_tsne = X_np[indices_of_data_for_tsne]
juxta_cluster_indices_temp = np.intersect1d(indices_of_data_for_tsne, indices_of_common_spikes_in_klusta)
juxta_cluster_indices = [i for i in np.arange(0, len(indices_of_data_for_tsne)) if
                         len(np.where(juxta_cluster_indices_temp == indices_of_data_for_tsne[i])[0])]


# T-SNE
perplexity = 800.0
early_exaggeration = 100.0
learning_rate = 3000.0
model = tsne(n_components=2, perplexity=perplexity, early_exaggeration=early_exaggeration,
             learning_rate=learning_rate, n_iter=1000, n_iter_without_progress=500,
             min_grad_norm=1e-7, metric="euclidean", init="random", verbose=3,
             random_state=None, method='barnes_hut', angle=0.5)
t_tsne = model.fit_transform(data_for_tsne)
t_tsne = t_tsne.T


file_name = r'D:\Data\George\Projects\SpikeSorting\Joana_Paired_128ch\2015-09-03\Analysis\t_tsne_ivm_data_{}sp_{}per_{}ee_{}lr.pkl'\
    .format(len(indices_of_data_for_tsne), perplexity, early_exaggeration, learning_rate)
file = open(file_name, 'bw')
pickle.dump((ivm_data_filtered, t_tsne, juxta_cluster_indices, perplexity, early_exaggeration, learning_rate), file)
file.close()


#  2D plot
fig = plt.figure()
ax = fig.add_subplot(111)
s = 5
ax.scatter(t_tsne[0], t_tsne[1], s=s)
ax.scatter(t_tsne[0][juxta_cluster_indices], t_tsne[1][juxta_cluster_indices], s=s, color='r')
fig.suptitle('T-SNE with Per={}, exag={}, l_rate={}, spikes={}'.format(perplexity, early_exaggeration, learning_rate,
                                                                       len(indices_of_data_for_tsne)))




