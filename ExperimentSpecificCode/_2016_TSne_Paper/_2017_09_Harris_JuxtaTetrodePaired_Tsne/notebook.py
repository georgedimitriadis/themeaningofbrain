
import numpy as np
from os.path import join
from IO import ephys
import BrainDataAnalysis.timelocked_analysis_functions as tf
import mne.filter as filters
import Layouts.Probes.klustakwik_prb_generator as prb_gen
import h5py as h5
import BrainDataAnalysis.Graphics.ploting_functions as pf
from spikesorting_tsne import tsne as TSNE
import matplotlib.pyplot as plt
from tsne_for_spikesort_old import io_with_cpp as io

# DEFINE THE CELL TO WORK ON _________
recording = 'd5331'
cell = '01'
# ____________________________________


full_cell_name = recording + cell

base_folder = join('Z:\g\George\DataAndResults\Harris_JuxtaTetrodePairedData', recording)
analysis_folder = join(base_folder, 'Analysis')


data = ephys.load_raw_data(join(base_folder, 'Data', full_cell_name+'.dat'), numchannels=8, dtype=np.int16)


ic_channel = 5
ec_channels = [1, 2, 3, 4]

# Make the binary data file for klustakwik--
extra_data = data.dataMatrix[ec_channels, :]
extra_data_to_bin = np.reshape(extra_data.T, (extra_data.size))
extra_data_to_bin.tofile(join(analysis_folder, 'KlustaKwik', cell, full_cell_name+'_extra.dat'))
# ------------------------------------------

sampling_freq = 10000

juxta_spike_triggers, juxta_spike_peaks, juxta_spike_data_in_V = \
 tf.create_spike_triggered_events(data_raw_spikes=data.dataMatrix[ic_channel, :], threshold=16e-5,
                                  inter_spike_time_distance=0.005, amp_gain=1000,
                                  sampling_freq=sampling_freq, amp_y_digitization=2**16, amp_y_range=20)
num_of_spikes = len(juxta_spike_triggers)

# ---------------------------------------
# DO THE KLUSTA HERE---------------------
# ---------------------------------------

# Grab the spike times from the .kwik file
filename_kwik = join(analysis_folder, 'KlustaKwik', cell, full_cell_name + '.kwik')
h5file = h5.File(filename_kwik, mode='r')
negative_extra_spike_times = np.array(list(h5file['channel_groups/0/spikes/time_samples']))
h5file.close()
print('All extra spikes = {}'.format(len(negative_extra_spike_times)))


# Function to find the common spikes between the klusta spikedetct results and the juxtacellular spikes
def find_juxta_spikes_in_extra_detected(juxta_spikes_to_include, juxta_spike_triggers, all_extra_spike_times, d_time_points):
    common_spikes = []
    indices_of_common_spikes_in_klusta = []
    prev_spikes_added = 0
    curr_spikes_added = 0
    juxta_spikes_not_found = []
    index_of_klusta_spike_found = 0
    for juxta_spike in juxta_spike_triggers[index_of_klusta_spike_found:juxta_spikes_to_include]:
        for possible_extra_spike in np.arange(juxta_spike - d_time_points, juxta_spike + d_time_points):
            possible_possitions = np.where(all_extra_spike_times == possible_extra_spike)[0]
            if len(possible_possitions) != 0:
                index_of_klusta_spike_found = possible_possitions[0]
                common_spikes.append(all_extra_spike_times[index_of_klusta_spike_found])
                indices_of_common_spikes_in_klusta.append(index_of_klusta_spike_found)
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

# Find the common spikes between the klusta spikedetct results and
# the juxtacellular spikes for all the juxta spikes and for all the sub groups of juxta spikes
common_spikes, indices_of_common_spikes_in_klusta, juxta_spikes_not_found = \
    find_juxta_spikes_in_extra_detected(juxta_spikes_to_include=spikes_to_include,
                                        juxta_spike_triggers=juxta_spike_triggers,
                                        all_extra_spike_times=negative_extra_spike_times,
                                        d_time_points=10)


# Grab the mask and the PCA components for all spikes from the .kwx file
filename_kwx = join(analysis_folder, 'KlustaKwik', cell, full_cell_name + '.kwx')
h5file = h5.File(filename_kwx, mode='r')
pca_and_masks = np.array(list(h5file['channel_groups/0/features_masks']))
masks = np.array(pca_and_masks[:, :, 1])
pca_features = np.array(pca_and_masks[:, :, 0])
masked_pca_features = pca_features * masks


# Do the TSne on the masked pca features
files_dir = join(analysis_folder, 'KlustaKwik', cell, 'tsne')
exe_dir = r'E:\Software\Develop\Source\Repos\spikesorting_tsne_bhpart\Barnes_Hut\x64\Release'
theta = 0.1
eta = 200.0
num_dims = 2
perplexity = 100
iterations = 4000
random_seed = 1
verbose = 2
tsne = TSNE.t_sne(samples=masked_pca_features, files_dir=files_dir, exe_dir=exe_dir, num_dims=num_dims,
                  perplexity=perplexity, theta=theta, eta=eta, iterations=iterations, random_seed=random_seed,
                  verbose=verbose)

# OR
tsne = io.load_tsne_result(files_dir)


# Grab the clusters from klustakwik
clusters_text_file = open(join(analysis_folder, 'KlustaKwik', cell, '.klustakwik2', "spike_clusters.0.txt"), "r")
clusters_of_all_extra_spikes = np.array([int(line) for line in clusters_text_file.readlines()])
clusters_of_all_extra_spikes[np.argwhere(clusters_of_all_extra_spikes==0)] = 1


# Make labels for tsne pic
#  Show only juxta
labels_dict = {1: indices_of_common_spikes_in_klusta,
               2: np.argwhere(np.invert(np.in1d(negative_extra_spike_times, common_spikes)))
               }
markers = ['o', '*']

#  Show clusters
num_of_clusters = len(np.unique(clusters_of_all_extra_spikes))
labels_dict = {}
for g in np.arange(len(np.unique(clusters_of_all_extra_spikes))) + 1:
    if g not in labels_dict.keys():
        labels_dict[g] = []
    for spike in np.arange(len(clusters_of_all_extra_spikes)):
        if clusters_of_all_extra_spikes[spike] == g:
            labels_dict[g].append(spike)
markers = ['.', '*', 'o', '>', '<', '_', ',']

fig, ax = pf.plot_tsne(tsne.T, cm=plt.cm.jet, labels_dict=labels_dict, legent_on=False, markers=None, labeled_sizes=[30])

# and add the juxta spikes
ax.scatter(tsne[indices_of_common_spikes_in_klusta, 0], tsne[indices_of_common_spikes_in_klusta, 1], marker='*',
           c='r', s=20)

# CALCULATE PRECISION AND RECALL OF JUXTA ON T-SNE BLOB
spikes_in_tsne_blob = 1070  # Taken from using the spikesorting gui on the data
number_of_juxta_not_in_blob = 17  # Found by counting on the t-sne plot
number_of_juxta = len(common_spikes)
tp = number_of_juxta - number_of_juxta_not_in_blob
all_pos = number_of_juxta
all_true = number_of_juxta
precision = tp / all_pos
recall = tp / all_true
f_factor = 2*(precision*recall)/(precision+recall)
print('Precision = {}, Recall = {}, F1 factor = {}'.format(precision, recall, f_factor))

# ________________________________________
# ONE OFF FUNCTIONS
# ________________________________________


# Generate prb file ---------------
all_electrodes = np.array([0, 1, 2, 3]).reshape((2, 2))
prb_filename = join(base_folder, 'Analysis', 'KlustaKwik', 'tetrode.prb')
prb_gen.generate_prb_file(filename=prb_filename, all_electrodes_array=all_electrodes,
                          steps_r=2, steps_c=2)
# ---------------------------------

# SPIKEDETECT AND PCA---------------------
# filter all data
high_pass_freq = 500
filtered_data_type = np.float64
temp_unfiltered = extra_data
temp_unfiltered = temp_unfiltered.astype(filtered_data_type)
iir_params = {'order': 4, 'ftype': 'butter', 'padlen': 0}
extra_data_hp = filters.filter_data(temp_unfiltered, sampling_freq, l_freq=high_pass_freq, h_freq=None, method='iir',
                                    iir_params=iir_params)
# Find thresholds
stdvs  = np.median(np.abs(extra_data_hp)/0.6745, axis=1)

large_thresholds = np.zeros(np.shape(extra_data_hp))
small_thresholds = np.zeros(np.shape(extra_data_hp))
for c in range(len(ec_channels)):
    large_thresholds[c, :] = 4.5 * stdvs[c]
    small_thresholds[c, :] = 2 * stdvs[c]

# Generate thresholded array of -1 (if negative threshold is passed), +1 (if possitive threshold is passed)
# and 0 otherwise
threshold_crossing_regions = np.zeros(np.shape(extra_data_hp))
threshold_crossing_regions[extra_data_hp < -large_thresholds] = -1
threshold_crossing_regions[extra_data_hp > large_thresholds] = 1

negative_extra_spike_times = np.argwhere(threshold_crossing_regions[0] < -0.5)
for c in range(1, len(ec_channels)):
    tcr = np.argwhere(threshold_crossing_regions[c] < -0.5)
    for i in range(len(tcr)):
        if np.abs(negative_extra_spike_times - tcr[i]).min() > 3:
            negative_extra_spike_times = np.append(negative_extra_spike_times, tcr[i])

# ------------------------------------------


# ----------------------------------------------------------------------------------
# Cut the extracelular traces into a channels x time x spikes cube and high pass them

def make_cube_on_spike_triggers(filename, spike_times, high_pass_freq=500, num_of_points_in_spike_triggered_cube = 100):
    filtered_data_type = np.float64

    num_of_spikes = len(spike_times)

    num_of_points_for_padding = 50

    shape_of_filt_spike_triggered_cube = ((len(ec_channels),
                                          num_of_points_in_spike_triggered_cube,
                                          num_of_spikes))
    spike_triggered_cube_hp = np.memmap(filename,
                                              dtype=np.int16,
                                              mode='w+',
                                              shape=shape_of_filt_spike_triggered_cube)

    for spike in np.arange(0, num_of_spikes):
        trigger_point = negative_extra_spike_times[spike]
        start_point = int(trigger_point - (num_of_points_in_spike_triggered_cube + num_of_points_for_padding)/2)
        if start_point < 0:
            break
        end_point = int(trigger_point + (num_of_points_in_spike_triggered_cube + num_of_points_for_padding)/2)
        if end_point > extra_data.shape[1]:
            break
        temp_unfiltered = extra_data[:, start_point:end_point]
        temp_unfiltered = temp_unfiltered.astype(filtered_data_type)
        iir_params = {'order': 4, 'ftype': 'butter', 'padlen': 0}
        temp_filtered = filters.filter_data(temp_unfiltered, sampling_freq, l_freq=high_pass_freq, h_freq=None,
                                            method='iir', iir_params=iir_params)  # 4th order Butter with no padding
        temp_filtered = temp_filtered[:, int(num_of_points_for_padding / 2):-int(num_of_points_for_padding / 2)]
        spike_triggered_cube_hp[:, :, spike] = temp_filtered
        if spike % 100 == 0:
            print('Done ' + str(spike) + 'spikes')

    return spike_triggered_cube_hp


num_of_points_in_spike_triggered_cube = 100

filename_juxta = join(analysis_folder, 'Basic', full_cell_name+'_juxta_spike_triggered_cube.data')
juxta_spike_triggered_cube_hp = make_cube_on_spike_triggers(filename_juxta, juxta_spike_triggers,
                                                            num_of_points_in_spike_triggered_cube=
                                                            num_of_points_in_spike_triggered_cube)


filename_extra = join(analysis_folder, 'Basic', full_cell_name+'_extra_spike_triggered_cube.data')
extra_spike_triggered_cube_hp = make_cube_on_spike_triggers(filename_extra, negative_extra_spike_times,
                                                            num_of_points_in_spike_triggered_cube=
                                                            num_of_points_in_spike_triggered_cube)


timeAxis = np.arange(-num_of_points_in_spike_triggered_cube / (2 * sampling_freq),
                     num_of_points_in_spike_triggered_cube / (2 * sampling_freq), 1 / sampling_freq)
pf.scan_through_3rd_dim(extra_spike_triggered_cube_hp, timeAxis)
# ----------------------------------------------------------------------------------

