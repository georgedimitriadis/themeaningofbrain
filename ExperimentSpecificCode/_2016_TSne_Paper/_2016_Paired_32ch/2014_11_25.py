

from Layouts.Probes import probes_neuronexus as prnn
from IO import ephys as ioep
from IO import klustakwik as iokl
from os.path import join
import h5py as h5
import numpy as np
import BrainDataAnalysis.Utilities as ut
import BrainDataAnalysis.timelocked_analysis_functions as tf
import BrainDataAnalysis.ploting_functions as pf
import matplotlib.pyplot as plt
import IO.ephys as ephys
import ExperimentSpecificCode._2016_TSne_Paper.t_sne_bhcuda.t_sne_spikes as tsne_spikes
from sklearn.cluster import DBSCAN
from sklearn import metrics
import BrainDataAnalysis.tsne_analysis_functions as taf


base_directory = r'D:\Data\George\Projects\SpikeSorting\Joana_Paired_32ch\2014-11-25'

num_ivm_channels = 32

spike_thresholds = 8e-4
adc_channel_used = 0
adc_dtype = np.uint16
inter_spike_time_distance = 0.002
amp_gain = 100
amp_dtype = np.uint16

good_channels = [8, 18, 13, 23, 28, 3, 9, 29, 2, 22]

# Generate the data phy requires
# 1) Generate the prb file
filename_prb = join(base_directory, 'Analysis\klustakwik_cell3\passive_cerebro_dense.prb')
all_electrodes, channel_positions = prnn.create_32channels_nn_prb(filename_prb)


# 2) Generate the .dat file
filename_raw_data = join(base_directory, r'Data\amplifier2014-11-25T23_00_08.bin')
raw_data = ioep.load_raw_data(filename=filename_raw_data)
filename_kl_data = join(base_directory, r'Analysis\klustakwik_cell3\raw_data_klusta.dat')
iokl.make_dat_file(raw_data=raw_data.dataMatrix, num_channels=num_ivm_channels, filename=filename_kl_data)



# Get extra spike times (as found by phy detect)
kwik_file_path = join(base_directory, r'Analysis\klustakwik_cell3\threshold_6_5std', 'amplifier_depth_70um_s16.kwik')
h5file = h5.File(kwik_file_path, mode='r')
extra_spike_times = np.array(list(h5file['channel_groups/0/spikes/time_samples']))
h5file.close()

spikes_used = len(extra_spike_times)
print(len(extra_spike_times))

# Get the juxta spikes and generate labels
# 1) Generate the juxta spike time triggers (and the adc traces in Volts)
raw_data_file_patch = join(base_directory, r'Data/adc2014-11-25T23_00_08.bin')

raw_data_patch = ephys.load_raw_event_trace(raw_data_file_patch, number_of_channels=8,
                                              channel_used=adc_channel_used, dtype=adc_dtype)
juxta_spike_triggers, juxta_spike_peaks, juxta_spike_data_in_V = tf.create_spike_triggered_events(raw_data_patch.dataMatrix,
                                                                   threshold=spike_thresholds,
                                                                   inter_spike_time_distance=inter_spike_time_distance,
                                                                   amp_gain=amp_gain)
num_of_spikes = len(juxta_spike_triggers)
print(num_of_spikes)




# 2) Seperate the juxta spikes into a number of groups according to their size
num_of_spike_groups = 10
juxta_spikes_grouped, juxta_spike_peaks_grouped, juxta_spike_triggers_grouped_withnans,\
        juxta_spike_peaks_grouped_withnans, spike_thresholds_groups = \
    taf.split_juxta_spikes_into_groups_by_size(num_of_spike_groups=num_of_spike_groups,
                                               juxta_spike_peaks=juxta_spike_peaks,
                                               juxta_spike_triggers=juxta_spike_triggers)


def select_spikes_in_certain_channels(common_spikes, indices_of_common_spikes_in_extra, raw_data, good_channels):
    common_spikes = np.array(common_spikes)
    indices_of_common_spikes_in_extra = np.array(indices_of_common_spikes_in_extra)
    t = raw_data.dataMatrix[:, common_spikes]
    spike_channels = np.argmin(t, axis=0)
    good_spike_indices = [i for i,x in list(enumerate(spike_channels)) if np.in1d(x, good_channels)]
    common_spikes = common_spikes[good_spike_indices]
    indices_of_common_spikes_in_extra = indices_of_common_spikes_in_extra[good_spike_indices]
    return common_spikes, indices_of_common_spikes_in_extra


# 3) Find the common spikes between the extra apikes and the juxta spikes for all the juxta spikes and for all the
# sub groups of juxta spikes
common_spikes, indices_of_common_spikes_in_extra, juxta_spikes_not_found = \
     ut.find_points_in_array_with_jitter(array_of_points_to_be_found=juxta_spike_triggers,
                                         array_to_search=extra_spike_times[:spikes_used],
                                         jitter_around_each_point=7)
common_spikes_chan_selected = select_spikes_in_certain_channels(common_spikes, indices_of_common_spikes_in_extra,
                                                                raw_data, good_channels)

# 4) Group the good spikes
common_spikes_grouped = {}
common_spikes_grouped_chan_selected = {}
juxta_spikes_not_found_grouped = {}
indices_of_common_extra_spikes_grouped = {}
indices_of_common_extra_spikes_grouped_chan_selected = {}
for g in range(1, num_of_spike_groups+1):
    common_spikes_grouped[g], indices_of_common_extra_spikes_grouped[g], juxta_spikes_not_found_grouped[g] = \
         ut.find_points_in_array_with_jitter(array_of_points_to_be_found=juxta_spikes_grouped[g],
                                             array_to_search=extra_spike_times[:spikes_used],
                                             jitter_around_each_point=7)
    common_spikes_grouped_chan_selected[g], indices_of_common_extra_spikes_grouped_chan_selected[g] \
        = select_spikes_in_certain_channels(common_spikes_grouped[g], indices_of_common_extra_spikes_grouped[g],
                                            raw_data, good_channels)






# Run t-sne
kwx_file_path = join(base_directory, r'Analysis\klustakwik_cell3\threshold_6_5std', 'amplifier_depth_70um_s16.kwx')
perplexity = 100
theta = 0.2
iterations = 2000
gpu_mem = 0.8
eta = 200
early_exaggeration = 4.0
seed = 0
verbose = 2
tsne = tsne_spikes.t_sne_spikes(kwx_file_path, hdf5_dir_to_pca=r'channel_groups/0/features_masks',
                                mask_data=True, perplexity=perplexity, theta=theta, iterations=iterations,
                                gpu_mem=gpu_mem, seed=seed, eta=eta, early_exaggeration=early_exaggeration,
                                verbose=verbose, indices_of_spikes_to_tsne=range(spikes_used))


# Load t-sne results
tsne = np.load(join(join(base_directory, r'Analysis\klustakwik_cell3\threshold_6_5std'),
                    't_sne_results_it10k_p1000_th02_eta200.npy'))


# 2D plot
indices_of_data_for_tsne = range(spikes_used)
juxta_cluster_indices_grouped = {}
for g in range(0, num_of_spike_groups):
    juxta_cluster_indices_temp = np.intersect1d(indices_of_data_for_tsne, indices_of_common_extra_spikes_grouped[g+1])
    juxta_cluster_indices_grouped[g] = [i for i in np.arange(0, len(indices_of_data_for_tsne)) if
                             len(np.where(juxta_cluster_indices_temp == indices_of_data_for_tsne[i])[0])]
    print(len(juxta_cluster_indices_grouped[g]))

pf.plot_tsne(tsne, juxta_cluster_indices_grouped, cm=plt.cm.coolwarm,
             subtitle='T-sne of 74000 spikes from the 32 channel probe',
             label_name='Peak size in uV',
             label_array=(spike_thresholds_groups*1e6).astype(int),
             sizes=[2, 15])



#------------------------------------------------------------------
# Clustering


# Dbscan
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



db, n_clusters_, labels, core_samples_mask, score = fit_dbscan(tsne, 0.75, 10, show=True)
pf.show_clustered_tsne(db, tsne, juxta_cluster_indices_grouped=None, threshold_legend=None)



# Define TP / FP / TN and FN
X = np.transpose(tsne)
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