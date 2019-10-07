

import numpy as np
import BrainDataAnalysis.Graphics.ploting_functions as pf
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics
import BrainDataAnalysis.Utilities as ut
import IO.ephys as ephys
import BrainDataAnalysis.timelocked_analysis_functions as tf
import h5py as h5


def select_spikes_in_certain_channels(spike_threshold, raw_data_file, common_spikes, indices_of_common_spikes_in_extra,
                                      good_channels, num_of_raw_data_channels):
    """
    Select spikes that appear only on the good_channels

    Parameters
    ----------
    spike_threshold
    raw_data_file
    common_spikes
    indices_of_common_spikes_in_extra
    good_channels
    num_of_raw_data_channels

    Returns
    -------

    """
    raw_data = ephys.load_raw_data(filename=raw_data_file, numchannels=num_of_raw_data_channels)
    common_spikes = np.array(common_spikes)
    indices_of_common_spikes_in_extra = np.array(indices_of_common_spikes_in_extra)
    t = raw_data.dataMatrix[:, common_spikes]
    if spike_threshold > 0:
        spike_channels = np.argmin(t, axis=0)
    if spike_threshold < 0:
        spike_channels = np.argmax(t, axis=0)
    good_spike_indices = [i for i, x in list(enumerate(spike_channels)) if np.in1d(x, good_channels)]
    common_spikes = common_spikes[good_spike_indices]
    indices_of_common_spikes_in_extra = indices_of_common_spikes_in_extra[good_spike_indices]
    return common_spikes, indices_of_common_spikes_in_extra


def create_juxta_label(kwik_file, spike_thresholds, num_of_spike_groups=1,
                       adc_channel_used=0, adc_dtype=np.uint16, inter_spike_time_distance=0.002,
                       amp_gain=100,
                       num_of_raw_data_channels=None,
                       spike_channels=None,
                       verbose=True):
    """
    Find the juxta spikes in the extra spike train and label them according to size splitting them into
    num_of_spike_groups groups

    Parameters
    ----------
    kwik_file
    spike_thresholds
    num_of_spike_groups
    adc_channel_used
    adc_dtype
    inter_spike_time_distance
    amp_gain
    num_of_raw_data_channels
    spike_channels
    verbose

    Returns
    -------

    """

    h5file = h5.File(kwik_file, mode='r')
    extra_spike_times = np.array(list(h5file['channel_groups/0/spikes/time_samples']))
    h5file.close()

    spikes_used = len(extra_spike_times)
    if verbose:
        print("Total spikes in extra = " + str(len(extra_spike_times)))


    # Get the juxta spikes and generate labels
    # 1) Generate the juxta spike time triggers (and the adc traces in Volts)
    raw_juxta_data_file = r'D:\Data\George\Projects\SpikeSorting\Joana_Paired_128ch\2015-09-03\Data' + \
                          r'\adc2015-09-03T21_18_47.bin'
    raw_data_patch = ephys.load_raw_event_trace(raw_juxta_data_file, number_of_channels=8,
                                                  channel_used=adc_channel_used, dtype=adc_dtype)
    juxta_spike_triggers, juxta_spike_peaks, juxta_spike_data_in_V = tf.create_spike_triggered_events(raw_data_patch.dataMatrix,
                                                                       threshold=spike_thresholds,
                                                                       inter_spike_time_distance=inter_spike_time_distance,
                                                                       amp_gain=amp_gain)
    num_of_spikes = len(juxta_spike_triggers)
    if verbose:
        print('Total spikes in Juxta = ' + str(num_of_spikes))



    # 2) Seperate the juxta spikes into a number of groups according to their size
    juxta_spikes_grouped, juxta_spike_peaks_grouped, juxta_spike_triggers_grouped_withnans,\
            juxta_spike_peaks_grouped_withnans, spike_thresholds_groups = \
        split_juxta_spikes_into_groups_by_size(num_of_spike_groups=num_of_spike_groups,
                                                   juxta_spike_peaks=juxta_spike_peaks,
                                                   juxta_spike_triggers=juxta_spike_triggers)

    # 3) Find the common spikes between the extra apikes and the juxta spikes for all the juxta spikes and for all the
    # sub groups of juxta spikes and group the good spikes
    common_spikes_grouped = {}
    juxta_spikes_not_found_grouped = {}
    indices_of_common_extra_spikes_grouped = {}
    for g in range(1, num_of_spike_groups+1):
        common_spikes_grouped[g], indices_of_common_extra_spikes_grouped[g], juxta_spikes_not_found_grouped[g] = \
             ut.find_points_in_array_with_jitter(array_of_points_to_be_found=juxta_spikes_grouped[g],
                                                 array_to_search=extra_spike_times[:spikes_used],
                                                 jitter_around_each_point=7)
    if spike_channels is not None and num_of_raw_data_channels is not None:
        common_spikes_grouped[g], indices_of_common_extra_spikes_grouped[g] \
            = select_spikes_in_certain_channels(spike_thresholds, common_spikes_grouped[g],
                                                indices_of_common_extra_spikes_grouped[g],
                                                spike_channels, num_of_raw_data_channels)


    # 5) Get the t-sne indices of the grouped juxta spikes
    indices_of_data_for_tsne = range(spikes_used)
    juxta_cluster_indices_grouped = {}
    for g in range(0, num_of_spike_groups):
        juxta_cluster_indices_temp = np.intersect1d(indices_of_data_for_tsne, indices_of_common_extra_spikes_grouped[g+1])
        juxta_cluster_indices_grouped[g] = [i for i in np.arange(0, len(indices_of_data_for_tsne)) if
                                 len(np.where(juxta_cluster_indices_temp == indices_of_data_for_tsne[i])[0])]
        if verbose and spike_channels is not None:
                print('Labeled after cleaning = ' + str(len(juxta_cluster_indices_grouped[g])))

    return juxta_cluster_indices_grouped, spike_thresholds_groups


def split_juxta_spikes_into_groups_by_size(num_of_spike_groups, juxta_spike_peaks, juxta_spike_triggers):
    spike_thresholds_groups = np.arange(np.min(juxta_spike_peaks), np.max(juxta_spike_peaks),
                                        (np.max(juxta_spike_peaks) - np.min(juxta_spike_peaks)) / num_of_spike_groups)
    spike_thresholds_groups = np.append(spike_thresholds_groups, np.max(juxta_spike_peaks))
    juxta_spikes_grouped = {}
    juxta_spike_peaks_grouped = {}
    juxta_spike_triggers_grouped_withnans = {}
    juxta_spike_peaks_grouped_withnans = {}
    for t in range(1, len(spike_thresholds_groups)):
        juxta_spikes_grouped[t] = []
        juxta_spike_peaks_grouped[t] = []
        juxta_spike_peaks_grouped_withnans[t] = np.empty(len(juxta_spike_peaks))
        juxta_spike_peaks_grouped_withnans[t][:] = np.NAN
        juxta_spike_triggers_grouped_withnans[t] = np.empty(len(juxta_spike_peaks))
        juxta_spike_triggers_grouped_withnans[t][:] = np.NAN
    for s in range(len(juxta_spike_peaks)):
        for t in range(1, len(spike_thresholds_groups)):
            if juxta_spike_peaks[s] < spike_thresholds_groups[t]:
                juxta_spikes_grouped[t].append(juxta_spike_triggers[s])
                juxta_spike_peaks_grouped[t].append(juxta_spike_peaks[s])
                break
    for t in range(1, len(spike_thresholds_groups)):
        juxta_spike_peaks_grouped_withnans[t][np.in1d(juxta_spike_peaks, juxta_spikes_grouped[t])] = \
            juxta_spike_peaks[np.in1d(juxta_spike_peaks, juxta_spikes_grouped[t])]
        juxta_spike_triggers_grouped_withnans[t][np.in1d(juxta_spike_peaks, juxta_spikes_grouped[t])] = \
            juxta_spike_peaks[np.in1d(juxta_spike_peaks, juxta_spikes_grouped[t])]

    return juxta_spikes_grouped, juxta_spike_peaks_grouped, \
        juxta_spike_triggers_grouped_withnans, juxta_spike_peaks_grouped_withnans, spike_thresholds_groups


def fit_dbscan(data, eps, min_samples, normalize=True,
               show=True, juxta_cluster_indices_grouped=None, threshold_legend=None):
    X = np.transpose(data)

    if normalize:
        from sklearn.preprocessing import minmax_scale
        minmax_scale(X, feature_range=(-1, 1), axis=0, copy=False)

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


def calculate_precision_recall_for_single_label_grouped(tsne, cluster_indices_grouped, n_clusters, labels,
                                                        core_samples_mask, show_means=False):
    # Define TP / FP / TN and FN
    X = np.transpose(tsne)
    juxta_cluster_indices = []
    num_of_spike_groups = len(cluster_indices_grouped)
    for g in range(0, num_of_spike_groups):
        juxta_cluster_indices.extend(cluster_indices_grouped[g])
    means_of_juxta = np.array([np.median(X[juxta_cluster_indices, 0]), np.median(X[juxta_cluster_indices, 1])])

    means_of_labels = np.zeros((n_clusters, 2))
    dmeans = np.zeros(n_clusters)
    for l in range(n_clusters):
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

    if show_means:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(means_of_juxta[0], means_of_juxta[1])
        ax.scatter(means_of_labels[:, 0], means_of_labels[:, 1], color='r')

    return precision, recall, f_factor


def calculate_precision_recal_for_many_labels(tsne, labels, core_samples_mask, n_clusters,
                                              db_labels, spikes_labeled_dict, show_means=False):
    #Define TP / FP / TN and FN
    X = np.transpose(tsne)
    number_of_labels = len(labels)
    means_of_labels = np.zeros((number_of_labels, 2))
    for l in range(number_of_labels):
        class_member_mask = (labels == l)
        xy = X[class_member_mask & core_samples_mask]
        means_of_labels[l, 0] = np.median(xy[:, 0])
        means_of_labels[l, 1] = np.median(xy[:, 1])

    means_of_DBclusters = np.zeros((n_clusters, 2))
    for c in range(n_clusters):
        class_member_mask = (db_labels == c)
        xy = X[class_member_mask & core_samples_mask]
        means_of_DBclusters[c, 0] = np.median(xy[:, 0])
        means_of_DBclusters[c, 1] = np.median(xy[:, 1])

    dmeans = np.zeros(n_clusters)
    labeled_cluster_db_index = []
    for l in range(number_of_labels):
        for c in range(n_clusters):
            dmeans[c] = np.linalg.norm((means_of_DBclusters[c, :]-means_of_labels[l]))
        labeled_cluster_db_index.append(np.argmin(dmeans))

    for l in range(number_of_labels):
        class_member_mask = (db_labels == labeled_cluster_db_index[l])
        db_cluster_indices = [i for i, x in enumerate(class_member_mask & core_samples_mask) if x]
        tp_indices = np.intersect1d(spikes_labeled_dict[l], db_cluster_indices)
        tp = len(tp_indices)
        all_pos = len(db_cluster_indices)
        all_true = len(spikes_labeled_dict[l])
        precision = tp / all_pos
        recall = tp / all_true
        f_factor = 2*(precision*recall)/(precision+recall)
        print('Precision = {}, Recall = {}, F1 factor = {}'.format(precision, recall, f_factor))

    if show_means:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(means_of_DBclusters[:, 0], means_of_DBclusters[:, 1])
        for i, txt in enumerate(range(n_clusters)):
            ax.annotate(str(txt), (means_of_DBclusters[i, 0], means_of_DBclusters[i, 1]), verticalalignment='top')
        ax.scatter(means_of_labels[:, 0], means_of_labels[:, 1], color='r')
        for i, txt in enumerate(range(number_of_labels)):
            ax.annotate(str(txt), (means_of_labels[i, 0], means_of_labels[i, 1]), verticalalignment='bottom')

    return precision, recall, f_factor

